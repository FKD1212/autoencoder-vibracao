"""
Features, modelo e perfil de calibração para detectar anomalias de vibração.

Cada janela (N amostras × C eixos) vira um vetor de features que não depende da fase
do sinal — o que permite comparar janelas quaisquer de um fluxo contínuo:
  - energia em `n_bands` faixas lineares de 0 a fs/2, em dB, por eixo
  - log10(RMS), log10(pico), fator de crista e curtose, por eixo

Calibrar = gravar a máquina em operação NORMAL, padronizar as features (z-score),
treinar um autoencoder que reconstrói essas features e fixar o threshold em
μ + k·σ do erro de reconstrução nas janelas de validação (os últimos 20 % da
gravação, que o modelo não viu no treino).
"""
import json
import os
import re
import time

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

STATS = ["log_rms", "log_pico", "crista", "curtose"]
STAT_LABELS = {"log_rms": "RMS", "log_pico": "pico", "crista": "fator de crista", "curtose": "curtose"}
# desvio-padrão mínimo por tipo de feature: evita que features quase constantes
# na calibração disparem alarmes por variações irrelevantes
STD_FLOOR = {"band": 0.25, "log_rms": 0.005, "log_pico": 0.005, "crista": 0.02, "curtose": 0.05}

PROFILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrations")


def safe_name(name):
    name = re.sub(r"[^\w\-]+", "_", str(name or "").strip()).strip("_")
    return name[:60]


class FeatureExtractor:
    def __init__(self, fs, window, n_bands=64):
        self.fs = float(fs)
        self.window = int(window)
        self.n_bands = int(n_bands)
        n_bins = self.window // 2 + 1
        if n_bins < self.n_bands:
            raise ValueError(f"janela de {self.window} amostras é curta para {self.n_bands} bandas")
        self._hann = np.hanning(self.window).astype(np.float32)[:, None]
        edges = np.linspace(0, n_bins, self.n_bands + 1).astype(int)
        self._starts = edges[:-1]
        self.edges_hz = (edges * self.fs / self.window).tolist()
        self.edges_hz[-1] = self.fs / 2

    def config(self):
        return {"fs": self.fs, "window": self.window, "n_bands": self.n_bands}

    def __call__(self, x):
        """x: (window, C) -> (features, bandas em dB (C, B), rms (C,), pico (C,))"""
        x = x - x.mean(axis=0)
        power = np.abs(np.fft.rfft(x * self._hann, axis=0)) ** 2
        band_db = 10 * np.log10(np.add.reduceat(power, self._starts, axis=0) + 1e-12).T
        rms = np.sqrt(np.mean(x ** 2, axis=0)) + 1e-12
        peak = np.max(np.abs(x), axis=0) + 1e-12
        stats = np.stack([np.log10(rms), np.log10(peak), peak / rms, np.mean(x ** 4, axis=0) / rms ** 4], axis=1)
        feats = np.concatenate([band_db.ravel(), stats.ravel()]).astype(np.float32)
        return feats, band_db, rms, peak


def build_autoencoder(n_features, latent_dim=8):
    model = tf.keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(latent_dim),                 # gargalo
        layers.Dense(32, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_features),                 # saída linear: features em z-score
    ])
    model.compile(optimizer="adam", loss="mae")
    return model


def _errors(model, z):
    return np.mean(np.abs(z - model(z, training=False).numpy()), axis=1)


class Profile:
    """Uma calibração: extrator + padronização + autoencoder + threshold."""

    def __init__(self, name, config, channels, mean, std, model, train_errors, val_errors, k,
                 baseline_mean, baseline_std, meta):
        self.name = name
        self.config = config
        self.channels = list(channels)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.model = model
        self.train_errors = np.asarray(train_errors, dtype=np.float64)
        self.val_errors = np.asarray(val_errors, dtype=np.float64)
        self.k = float(k)
        self.baseline_mean = np.asarray(baseline_mean)
        self.baseline_std = np.asarray(baseline_std)
        self.meta = meta
        self.extractor = FeatureExtractor(**config)

    @property
    def mu(self):
        return float(self.val_errors.mean())

    @property
    def sigma(self):
        return float(self.val_errors.std())

    @property
    def threshold(self):
        return self.mu + self.k * self.sigma

    def false_alarm_rate(self):
        """Fração das janelas normais de validação acima do threshold."""
        return float(np.mean(self.val_errors > self.threshold))

    def compatibility(self, fs, n_channels):
        """None se a fonte serve para este perfil; senão, o motivo."""
        if n_channels != len(self.channels):
            return f"o perfil tem {len(self.channels)} eixos e a fonte envia {n_channels}"
        if not fs or abs(fs - self.config["fs"]) / self.config["fs"] > 0.03:
            return f"o perfil foi calibrado a {self.config['fs']:.0f} Hz e a fonte está a {fs or 0:.0f} Hz"
        return None

    def feature_label(self, i):
        c_count, bands = len(self.channels), self.config["n_bands"]
        edges = self.extractor.edges_hz
        if i < c_count * bands:
            c, b = divmod(i, bands)
            return f"{self.channels[c].upper()} · {edges[b]:.0f}–{edges[b + 1]:.0f} Hz"
        c, s = divmod(i - c_count * bands, len(STATS))
        return f"{self.channels[c].upper()} · {STAT_LABELS[STATS[s]]}"

    def score(self, feats, top=3):
        z = (feats - self.mean) / self.std
        rec = self.model(z[None], training=False).numpy()[0]
        diff = np.abs(z - rec)
        order = np.argsort(diff)[::-1][:top]
        contributions = [{"label": self.feature_label(int(i)), "z": float(z[i]), "err": float(diff[i])}
                         for i in order]
        return float(diff.mean()), contributions

    # ------------------------------------------------------------------ disco
    def save(self, root=PROFILES_DIR):
        folder = os.path.join(root, self.name)
        os.makedirs(folder, exist_ok=True)
        self.model.save(os.path.join(folder, "model.keras"))
        data = {
            "name": self.name, "config": self.config, "channels": self.channels,
            "mean": self.mean.tolist(), "std": self.std.tolist(), "k": self.k,
            "train_errors": self.train_errors.tolist(), "val_errors": self.val_errors.tolist(),
            "baseline_mean": self.baseline_mean.tolist(), "baseline_std": self.baseline_std.tolist(),
            "meta": self.meta,
        }
        with open(os.path.join(folder, "profile.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        return folder

    @classmethod
    def load(cls, name, root=PROFILES_DIR):
        folder = os.path.join(root, safe_name(name))
        with open(os.path.join(folder, "profile.json"), encoding="utf-8") as fh:
            d = json.load(fh)
        model = tf.keras.models.load_model(os.path.join(folder, "model.keras"))
        return cls(d["name"], d["config"], d["channels"], d["mean"], d["std"], model,
                   d["train_errors"], d["val_errors"], d["k"], d["baseline_mean"], d["baseline_std"], d["meta"])

    def summary(self):
        """Resumo para a interface."""
        errors = np.concatenate([self.train_errors, self.val_errors])
        edges = np.histogram_bin_edges(errors, bins=40)
        return {
            "name": self.name, "channels": self.channels, **self.config,
            "edges_hz": self.extractor.edges_hz,
            "k": self.k, "mu": self.mu, "sigma": self.sigma, "threshold": self.threshold,
            "false_alarm_rate": self.false_alarm_rate(),
            "n_train": len(self.train_errors), "n_val": len(self.val_errors),
            "baseline": {"mean_db": self.baseline_mean, "std_db": self.baseline_std},
            "hist": {
                "edges": edges,
                "train": np.histogram(self.train_errors, bins=edges)[0] / max(len(self.train_errors), 1),
                "val": np.histogram(self.val_errors, bins=edges)[0] / max(len(self.val_errors), 1),
            },
            "meta": self.meta,
        }


def list_profiles(root=PROFILES_DIR):
    if not os.path.isdir(root):
        return []
    out = []
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "profile.json")
        if os.path.isfile(path):
            out.append({"name": name, "modified": os.path.getmtime(path)})
    return sorted(out, key=lambda p: p["modified"], reverse=True)


def train_profile(name, feats, bands_db, extractor, channels, k=3.0, callbacks=(), meta=None):
    """
    feats: (n_janelas, n_features) gravadas em operação normal, em ordem temporal.
    bands_db: (n_janelas, C, B) — espectro de referência mostrado na interface.
    """
    n = len(feats)
    split = int(n * 0.8)
    if split < 20 or n - split < 8:
        raise ValueError(f"poucas janelas para calibrar ({n}); grave por mais tempo")

    train = feats[:split]
    mean = train.mean(axis=0)
    n_band_feats = len(channels) * extractor.n_bands
    floor = np.concatenate([np.full(n_band_feats, STD_FLOOR["band"]),
                            np.tile([STD_FLOOR[s] for s in STATS], len(channels))])
    std = np.maximum(train.std(axis=0), floor)
    z = ((feats - mean) / std).astype(np.float32)

    model = build_autoencoder(z.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    model.fit(
        z[:split], z[:split],
        validation_data=(z[split:], z[split:]),
        epochs=400, batch_size=32, shuffle=True, verbose=0,
        callbacks=[early_stopping, *[cb(early_stopping) for cb in callbacks]],
    )

    return Profile(
        name, extractor.config(), channels, mean, std, model,
        _errors(model, z[:split]), _errors(model, z[split:]), k,
        bands_db.mean(axis=0), bands_db.std(axis=0),
        {"created_at": time.time(), "n_windows": n, **(meta or {})},
    )
