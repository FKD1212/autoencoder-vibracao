# Antes de tudo: se este Python não tiver as dependências, reinicia no .venv312.
from venv_guard import ensure
ensure("tensorflow", "pandas", "sklearn", "scipy", "matplotlib")

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from scipy.stats import norm

from dashboard import Dashboard, PredictMonitor, TrainingMonitor, describe_layers

DASHBOARD_PORT = 8050

# ---------------------------------------------------------------------------
# Dashboard em tempo real (front end em ./frontend)
# ---------------------------------------------------------------------------
dash = Dashboard(port=DASHBOARD_PORT).start()
dash.plan([
    ("setup",      "Configuração"),
    ("load_train", "Carregando treino"),
    ("train",      "Treinamento"),
    ("threshold",  "Threshold"),
    ("eval_test",  "Validação interna"),
    ("load_val",   "Carregando validação"),
    ("eval_val",   "Validação externa"),
])
dash.phase("setup")


def info(msg):
    """print + log no dashboard."""
    print(msg)
    dash.log(str(msg).strip())


# ---------------------------------------------------------------------------
# GPU — habilita memory growth para evitar alocar toda a VRAM de uma vez
# ---------------------------------------------------------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        info(f"GPU(s) detectada(s): {[g.name for g in gpus]}")
    except RuntimeError as e:
        info(f"Erro ao configurar GPU: {e}")
else:
    info("Nenhuma GPU encontrada — usando CPU.")

# ---------------------------------------------------------------------------
# Caminhos — funciona no Windows e no WSL2 (/mnt/d/...)
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(_BASE, "train_df.csv")
VAL_PATH    = os.path.join(_BASE, "validation_df.csv")
GT_PATH     = os.path.join(_BASE, "dfvalid_groundtruth.csv")
MODEL_PATH  = os.path.join(_BASE, "autoencoder.keras")

EPOCHS      = 50
BATCH_SIZE  = 64
LATENT_DIM  = 8
THRESHOLD_K = 3.0   # threshold = μ + k·σ sobre os erros do treino normal
VAL_SPLIT   = 0.2   # fração do treino reservada como validação interna (early stopping + falsos positivos)

dash.set("run", {
    "device": "GPU" if gpus else "CPU",
    "gpus": [g.name for g in gpus],
    "config": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "latent_dim": LATENT_DIM,
        "threshold_k": THRESHOLD_K,
    },
})

# ---------------------------------------------------------------------------
# Carregamento e pré-processamento
# ---------------------------------------------------------------------------
def load_sequences(csv_path):
    """
    Matriz float32 (n_sequências, 61440) de um CSV do dataset.

    A 1ª coluna do CSV é só o índice da linha e não há coluna de rótulo: todas as
    outras colunas são amostras do sinal. Ler o CSV é lento (~1 min para o treino),
    então na primeira vez ele é convertido para um .npy ao lado; as próximas
    execuções carregam o .npy em ~1 s. Se o CSV mudar, o .npy é refeito.
    """
    npy_path = os.path.splitext(csv_path)[0] + ".npy"
    if os.path.exists(npy_path) and os.path.getmtime(npy_path) >= os.path.getmtime(csv_path):
        return np.load(npy_path)
    info(f"Convertendo {os.path.basename(csv_path)} para .npy (só na primeira execução)...")
    data = pd.read_csv(csv_path, index_col=0, dtype=np.float32).to_numpy()
    np.save(npy_path, data)
    return data


dash.phase("load_train")
info("Carregando dados de treino...")
# O conjunto de treino só tem sequências normais — o autoencoder aprende o "normal".
data       = load_sequences(TRAIN_PATH)
n_features = data.shape[1]

# Validação interna: sequências normais que o modelo não vê no treino. Guiam o
# early stopping e medem a taxa de falsos positivos do threshold.
X_train, X_int = train_test_split(data, test_size=VAL_SPLIT, random_state=21)
del data

# Normalização min-max por feature, calculada apenas no treino
min_val = X_train.min(axis=0, keepdims=True)
max_val = X_train.max(axis=0, keepdims=True)
den     = max_val - min_val
den[den == 0.0] = 1.0

X_train_n = (X_train - min_val) / den
X_int_n   = (X_int   - min_val) / den
del X_train, X_int

dash.set("data", {
    "n_features": n_features,
    "train":      len(X_train_n),
    "internal":   len(X_int_n),
})
info(f"{len(X_train_n) + len(X_int_n)} sequências normais · {n_features} amostras cada · "
     f"{len(X_train_n)} para treino e {len(X_int_n)} para validação interna")

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class AnomalyDetector(Model):
    def __init__(self, n_features: int, latent_dim: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(32, activation="relu"),
            layers.Dense(16, activation="relu"),
            layers.Dense(latent_dim, activation="relu"),  # relu preserva variância no bottleneck
        ])
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(16, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(64, activation="relu"),
            layers.Dense(n_features, activation="sigmoid"),
        ])

    def call(self, x, training=False):
        z     = self.encoder(x, training=training)
        x_hat = self.decoder(z, training=training)
        return x_hat

    def get_config(self):
        config = super().get_config()
        config.update({"n_features": self.n_features, "latent_dim": self.latent_dim})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

autoencoder = AnomalyDetector(n_features, LATENT_DIM)
autoencoder.compile(optimizer="adam", loss="mae")
autoencoder.encoder.summary()

dash.set("model", {
    "n_features": n_features,
    "encoder":    describe_layers(autoencoder.encoder),
    "decoder":    describe_layers(autoencoder.decoder),
    "params":     autoencoder.encoder.count_params() + autoencoder.decoder.count_params(),
})

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
)
callbacks = [
    early_stopping,
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
    TrainingMonitor(dash, early_stopping),
]

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------
dash.phase("train")
info("\nTreinando autoencoder...")
history = autoencoder.fit(
    X_train_n, X_train_n,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    shuffle=True,
    validation_data=(X_int_n, X_int_n),
    callbacks=callbacks,
    verbose=1,
)

autoencoder.save(MODEL_PATH)
info(f"\nModelo salvo em {MODEL_PATH}")

# ---------------------------------------------------------------------------
# Curva de treinamento (as figuras são salvas; o acompanhamento é no dashboard)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"],     label="Treino")
plt.plot(history.history["val_loss"], label="Validação interna")
plt.xlabel("Época")
plt.ylabel("MAE")
plt.title("Curva de Treinamento")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Threshold: μ + k·σ dos erros de reconstrução no treino normal
# ---------------------------------------------------------------------------
dash.phase("threshold")
recon_train = autoencoder.predict(
    X_train_n, verbose=0, callbacks=[PredictMonitor(dash, "Reconstruindo o treino normal")]
)
train_loss  = tf.keras.losses.mae(X_train_n, recon_train).numpy()

mu, sigma = norm.fit(train_loss)
threshold = mu + THRESHOLD_K * sigma
info(f"\nErros de treino → μ={mu:.4f}  σ={sigma:.4f}  threshold={threshold:.4f}")

x = np.linspace(train_loss.min(), train_loss.max(), 500)
hist_density, hist_edges = np.histogram(train_loss, bins=50, density=True)
dash.set("threshold", {
    "mu": mu,
    "sigma": sigma,
    "k": THRESHOLD_K,
    "value": threshold,
    "hist": {"edges": hist_edges, "values": hist_density},
    "pdf": {"x": x, "y": norm.pdf(x, mu, sigma)},
})

plt.figure(figsize=(8, 4))
plt.hist(train_loss, bins=50, density=True, alpha=0.6, label="Erros (treino normal)")
plt.plot(x, norm.pdf(x, mu, sigma), linewidth=2, label="Normal ajustada")
plt.axvline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.4f}")
plt.xlabel("MAE de reconstrução")
plt.ylabel("Densidade")
plt.title(f"Distribuição do erro  (μ={mu:.4f}, σ={sigma:.4f})")
plt.legend()
plt.tight_layout()
plt.savefig("error_distribution.png", dpi=150)
plt.close()


def publish_evaluation(key, y_true, errors, y_pred):
    """Envia ao dashboard as métricas de um conjunto avaliado."""
    both_classes = len(np.unique(y_true)) == 2   # a validação interna só tem normais
    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=["Normal", "Anomalia"],
        output_dict=True, zero_division=0,
    )
    roc = None
    if both_classes:
        fpr, tpr, _ = roc_curve(y_true, errors)
        keep = np.unique(np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int))
        roc = {"fpr": fpr[keep], "tpr": tpr[keep]}

    edges = np.histogram_bin_edges(errors, bins=60)
    hist = {"edges": edges}
    for cls, name in ((0, "normal"), (1, "anomaly")):
        class_errors = errors[y_true == cls]
        counts, _ = np.histogram(class_errors, bins=edges)
        hist[name] = counts / max(len(class_errors), 1)   # fração da classe por bin

    dash.set(key, {
        "n":               len(y_true),
        "n_anomaly":       int(y_true.sum()),
        "false_positives": int(((y_pred == 1) & (y_true == 0)).sum()),
        "auc":             roc_auc_score(y_true, errors) if both_classes else None,
        "report":          report,
        "confusion":       confusion_matrix(y_true, y_pred, labels=[0, 1]),
        "roc":             roc,
        "hist":            hist,
        "threshold":       threshold,
    })


# ---------------------------------------------------------------------------
# Validação interna — sequências normais fora do treino: mede falsos positivos
# ---------------------------------------------------------------------------
dash.phase("eval_test")
info("\n--- Validação interna (só sequências normais, fora do treino) ---")
recon_int = autoencoder.predict(
    X_int_n, verbose=0, callbacks=[PredictMonitor(dash, "Reconstruindo a validação interna")]
)
int_loss  = tf.keras.losses.mae(X_int_n, recon_int).numpy()

y_pred_int = (int_loss > threshold).astype(int)
y_true_int = np.zeros(len(int_loss), dtype=int)

info(f"{y_pred_int.sum()} de {len(y_pred_int)} sequências normais acima do threshold "
     f"→ taxa de falso positivo {y_pred_int.mean():.1%}")
publish_evaluation("eval_test", y_true_int, int_loss, y_pred_int)

# ---------------------------------------------------------------------------
# Avaliação — conjunto de validação externo, rótulos do dfvalid_groundtruth.csv
# ---------------------------------------------------------------------------
dash.phase("load_val")
info("\n--- Avaliação: validação externa (validation_df.csv) ---")
info("Carregando validation_df.csv...")
X_val = load_sequences(VAL_PATH)

# seqID = número da linha em validation_df.csv; anomaly = 1 (anômala) ou 0 (normal)
ground_truth = pd.read_csv(GT_PATH, index_col="seqID")["anomaly"]
y_val = ground_truth.reindex(np.arange(len(X_val))).to_numpy()
if np.isnan(y_val).any():
    raise ValueError("dfvalid_groundtruth.csv não tem rótulo para todas as sequências de validação")
y_val = y_val.astype(int)

X_val_n = (X_val - min_val) / den   # mesma normalização do treino
del X_val

dash.set("data", {
    "n_features":       n_features,
    "train":            len(X_train_n),
    "internal":         len(X_int_n),
    "external":         len(y_val),
    "external_anomaly": int(y_val.sum()),
})

dash.phase("eval_val")
recon_val = autoencoder.predict(
    X_val_n, batch_size=BATCH_SIZE, verbose=1,
    callbacks=[PredictMonitor(dash, "Reconstruindo a validação externa")],
)
val_loss  = tf.keras.losses.mae(X_val_n, recon_val).numpy()

y_pred_val = (val_loss > threshold).astype(int)
y_true_val = y_val

info(classification_report(y_true_val, y_pred_val, target_names=["Normal", "Anomalia"]))
auc_val = roc_auc_score(y_true_val, val_loss)
info(f"ROC-AUC (validação externa): {auc_val:.4f}")
publish_evaluation("eval_val", y_true_val, val_loss, y_pred_val)

# Curva ROC
fpr, tpr, _ = roc_curve(y_true_val, val_loss)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("Taxa de Falsos Positivos")
plt.ylabel("Taxa de Verdadeiros Positivos")
plt.title("Curva ROC — Validação Externa")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

info("\nExecução concluída.")
dash.finish()
dash.wait()
