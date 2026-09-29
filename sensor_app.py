"""
Monitor de vibração em tempo real com calibração por autoencoder.

    python sensor_app.py                     # abre a interface; escolha a fonte nela
    python sensor_app.py --sim               # conecta direto no simulador
    python sensor_app.py --serial COM3       # conecta direto na placa (921600 baud)
    python sensor_app.py --profile motor1    # carrega uma calibração salva

Interface: http://localhost:8060/sensor.html

Fluxo: conectar -> conferir o sinal -> calibrar com a máquina em operação NORMAL
-> monitorar. Calibrações ficam em ./calibrations e gravações em ./recordings.

Detecção: a cada 0,5 s uma janela de 1 s vira features (vibration.py) e o
autoencoder da calibração mede o erro de reconstrução. score = erro / threshold.
Uma janela acima do threshold é "alerta"; ALARM_MIN das últimas ALARM_WINDOW
janelas acima é "anomalia" — a persistência evita alarme por um pico isolado.
"""
# Antes de tudo: se este Python não tiver as dependências, reinicia no .venv312.
from venv_guard import ensure
ensure("tensorflow", "numpy", "serial")

import argparse
import os
import threading
import time
import traceback
from collections import deque
from datetime import datetime

import numpy as np

from dashboard import Dashboard, TrainingMonitor
from sensor_sources import FileSource, SerialSource, SimulatedSource, list_serial_ports
from vibration import FeatureExtractor, Profile, list_profiles, safe_name, train_profile

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")

WINDOW_S = 1.0        # duração de cada janela analisada
HOP_S = 0.5           # uma janela nova a cada 0,5 s (50 % de sobreposição)
N_BANDS = 64          # faixas de frequência de 0 a fs/2
WAVE_POINTS = 512     # amostras mostradas no gráfico de sinal
ALARM_WINDOW = 5
ALARM_MIN = 3
MIN_CALIBRATION_S = 15


class RingBuffer:
    def __init__(self, capacity, channels):
        self.data = np.zeros((capacity, channels), dtype=np.float32)
        self.channels = channels
        self.total = 0
        self._lock = threading.Lock()

    def push(self, chunk):
        cap = len(self.data)
        with self._lock:
            if len(chunk) > cap:
                self.total += len(chunk) - cap
                chunk = chunk[-cap:]
            self.data[(self.total + np.arange(len(chunk))) % cap] = chunk
            self.total += len(chunk)

    def latest(self, n):
        with self._lock:
            n = min(n, self.total, len(self.data))
            return self.data[(self.total - n + np.arange(n)) % len(self.data)].copy(), self.total


class RateMeter:
    """Amostras por segundo que realmente chegam (média dos últimos segundos)."""

    def __init__(self, span=3.0):
        self.span = span
        self._events = deque()
        self._lock = threading.Lock()

    def add(self, n):
        now = time.monotonic()
        with self._lock:
            self._events.append((now, n))
            while self._events and now - self._events[0][0] > self.span:
                self._events.popleft()

    def rate(self):
        with self._lock:
            if len(self._events) < 2:
                return 0.0
            dt = self._events[-1][0] - self._events[0][0]
            total = sum(n for _, n in self._events) - self._events[0][1]
        return total / dt if dt > 0.5 else 0.0

    def reset(self):
        with self._lock:
            self._events.clear()


class Recorder:
    """Grava as amostras brutas em CSV (mesmo formato que o FileSource lê)."""

    def __init__(self, fs, channels, units):
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        self.path = os.path.join(RECORDINGS_DIR, datetime.now().strftime("gravacao-%Y%m%d-%H%M%S.csv"))
        self.fs = fs
        self.samples = 0
        self._lock = threading.Lock()
        self._fh = open(self.path, "w", encoding="utf-8", newline="\n")
        self._fh.write(f"# fs={fs:g} channels={','.join(channels)} units={units or 'raw'}\n")

    def write(self, chunk):
        with self._lock:
            if self._fh is None:
                return
            np.savetxt(self._fh, chunk, delimiter=",", fmt="%.6g")
            self.samples += len(chunk)

    def close(self):
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def list_recordings():
    if not os.path.isdir(RECORDINGS_DIR):
        return []
    files = [f for f in os.listdir(RECORDINGS_DIR) if f.lower().endswith(".csv")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(RECORDINGS_DIR, f)), reverse=True)
    return [{"name": f, "size": os.path.getsize(os.path.join(RECORDINGS_DIR, f))} for f in files]


class VibrationMonitor:
    def __init__(self, dash):
        self.dash = dash
        self.lock = threading.RLock()
        self.source = None
        self.ring = None
        self.rate = RateMeter()
        self.recorder = None
        self.extractor = None
        self.profile = None
        self.calib = None
        self.mode = "idle"          # idle | preview | calibrating | training | monitoring
        self.alarm = "normal"
        self.recent = deque(maxlen=ALARM_WINDOW)
        self.events = []
        self.last_error = None
        self._incompatible = None
        self._last_end = 0
        self._last_wave = 0.0
        self._last_acq = 0.0
        self.dash.set("mode", self.mode)
        self.dash.set("alarm_rule", {"window": ALARM_WINDOW, "min": ALARM_MIN, "window_s": WINDOW_S, "hop_s": HOP_S})
        threading.Thread(target=self._loop, name="processamento", daemon=True).start()

    # ------------------------------------------------------------------ comandos
    def handle(self, payload):
        action = str(payload.get("action", ""))
        fn = getattr(self, f"cmd_{action}", None)
        if fn is None:
            raise ValueError(f"ação desconhecida: {action!r}")
        params = {k: v for k, v in payload.items() if k != "action"}
        return {"ok": True, **(fn(**params) or {})}

    def cmd_options(self):
        options = {
            "ports": list_serial_ports(),
            "recordings": list_recordings(),
            "profiles": list_profiles(),
            "faults": SimulatedSource.FAULTS,
        }
        self.dash.set("options", options)
        return options

    def cmd_connect(self, kind, port=None, baud=921600, fs=None, file=None):
        if kind == "sim":
            source = SimulatedSource()
        elif kind == "serial":
            if list_serial_ports() is None:
                raise RuntimeError("pyserial não instalado — rode: pip install pyserial")
            if not port:
                raise ValueError("escolha a porta serial")
            source = SerialSource(str(port), baud=int(baud), fs=float(fs) if fs else None)
        elif kind == "file":
            path = os.path.join(RECORDINGS_DIR, os.path.basename(str(file or "")))
            if not os.path.isfile(path):
                raise ValueError("gravação não encontrada")
            source = FileSource(path)
        else:
            raise ValueError(f"fonte desconhecida: {kind!r}")

        self.cmd_disconnect(quiet=True)
        with self.lock:
            self.source = source
            self.ring = None
            self.extractor = None
            self.last_error = None
            self._last_end = 0
            self._reset_detection()
            if self.mode != "training":               # o treino em andamento define o modo ao terminar
                self._set_mode("monitoring" if self.profile else "preview")
        self.rate.reset()
        source.start(lambda chunk: self._on_samples(source, chunk))
        self.dash.set("scores", [])
        self.dash.log(f"Conectado: {source.label}")
        self._publish_acq()

    def cmd_disconnect(self, quiet=False):
        with self.lock:
            source, self.source = self.source, None
            self.ring = None
            if self.mode == "calibrating":
                self._publish_calib(self.calib, "cancelled")
                self.calib = None
            if self.mode != "training":
                self._set_mode("idle")
        self._stop_recording()
        if source is not None:
            source.stop()
            if not quiet:
                self.dash.log(f"Desconectado: {source.label}")
        self._publish_acq()

    def cmd_record(self, on=True):
        if not on:
            self._stop_recording()
            return
        with self.lock:
            if self.source is None:
                raise ValueError("conecte uma fonte antes de gravar")
            if self.recorder is not None:
                return
            fs = self._fs()
            if not fs:
                raise ValueError("aguardando dados para saber a taxa de amostragem")
            self.recorder = Recorder(fs, self.source.channels, self.source.units)
        self.dash.log(f"Gravando em recordings/{os.path.basename(self.recorder.path)}")
        self._publish_acq()

    def cmd_calibrate(self, seconds=120, k=3.0, name=None):
        seconds = float(seconds)
        if seconds < MIN_CALIBRATION_S:
            raise ValueError(f"grave pelo menos {MIN_CALIBRATION_S} s de operação normal")
        with self.lock:
            if self.source is None:
                raise ValueError("conecte uma fonte antes de calibrar")
            if self.mode in ("calibrating", "training"):
                raise ValueError("já há uma calibração em andamento")
            fs = self._fs()
            if not fs:
                raise ValueError("aguardando dados para saber a taxa de amostragem")
            self.calib = {
                "name": safe_name(name) or datetime.now().strftime("calib-%Y%m%d-%H%M%S"),
                "k": float(k),
                "seconds": seconds,
                "target": int(seconds / HOP_S),
                "extractor": FeatureExtractor(fs, int(round(fs * WINDOW_S)), N_BANDS),
                "feats": [],
                "bands": [],
                "source": self.source.label,
                "units": self.source.units,
                "channels": list(self.source.channels),
            }
            self._set_mode("calibrating")
            calib = self.calib
        self.dash.set("epochs", [])
        self.dash.set("training", None)
        self._publish_calib(calib, "collecting")
        self.dash.log(f"Calibração '{calib['name']}': gravando {seconds:.0f} s de operação normal…")

    def cmd_cancel(self):
        with self.lock:
            if self.mode != "calibrating":
                raise ValueError("não há coleta de calibração em andamento")
            calib, self.calib = self.calib, None
            self._set_mode(self._idle_mode())
        self._publish_calib(calib, "cancelled")
        self.dash.log("Calibração cancelada.")

    def cmd_set_k(self, k):
        with self.lock:
            if self.profile is None:
                raise ValueError("nenhuma calibração carregada")
            self.profile.k = float(np.clip(float(k), 0.5, 10.0))
            self.profile.save()
            self._reset_detection()
        self._publish_profile()
        p = self.profile
        self.dash.log(f"k = {p.k:.1f} → threshold {p.threshold:.4f} "
                      f"({p.false_alarm_rate():.1%} das janelas normais de validação acima)")

    def cmd_load_profile(self, name):
        if safe_name(name) not in {p["name"] for p in list_profiles()}:
            raise ValueError(f"calibração não encontrada: {name}")
        profile = Profile.load(name)
        with self.lock:
            self.profile = profile
            self._incompatible = None
            self._reset_detection()
            if self.mode in ("preview", "monitoring"):
                self._set_mode("monitoring")
        self._publish_profile()
        self.dash.log(f"Calibração carregada: {profile.name} (threshold {profile.threshold:.4f})")

    def cmd_sim(self, fault, level):
        source = self.source
        if not isinstance(source, SimulatedSource):
            raise ValueError("a fonte atual não é o simulador")
        source.set_fault(fault, float(level))
        self._publish_acq()

    # ------------------------------------------------------------------ aquisição
    def _on_samples(self, source, chunk):
        if source is not self.source:
            return                                  # fonte antiga ainda terminando
        ring = self.ring
        if ring is None or ring.channels != chunk.shape[1]:
            ring = self.ring = RingBuffer(64000, chunk.shape[1])
            self._last_end = 0
        ring.push(chunk)
        self.rate.add(len(chunk))
        recorder = self.recorder
        if recorder is not None:
            recorder.write(chunk)

    def _stop_recording(self):
        with self.lock:
            recorder, self.recorder = self.recorder, None
        if recorder is None:
            return
        recorder.close()
        self.dash.log(f"Gravação salva: recordings/{os.path.basename(recorder.path)} "
                      f"({recorder.samples / recorder.fs:.0f} s)")
        self.cmd_options()
        self._publish_acq()

    def _fs(self):
        source = self.source
        if source is None:
            return 0.0
        if source.fs:
            return source.fs
        measured = self.rate.rate()                 # sem cabeçalho: taxa medida, em passos de 10 Hz
        return float(round(measured / 10) * 10) if measured > 0 else 0.0

    # ------------------------------------------------------------------ processamento
    def _loop(self):
        while True:
            time.sleep(0.05)
            try:
                self._tick()
            except Exception:
                self.dash.log(traceback.format_exc(limit=4), level="error")
                time.sleep(1)

    def _tick(self):
        now = time.time()
        if now - self._last_acq >= 1.0:
            self._last_acq = now
            self._publish_acq()

        source = self.source
        while source is not None and source.messages:
            self.dash.log(f"{source.label}: {source.messages.popleft()}")
        ring = self.ring
        if source is None or ring is None:
            return
        if source.error:
            self.last_error = f"{source.label}: {source.error}"
            self.dash.log(f"Falha na fonte — {self.last_error}", level="error")
            self.cmd_disconnect(quiet=True)
            return
        if now - self._last_wave >= 0.2:
            self._last_wave = now
            self._publish_wave(ring, source)

        fs = self._fs()
        if not fs:
            return
        extractor = self._extractor(fs, ring.channels)
        if ring.total < extractor.window or ring.total - self._last_end < int(round(fs * HOP_S)):
            return
        x, self._last_end = ring.latest(extractor.window)
        self._process(x, extractor, source, now)

    def _extractor(self, fs, n_channels):
        """Calibrando: o extrator fixo da coleta. Monitorando: o do perfil (se compatível)."""
        with self.lock:
            if self.mode == "calibrating" and self.calib:
                return self.calib["extractor"]
            if self.mode == "monitoring" and self.profile:
                reason = self.profile.compatibility(fs, n_channels)
                if reason != self._incompatible:
                    self._incompatible = reason
                    if reason:
                        self.dash.log(f"Calibração incompatível com a fonte: {reason}", level="error")
                    self._publish_profile()
                if reason is None:
                    return self.profile.extractor
            if self.extractor is None or abs(self.extractor.fs - fs) > 0.5:
                self.extractor = FeatureExtractor(fs, int(round(fs * WINDOW_S)), N_BANDS)
            return self.extractor

    def _process(self, x, extractor, source, now):
        feats, band_db, rms, peak = extractor(x)
        self.dash.set("spectrum", {
            "t": now, "edges_hz": extractor.edges_hz, "channels": source.channels,
            "db": np.round(band_db, 2), "rms": rms, "peak": peak,
        })
        with self.lock:
            mode, calib, profile = self.mode, self.calib, self.profile
        if mode == "calibrating" and calib is not None and extractor is calib["extractor"]:
            self._collect(calib, feats, band_db)
        elif mode == "monitoring" and profile is not None and self._incompatible is None:
            self._detect(profile, feats, now)

    def _collect(self, calib, feats, band_db):
        with self.lock:
            if self.calib is not calib:
                return
            calib["feats"].append(feats)
            calib["bands"].append(band_db)
            done = len(calib["feats"]) >= calib["target"]
            if done:
                self._set_mode("training")
        self._publish_calib(calib, "training" if done else "collecting")
        if done:
            threading.Thread(target=self._train, args=(calib,), name="calibracao", daemon=True).start()

    def _detect(self, profile, feats, now):
        err, top = profile.score(feats)
        score = err / profile.threshold
        above = score > 1.0
        with self.lock:
            self.recent.append(above)
            n_above = sum(self.recent)
            alarm = "anomalia" if n_above >= ALARM_MIN else ("alerta" if above else "normal")
            self._update_events(alarm, score, top, now)
            self.alarm = alarm
        self.dash.append("scores", {"t": now, "score": score, "alarm": alarm}, limit=1200)
        self.dash.set("detection", {
            "t": now, "score": score, "err": err, "threshold": profile.threshold,
            "alarm": alarm, "above": n_above, "top": top,
        })

    def _update_events(self, alarm, score, top, now):
        if alarm == "anomalia" and self.alarm != "anomalia":
            self.events.append({"start": now, "end": None, "peak": score, "top": top})
            self.dash.log(f"ANOMALIA — score {score:.2f} · " + " | ".join(c["label"] for c in top), level="alarm")
        elif alarm == "anomalia":
            event = self.events[-1]
            if score > event["peak"]:
                event["peak"], event["top"] = score, top
        elif self.alarm == "anomalia":
            event = self.events[-1]
            event["end"] = now
            self.dash.log(f"Voltou ao normal após {now - event['start']:.0f} s de anomalia")
        else:
            return
        self.events = self.events[-50:]
        self.dash.set("events", self.events)

    def _reset_detection(self):
        self.recent.clear()
        self.alarm = "normal"
        self.dash.set("detection", None)

    def _train(self, calib):
        n = len(calib["feats"])
        self.dash.log(f"Treinando o autoencoder com {n} janelas de operação normal…")
        try:
            profile = train_profile(
                calib["name"], np.asarray(calib["feats"]), np.asarray(calib["bands"]),
                calib["extractor"], calib["channels"], k=calib["k"],
                callbacks=[lambda es: TrainingMonitor(self.dash, es, log_every=50)],
                meta={"source": calib["source"], "units": calib["units"], "seconds": calib["seconds"]},
            )
            profile.save()
        except Exception as exc:
            with self.lock:
                self.calib = None
                self._set_mode(self._idle_mode())
            self._publish_calib(calib, "failed", error=f"{type(exc).__name__}: {exc}")
            self.dash.log(f"Falha na calibração: {exc}", level="error")
            return

        with self.lock:
            self.profile = profile
            self.calib = None
            self._incompatible = None
            self._reset_detection()
            self._set_mode("monitoring" if self.source else "idle")
        self._publish_calib(calib, "done")
        self._publish_profile()
        self.cmd_options()
        self.dash.log(
            f"Calibração '{profile.name}' pronta: threshold {profile.threshold:.4f} "
            f"(μ {profile.mu:.4f} + {profile.k:.1f}·σ {profile.sigma:.4f}); "
            f"{profile.false_alarm_rate():.1%} das janelas normais de validação acima dele."
        )

    # ------------------------------------------------------------------ publicação
    def _idle_mode(self):
        if self.source is None:
            return "idle"
        return "monitoring" if self.profile else "preview"

    def _set_mode(self, mode):
        self.mode = mode
        self.dash.set("mode", mode)

    def _publish_acq(self):
        source, ring, recorder = self.source, self.ring, self.recorder
        acq = {"connected": source is not None, "last_error": self.last_error}
        if source is not None:
            acq.update(source.describe())
            acq.update({
                "fs_measured": self.rate.rate(),
                "fs_used": self._fs(),
                "samples": ring.total if ring else 0,
                "recording": ({"file": os.path.basename(recorder.path), "seconds": recorder.samples / recorder.fs}
                              if recorder else None),
            })
        self.dash.set("acq", acq)

    def _publish_wave(self, ring, source):
        data, _ = ring.latest(WAVE_POINTS)
        if len(data):
            data = data - data.mean(axis=0)
            self.dash.set("wave", {"fs": self._fs(), "channels": source.channels, "data": np.round(data.T, 5)})

    def _publish_calib(self, calib, state, error=None):
        if calib is None:
            return
        self.dash.set("calib", {
            "state": state, "name": calib["name"], "k": calib["k"], "seconds": calib["seconds"],
            "collected": len(calib["feats"]), "target": calib["target"], "error": error,
        })

    def _publish_profile(self):
        profile = self.profile
        if profile is None:
            self.dash.set("profile", None)
            return
        summary = profile.summary()
        summary["val_errors"] = profile.val_errors
        summary["incompatible"] = self._incompatible
        self.dash.set("profile", summary)


def main():
    parser = argparse.ArgumentParser(description="Monitor de vibração com calibração por autoencoder")
    parser.add_argument("--sim", action="store_true", help="conecta no simulador ao iniciar")
    parser.add_argument("--serial", metavar="PORTA", help="conecta na porta serial ao iniciar (ex.: COM3, /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=921600, help="baud rate da serial (padrão 921600)")
    parser.add_argument("--fs", type=float, help="taxa de amostragem da placa, se ela não enviar '# fs=...'")
    parser.add_argument("--profile", help="nome de uma calibração salva em ./calibrations")
    parser.add_argument("--port", type=int, default=8060, help="porta HTTP da interface (padrão 8060)")
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 para acessar de outra máquina/WSL")
    parser.add_argument("--no-browser", action="store_true", help="não abre o navegador automaticamente")
    args = parser.parse_args()

    dash = Dashboard(host=args.host, port=args.port, page="sensor.html", open_browser=not args.no_browser)
    monitor = VibrationMonitor(dash)
    dash.on_command = monitor.handle
    monitor.cmd_options()
    if args.profile:
        monitor.cmd_load_profile(args.profile)
    if args.sim:
        monitor.cmd_connect("sim")
    elif args.serial:
        monitor.cmd_connect("serial", port=args.serial, baud=args.baud, fs=args.fs)
    dash.start()
    dash.wait()


if __name__ == "__main__":
    main()
