"""
Dashboard em tempo real do treinamento.

Sobe um servidor HTTP (somente biblioteca padrão) em uma thread que:
  - serve o front end estático de ./frontend
  - expõe /events  (Server-Sent Events) com cada atualização de métrica
  - expõe /api/state com o estado completo em JSON
  - aceita POST /api/command (JSON) repassado a `on_command`, se houver

O estado é um dicionário. Toda atualização é uma operação "set" (substitui
uma chave) ou "append" (acrescenta a uma lista) aplicada aqui e repassada,
igual, aos navegadores conectados — que aplicam a mesma operação no seu estado.
Quem conecta no meio da execução recebe primeiro um "snapshot" completo.
"""
import functools
import json
import math
import os
import queue
import sys
import threading
import time
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import tensorflow as tf

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")


def _clean(obj):
    """Converte numpy/tensores para tipos JSON e NaN/inf para None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    if hasattr(obj, "numpy"):
        return _clean(obj.numpy())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _num(value):
    """float ou None — aceita float, numpy e tensores."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _apply(state, op):
    if op["op"] == "set":
        state[op["key"]] = op["value"]
    elif op["op"] == "append":
        items = state.setdefault(op["key"], [])
        items.append(op["value"])
        limit = op.get("limit")
        if limit and len(items) > limit:
            del items[: len(items) - limit]


class Dashboard:
    def __init__(self, host="127.0.0.1", port=8050, open_browser=True, max_logs=500,
                 page="", on_command=None):
        self.host = host
        self.port = port
        self.open_browser = open_browser
        self.max_logs = max_logs
        self.page = page              # página aberta no navegador (ex.: "sensor.html")
        self.on_command = on_command  # função(dict) -> dict chamada em POST /api/command
        self._lock = threading.Lock()
        self._clients = set()
        self._server = None
        self._state = {
            "started_at": time.time(),
            "status": {"state": "running"},
            "plan": [],
            "phases": [],
            "epochs": [],
            "logs": [],
        }

    @property
    def url(self):
        host = "localhost" if self.host in ("127.0.0.1", "0.0.0.0") else self.host
        return f"http://{host}:{self.port}/{self.page}"

    # ------------------------------------------------------------------ servidor
    def start(self):
        handler = functools.partial(_Handler, dashboard=self)
        for port in range(self.port, self.port + 20):
            try:
                self._server = _Server((self.host, port), handler)
                break
            except OSError:
                continue
        else:
            raise OSError(f"Nenhuma porta livre entre {self.port} e {self.port + 19}")
        self.port = port

        threading.Thread(target=self._server.serve_forever, name="dashboard", daemon=True).start()
        print(f"Dashboard em tempo real: {self.url}")
        self._install_excepthook()

        # No WSL o navegador fica no Windows — basta abrir a URL manualmente.
        if self.open_browser and sys.platform in ("win32", "darwin"):
            try:
                webbrowser.open(self.url)
            except Exception:
                pass
        return self

    def wait(self):
        """Mantém o dashboard no ar até Ctrl+C."""
        print(f"\nDashboard segue disponível em {self.url} — Ctrl+C para encerrar.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        with self._lock:
            for q in self._clients:
                q.put(None)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def _install_excepthook(self):
        previous = sys.excepthook

        def hook(exc_type, exc, tb):
            previous(exc_type, exc, tb)
            if issubclass(exc_type, KeyboardInterrupt):
                return
            message = f"{exc_type.__name__}: {exc}"
            self.log(message, level="error")
            self.finish("error", error=message)
            self.wait()

        sys.excepthook = hook

    # ------------------------------------------------------------------ estado
    def set(self, key, value):
        self._emit({"op": "set", "key": key, "value": _clean(value)})

    def append(self, key, value, limit=None):
        self._emit({"op": "append", "key": key, "value": _clean(value), "limit": limit})

    def log(self, message, level="info"):
        self.append("logs", {"t": time.time(), "level": level, "msg": str(message)}, limit=self.max_logs)

    def plan(self, steps):
        """Etapas previstas da execução: lista de (id, rótulo)."""
        self.set("plan", [{"id": step_id, "label": label} for step_id, label in steps])

    def phase(self, phase_id):
        """Encerra a etapa atual e inicia `phase_id`."""
        phases = self._close_phase()
        phases.append({"id": phase_id, "started_at": time.time(), "ended_at": None})
        self.set("phases", phases)
        self.set("progress", None)

    def finish(self, state="done", error=None):
        with self._lock:
            phases = self._state["phases"]
            current = phases[-1]["id"] if phases else None
        self.set("phases", self._close_phase())
        self.set("progress", None)
        self.set("status", {"state": state, "error": error, "phase": current, "ended_at": time.time()})

    def _close_phase(self):
        with self._lock:
            phases = [dict(p) for p in self._state["phases"]]
        if phases and phases[-1]["ended_at"] is None:
            phases[-1]["ended_at"] = time.time()
        return phases

    def _emit(self, op):
        with self._lock:
            _apply(self._state, op)
            message = json.dumps(op, separators=(",", ":"))
            for q in self._clients:
                q.put(message)

    def _subscribe(self):
        q = queue.Queue()
        with self._lock:
            snapshot = dict(self._state, server_time=time.time())
            message = json.dumps({"op": "snapshot", "value": snapshot}, separators=(",", ":"))
            self._clients.add(q)
        return q, message

    def _unsubscribe(self, q):
        with self._lock:
            self._clients.discard(q)

    def snapshot(self):
        with self._lock:
            return json.dumps(dict(self._state, server_time=time.time()))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # No Windows, SO_REUSEADDR deixa dois processos escutarem a mesma porta.
    allow_reuse_address = sys.platform != "win32"


class _Handler(SimpleHTTPRequestHandler):
    # O registro do Windows às vezes mapeia .js para text/plain.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".html": "text/html",
        ".css": "text/css",
        ".js": "text/javascript",
        ".json": "application/json",
        ".svg": "image/svg+xml",
    }

    def __init__(self, *args, dashboard, **kwargs):
        self.dashboard = dashboard
        super().__init__(*args, directory=FRONTEND_DIR, **kwargs)

    def do_GET(self):
        route = self.path.split("?", 1)[0]
        if route == "/events":
            return self._events()
        if route == "/api/state":
            body = self.dashboard.snapshot().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return None
        return super().do_GET()

    def do_POST(self):
        if self.path.split("?", 1)[0] != "/api/command" or self.dashboard.on_command is None:
            return self.send_error(404)
        # Só JSON e só da própria origem: impede que outro site no navegador dispare comandos.
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            return self.send_error(415)
        origin = self.headers.get("Origin")
        if origin and origin.split("://", 1)[-1] != self.headers.get("Host"):
            return self.send_error(403)

        try:
            length = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(length) or b"{}")
            result, status = self.dashboard.on_command(payload), 200
        except Exception as exc:
            result, status = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, 400
        body = json.dumps(_clean(result)).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        return None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _events(self):
        q, first = self.dashboard._subscribe()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.end_headers()
            self._send(first)
            while True:
                try:
                    message = q.get(timeout=15)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")  # mantém a conexão viva
                    self.wfile.flush()
                    continue
                if message is None:
                    break
                self._send(message)
        except OSError:
            pass  # aba fechada / conexão reiniciada
        finally:
            self.dashboard._unsubscribe(q)
            self.close_connection = True

    def _send(self, message):
        self.wfile.write(f"data: {message}\n\n".encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format, *args):
        pass  # não polui o console do treino


# ---------------------------------------------------------------------------
# Callbacks Keras
# ---------------------------------------------------------------------------
class TrainingMonitor(tf.keras.callbacks.Callback):
    """Publica o progresso por batch (limitado a `interval` s) e as métricas de cada época."""

    def __init__(self, dashboard, early_stopping=None, interval=0.25, log_every=1):
        super().__init__()
        self.dash = dashboard
        self.early_stopping = early_stopping
        self.interval = interval
        self.log_every = log_every

    def on_train_begin(self, logs=None):
        self._epochs = self.params.get("epochs")
        self._steps = self.params.get("steps")
        self._best = None
        self._best_epoch = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch = epoch + 1
        self._t0 = self._last = time.time()
        self._lr = self._current_lr()
        self._progress(0, None)

    def on_train_batch_end(self, batch, logs=None):
        now = time.time()
        if now - self._last < self.interval:
            return
        self._last = now
        self._progress(batch + 1, (logs or {}).get("loss"))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = _num(logs.get("loss"))
        val_loss = _num(logs.get("val_loss"))
        if val_loss is not None and (self._best is None or val_loss < self._best):
            self._best, self._best_epoch = val_loss, epoch + 1

        self.dash.append("epochs", {
            "epoch": epoch + 1,
            "loss": loss,
            "val_loss": val_loss,
            "lr": self._lr,
            "duration": time.time() - self._t0,
        })
        self._publish_training(epoch + 1)
        if (epoch + 1) % self.log_every:
            return

        parts = [f"Época {epoch + 1}/{self._epochs}"]
        if loss is not None:
            parts.append(f"loss {loss:.5f}")
        if val_loss is not None:
            parts.append(f"val_loss {val_loss:.5f}")
        if self._lr is not None:
            parts.append(f"lr {self._lr:.1e}")
        self.dash.log(" · ".join(parts))

    def on_train_end(self, logs=None):
        es = self.early_stopping
        stopped = es.stopped_epoch + 1 if es is not None and es.stopped_epoch else None
        self._publish_training(None, stopped)
        self.dash.set("progress", None)

    def _publish_training(self, epoch, stopped=None):
        es = self.early_stopping
        self.dash.set("training", {
            "epochs": self._epochs,
            "best_val_loss": self._best,
            "best_epoch": self._best_epoch,
            # mesma regra do EarlyStopping (min_delta=0): épocas desde a última melhora
            "wait": (epoch - self._best_epoch) if epoch and self._best_epoch else 0,
            "patience": es.patience if es is not None else None,
            "stopped_epoch": stopped,
            "done": epoch is None,
        })

    def _progress(self, step, loss):
        self.dash.set("progress", {
            "task": "train",
            "epoch": self._epoch,
            "epochs": self._epochs,
            "step": step,
            "steps": self._steps,
            "loss": _num(loss),
            "elapsed": time.time() - self._t0,
        })

    def _current_lr(self):
        try:
            return _num(np.asarray(self.model.optimizer.learning_rate))
        except Exception:
            return None


class PredictMonitor(tf.keras.callbacks.Callback):
    """Publica o progresso de um model.predict()."""

    def __init__(self, dashboard, label, interval=0.25):
        super().__init__()
        self.dash = dashboard
        self.label = label
        self.interval = interval

    def on_predict_begin(self, logs=None):
        self._steps = self.params.get("steps")
        self._t0 = self._last = time.time()
        self._progress(0)

    def on_predict_batch_end(self, batch, logs=None):
        now = time.time()
        if now - self._last < self.interval:
            return
        self._last = now
        self._progress(batch + 1)

    def on_predict_end(self, logs=None):
        self._progress(self._steps)

    def _progress(self, step):
        self.dash.set("progress", {
            "task": "predict",
            "label": self.label,
            "step": step,
            "steps": self._steps,
            "elapsed": time.time() - self._t0,
        })


def describe_layers(model):
    """Resumo das camadas de um Sequential para o card de arquitetura."""
    layers = []
    for layer in model.layers:
        activation = getattr(layer, "activation", None)
        layers.append({
            "name": layer.name,
            "type": type(layer).__name__,
            "units": getattr(layer, "units", None),
            "rate": getattr(layer, "rate", None),
            "activation": getattr(activation, "__name__", None),
            "params": layer.count_params(),
        })
    return layers
