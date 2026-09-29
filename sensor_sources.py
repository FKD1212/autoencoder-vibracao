"""
Fontes de amostras de vibração.

Toda fonte roda numa thread própria e entrega blocos numpy (n_amostras, n_eixos)
para um `sink`. Implementações:
  SerialSource    — placa (ESP32/Arduino) enviando texto pela USB/serial
  SimulatedSource — máquina rotativa sintética com falhas injetáveis
  FileSource      — reproduz em tempo real uma gravação feita pelo sensor_app

Protocolo serial (texto ASCII, uma amostra por linha, terminada em \\n):
  # fs=1600 channels=x,y,z units=g     <- metadados opcionais (linhas com '#')
  0.0123,-0.9981,0.0457                <- uma amostra; separador: vírgula, ';', espaço ou tab
A taxa de amostragem é ditada pela placa. Sem o cabeçalho, informe `fs` ou o app
usa a taxa medida na chegada dos dados.
"""
import os
import re
import threading
import time
from collections import deque

import numpy as np

DEFAULT_CHANNELS = {1: ["a"], 2: ["x", "y"], 3: ["x", "y", "z"]}

_META = re.compile(r"(\w+)\s*=\s*(\S+)")
_SPLIT = re.compile(r"[,;\s]+")


def default_channels(n):
    return DEFAULT_CHANNELS.get(n, [f"c{i + 1}" for i in range(n)])


def parse_meta(line):
    """'# fs=1600 channels=x,y,z' -> {'fs': '1600', 'channels': 'x,y,z'}"""
    return dict(_META.findall(line.lstrip("#")))


def parse_sample(line):
    """'0.1, -0.2, 0.3' -> [0.1, -0.2, 0.3]; None se não for uma linha numérica."""
    parts = [p for p in _SPLIT.split(line.strip()) if p]
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def list_serial_ports():
    """Portas disponíveis, ou None se o pyserial não estiver instalado."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    return [{"device": p.device, "description": p.description} for p in list_ports.comports()]


class Source:
    kind = "base"

    def __init__(self, fs=0.0, channels=None, units=""):
        self.fs = float(fs or 0.0)        # taxa nominal (Hz); 0 = desconhecida
        self.channels = list(channels or [])
        self.units = units
        self.error = None
        self.messages = deque(maxlen=20)    # avisos da placa ('# texto' sem chave=valor)
        self._sink = None
        self._thread = None
        self._stop = threading.Event()

    @property
    def label(self):
        return self.kind

    def start(self, sink):
        self._sink = sink
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_safe, name=f"fonte-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self._close()

    def describe(self):
        return {"kind": self.kind, "label": self.label, "fs": self.fs,
                "channels": self.channels, "units": self.units, "error": self.error}

    def _run_safe(self):
        try:
            self._run()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _run(self):
        raise NotImplementedError

    def _close(self):
        pass


# ---------------------------------------------------------------------------
class SerialSource(Source):
    kind = "serial"

    def __init__(self, port, baud=921600, fs=None):
        super().__init__(fs=fs or 0.0, units="")
        self.port = port
        self.baud = int(baud)
        self.bad_lines = 0
        self._fs_fixed = bool(fs)
        self._ser = None

    @property
    def label(self):
        return f"Serial {self.port} @ {self.baud}"

    def describe(self):
        return {**super().describe(), "bad_lines": self.bad_lines}

    def _apply_meta(self, meta):
        if "fs" in meta and not self._fs_fixed:
            try:
                self.fs = float(meta["fs"])
            except ValueError:
                pass
        if "channels" in meta:
            self.channels = [c for c in meta["channels"].split(",") if c]
        if "units" in meta:
            self.units = meta["units"]

    def _run(self):
        import serial  # pyserial

        # serial_for_url aceita "COM3", "/dev/ttyUSB0" e também "loop://" (testes)
        self._ser = serial.serial_for_url(self.port, baudrate=self.baud, timeout=0.1)
        pending = b""
        synced = False      # a primeira linha após abrir a porta pode vir cortada
        n_channels = None
        while not self._stop.is_set():
            data = self._ser.read(self._ser.in_waiting or 1)
            if not data:
                continue
            pending += data
            *lines, pending = pending.split(b"\n")
            if not synced and lines:
                lines, synced = lines[1:], True

            rows = []
            for raw in lines:
                line = raw.decode("ascii", "ignore").strip()
                if not line:
                    continue
                if line.startswith("#"):
                    meta = parse_meta(line)
                    if meta:
                        self._apply_meta(meta)
                    else:
                        self.messages.append(line.lstrip("# ").strip())
                    continue
                values = parse_sample(line)
                if values is None:
                    self.bad_lines += 1
                    continue
                if n_channels is None:
                    n_channels = len(self.channels) or len(values)
                    if len(self.channels) != n_channels:
                        self.channels = default_channels(n_channels)
                if len(values) != n_channels:
                    self.bad_lines += 1
                    continue
                rows.append(values)
            if rows:
                self._sink(np.asarray(rows, dtype=np.float32))

    def _close(self):
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
class SimulatedSource(Source):
    """
    Acelerômetro triaxial (g) num mancal de máquina rotativa a ~1770 rpm.
    Operação normal: 1×, 2× e 3× da rotação, ruído, gravidade no eixo y e
    pequenas variações de carga/velocidade. Falhas (nível 0–1):
      desbalanceamento — 1× radial cresce
      desalinhamento   — 2× axial/radial cresce
      rolamento        — impactos na BPFO excitando uma ressonância de 520 Hz
      folga            — muitos harmônicos + sub-harmônico 0,5×
    """
    kind = "sim"
    FAULTS = {
        "desbalanceamento": "Desbalanceamento",
        "desalinhamento": "Desalinhamento",
        "rolamento": "Rolamento (pista externa)",
        "folga": "Folga mecânica",
    }

    def __init__(self, fs=1600.0, rpm=1770.0, seed=None):
        super().__init__(fs=fs, channels=["x", "y", "z"], units="g")
        self.rpm = rpm
        self.faults = {name: 0.0 for name in self.FAULTS}
        self._rng = np.random.default_rng(seed)
        self._phase = 0.0
        self._f0 = rpm / 60.0
        self._t = 0.0
        self._next_impact = 0.0
        self._tail = np.zeros(0)
        kt = np.arange(int(0.02 * fs)) / fs
        self._kernel = np.exp(-kt / 0.003) * np.sin(2 * np.pi * 520.0 * kt)

    @property
    def label(self):
        return f"Simulador · {self.rpm:.0f} rpm"

    def describe(self):
        return {**super().describe(), "faults": dict(self.faults), "fault_labels": self.FAULTS}

    def set_fault(self, name, level):
        if name not in self.faults:
            raise ValueError(f"falha desconhecida: {name}")
        self.faults[name] = float(np.clip(level, 0.0, 1.0))

    def _run(self):
        start = time.perf_counter()
        produced = 0
        while not self._stop.is_set():
            time.sleep(0.05)
            due = int((time.perf_counter() - start) * self.fs) - produced
            if due > 0:
                self._sink(self._generate(due))
                produced += due

    def _generate(self, n):
        fs, rng, f = self.fs, self._rng, self.faults
        # velocidade com deriva lenta (±0,3 %) e carga variando ±5 %
        self._f0 = float(np.clip(self._f0 + rng.normal(0, 0.002), 0.997 * self.rpm / 60, 1.003 * self.rpm / 60))
        t = self._t + np.arange(n) / fs
        phi = self._phase + 2 * np.pi * self._f0 * np.arange(n) / fs
        self._phase = float((phi[-1] + 2 * np.pi * self._f0 / fs) % (2 * np.pi))
        self._t = float(t[-1] + 1 / fs)
        load = 1.0 + 0.05 * np.sin(2 * np.pi * 0.07 * t)

        x = load * (0.050 * np.sin(phi) + 0.015 * np.sin(2 * phi + 0.3) + 0.006 * np.sin(3 * phi + 1.1))
        y = load * (0.040 * np.cos(phi) + 0.012 * np.sin(2 * phi + 1.3) + 0.005 * np.sin(3 * phi))
        z = load * (0.020 * np.sin(phi + 0.8) + 0.010 * np.sin(2 * phi + 0.1))

        if f["desbalanceamento"]:
            a = 0.15 * f["desbalanceamento"]
            x += a * np.sin(phi + 0.2)
            y += 0.8 * a * np.cos(phi + 0.2)
        if f["desalinhamento"]:
            a = 0.10 * f["desalinhamento"]
            z += a * np.sin(2 * phi) + 0.5 * a * np.sin(phi + 0.4)
            x += 0.4 * a * np.sin(2 * phi + 0.7)
        if f["folga"]:
            a = 0.03 * f["folga"]
            loose = sum(a / h ** 0.5 * np.sin(h * phi + h) for h in range(2, 10))
            loose += a * np.sin(0.5 * phi)
            x += loose
            y += 1.3 * loose

        # impactos de rolamento: trem de impulsos na BPFO convoluído com a ressonância
        bpfo = 3.58 * self._f0
        impulses = np.zeros(n)
        t_end = t[0] + n / fs
        while self._next_impact < t_end:
            idx = int((self._next_impact - t[0]) * fs)
            if 0 <= idx < n:
                impulses[idx] += 1.0 + 0.2 * rng.normal()
            self._next_impact += (1.0 + 0.01 * rng.normal()) / bpfo
        ring = np.convolve(impulses * 0.6 * f["rolamento"], self._kernel)
        ring[: len(self._tail)] += self._tail
        self._tail = ring[n:]
        x += ring[:n]
        y += 0.7 * ring[:n]

        noise = rng.normal(0, 0.008, size=(n, 3))
        out = np.column_stack([x, y + 1.0, z]) + noise   # +1 g de gravidade no eixo y
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
class FileSource(Source):
    """Reproduz em tempo real (em loop) um CSV gravado pelo sensor_app."""
    kind = "file"

    def __init__(self, path, loop=True):
        fs, channels, units = 0.0, None, ""
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith("#"):
                    break
                meta = parse_meta(line)
                fs = float(meta.get("fs", fs))
                channels = meta["channels"].split(",") if "channels" in meta else channels
                units = meta.get("units", units)
        data = np.loadtxt(path, delimiter=",", comments="#", dtype=np.float32, ndmin=2)
        if not fs:
            raise ValueError("gravação sem '# fs=' no cabeçalho")
        super().__init__(fs=fs, channels=channels or default_channels(data.shape[1]), units=units)
        self.path = path
        self.data = data
        self.loop = loop

    @property
    def label(self):
        return f"Gravação {os.path.basename(self.path)}"

    def _run(self):
        start = time.perf_counter()
        produced = 0
        total = len(self.data)
        while not self._stop.is_set():
            time.sleep(0.05)
            due = int((time.perf_counter() - start) * self.fs) - produced
            if due <= 0:
                continue
            idx = (produced + np.arange(due)) % total
            if not self.loop and produced + due > total:
                idx = idx[: max(total - produced, 0)]
            if len(idx):
                self._sink(self.data[idx])
            produced += due
            if not self.loop and produced >= total:
                return
