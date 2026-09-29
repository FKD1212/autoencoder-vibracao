/*
 * Monitor de vibração (sensor.html): fonte do sinal, detecção ao vivo e calibração.
 * Conexão ao vivo, formatação, tema e gráficos base vêm de common.js.
 * Ações do usuário viram POST /api/command, tratados pelo sensor_app.py.
 */
"use strict";

const ui = { kind: null, axis: 0, kPreview: null, waveScale: {}, faultTimers: {} };

const MODE_LABELS = {
  idle: "Parado", preview: "Pré-visualização", calibrating: "Calibrando",
  training: "Treinando", monitoring: "Monitorando",
};
const ALARM_TEXT = {
  normal: { icon: "✓", title: "Normal" },
  alerta: { icon: "!", title: "Alerta" },
  anomalia: { icon: "✕", title: "Anomalia" },
};

const RENDERERS = {
  mode: [renderHeader, renderStatus, renderCalib, renderSource, renderScore],
  acq: [renderHeader, renderSource, renderStatus, renderAxes, renderCalib],
  status: [renderHeader],
  options: [renderOptions],
  wave: [renderWave],
  spectrum: [renderSpectrum, renderStatus],
  detection: [renderStatus, renderHeader],
  scores: [renderScore],
  events: [renderEvents],
  profile: [renderCalib, renderCalHist, renderSpectrum, renderStatus, renderScore, renderOptions],
  calib: [renderCalib],
  epochs: [renderCalTrain, renderCalib],
  alarm_rule: [renderStatus],
  logs: [renderLogs],
};

const channels = () => (state.acq && state.acq.channels) || [];
const units = () => (state.acq && state.acq.units) || "";
const axisName = () => String(channels()[ui.axis] || "").toUpperCase();
const profileK = () => (ui.kPreview != null ? ui.kPreview : state.profile ? state.profile.k : Number($("cal-k").value));

// ------------------------------------------------------------------ comandos
async function command(action, params = {}, errorId = "src-error") {
  const el = $(errorId);
  el.textContent = "";
  try {
    const res = await fetch("api/command", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, ...params }),
    });
    const data = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
    if (!data.ok) throw new Error(data.error || `HTTP ${res.status}`);
    return data;
  } catch (err) {
    el.textContent = err.message.replace(/^\w+(Error|Exception): /, "");
    return null;
  }
}

function fillSelect(select, items, preferred) {
  const current = preferred != null ? preferred : select.value;
  select.innerHTML = items.map((i) => `<option value="${esc(i.value)}">${esc(i.label)}</option>`).join("");
  if (items.some((i) => i.value === current)) select.value = current;
}

// ------------------------------------------------------------------ topo
function renderHeader() {
  const acq = state.acq || {};
  const mode = state.mode || "idle";
  const parts = [];
  if (acq.connected) {
    parts.push(acq.label);
    if (acq.fs_used) parts.push(`${fmtInt(acq.fs_used)} Hz`);
    if (acq.channels && acq.channels.length) parts.push(`eixos ${acq.channels.join(", ")}`);
    if (acq.units) parts.push(acq.units);
  }
  $("src-meta").textContent = parts.length ? parts.join(" · ") : "Nenhuma fonte conectada";
  $("mode-pill").dataset.state = mode === "idle" ? "idle" : "running";
  $("mode-label").textContent = MODE_LABELS[mode] || mode;

  const err = (state.status && state.status.error) || acq.last_error;
  $("error-banner").hidden = !err;
  $("error-text").textContent = err || "";

  const det = state.detection;
  document.title = mode === "monitoring" && det
    ? `${ALARM_TEXT[det.alarm].icon} ${ALARM_TEXT[det.alarm].title} · Monitor de Vibração`
    : "Monitor de Vibração";
}

// ------------------------------------------------------------------ fonte
function renderSource() {
  const acq = state.acq || {};
  if (ui.kind == null) ui.kind = acq.connected ? acq.kind : "sim";
  document.querySelectorAll("#src-kind button").forEach((b) =>
    b.setAttribute("aria-selected", String(b.dataset.kind === ui.kind))
  );
  document.querySelectorAll("#card-source .form[data-kind]").forEach((f) => {
    f.hidden = f.dataset.kind !== ui.kind;
  });

  const connected = !!acq.connected;
  $("btn-connect").textContent = connected && acq.kind === ui.kind ? "Reconectar" : "Conectar";
  $("btn-disconnect").disabled = !connected;
  const rec = $("btn-record");
  rec.disabled = !connected;
  rec.setAttribute("aria-pressed", String(!!acq.recording));
  rec.textContent = acq.recording ? `■ Parar · ${fmtDuration(acq.recording.seconds)}` : "● Gravar";

  const kv = [];
  if (connected) {
    const drift = acq.fs && acq.fs_measured && Math.abs(acq.fs_measured - acq.fs) / acq.fs > 0.05;
    kv.push(["Taxa medida", `${fmtInt(acq.fs_measured)} Hz`]);
    kv.push(["Taxa usada", acq.fs ? `${fmtInt(acq.fs)} Hz (nominal)` : `${fmtInt(acq.fs_used)} Hz (medida)`]);
    if (drift) kv.push(["⚠ Atenção", "taxa medida difere da nominal — amostras se perdendo?"]);
    kv.push(["Amostras recebidas", fmtInt(acq.samples)]);
    if (acq.kind === "serial") kv.push(["Linhas inválidas", fmtInt(acq.bad_lines)]);
    if (acq.recording) kv.push(["Gravando em", `recordings/${acq.recording.file}`]);
  }
  $("acq-kv").innerHTML = kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
  renderFaults(acq);
}

function renderFaults(acq) {
  const isSim = !!(acq.connected && acq.kind === "sim");
  $("sim-faults").hidden = !isSim;
  if (!isSim) return;
  const list = $("fault-list");
  if (list.dataset.built !== "1") {
    list.dataset.built = "1";
    list.innerHTML = Object.entries(acq.fault_labels || {})
      .map(
        ([key, label]) => `<label class="fault-row"><span>${esc(label)}</span>
          <input type="range" min="0" max="1" step="0.05" value="0" data-fault="${esc(key)}">
          <output data-for="${esc(key)}">0%</output></label>`
      )
      .join("");
    list.querySelectorAll("input").forEach((input) =>
      input.addEventListener("input", () => {
        const fault = input.dataset.fault;
        list.querySelector(`output[data-for="${fault}"]`).textContent = fmtPct(Number(input.value), 0);
        clearTimeout(ui.faultTimers[fault]);
        ui.faultTimers[fault] = setTimeout(() => command("sim", { fault, level: Number(input.value) }), 120);
      })
    );
  }
  for (const [fault, level] of Object.entries(acq.faults || {})) {
    const input = list.querySelector(`input[data-fault="${fault}"]`);
    if (!input || document.activeElement === input) continue;   // não briga com quem está arrastando
    input.value = level;
    list.querySelector(`output[data-for="${fault}"]`).textContent = fmtPct(level, 0);
  }
}

function renderOptions() {
  const o = state.options || {};
  const ports =
    o.ports == null
      ? [{ value: "", label: "pyserial não instalado" }]
      : o.ports.length
        ? o.ports.map((p) => ({ value: p.device, label: `${p.device} — ${p.description}` }))
        : [{ value: "", label: "Nenhuma porta encontrada" }];
  fillSelect($("serial-port"), ports);

  const recs = o.recordings || [];
  fillSelect(
    $("file-name"),
    recs.length
      ? recs.map((r) => ({ value: r.name, label: `${r.name} (${fmt(r.size / 1e6, 1)} MB)` }))
      : [{ value: "", label: "Nenhuma gravação ainda" }]
  );

  const profiles = o.profiles || [];
  fillSelect(
    $("profile-select"),
    profiles.length ? profiles.map((p) => ({ value: p.name, label: p.name })) : [{ value: "", label: "Nenhuma calibração salva" }],
    state.profile ? state.profile.name : null
  );
  $("btn-load").disabled = !profiles.length;
}

function renderAxes() {
  const ch = channels();
  const box = $("axis-select");
  const signature = ch.join(",");
  if (box.dataset.signature !== signature) {
    box.dataset.signature = signature;
    box.innerHTML = ch
      .map((c, i) => `<button type="button" role="tab" data-axis="${i}">${esc(c.toUpperCase())}</button>`)
      .join("");
    box.querySelectorAll("button").forEach((b) =>
      b.addEventListener("click", () => {
        ui.axis = Number(b.dataset.axis);
        renderAxes();
        renderWave();
        renderSpectrum();
      })
    );
  }
  if (ui.axis >= ch.length) ui.axis = 0;
  box.querySelectorAll("button").forEach((b) =>
    b.setAttribute("aria-selected", String(Number(b.dataset.axis) === ui.axis))
  );
}

// ------------------------------------------------------------------ estado da detecção
function renderStatus() {
  const acq = state.acq || {};
  const mode = state.mode || "idle";
  const p = state.profile;
  const det = state.detection;
  const rule = state.alarm_rule || { window: 5, min: 3 };

  let st = "off";
  let icon = "–";
  let label = "Desconectado";
  let sub = "Conecte uma fonte para começar.";
  if (mode === "training") {
    [st, icon, label, sub] = ["busy", "", "Treinando o modelo…", "O autoencoder está aprendendo a linha de base gravada."];
  } else if (!acq.connected) {
    // mantém "Desconectado"
  } else if (mode === "calibrating") {
    [st, icon, label, sub] = ["busy", "", "Calibrando…", "Gravando a linha de base — mantenha a máquina em operação normal."];
  } else if (!p) {
    [st, icon, label, sub] = ["off", "?", "Sem calibração",
      "Confira o sinal abaixo e calibre com a máquina em operação normal para ativar a detecção."];
  } else if (p.incompatible) {
    [st, icon, label, sub] = ["off", "!", "Calibração incompatível", `${p.incompatible}. Recalibre ou carregue outra.`];
  } else if (!det) {
    [st, icon, label, sub] = ["busy", "", "Iniciando a detecção…", `Calibração “${p.name}” ativa.`];
  } else {
    st = det.alarm;
    ({ icon, title: label } = ALARM_TEXT[det.alarm]);
    sub =
      det.alarm === "anomalia"
        ? `${det.above} das últimas ${rule.window} janelas acima do threshold.`
        : det.alarm === "alerta"
          ? `Janela acima do threshold — vira anomalia com ${rule.min} de ${rule.window} janelas.`
          : `Vibração dentro do padrão da calibração “${p.name}”.`;
  }
  $("alarm").dataset.state = st;
  $("alarm-icon").textContent = icon;
  $("alarm-label").textContent = label;
  $("alarm-sub").textContent = sub;

  const showScore = !!(mode === "monitoring" && acq.connected && det && p && !p.incompatible);
  $("score-block").hidden = !showScore;
  if (showScore) {
    $("score-value").textContent = `${fmt(det.score, 2)}  (erro ${fmt(det.err, 3)} ÷ ${fmt(det.threshold, 3)})`;
    $("score-meter").dataset.state = det.alarm;
    $("score-fill").style.width = `${Math.min(det.score / 3, 1) * 100}%`;
    $("top-devs").innerHTML =
      det.alarm === "normal"
        ? ""
        : det.top
            .map((c) => `<li><span>${esc(c.label)}</span><b>${c.z >= 0 ? "+" : ""}${fmt(c.z, 1)}σ</b></li>`)
            .join("");
  }

  const s = state.spectrum;
  const ch = channels();
  const tiles = [];
  if (acq.connected && s && s.rms) {
    s.rms.forEach((v, i) => tiles.push([`RMS ${String(ch[i] || i + 1).toUpperCase()}`, fmt(v, 4), units()]));
    tiles.push(["Pico", fmt(Math.max(...s.peak), 3), units()]);
  }
  if (p) tiles.push(["Threshold", fmt(p.threshold, 3), `k = ${fmt(p.k, 1)}`]);
  $("status-tiles").innerHTML = tiles
    .map(([l, v, u]) => `<div class="mini"><span>${esc(l)}</span><strong>${v}</strong><span>${esc(u || " ")}</span></div>`)
    .join("");
}

// ------------------------------------------------------------------ sinal
// arredonda para cima até 1, 2, 2,5 ou 5 × 10^n — limites de eixo "redondos"
function niceCeil(v) {
  const e = 10 ** Math.floor(Math.log10(v));
  const m = v / e;
  return (m <= 1 ? 1 : m <= 2 ? 2 : m <= 2.5 ? 2.5 : m <= 5 ? 5 : 10) * e;
}

function createWaveChart() {
  const t = tokens();
  return new Chart($("c-wave"), {
    type: "line",
    data: { datasets: [lineDataset(t, "Sinal", t.s1, { pointRadius: 0, borderWidth: 1.5, pointHitRadius: 4 })] },
    options: {
      ...baseOptions(),
      scales: {
        x: scaleOptions(t, { title: "Tempo (ms)", min: 0, ticks: { callback: (v) => fmt(v, 0) } }),
        y: scaleOptions(t, { title: "", border: false, ticks: { callback: (v) => fmt(v, 2) } }),
      },
      plugins: {
        legend: { display: false },
        tooltip: tooltipOptions(t, {
          title: (items) => (items[0] ? `${fmt(items[0].parsed.x, 1)} ms` : ""),
          label: (item) => ` Eixo ${axisName()}: ${fmt(item.parsed.y, 4)} ${units()}`,
        }),
      },
    },
    plugins: [crosshairPlugin],
  });
}

function renderWave() {
  const w = state.wave;
  const has = !!(w && w.data && w.data.length && state.acq && state.acq.connected);
  $("card-wave").classList.toggle("has-data", has);
  if (!has) return;
  const axis = Math.min(ui.axis, w.data.length - 1);
  const series = w.data[axis];
  const dt = w.fs ? 1000 / w.fs : 1;
  const chart = ensureChart("wave", createWaveChart);
  chart.data.datasets[0].data = series.map((y, i) => ({ x: i * dt, y }));
  // escala simétrica que cresce na hora e encolhe devagar: o gráfico não "pula" a cada quadro
  const peak = Math.max(1e-6, ...series.map(Math.abs));
  const scale = (ui.waveScale[axis] = Math.max(peak * 1.15, (ui.waveScale[axis] || 0) * 0.97));
  const niceScale = niceCeil(scale);
  chart.options.scales.y.min = -niceScale;
  chart.options.scales.y.max = niceScale;
  chart.options.scales.y.title.text = `Eixo ${axisName()}${units() ? ` (${units()})` : ""}`;
  chart.options.scales.y.title.display = true;
  chart.options.scales.x.max = (series.length - 1) * dt;
  chart.update("none");
  $("wave-sub").textContent = `Últimas ${fmtInt(series.length)} amostras (${fmt(series.length * dt, 0)} ms), sem a componente DC`;
}

// ------------------------------------------------------------------ espectro
function bandTitle(chart, index) {
  const e = chart.$edges;
  return e ? `${fmt(e[index], 0)}–${fmt(e[index + 1], 0)} Hz` : "";
}

function createSpecChart() {
  const t = tokens();
  const band = rgba(t.muted, 0.22);
  const envelope = (label, extra) => ({
    label, data: [], borderWidth: 0, borderColor: band, backgroundColor: band,
    pointRadius: 0, pointHoverRadius: 0, pointHitRadius: 0, tension: 0, ...extra,
  });
  const chart = new Chart($("c-spec"), {
    type: "line",
    data: {
      datasets: [
        lineDataset(t, "Agora", t.s1, { pointRadius: 0 }),
        lineDataset(t, "Média da calibração", t.ink2, { pointRadius: 0, borderWidth: 1.5 }),
        envelope("Faixa normal", { fill: "+1", envelope: true }),
        envelope("Faixa normal (mín.)", { fill: false, noTooltip: true }),
      ],
    },
    options: {
      ...baseOptions(),
      scales: {
        x: scaleOptions(t, { title: "Frequência (Hz)", min: 0, ticks: { callback: (v) => fmt(v, 0) } }),
        y: scaleOptions(t, { title: "dB", border: false, ticks: { callback: (v) => fmt(v, 0) } }),
      },
      plugins: {
        legend: { display: false },
        crosshair: { sameX: true },
        tooltip: tooltipOptions(t, {
          title: (items) => (items[0] ? bandTitle(items[0].chart, items[0].dataIndex) : ""),
          label: (item) => {
            if (item.dataset.envelope) {
              const lo = item.chart.data.datasets[3].data[item.dataIndex];
              return ` Faixa normal: ${fmt(lo && lo.y, 1)} a ${fmt(item.parsed.y, 1)} dB`;
            }
            return ` ${item.dataset.label}: ${fmt(item.parsed.y, 1)} dB`;
          },
        }),
      },
    },
    plugins: [crosshairPlugin],
  });
  return chart;
}

function renderSpectrum() {
  const s = state.spectrum;
  const has = !!(s && s.db && state.acq && state.acq.connected);
  const card = $("card-spec");
  card.classList.toggle("has-data", has);
  if (!has) return;
  const p = state.profile;
  const edges = s.edges_hz;
  const withProfile = !!(
    p && !p.incompatible && p.channels.length === s.db.length && p.n_bands === s.db[0].length &&
    Math.abs(last(p.edges_hz) - last(edges)) < 1
  );
  card.classList.toggle("has-profile", withProfile);

  const axis = Math.min(ui.axis, s.db.length - 1);
  const centers = edges.slice(0, -1).map((e, i) => (e + edges[i + 1]) / 2);
  const chart = ensureChart("spec", createSpecChart);
  const ds = chart.data.datasets;
  ds[0].data = s.db[axis].map((y, i) => ({ x: centers[i], y }));
  if (withProfile) {
    const k = profileK();
    const mean = p.baseline.mean_db[axis];
    const std = p.baseline.std_db[axis];
    ds[1].data = mean.map((y, i) => ({ x: centers[i], y }));
    ds[2].data = mean.map((y, i) => ({ x: centers[i], y: y + k * std[i] }));
    ds[3].data = mean.map((y, i) => ({ x: centers[i], y: y - k * std[i] }));
  } else {
    ds[1].data = [];
    ds[2].data = [];
    ds[3].data = [];
  }
  chart.$edges = edges;
  chart.options.scales.x.max = last(edges);
  chart.update("none");
}

// ------------------------------------------------------------------ score no tempo
function fmtAgo(seconds) {
  const s = Math.round(-seconds);
  if (s <= 0) return "agora";
  return `−${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

function logTick(v) {
  const mantissa = Math.round(v / 10 ** Math.floor(Math.log10(v)));
  if (![1, 2, 5].includes(mantissa)) return "";
  return v < 1 ? fmt(v, 1) : fmt(v, 0);
}

function createScoreChart() {
  const t = tokens();
  return new Chart($("c-score"), {
    type: "line",
    data: {
      datasets: [
        {
          label: "Acima do threshold", data: [], showLine: false, noTooltip: true,
          pointRadius: 3.5, pointBackgroundColor: t.critical, pointBorderColor: t.surface, pointBorderWidth: 1.5,
          pointHoverRadius: 3.5, borderColor: t.critical, backgroundColor: t.critical,
        },
        lineDataset(t, "Score", t.s1, { borderWidth: 1.5 }),
      ],
    },
    options: {
      ...baseOptions(),
      scales: {
        x: scaleOptions(t, { ticks: { callback: fmtAgo, maxTicksLimit: 7 } }),
        y: scaleOptions(t, { type: "logarithmic", min: 0.1, suggestedMax: 3, border: false, ticks: { callback: logTick } }),
      },
      plugins: {
        legend: { display: false },
        refLines: { lines: [{ y: 1, label: "threshold" }] },
        tooltip: tooltipOptions(t, {
          title: (items) => (items[0] ? `${fmtClock(items[0].raw.t)} (${fmtAgo(items[0].parsed.x)})` : ""),
          label: (item) => ` Score ${fmt(item.raw.score, 2)} · ${ALARM_TEXT[item.raw.alarm].title.toLowerCase()}`,
        }),
      },
    },
    plugins: [crosshairPlugin, refLinesPlugin],
  });
}

function renderScore() {
  const scores = state.scores || [];
  $("score-empty").textContent = state.profile ? "Aguardando janelas…" : "Calibre para ativar a detecção";
  const has = scores.length > 0;
  $("card-score").classList.toggle("has-data", has);
  if (!has) return;
  const now = serverNow();
  const pts = scores.map((s) => ({ x: s.t - now, y: Math.max(s.score, 0.1), t: s.t, score: s.score, alarm: s.alarm }));
  const chart = ensureChart("score", createScoreChart);
  chart.data.datasets[1].data = pts;
  chart.data.datasets[0].data = pts.filter((p) => p.score > 1);
  chart.options.scales.x.min = Math.max(-600, pts[0].x);
  chart.options.scales.x.max = 0;
  chart.update("none");
}

function renderEvents() {
  const events = (state.events || []).slice().reverse();
  $("events-count").textContent = events.length ? `${events.length}` : "";
  $("events").innerHTML = events.length
    ? events
        .map((e) => {
          const open = e.end == null;
          const when = open ? "em andamento" : fmtDuration(e.end - e.start);
          return `<li class="event${open ? " open" : ""}">
            <div class="event-head"><span><span class="icon">✕</span>${fmtClock(e.start)} · ${when}</span>
              <span class="tabular">pico ${fmt(e.peak, 2)}×</span></div>
            <div class="event-top">${e.top.map((c) => esc(c.label)).join(" · ")}</div>
          </li>`;
        })
        .join("")
    : '<li class="muted small">Nenhum evento.</li>';
}

// ------------------------------------------------------------------ calibração
function renderCalib() {
  const mode = state.mode || "idle";
  const acq = state.acq || {};
  const c = state.calib;
  const p = state.profile;
  const busy = mode === "calibrating" || mode === "training";

  $("btn-calibrate").disabled = !acq.connected || busy;
  $("btn-calibrate").textContent = p ? "Recalibrar" : "Iniciar calibração";
  $("btn-cancel").hidden = mode !== "calibrating";

  const slider = $("cal-k");
  if (p && ui.kPreview == null && document.activeElement !== slider) slider.value = p.k;
  $("cal-k-value").textContent = fmt(Number(slider.value), 1);

  const bar = $("cal-bar");
  const setProgress = (label, right, frac, indeterminate = false) => {
    $("cal-progress").hidden = false;
    $("cal-progress-label").textContent = label;
    $("cal-progress-right").textContent = right;
    bar.classList.toggle("indeterminate", indeterminate);
    $("cal-fill").style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`;
  };
  if (c && c.state === "collecting") {
    const hop = (state.alarm_rule && state.alarm_rule.hop_s) || 0.5;
    setProgress("Gravando a linha de base", `${fmtInt(c.collected)} / ${fmtInt(c.target)} janelas · ≈ ${fmtDuration((c.target - c.collected) * hop)}`, c.collected / c.target);
  } else if (c && c.state === "training") {
    const n = (state.epochs || []).length;
    setProgress("Treinando o autoencoder", n ? `época ${n}` : "", 0, true);
  } else if (c && c.state === "done" && p && p.name === c.name) {
    setProgress(`Calibração “${c.name}” pronta`, "", 1);
  } else if (c && c.state === "failed") {
    setProgress(`Falhou: ${c.error}`, "", 0);
  } else {
    $("cal-progress").hidden = true;
  }

  const kv = [];
  if (p) {
    const k = profileK();
    const thr = p.mu + k * p.sigma;
    const far = p.val_errors.length ? p.val_errors.filter((e) => e > thr).length / p.val_errors.length : 0;
    kv.push(["Nome", p.name]);
    kv.push(["Criada em", new Date(p.meta.created_at * 1000).toLocaleString("pt-BR")]);
    kv.push(["Linha de base", `${fmtDuration(p.meta.seconds)} · ${fmtInt(p.n_train + p.n_val)} janelas`]);
    kv.push(["Sinal", `${fmtInt(p.fs)} Hz · ${p.channels.length} eixos · ${p.n_bands} faixas`]);
    kv.push(["Threshold", `${fmt(thr, 4)} = μ ${fmt(p.mu, 4)} + ${fmt(k, 1)}·σ`]);
    kv.push(["Janelas normais acima", `${fmtPct(far)} (validação)`]);
    if (p.incompatible) kv.push(["⚠ Incompatível", p.incompatible]);
  }
  $("profile-kv").innerHTML = kv.length
    ? kv.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")
    : '<dt class="muted">Nenhuma carregada</dt><dd></dd>';
}

function createCalTrainChart() {
  const t = tokens();
  return new Chart($("c-caltrain"), {
    type: "line",
    data: { datasets: [lineDataset(t, "Treino", t.s1), lineDataset(t, "Validação", t.s2)] },
    options: {
      ...baseOptions(),
      scales: {
        x: scaleOptions(t, { title: "Época", min: 1, ticks: { precision: 0 } }),
        y: scaleOptions(t, { title: "MAE", border: false, ticks: { callback: (v) => fmt(v, 2) } }),
      },
      plugins: {
        legend: { display: false },
        crosshair: { sameX: true },
        tooltip: tooltipOptions(t, {
          title: (items) => (items[0] ? `Época ${items[0].raw.x}` : ""),
          label: (item) => ` ${item.dataset.label}: ${fmt(item.parsed.y)}`,
        }),
      },
    },
    plugins: [crosshairPlugin],
  });
}

function renderCalTrain() {
  const epochs = state.epochs || [];
  $("card-caltrain").classList.toggle("has-data", epochs.length > 0);
  if (!epochs.length) return;
  const chart = ensureChart("caltrain", createCalTrainChart);
  chart.data.datasets[0].data = epochs.filter((e) => ok(e.loss)).map((e) => ({ x: e.epoch, y: e.loss }));
  chart.data.datasets[1].data = epochs.filter((e) => ok(e.val_loss)).map((e) => ({ x: e.epoch, y: e.val_loss }));
  chart.options.scales.x.max = Math.max(5, last(epochs).epoch);
  chart.update("none");
}

function createCalHistChart() {
  const t = tokens();
  return new Chart($("c-calhist"), {
    type: "line",
    data: { datasets: [histDataset(t, "Treino", t.s1), histDataset(t, "Validação", t.s2)] },
    options: {
      ...baseOptions(),
      scales: {
        x: scaleOptions(t, { title: "Erro de reconstrução (MAE)", ticks: { callback: (v) => fmt(v, 2) } }),
        y: scaleOptions(t, { title: "% das janelas", border: false, beginAtZero: true, ticks: { callback: (v) => fmtPct(v, 0) } }),
      },
      plugins: {
        legend: { display: false },
        refLines: { lines: [] },
        tooltip: tooltipOptions(t, {
          title: binTitle,
          label: (item) => ` ${item.dataset.label}: ${fmtPct(item.parsed.y)} das janelas`,
        }),
      },
    },
    plugins: [crosshairPlugin, refLinesPlugin],
  });
}

function renderCalHist() {
  const p = state.profile;
  $("card-calhist").classList.toggle("has-data", !!p);
  if (!p) {
    $("calhist-sub").textContent = "% das janelas por faixa de erro";
    return;
  }
  const k = profileK();
  const thr = p.mu + k * p.sigma;
  $("calhist-sub").textContent = `μ ${fmt(p.mu, 3)} · σ ${fmt(p.sigma, 3)} · threshold μ + ${fmt(k, 1)}·σ = ${fmt(thr, 3)}`;
  const chart = ensureChart("calhist", createCalHistChart);
  const edges = p.hist.edges;
  chart.data.datasets[0].data = stepPoints(edges, p.hist.train);
  chart.data.datasets[0].edges = edges;
  chart.data.datasets[1].data = stepPoints(edges, p.hist.val);
  chart.data.datasets[1].edges = edges;
  const lo = edges[0];
  const hi = Math.max(last(edges), thr);
  chart.options.scales.x.min = lo;
  chart.options.scales.x.max = hi + (hi - lo) * 0.05;
  chart.options.plugins.refLines.lines = [{ x: thr, label: `threshold ${fmt(thr, 3)}` }];
  chart.update("none");
}

// ------------------------------------------------------------------ ações
document.querySelectorAll("#src-kind button").forEach((b) =>
  b.addEventListener("click", () => {
    ui.kind = b.dataset.kind;
    renderSource();
  })
);
$("ports-refresh").addEventListener("click", () => command("options"));
$("btn-connect").addEventListener("click", () => {
  const params = { kind: ui.kind };
  if (ui.kind === "serial") {
    params.port = $("serial-port").value;
    params.baud = Number($("serial-baud").value);
    params.fs = $("serial-fs").value ? Number($("serial-fs").value) : null;
  } else if (ui.kind === "file") {
    params.file = $("file-name").value;
  }
  command("connect", params);
});
$("btn-disconnect").addEventListener("click", () => command("disconnect"));
$("btn-record").addEventListener("click", () => command("record", { on: !(state.acq && state.acq.recording) }));

$("btn-calibrate").addEventListener("click", () =>
  command("calibrate", {
    name: $("cal-name").value,
    seconds: Number($("cal-seconds").value),
    k: Number($("cal-k").value),
  }, "cal-error")
);
$("btn-cancel").addEventListener("click", () => command("cancel", {}, "cal-error"));
$("btn-load").addEventListener("click", () => command("load_profile", { name: $("profile-select").value }, "cal-error"));

const kSlider = $("cal-k");
kSlider.addEventListener("input", () => {
  ui.kPreview = Number(kSlider.value);
  renderCalib();
  renderCalHist();
  renderSpectrum();
});
kSlider.addEventListener("change", async () => {
  if (state.profile) await command("set_k", { k: Number(kSlider.value) }, "cal-error");
  ui.kPreview = null;
  renderCalib();
  renderCalHist();
});

startLive(RENDERERS);
command("options");
