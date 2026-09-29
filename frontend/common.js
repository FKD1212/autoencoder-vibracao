/*
 * Base compartilhada pelas páginas do dashboard (index.html e sensor.html):
 * estado ao vivo via Server-Sent Events, formatação pt-BR, tema claro/escuro
 * e utilitários de gráfico sobre o Chart.js.
 *
 * O servidor (dashboard.py) envia operações:
 *   {op: "snapshot", value}       estado completo (ao conectar)
 *   {op: "set", key, value}       substitui uma chave
 *   {op: "append", key, value}    acrescenta a uma lista
 * Cada página chama startLive({chave: [funções de render]}) e só as funções
 * ligadas às chaves alteradas rodam, no máximo uma vez por frame.
 */
"use strict";

const state = {};
const charts = {};
const live = { renderers: {}, dirty: new Set(), queued: false, clockOffset: 0, connLabel: null };

const $ = (id) => document.getElementById(id);

// ------------------------------------------------------------------ formatação
const numberFormats = {};
const nf = (d) =>
  (numberFormats[d] ||= new Intl.NumberFormat("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d }));
const ok = (v) => v != null && Number.isFinite(v);
const fmt = (v, d = 4) => (ok(v) ? nf(d).format(v) : "—");
const fmtInt = (v) => (ok(v) ? new Intl.NumberFormat("pt-BR").format(Math.round(v)) : "—");
const fmtPct = (v, d = 1) => (ok(v) ? `${nf(d).format(v * 100)}%` : "—");

function fmtSci(v) {
  if (!ok(v)) return "—";
  const [mantissa, exp] = v.toExponential(1).split("e");
  return `${mantissa.replace(".", ",")}e${exp.replace("+", "")}`;
}

function fmtDuration(s) {
  if (!ok(s)) return "—";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h) return `${h}h ${String(m).padStart(2, "0")}min`;
  if (m) return `${m}min ${String(sec).padStart(2, "0")}s`;
  return `${sec}s`;
}

const fmtClock = (t) => new Date(t * 1000).toLocaleTimeString("pt-BR");
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const serverNow = () => Date.now() / 1000 + live.clockOffset;
const last = (arr) => (arr && arr.length ? arr[arr.length - 1] : undefined);

// ------------------------------------------------------------------ estado ao vivo
function startLive(renderers, { connLabel } = {}) {
  live.renderers = renderers;
  live.connLabel = connLabel || null;
  initTheme();
  markAll();
  render();
  connect();
}

function connect() {
  const source = new EventSource("events");
  source.onopen = () => setConn("live");
  source.onmessage = (ev) => applyOp(JSON.parse(ev.data));
  source.onerror = () => setConn("retry");   // o EventSource reconecta sozinho
}

function setConn(s) {
  $("conn").dataset.state = s;
  $("conn-label").textContent = live.connLabel
    ? live.connLabel(s)
    : s === "live" ? "Ao vivo" : s === "retry" ? "Reconectando…" : "Conectando…";
}

function applyOp(op) {
  if (op.op === "snapshot") {
    for (const k of Object.keys(state)) delete state[k];
    Object.assign(state, op.value);
    live.clockOffset = op.value.server_time - Date.now() / 1000;
    markAll();
  } else if (op.op === "set") {
    state[op.key] = op.value;
    live.dirty.add(op.key);
  } else if (op.op === "append") {
    const list = (state[op.key] ||= []);
    list.push(op.value);
    if (op.limit && list.length > op.limit) list.splice(0, list.length - op.limit);
    live.dirty.add(op.key);
  }
  if (!live.queued) {
    live.queued = true;
    requestAnimationFrame(render);
  }
}

function markAll() {
  Object.keys(live.renderers).forEach((k) => live.dirty.add(k));
}

function render() {
  live.queued = false;
  const fns = new Set();
  for (const key of live.dirty) for (const fn of live.renderers[key] || []) fns.add(fn);
  live.dirty.clear();
  for (const fn of fns) {
    try {
      fn();
    } catch (err) {
      console.error(`Falha em ${fn.name}`, err);
    }
  }
}

// ------------------------------------------------------------------ tema
let tokenCache = null;
function tokens() {
  if (tokenCache) return tokenCache;
  const cs = getComputedStyle(document.documentElement);
  const v = (name) => cs.getPropertyValue(name).trim();
  tokenCache = {
    surface: v("--surface"), ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"),
    grid: v("--grid"), axis: v("--axis"), border: v("--border"), hover: v("--hover"),
    s1: v("--s1"), s2: v("--s2"),
    good: v("--good"), warning: v("--warning"), critical: v("--critical"),
    ramp: [0, 1, 2, 3, 4, 5, 6].map((i) => v(`--ramp-${i}`)),
  };
  return tokenCache;
}

function rgba(hex, a) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

function inkOn(hex) {
  const n = parseInt(hex.slice(1), 16);
  const lin = (c) => ((c /= 255) <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4);
  const L = 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
  return (L + 0.05) / 0.05 >= 1.05 / (L + 0.05) ? "#0b0b0b" : "#ffffff";
}

function currentTheme() {
  const forced = document.documentElement.dataset.theme;
  if (forced) return forced;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function retheme() {
  tokenCache = null;
  for (const k of Object.keys(charts)) {
    charts[k].destroy();
    delete charts[k];
  }
  markAll();
  render();
}

function initTheme() {
  Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
  Chart.defaults.font.size = 12;
  $("theme-toggle").addEventListener("click", () => {
    const next = currentTheme() === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch (e) {
      /* sem storage: vale só para esta aba */
    }
    retheme();
  });
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!document.documentElement.dataset.theme) retheme();
  });
}

const ensureChart = (key, factory) => (charts[key] ||= factory());

// ------------------------------------------------------------------ gráficos
function nearestIndexByX(data, xv) {
  let lo = 0;
  let hi = data.length - 1;
  if (hi < 0) return -1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (data[mid].x < xv) lo = mid;
    else hi = mid;
  }
  return Math.abs(data[lo].x - xv) <= Math.abs(data[hi].x - xv) ? lo : hi;
}

// Modo de hover: em cada série pega o ponto mais próximo no eixo x;
// em histogramas (dataset.edges) pega o bin sob o cursor.
Chart.Interaction.modes.xslice = function (chart, e) {
  const { left, right, top, bottom } = chart.chartArea;
  if (e.x == null || e.x < left || e.x > right || e.y < top || e.y > bottom) return [];
  const xv = chart.scales.x.getValueForPixel(e.x);
  const items = [];
  chart.data.datasets.forEach((ds, di) => {
    if (ds.noTooltip || !ds.data.length || !chart.isDatasetVisible(di)) return;
    let index;
    if (ds.edges) {
      const ed = ds.edges;
      if (xv < ed[0] || xv > ed[ed.length - 1]) return;
      let b = 0;
      while (b < ed.length - 2 && xv >= ed[b + 1]) b++;
      index = 1 + 2 * b;
    } else {
      index = nearestIndexByX(ds.data, xv);
    }
    const element = chart.getDatasetMeta(di).data[index];
    if (element) items.push({ element, datasetIndex: di, index });
  });
  const opts = chart.options.plugins.crosshair || {};
  if (opts.sameX && items.length > 1) {
    const xOf = (it) => chart.data.datasets[it.datasetIndex].data[it.index].x;
    const target = xOf(items.reduce((a, b) => (Math.abs(xOf(b) - xv) < Math.abs(xOf(a) - xv) ? b : a)));
    return items.filter((it) => xOf(it) === target);
  }
  return items;
};

// Linha vertical (ou faixa do bin) sob o cursor.
const crosshairPlugin = {
  id: "crosshair",
  beforeDatasetsDraw(chart) {
    const active = chart.tooltip ? chart.tooltip.getActiveElements() : [];
    if (!active.length) return;
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    const t = tokens();
    const first = active[0];
    const ds = chart.data.datasets[first.datasetIndex];
    ctx.save();
    if (ds.edges) {
      const b = (first.index - 1) / 2;
      const xs = chart.scales.x;
      const x0 = xs.getPixelForValue(ds.edges[b]);
      const x1 = xs.getPixelForValue(ds.edges[b + 1]);
      ctx.fillStyle = t.hover;
      ctx.fillRect(x0, top, x1 - x0, bottom - top);
    } else {
      const x = Math.round(first.element.x) + 0.5;
      ctx.strokeStyle = t.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    }
    ctx.restore();
  },
};

// Linhas de referência tracejadas e rotuladas: verticais ({x}) ou horizontais ({y}).
const refLinesPlugin = {
  id: "refLines",
  afterDatasetsDraw(chart, _args, opts) {
    const lines = (opts && opts.lines) || [];
    const { top, bottom, left, right } = chart.chartArea;
    const ctx = chart.ctx;
    const t = tokens();
    for (const line of lines) {
      const vertical = ok(line.x);
      if (!vertical && !ok(line.y)) continue;
      const pos = vertical
        ? Math.round(chart.scales.x.getPixelForValue(line.x)) + 0.5
        : Math.round(chart.scales.y.getPixelForValue(line.y)) + 0.5;
      if (vertical ? pos < left || pos > right : pos < top || pos > bottom) continue;
      ctx.save();
      ctx.strokeStyle = t.ink2;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      if (vertical) {
        ctx.moveTo(pos, top);
        ctx.lineTo(pos, bottom);
      } else {
        ctx.moveTo(left, pos);
        ctx.lineTo(right, pos);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.font = `600 12px ${Chart.defaults.font.family}`;
      const w = ctx.measureText(line.label).width;
      let tx;
      let ty;
      if (vertical) {
        const flip = pos > left + (right - left) * 0.7;
        tx = flip ? pos - 6 - w : pos + 6;
        ty = top + 4;
      } else {
        tx = right - w - 6;
        ty = pos - 19;
      }
      ctx.textBaseline = "top";
      ctx.fillStyle = t.surface;
      ctx.fillRect(tx - 3, ty - 2, w + 6, 17);
      ctx.fillStyle = t.ink;
      ctx.fillText(line.label, tx, ty);
      ctx.restore();
    }
  },
};

function scaleOptions(t, { title, type = "linear", ticks = {}, border = true, ...rest } = {}) {
  return {
    type,
    title: { display: !!title, text: title, color: t.muted, font: { size: 12 } },
    grid: { color: t.grid, drawTicks: false },
    border: { display: border, color: t.axis },
    ticks: { color: t.muted, padding: 6, maxRotation: 0, ...ticks },
    ...rest,
  };
}

function tooltipOptions(t, callbacks) {
  return {
    backgroundColor: t.surface,
    borderColor: t.border,
    borderWidth: 1,
    titleColor: t.ink,
    bodyColor: t.ink2,
    titleFont: { weight: "600" },
    padding: 10,
    cornerRadius: 8,
    caretPadding: 8,
    boxWidth: 8,
    boxHeight: 8,
    boxPadding: 4,
    callbacks: {
      labelColor: (item) => {
        const c = item.dataset.borderColor;
        return { borderColor: c, backgroundColor: c, borderWidth: 0, borderRadius: 2 };
      },
      ...callbacks,
    },
  };
}

function baseOptions() {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: "xslice", intersect: false },
    layout: { padding: { top: 6, right: 10 } },
    plugins: { legend: { display: false } },
  };
}

const isLastPoint = (ctx) => ctx.dataIndex === ctx.dataset.data.length - 1;

// Série de linha 2px, marcador só no fim (anel da cor da superfície);
// um ponto com raw.live=true é desenhado vazado (em andamento).
function lineDataset(t, label, color, extra = {}) {
  return {
    label,
    data: [],
    borderColor: color,
    backgroundColor: color,
    borderWidth: 2,
    borderCapStyle: "round",
    borderJoinStyle: "round",
    tension: 0,
    pointRadius: (ctx) => (isLastPoint(ctx) ? 4 : 0),
    pointBorderWidth: 2,
    pointBackgroundColor: (ctx) => (ctx.raw && ctx.raw.live ? t.surface : color),
    pointBorderColor: (ctx) => (ctx.raw && ctx.raw.live ? color : t.surface),
    pointHoverRadius: 5,
    pointHoverBorderWidth: 2,
    pointHoverBackgroundColor: color,
    pointHoverBorderColor: t.surface,
    pointHitRadius: 12,
    ...extra,
  };
}

// Histograma como área em degraus: dois pontos por bin (bordas esquerda/direita).
function stepPoints(edges, values) {
  const pts = [{ x: edges[0], y: 0 }];
  values.forEach((v, i) => {
    pts.push({ x: edges[i], y: v, bin: i }, { x: edges[i + 1], y: v, bin: i });
  });
  pts.push({ x: edges[edges.length - 1], y: 0 });
  return pts;
}

function histDataset(t, label, color) {
  return {
    label,
    data: [],
    edges: null,
    borderColor: color,
    backgroundColor: rgba(color, 0.12),
    borderWidth: 1.5,
    fill: "origin",
    tension: 0,
    pointRadius: 0,
    pointHoverRadius: 0,
    pointHitRadius: 0,
  };
}

function binTitle(items) {
  const withBin = items.find((it) => it.dataset.edges && it.raw.bin != null);
  if (withBin) {
    const e = withBin.dataset.edges;
    const b = withBin.raw.bin;
    return `MAE ${fmt(e[b])} – ${fmt(e[b + 1])}`;
  }
  return items.length ? `MAE ${fmt(items[0].parsed.x)}` : "";
}

// ------------------------------------------------------------------ log
function renderLogs() {
  const el = $("log");
  el.innerHTML = (state.logs || [])
    .map((l) => `<div class="${esc(l.level)}"><span class="t">${fmtClock(l.t)}</span>  ${esc(l.msg)}</div>`)
    .join("");
  if ($("log-follow").checked) el.scrollTop = el.scrollHeight;
}
