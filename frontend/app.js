/*
 * Monitor do treinamento (index.html) — o que é específico desta página.
 * Conexão ao vivo, formatação, tema e utilitários de gráfico vêm de common.js.
 */
"use strict";

let distSet = null;           // conjunto escolhido em "Erro por classe" (null = automático)

const RENDERERS = {
  run: [renderHeader, renderModel, renderTiles],
  data: [renderHeader, renderModel],
  model: [renderModel],
  status: [renderHeader, renderStepper, renderProgress],
  started_at: [renderHeader],
  plan: [renderStepper, renderProgress],
  phases: [renderStepper, renderProgress],
  progress: [renderProgress, renderTrainChart, renderTiles, renderHeader],
  epochs: [renderTrainChart, renderLrChart, renderTiles, renderEpochTable],
  training: [renderTiles, renderEpochTable],
  threshold: [renderThreshold, renderTiles],
  eval_test: [renderEvals, renderRoc, renderDist, renderTiles],
  eval_val: [renderEvals, renderRoc, renderDist, renderTiles],
  logs: [renderLogs],
};

const labelOf = (id) => ((state.plan || []).find((s) => s.id === id) || {}).label || id;
const currentPhase = () => last(state.phases);

// ------------------------------------------------------------------ topo
function renderHeader() {
  const { run, data } = state;
  const st = state.status || { state: "running" };
  const parts = [];
  if (run) parts.push(run.device === "GPU" ? `GPU × ${run.gpus.length}` : "CPU");
  if (data) parts.push(`${fmtInt(data.n_features)} features`);
  if (run) parts.push(`batch ${run.config.batch_size}`, `latente ${run.config.latent_dim}`);
  if (state.started_at) {
    const end = st.ended_at || serverNow();
    parts.push(`${st.state === "running" ? "rodando há" : "duração"} ${fmtDuration(end - state.started_at)}`);
  }
  $("run-meta").textContent = parts.length ? parts.join(" · ") : "Conectando ao treinamento…";

  $("status-pill").dataset.state = st.state;
  $("status-label").textContent = { running: "Em execução", done: "Concluído", error: "Erro" }[st.state] || st.state;

  $("error-banner").hidden = st.state !== "error";
  $("error-text").textContent = st.error || "";

  const p = state.progress;
  document.title =
    st.state === "running" && p && p.task === "train"
      ? `Época ${p.epoch}/${p.epochs} · Monitor do Autoencoder`
      : st.state === "done" ? "✓ Monitor do Autoencoder" : "Monitor do Autoencoder";
}

function renderStepper() {
  const plan = state.plan || [];
  const ol = $("stepper");
  const signature = plan.map((s) => s.id).join("|");
  if (ol.dataset.signature !== signature) {
    ol.dataset.signature = signature;
    ol.innerHTML = plan
      .map(
        (s, i) => `<li class="step pending" data-id="${esc(s.id)}">
          <span class="step-mark"><span>${i + 1}</span></span>
          <span class="step-text"><span class="step-label">${esc(s.label)}</span><span class="step-time">&nbsp;</span></span>
        </li>`
      )
      .join("");
  }

  const phases = Object.fromEntries((state.phases || []).map((p) => [p.id, p]));
  const st = state.status || {};
  ol.querySelectorAll(".step").forEach((li, i) => {
    const p = phases[li.dataset.id];
    let cls = "pending";
    let mark = String(i + 1);
    let time = " ";
    if (p) {
      cls = st.state === "error" && st.phase === p.id ? "error" : p.ended_at != null ? "done" : "active";
      mark = cls === "done" ? "✓" : cls === "error" ? "✕" : "";
      time = fmtDuration((p.ended_at || serverNow()) - p.started_at);
    }
    li.className = `step ${cls}`;
    li.querySelector(".step-mark span").textContent = mark;
    li.querySelector(".step-time").textContent = time;
    if (cls === "active") li.setAttribute("aria-current", "step");
    else li.removeAttribute("aria-current");
  });
}

// ------------------------------------------------------------------ progresso
function setMeter(n, { label = "", right = "", frac = 0, indeterminate = false, hidden = false }) {
  $(`m${n}`).hidden = hidden;
  if (hidden) return;
  $(`m${n}-label`).textContent = label;
  $(`m${n}-right`).textContent = right;
  const bar = $(`m${n}-bar`);
  bar.classList.toggle("indeterminate", indeterminate);
  if (indeterminate) {
    bar.removeAttribute("aria-valuenow");
    return;
  }
  const pct = Math.max(0, Math.min(1, frac || 0)) * 100;
  $(`m${n}-fill`).style.width = `${pct}%`;
  bar.setAttribute("aria-valuenow", String(Math.round(pct)));
}

function renderProgress() {
  const st = state.status || { state: "running" };
  const phase = currentPhase();
  const p = state.progress;
  const note = $("task-note");

  if (st.state !== "running") {
    const plan = state.plan || [];
    const done = (state.phases || []).filter((ph) => ph.ended_at != null && ph.id !== st.phase).length;
    $("task-title").textContent = st.state === "done" ? "Execução concluída" : "Execução interrompida";
    $("task-elapsed").textContent =
      state.started_at && st.ended_at ? `total ${fmtDuration(st.ended_at - state.started_at)}` : "";
    setMeter(1, {
      label: st.state === "done" ? "Todas as etapas" : `Falhou em: ${labelOf(st.phase)}`,
      right: `${st.state === "done" ? plan.length : done} de ${plan.length} etapas`,
      frac: st.state === "done" ? 1 : plan.length ? done / plan.length : 0,
    });
    setMeter(2, { hidden: true });
    note.textContent =
      st.state === "done"
        ? "O servidor segue no ar até Ctrl+C no terminal."
        : "Veja o erro no topo e o log abaixo.";
    return;
  }

  $("task-title").textContent = phase ? labelOf(phase.id) : "Aguardando o script…";
  $("task-elapsed").textContent = phase ? `nesta etapa há ${fmtDuration(serverNow() - phase.started_at)}` : "";

  if (p && p.task === "train") {
    const f = p.steps ? p.step / p.steps : 0;
    const overall = (p.epoch - 1 + f) / p.epochs;
    const epochs = state.epochs || [];
    const avg = epochs.length ? epochs.reduce((a, e) => a + (e.duration || 0), 0) / epochs.length : null;
    const eta = p.step > 0 && p.steps ? (p.elapsed / p.step) * (p.steps - p.step) : null;
    setMeter(1, { label: `Época ${p.epoch} de ${p.epochs}`, right: fmtPct(overall, 0), frac: overall });
    setMeter(2, {
      label: `Batch ${fmtInt(p.step)} de ${fmtInt(p.steps)}`,
      right: eta != null ? `≈ ${fmtDuration(eta)} para fechar a época` : "",
      frac: f,
    });
    note.textContent = avg
      ? `Média de ${fmtDuration(avg)} por época · o early stopping pode encerrar antes da época ${p.epochs}.`
      : "A loss parcial da época aparece como ponto vazado na curva de treinamento.";
  } else if (p && p.task === "predict") {
    const eta = p.step > 0 && p.steps ? (p.elapsed / p.step) * (p.steps - p.step) : null;
    setMeter(1, {
      label: p.label,
      right: p.steps ? `${fmtInt(p.step)} / ${fmtInt(p.steps)} batches` : "",
      frac: p.steps ? p.step / p.steps : 0,
    });
    setMeter(2, { hidden: true });
    note.textContent = eta != null && p.step < p.steps ? `≈ ${fmtDuration(eta)} restantes` : "";
  } else {
    setMeter(1, { label: phase ? `${labelOf(phase.id)}…` : "Aguardando", indeterminate: true });
    setMeter(2, { hidden: true });
    note.textContent =
      phase && phase.id.startsWith("load") ? "Lendo o CSV — esta etapa não tem progresso granular." : "";
  }
}

// ------------------------------------------------------------------ indicadores
function setTile(id, value, sub) {
  $(id).textContent = value;
  $(`${id}-sub`).textContent = sub || " ";
}

function renderTiles() {
  const epochs = state.epochs || [];
  const tr = state.training || {};
  const p = state.progress;
  const lastEpoch = last(epochs);
  const total = tr.epochs || (state.run && state.run.config.epochs);
  const training = p && p.task === "train";

  let epochSub = total ? `de ${total}` : "";
  if (tr.stopped_epoch) epochSub = "encerrado pelo early stopping";
  else if (tr.done) epochSub = "treino concluído";
  else if (tr.patience != null && epochs.length) epochSub += ` · paciência ${tr.wait}/${tr.patience}`;
  setTile("t-epoch", String(training ? p.epoch : lastEpoch ? lastEpoch.epoch : "—"), epochSub);

  setTile(
    "t-loss",
    fmt(lastEpoch && lastEpoch.loss),
    training && ok(p.loss) ? `parcial da época ${p.epoch}: ${fmt(p.loss)}` : lastEpoch ? `época ${lastEpoch.epoch}` : ""
  );

  setTile("t-best", fmt(tr.best_val_loss), tr.best_epoch ? `época ${tr.best_epoch}` : "");

  const lrs = epochs.map((e) => e.lr).filter(ok);
  const cuts = lrs.filter((v, i) => i > 0 && v < lrs[i - 1]).length;
  setTile("t-lr", fmtSci(last(lrs)), lrs.length ? (cuts ? `reduzido ${cuts}×` : "sem reduções") : "");

  const th = state.threshold;
  setTile("t-thr", fmt(th && th.value), th ? `μ + ${fmt(th.k, 1)}·σ` : "");

  const internal = state.eval_test;
  setTile(
    "t-auc",
    fmt(state.eval_val && state.eval_val.auc),
    internal ? `falso positivo interno: ${fmtPct(internal.false_positives / internal.n)}` : ""
  );
}

// ------------------------------------------------------------------ curva de treinamento
function createTrainChart() {
  const t = tokens();
  return new Chart($("c-train"), {
    type: "line",
    data: {
      datasets: [lineDataset(t, "Treino", t.s1), lineDataset(t, "Validação interna", t.s2)],
    },
    options: {
      ...baseOptions(t),
      scales: {
        x: scaleOptions(t, { title: "Época", min: 0, ticks: { precision: 0 } }),
        y: scaleOptions(t, { title: "MAE", border: false, grace: "5%" }),
      },
      plugins: {
        legend: { display: false },
        crosshair: { sameX: true },
        tooltip: tooltipOptions(t, {
          title: (items) => {
            const raw = items[0] && items[0].raw;
            if (!raw) return "";
            if (raw.live) {
              const p = state.progress;
              return `Época ${p.epoch} · em andamento (${fmtPct(p.step / p.steps, 0)})`;
            }
            return `Época ${raw.x}`;
          },
          label: (item) => ` ${item.dataset.label}: ${fmt(item.parsed.y)}`,
        }),
      },
    },
    plugins: [crosshairPlugin],
  });
}

function renderTrainChart() {
  const epochs = state.epochs || [];
  const p = state.progress;
  const train = epochs.filter((e) => ok(e.loss)).map((e) => ({ x: e.epoch, y: e.loss }));
  const val = epochs.filter((e) => ok(e.val_loss)).map((e) => ({ x: e.epoch, y: e.val_loss }));

  if (p && p.task === "train" && p.steps && ok(p.loss) && p.step > 0) {
    const x = p.epoch - 1 + p.step / p.steps;
    const lastX = train.length ? last(train).x : 0;
    if (x > lastX) train.push({ x, y: p.loss, live: true });
  }

  const card = $("card-train");
  card.classList.toggle("has-data", train.length > 0);
  $("lg-loss").textContent = train.length ? fmt(last(train).y) : "—";
  $("lg-val").textContent = val.length ? fmt(last(val).y) : "—";
  if (!train.length) return;

  const chart = ensureChart("train", createTrainChart);
  chart.data.datasets[0].data = train;
  chart.data.datasets[1].data = val;
  chart.options.scales.x.max = Math.max(5, Math.ceil(last(train).x));
  chart.update("none");
}

function renderEpochTable() {
  const epochs = state.epochs || [];
  const best = state.training && state.training.best_epoch;
  $("epoch-rows").innerHTML = epochs
    .slice()
    .reverse()
    .map(
      (e) => `<tr class="${e.epoch === best ? "best" : ""}">
        <td>${e.epoch}${e.epoch === best ? ' <span class="muted">· melhor</span>' : ""}</td>
        <td>${fmt(e.loss, 5)}</td><td>${fmt(e.val_loss, 5)}</td>
        <td>${fmtSci(e.lr)}</td><td>${fmtDuration(e.duration)}</td>
      </tr>`
    )
    .join("");
}

// ------------------------------------------------------------------ learning rate
function createLrChart() {
  const t = tokens();
  return new Chart($("c-lr"), {
    type: "line",
    data: { datasets: [lineDataset(t, "Learning rate", t.s1, { stepped: true })] },
    options: {
      ...baseOptions(t),
      scales: {
        x: scaleOptions(t, { title: "Época", min: 1, ticks: { precision: 0 } }),
        y: scaleOptions(t, {
          type: "logarithmic",
          border: false,
          ticks: {
            callback: (v) => {
              const mantissa = Math.round(v / 10 ** Math.floor(Math.log10(v)));
              return [1, 2, 5].includes(mantissa) ? fmtSci(v) : "";
            },
          },
        }),
      },
      plugins: {
        legend: { display: false },
        crosshair: { sameX: true },
        tooltip: tooltipOptions(t, {
          title: (items) => (items[0] ? `Época ${items[0].raw.x}` : ""),
          label: (item) => ` Learning rate: ${fmtSci(item.parsed.y)}`,
        }),
      },
    },
    plugins: [crosshairPlugin],
  });
}

function renderLrChart() {
  const pts = (state.epochs || []).filter((e) => ok(e.lr) && e.lr > 0).map((e) => ({ x: e.epoch, y: e.lr }));
  $("card-lr").classList.toggle("has-data", pts.length > 0);
  if (!pts.length) return;
  const chart = ensureChart("lr", createLrChart);
  chart.data.datasets[0].data = pts;
  chart.options.scales.x.max = Math.max(5, last(pts).x);
  // folga de 2× em cada ponta para o marcador final não encostar no eixo
  const lrs = pts.map((p) => p.y);
  chart.options.scales.y.min = Math.min(...lrs) / 2;
  chart.options.scales.y.max = Math.max(...lrs) * 2;
  chart.update("none");
}

// ------------------------------------------------------------------ modelo e dados
function renderModel() {
  const m = state.model;
  const d = state.data;
  const cfg = state.run && state.run.config;
  let html = "";

  if (m) {
    const rows = [{ kind: "io", label: `entrada · ${fmtInt(m.n_features)}`, units: m.n_features }];
    const addLayers = (layers, part) => {
      layers.forEach((l) => {
        if (l.type === "Dropout") rows.push({ kind: "drop", label: `dropout ${fmt(l.rate, 1)}` });
        else if (l.units) rows.push({ kind: part, label: `${fmtInt(l.units)} · ${l.activation}`, units: l.units });
      });
    };
    addLayers(m.encoder, "enc");
    const bottleneck = last(rows.filter((r) => r.units));
    if (bottleneck) {
      bottleneck.kind = "latent";
      bottleneck.label += " · latente";
    }
    addLayers(m.decoder, "dec");
    const out = last(rows);
    if (out && out.units === m.n_features) {
      out.kind = "io";
      out.label = `saída · ${fmtInt(out.units)} · ${out.label.split(" · ")[1]}`;
    }
    const maxLog = Math.log2(Math.max(...rows.filter((r) => r.units).map((r) => r.units)));
    html += `<div class="arch" aria-label="Arquitetura do autoencoder">${rows
      .map((r) => {
        const width = r.units ? Math.max(6, (Math.log2(r.units) / maxLog) * 100) : 0;
        const bar = r.units ? `<div class="arch-bar" style="width:${width.toFixed(1)}%"></div>` : "";
        return `<div class="arch-row ${r.kind}"><div class="arch-bar-wrap">${bar}</div><span class="arch-label">${esc(r.label)}</span></div>`;
      })
      .join("")}</div>`;
  }

  const kv = [];
  if (d) {
    kv.push(["Treino", `${fmtInt(d.train)} normais`]);
    kv.push(["Validação interna", `${fmtInt(d.internal)} normais`]);
    if (d.external != null) {
      kv.push(["Validação externa", `${fmtInt(d.external - d.external_anomaly)} normais · ${fmtInt(d.external_anomaly)} anômalas`]);
    }
  }
  if (cfg) {
    kv.push(["Épocas (máx.) · batch", `${cfg.epochs} · ${cfg.batch_size}`]);
    kv.push(["Threshold", `μ + ${fmt(cfg.threshold_k, 1)}·σ`]);
  }
  if (kv.length) html += `<dl class="kv">${kv.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("")}</dl>`;

  $("model-body").innerHTML = html || '<p class="muted small">Aguardando o carregamento dos dados…</p>';
  $("model-params").textContent = m ? `${fmtInt(m.params)} parâmetros` : "";
}

// ------------------------------------------------------------------ threshold
function createThresholdChart() {
  const t = tokens();
  return new Chart($("c-thr"), {
    type: "line",
    data: {
      datasets: [
        histDataset(t, "Erros (treino normal)", t.s1),
        lineDataset(t, "Normal ajustada", t.s2, { pointRadius: 0 }),
      ],
    },
    options: {
      ...baseOptions(t),
      scales: {
        x: scaleOptions(t, { title: "MAE de reconstrução", ticks: { callback: (v) => fmt(v, 3) } }),
        y: scaleOptions(t, { title: "Densidade", border: false, beginAtZero: true }),
      },
      plugins: {
        legend: { display: false },
        refLines: { lines: [] },
        tooltip: tooltipOptions(t, {
          title: binTitle,
          label: (item) => ` ${item.dataset.label}: ${fmt(item.parsed.y, 1)}`,
        }),
      },
    },
    plugins: [crosshairPlugin, refLinesPlugin],
  });
}

function renderThreshold() {
  const th = state.threshold;
  $("card-thr").classList.toggle("has-data", !!th);
  if (!th) return;
  $("thr-sub").textContent =
    `Threshold = μ + ${fmt(th.k, 1)}·σ = ${fmt(th.mu)} + ${fmt(th.k, 1)} × ${fmt(th.sigma)} = ${fmt(th.value)}`;
  const chart = ensureChart("thr", createThresholdChart);
  const edges = th.hist.edges;
  chart.data.datasets[0].data = stepPoints(edges, th.hist.values);
  chart.data.datasets[0].edges = edges;
  chart.data.datasets[1].data = th.pdf.x.map((x, i) => ({ x, y: th.pdf.y[i] }));
  const lo = edges[0];
  const hi = Math.max(last(edges), th.value);
  chart.options.scales.x.min = lo;
  chart.options.scales.x.max = hi + (hi - lo) * 0.03;
  chart.options.plugins.refLines.lines = [{ x: th.value, label: `threshold ${fmt(th.value)}` }];
  chart.update("none");
}

// ------------------------------------------------------------------ avaliação
const EVAL_CARDS = { eval_test: "eval-test", eval_val: "eval-val" };

function renderEvals() {
  Object.keys(EVAL_CARDS).forEach(renderEval);
}

const mini = (label, value) => `<div class="mini"><span>${label}</span><strong>${value}</strong></div>`;

function renderEval(key) {
  const ev = state[key];
  const card = $(EVAL_CARDS[key]);
  if (!ev) return;
  card.querySelector('[data-role="n"]').textContent = `${fmtInt(ev.n)} sequências`;
  if (!ev.n_anomaly) {
    // só normais (validação interna): o que dá para medir são os falsos positivos
    const fpRate = ev.n ? ev.false_positives / ev.n : null;
    card.querySelector('[data-role="body"]').innerHTML = `
      <div class="mini-tiles">
        ${mini("Sequências normais", fmtInt(ev.n))}
        ${mini("Acima do threshold", fmtInt(ev.false_positives))}
        ${mini("Falso positivo", fmtPct(fpRate))}
        ${mini("Especificidade", fmt(ok(fpRate) ? 1 - fpRate : null, 3))}
      </div>
      <p class="muted small">Sequências normais que o modelo não viu no treino: mostram quantas o
        threshold marcaria como anomalia por engano. Não há anomalias neste conjunto, então
        recall e ROC só existem na validação externa.</p>`;
    return;
  }
  const r = ev.report || {};
  const an = r.Anomalia || {};
  card.querySelector('[data-role="body"]').innerHTML = `
    <div class="mini-tiles">
      ${mini("ROC-AUC", fmt(ev.auc, 3))}
      ${mini("Acurácia", fmt(r.accuracy, 3))}
      ${mini("Precisão · anomalia", fmt(an.precision, 3))}
      ${mini("Recall · anomalia", fmt(an.recall, 3))}
      ${mini("F1 · anomalia", fmt(an["f1-score"], 3))}
    </div>
    <div class="eval-body">${confusionTable(ev.confusion)}${reportTable(r)}</div>`;
}

function confusionTable(cm) {
  const t = tokens();
  const names = ["Normal", "Anomalia"];
  const cellNames = [["Verdadeiro negativo", "Falso positivo"], ["Falso negativo", "Verdadeiro positivo"]];
  const body = cm
    .map((row, i) => {
      const total = row.reduce((a, b) => a + b, 0);
      const cells = row
        .map((v, j) => {
          const frac = total ? v / total : 0;
          const bg = t.ramp[Math.round(frac * (t.ramp.length - 1))];
          return `<td style="background:${bg};color:${inkOn(bg)}" title="${cellNames[i][j]}: ${fmtInt(v)} (${fmtPct(frac)} da classe ${names[i].toLowerCase()})">
            <strong>${fmtInt(v)}</strong><small>${fmtPct(frac)}</small></td>`;
        })
        .join("");
      return `<tr><th scope="row">${names[i]}</th>${cells}</tr>`;
    })
    .join("");
  return `<table class="cm">
    <caption>Linhas: classe real · colunas: prevista</caption>
    <thead>
      <tr><th></th>${names.map((n) => `<th scope="col">${n}</th>`).join("")}</tr>
    </thead>
    <tbody>${body}</tbody>
  </table>`;
}

function reportTable(r) {
  const rows = [
    ["Normal", r.Normal],
    ["Anomalia", r.Anomalia],
    ["Macro", r["macro avg"]],
    ["Ponderada", r["weighted avg"]],
  ].filter(([, v]) => v);
  return `<div class="table-scroll"><table class="data-table">
    <thead><tr><th>Classe</th><th>Precisão</th><th>Recall</th><th>F1</th><th>Suporte</th></tr></thead>
    <tbody>${rows
      .map(
        ([name, v]) => `<tr><td>${name}</td><td>${fmt(v.precision, 3)}</td><td>${fmt(v.recall, 3)}</td>
          <td>${fmt(v["f1-score"], 3)}</td><td>${fmtInt(v.support)}</td></tr>`
      )
      .join("")}</tbody>
  </table></div>`;
}

// ------------------------------------------------------------------ ROC
function createRocChart() {
  const t = tokens();
  const rocLine = (label, color) => lineDataset(t, label, color, { pointRadius: 0 });
  return new Chart($("c-roc"), {
    type: "line",
    data: {
      datasets: [
        rocLine("Validação externa", t.s1),
        {
          label: "Aleatório",
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: t.axis,
          borderWidth: 1,
          borderDash: [4, 4],
          pointRadius: 0,
          pointHoverRadius: 0,
          noTooltip: true,
        },
      ],
    },
    options: {
      ...baseOptions(t),
      scales: {
        x: scaleOptions(t, { title: "Taxa de falsos positivos", min: 0, max: 1, ticks: { callback: (v) => fmt(v, 1) } }),
        y: scaleOptions(t, { title: "Taxa de verdadeiros positivos", min: 0, max: 1, border: false, ticks: { callback: (v) => fmt(v, 1) } }),
      },
      plugins: {
        legend: { display: false },
        tooltip: tooltipOptions(t, {
          title: () => "",
          label: (item) => ` ${item.dataset.label}: FPR ${fmt(item.parsed.x, 3)} · TPR ${fmt(item.parsed.y, 3)}`,
        }),
      },
    },
    plugins: [crosshairPlugin],
  });
}

function renderRoc() {
  const ev = state.eval_val;
  const has = !!(ev && ev.roc);
  $("roc-auc").textContent = has ? `AUC ${fmt(ev.auc, 3)}` : "";
  $("card-roc").classList.toggle("has-data", has);
  if (!has) return;
  const chart = ensureChart("roc", createRocChart);
  chart.data.datasets[0].data = ev.roc.fpr.map((x, j) => ({ x, y: ev.roc.tpr[j] }));
  chart.update("none");
}

// ------------------------------------------------------------------ erro por classe
function createDistChart() {
  const t = tokens();
  return new Chart($("c-dist"), {
    type: "line",
    data: { datasets: [histDataset(t, "Normal", t.s1), histDataset(t, "Anomalia", t.s2)] },
    options: {
      ...baseOptions(t),
      scales: {
        x: scaleOptions(t, { title: "MAE de reconstrução", ticks: { callback: (v) => fmt(v, 3) } }),
        y: scaleOptions(t, {
          title: "% da classe",
          border: false,
          beginAtZero: true,
          ticks: { callback: (v) => fmtPct(v, 0) },
        }),
      },
      plugins: {
        legend: { display: false },
        refLines: { lines: [] },
        tooltip: tooltipOptions(t, {
          title: binTitle,
          label: (item) => ` ${item.dataset.label}: ${fmtPct(item.parsed.y)} da classe`,
        }),
      },
    },
    plugins: [crosshairPlugin, refLinesPlugin],
  });
}

function renderDist() {
  const available = ["eval_test", "eval_val"].filter((k) => state[k]);
  const key = distSet && state[distSet] ? distSet : last(available);
  document.querySelectorAll("#card-dist .segmented button").forEach((b) => {
    b.setAttribute("aria-selected", String(b.dataset.set === key));
    b.disabled = !state[b.dataset.set];
  });
  $("card-dist").classList.toggle("has-data", !!key);
  if (!key) return;

  const ev = state[key];
  const h = ev.hist;
  const chart = ensureChart("dist", createDistChart);
  chart.data.datasets[0].data = stepPoints(h.edges, h.normal);
  chart.data.datasets[0].edges = h.edges;
  chart.data.datasets[1].data = ev.n_anomaly ? stepPoints(h.edges, h.anomaly) : [];
  chart.data.datasets[1].edges = h.edges;
  const lo = h.edges[0];
  const hi = Math.max(last(h.edges), ev.threshold);
  chart.options.scales.x.min = lo;
  chart.options.scales.x.max = hi + (hi - lo) * 0.03;
  chart.options.plugins.refLines.lines = [{ x: ev.threshold, label: `threshold ${fmt(ev.threshold)}` }];
  chart.update("none");
}

// ------------------------------------------------------------------ início
document.querySelectorAll("#card-dist .segmented button").forEach((b) =>
  b.addEventListener("click", () => {
    distSet = b.dataset.set;
    renderDist();
  })
);

// relógios (tempo decorrido, etapa atual) andam mesmo sem mensagens novas
setInterval(() => {
  renderHeader();
  renderStepper();
  if (!state.status || state.status.state === "running") renderProgress();
}, 1000);

startLive(RENDERERS, {
  connLabel: (s) => {
    const finished = state.status && state.status.state !== "running";
    return s === "live" ? "Ao vivo" : s === "retry" ? (finished ? "Servidor encerrado" : "Reconectando…") : "Conectando…";
  },
});
