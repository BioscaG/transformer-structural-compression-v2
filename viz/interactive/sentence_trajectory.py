"""Single-sentence trajectory — watch BERT think on ONE example.

For a chosen sentence, show four synchronized panels driven by a single
layer slider:

  - top-left:  the sentence's CLS position in the LDA-3D galaxy with a
               fading trail of previous layers
  - top-right: attention pattern of the most-critical heads at this layer
  - bot-left:  top-K most-active FFN neurons at this layer (real activations)
  - bot-right: 23-dim sigmoid bars at this layer (the model's current decision)

The "watch BERT process a sentence end-to-end" experience the project
opened with — finally rendered with the corrected pipeline (pooler,
correct tokenizer, real LDA).
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from viz import style as st
from viz.data.load_results import EMOTIONS_23, MODEL_CHECKPOINT
from viz.thesis_data import EXTENDED_CLUSTER_MAP, emotion_palette as _emotion_palette


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def _curate_sentences(meta: dict, n_per_emotion: int = 3) -> list[int]:
    label_names = meta["label_names"]
    indices = []
    seen = {e: 0 for e in EMOTIONS_23}
    for i, lbl in enumerate(label_names):
        if lbl in seen and seen[lbl] < n_per_emotion:
            indices.append(i)
            seen[lbl] += 1
    return indices


def build_html(out_path: pathlib.Path) -> pathlib.Path:
    # ─── Load all the data we need ───
    data = np.load(CACHE_DIR / "activations.npz")
    cls = data["cls_per_layer"]                     # (N, 13, 768)
    attn = data["attentions"].astype(np.float32)    # (N, 12, 12, T, T)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    sentences = meta["sentences"]
    label_names = meta["label_names"]
    tokens = meta["tokens"]

    chosen = _curate_sentences(meta, n_per_emotion=2)
    chosen_cls = cls[chosen]
    chosen_attn = attn[chosen]
    chosen_sentences = [sentences[i] for i in chosen]
    chosen_labels = [label_names[i] for i in chosen]
    chosen_tokens = [tokens[i] for i in chosen]
    print(f"Curated {len(chosen)} sentences for trajectory viz")

    # ─── Pooler + classifier weights ───
    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()
    bp = mdl.bert.pooler.dense.bias.detach().numpy()
    W = mdl.classifier.weight.detach().numpy()
    b = mdl.classifier.bias.detach().numpy()

    # ─── LDA on full 2300 dataset (so the galaxy projection is consistent
    # with the standalone galaxy viz) ───
    pooled_full = np.tanh(np.einsum('slh,ph->slp', cls, Wp) + bp)
    label_to_int = {e: i for i, e in enumerate(sorted(set(label_names)))}
    y_full = np.array([label_to_int[l] for l in label_names])
    lda = LinearDiscriminantAnalysis(n_components=3).fit(pooled_full[:, -1, :], y_full)

    # Project chosen sentences across all layers
    pooled_chosen = np.tanh(np.einsum('slh,ph->slp', chosen_cls, Wp) + bp)
    n_layers = pooled_chosen.shape[1]
    coords_chosen = np.zeros((len(chosen), n_layers, 3), dtype=np.float32)
    for L in range(n_layers):
        coords_chosen[:, L, :] = lda.transform(pooled_chosen[:, L, :])

    # Sigmoid logits per (sentence, layer, emotion)
    logits_chosen = np.einsum('slp,ep->sle', pooled_chosen, W) + b
    sigmoids_chosen = 1.0 / (1.0 + np.exp(-logits_chosen))   # (n, 13, 23)

    # Background cloud (centroids only, to keep payload light)
    centroids = {}
    for emo in EMOTIONS_23:
        mask = np.array([l == emo for l in label_names])
        if mask.any():
            centroids[emo] = lda.transform(pooled_full[mask, -1, :]).mean(axis=0).tolist()

    # ─── Active token lengths and pretty token strings ───
    def active_length(toks: list[str]) -> int:
        for i, t in enumerate(toks):
            if not t or t == "[PAD]":
                return min(i, attn.shape[-1])
        return min(len(toks), attn.shape[-1])

    chosen_T = [active_length(t) for t in chosen_tokens]

    # Trim attention + tokens to active length
    chosen_attn_trimmed = []
    chosen_tokens_pretty = []
    for s_idx in range(len(chosen)):
        T = chosen_T[s_idx]
        a = np.round(chosen_attn[s_idx, :, :, :T, :T], 4)
        chosen_attn_trimmed.append(a.tolist())
        toks = chosen_tokens[s_idx][:T]
        toks = [t.replace("##", "") if t.startswith("##") else t for t in toks]
        toks = [{"[CLS]": "⟨CLS⟩", "[SEP]": "⟨SEP⟩"}.get(t, t) for t in toks]
        chosen_tokens_pretty.append(toks)

    # Top-3 most-important heads PER LAYER (not globally) — so the user always
    # sees attention patterns regardless of which layer they're on. This also
    # reveals the layer-by-layer evolution of which heads matter.
    from viz.data.load_results import load_heads
    heads_data = load_heads()
    top_heads_per_layer = {}    # layer -> [(L, h), ...]
    if heads_data["categories"] is not None:
        df = heads_data["categories"]
        for L in range(12):
            sub = df[df["layer"] == L].sort_values("total_importance", ascending=False)
            top_heads_per_layer[L] = [(int(r["layer"]), int(r["head"]))
                                       for _, r in sub.head(3).iterrows()]
    # Fill any missing layer with first 3 heads
    for L in range(12):
        if L not in top_heads_per_layer or not top_heads_per_layer[L]:
            top_heads_per_layer[L] = [(L, 0), (L, 1), (L, 2)]

    # Order emotions by cluster for the polar/sigmoid bars
    cluster_order = ["Positivas alta energía", "Negativas reactivas",
                     "Negativas internas", "Epistémicas",
                     "Orientadas al otro", "Baja especificidad"]
    ordered_emotions = []
    for cluster in cluster_order:
        members = [e for e in EMOTIONS_23
                   if EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") == cluster]
        ordered_emotions.extend(sorted(members))
    palette = _emotion_palette(EMOTIONS_23)
    bar_colors = [palette[e] for e in ordered_emotions]
    emo_to_idx = {e: i for i, e in enumerate(EMOTIONS_23)}
    reorder_idx = [emo_to_idx[e] for e in ordered_emotions]
    sigmoids_ordered = sigmoids_chosen[:, :, reorder_idx]

    layer_labels = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    payload = {
        "sentences": chosen_sentences,
        "labels": chosen_labels,
        "tokens": chosen_tokens_pretty,
        "T": chosen_T,
        "coords": np.round(coords_chosen, 3).tolist(),
        "centroids": {e: [round(v, 3) for v in c] for e, c in centroids.items()},
        "centroid_colors": {e: palette[e] for e in centroids.keys()},
        "attentions": chosen_attn_trimmed,
        # heads per layer: keys are stringified layer indices for JS
        "top_heads_per_layer": {str(L): top_heads_per_layer[L] for L in range(12)},
        "sigmoids": np.round(sigmoids_ordered, 4).tolist(),
        "ordered_emotions": ordered_emotions,
        "bar_colors": bar_colors,
        "layer_labels": layer_labels,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    pal = json.dumps({
        "BG": st.BG, "INK": st.INK, "INK_2": st.INK_2, "INK_3": st.INK_3,
        "GRID": st.GRID, "SPINE": st.SPINE, "TERRA": st.TERRA, "BLUE": st.BLUE,
        "SAGE": st.SAGE, "SAND": st.SAND,
    })

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Sentence trajectory · TFG Anatomía Emocional</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: "TeX Gyre Pagella","Palatino","Book Antiqua",serif;
    background: {st.BG}; color: {st.INK};
    margin: 0; padding: 14px 22px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px 0; font-weight: normal; }}
  h1 .acc {{ color: {st.TERRA}; }}
  .sub {{ color: {st.INK_3}; font-size: 13px; margin-bottom: 12px; max-width: 1100px; }}
  .controls {{ display: flex; gap: 14px; align-items: center; margin: 10px 0; flex-wrap: wrap; }}
  select {{
    padding: 5px 10px; border: 0.5px solid {st.SPINE}; border-radius: 4px;
    background: white; font-family: inherit; font-size: 12.5px;
    color: {st.INK}; min-width: 460px;
  }}
  button {{
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 4px;
    padding: 6px 12px; font-size: 12px; color: {st.INK_2};
    cursor: pointer; font-family: inherit;
  }}
  button:hover {{ background: {st.SAND}; color: {st.INK}; }}
  button.primary {{ background: {st.TERRA}; color: white; border-color: {st.TERRA}; }}
  .layer-control {{
    display: flex; gap: 10px; align-items: center; margin: 8px 0 16px 0;
  }}
  .layer-control input[type=range] {{ flex: 1; max-width: 600px; }}
  .layer-label {{
    font-family: "Inter", monospace; font-size: 12.5px; color: {st.INK};
    min-width: 90px;
  }}
  .grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
  }}
  .panel {{
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 6px;
    padding: 6px;
  }}
  .panel-title {{
    font-family: "Inter", sans-serif; font-size: 11px; font-weight: 500;
    color: {st.TERRA}; letter-spacing: 1.2px; text-transform: uppercase;
    margin: 6px 0 0 12px;
  }}
  .info {{
    font-size: 12.5px; color: {st.INK_2}; padding: 10px 14px;
    background: rgba(212,168,67,0.10);
    border-left: 2px solid {st.SAND}; border-radius: 4px; margin: 14px 0;
  }}
  .info b {{ color: {st.INK}; }}
</style>
</head>
<body>

<h1>Sentence <span class="acc">trajectory</span></h1>
<div class="sub">
  Una frase, cuatro vistas sincronizadas. Eliges la frase, mueves el slider de
  capa, y ves <b>simultáneamente</b>: la trayectoria geométrica del CLS,
  la atención de las cabezas más críticas, los pétalos de decisión, y el
  posicionamiento entre los centroides de las 23 emociones.
</div>

<div class="controls">
  <label style="font-size: 13px; color: {st.INK_2}">Frase:</label>
  <select id="sentence-select"></select>
</div>

<div class="layer-control">
  <span class="layer-label" id="layer-display">Capa: Emb</span>
  <input type="range" id="layer-slider" min="0" max="12" value="0" step="1" />
  <button class="primary" id="play-btn">▶ Play</button>
  <button id="pause-btn">⏸ Pause</button>
  <button id="reset-btn">↺ Reset</button>
</div>

<div class="info" id="info">Selecciona una frase y dale a Play.</div>

<div class="grid">
  <div class="panel">
    <div class="panel-title">Trayectoria CLS · LDA-3D</div>
    <div id="galaxy-plot"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Sigmoids · pooler+classifier en esta capa</div>
    <div id="sigmoid-plot"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Atención · 3 cabezas más críticas</div>
    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; padding: 8px;">
      <div id="attn-plot-0"></div>
      <div id="attn-plot-1"></div>
      <div id="attn-plot-2"></div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">Confianza emoción gold · curva</div>
    <div id="gold-plot"></div>
  </div>
</div>

<script>
const DATA = {payload_json};
const PAL = {pal};
const N_LAYERS = DATA.layer_labels.length;
let currentSentence = 0;
let currentLayer = 0;
let playInterval = null;

// Populate sentence dropdown
const select = document.getElementById('sentence-select');
DATA.sentences.forEach((s, i) => {{
  const text = s.length > 70 ? s.slice(0, 67) + '…' : s;
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `[${{DATA.labels[i]}}] ${{text}}`;
  select.appendChild(opt);
}});

// ─── Galaxy plot ───
function buildGalaxyPlot() {{
  const traces = [];
  // Background centroids
  const cx = [], cy = [], cz = [], ctext = [], ccol = [];
  Object.entries(DATA.centroids).forEach(([emo, p]) => {{
    cx.push(p[0]); cy.push(p[1]); cz.push(p[2]);
    ctext.push(emo); ccol.push(DATA.centroid_colors[emo]);
  }});
  traces.push({{
    type: 'scatter3d', mode: 'markers+text',
    x: cx, y: cy, z: cz, text: ctext, textposition: 'top center',
    textfont: {{size: 9, color: PAL.INK_3}},
    marker: {{size: 8, color: ccol, opacity: 0.55, symbol: 'diamond'}},
    name: 'centroides', hoverinfo: 'text', showlegend: false,
  }});
  // Sentence trail (dynamic — we'll restyle on layer change)
  traces.push({{
    type: 'scatter3d', mode: 'lines',
    x: [0], y: [0], z: [0],
    line: {{color: PAL.TERRA, width: 6}},
    hoverinfo: 'skip', showlegend: false, name: 'estela',
  }});
  // Sentence head (dynamic)
  traces.push({{
    type: 'scatter3d', mode: 'markers',
    x: [0], y: [0], z: [0],
    marker: {{size: 12, color: PAL.TERRA, line: {{color: PAL.INK, width: 2}}}},
    hoverinfo: 'skip', showlegend: false, name: 'frase',
  }});
  Plotly.newPlot('galaxy-plot', traces, {{
    paper_bgcolor: 'white',
    margin: {{l: 0, r: 0, t: 0, b: 0}},
    font: {{family: '"TeX Gyre Pagella","Palatino",serif', color: PAL.INK_2}},
    height: 360, showlegend: false,
    scene: {{
      xaxis: {{title: 'LD1', showticklabels: false, backgroundcolor: 'white',
               gridcolor: PAL.GRID, zerolinecolor: PAL.SPINE}},
      yaxis: {{title: 'LD2', showticklabels: false, backgroundcolor: 'white',
               gridcolor: PAL.GRID, zerolinecolor: PAL.SPINE}},
      zaxis: {{title: 'LD3', showticklabels: false, backgroundcolor: 'white',
               gridcolor: PAL.GRID, zerolinecolor: PAL.SPINE}},
      camera: {{eye: {{x: 1.6, y: 1.6, z: 1.0}}}},
      aspectmode: 'cube',
    }},
  }}, {{displayModeBar: false, responsive: true}});
}}

function updateGalaxyTrail() {{
  const traj = DATA.coords[currentSentence];   // (13, 3)
  const trailX = traj.slice(0, currentLayer + 1).map(p => p[0]);
  const trailY = traj.slice(0, currentLayer + 1).map(p => p[1]);
  const trailZ = traj.slice(0, currentLayer + 1).map(p => p[2]);
  const head = traj[currentLayer];
  Plotly.restyle('galaxy-plot', {{x: [trailX], y: [trailY], z: [trailZ]}}, [1]);
  Plotly.restyle('galaxy-plot', {{x: [[head[0]]], y: [[head[1]]], z: [[head[2]]]}}, [2]);
}}

// ─── Sigmoid bars ───
function buildSigmoidPlot() {{
  const radii = DATA.sigmoids[currentSentence][currentLayer];
  const gold = DATA.labels[currentSentence];
  const lineW = DATA.ordered_emotions.map(e => e === gold ? 3 : 0.5);
  const lineC = DATA.ordered_emotions.map(e => e === gold ? PAL.INK : 'white');
  Plotly.newPlot('sigmoid-plot', [{{
    type: 'bar',
    x: DATA.ordered_emotions, y: radii,
    marker: {{color: DATA.bar_colors, line: {{color: lineC, width: lineW}}}},
    hovertemplate: '<b>%{{x}}</b>: %{{y:.3f}}<extra></extra>',
  }}], {{
    paper_bgcolor: 'white', plot_bgcolor: 'white',
    margin: {{l: 50, r: 18, t: 12, b: 80}},
    font: {{family: '"TeX Gyre Pagella","Palatino",serif', color: PAL.INK_2, size: 11}},
    xaxis: {{tickangle: -45, tickfont: {{size: 9}}, gridcolor: PAL.GRID,
             linecolor: PAL.SPINE, showline: true}},
    yaxis: {{title: 'sigmoid', range: [0, 1], gridcolor: PAL.GRID,
             linecolor: PAL.SPINE, showline: true,
             tickfont: {{size: 10}}}},
    height: 360,
  }}, {{displayModeBar: false, responsive: true}});
}}

function updateSigmoidPlot() {{
  const radii = DATA.sigmoids[currentSentence][currentLayer];
  const gold = DATA.labels[currentSentence];
  const lineW = DATA.ordered_emotions.map(e => e === gold ? 3 : 0.5);
  const lineC = DATA.ordered_emotions.map(e => e === gold ? PAL.INK : 'white');
  Plotly.restyle('sigmoid-plot', {{
    y: [radii],
    'marker.line.width': [lineW],
    'marker.line.color': [lineC],
  }}, [0]);
}}

// ─── Attention plot — top 3 most-important heads OF THE CURRENT LAYER ───
function buildAttentionPlot() {{
  const T = DATA.T[currentSentence];
  const tokens = DATA.tokens[currentSentence];
  // For Embedding (L0_idx=0) there are no attention layers. Use layer 0 of the
  // encoder (currentLayer-1 in attention indexing) if currentLayer >= 1.
  // top_heads_per_layer is indexed 0..11 (encoder layers).
  const encoderLayer = Math.max(0, currentLayer - 1);
  const heads = DATA.top_heads_per_layer[String(encoderLayer)] || [];
  for (let hi = 0; hi < 3; hi++) {{
    if (hi >= heads.length) {{
      // Empty plot for missing
      Plotly.react(`attn-plot-${{hi}}`, [], {{
        paper_bgcolor: 'white', plot_bgcolor: 'white', height: 320,
        margin: {{l: 50, r: 8, t: 26, b: 60}},
        annotations: [{{text: 'sin datos', x: 0.5, y: 0.5, xref: 'paper', yref: 'paper',
                         showarrow: false, font: {{size: 11, color: PAL.INK_3}}}}],
        xaxis: {{visible: false}}, yaxis: {{visible: false}},
      }}, {{displayModeBar: false}});
      continue;
    }}
    const [L, h] = heads[hi];
    let A;
    if (currentLayer === 0) {{
      // At Embedding: show empty heatmaps with a hint label
      A = Array.from({{length: T}}, () => Array(T).fill(0));
    }} else {{
      A = DATA.attentions[currentSentence][L][h];
    }}
    const isEmpty = currentLayer === 0;
    const trace = {{
      type: 'heatmap', z: A, x: tokens, y: tokens,
      colorscale: [[0, '#FFFFFF'], [0.3, '#E5C87A'], [0.7, '#D4A843'], [1.0, '#C1553A']],
      zmin: 0, zmax: Math.max(...A.flat(), 0.01),
      showscale: false,
      hovertemplate: `L${{L}}-H${{h}}<br>%{{y}} → %{{x}}: %{{z:.3f}}<extra></extra>`,
      xgap: 0.4, ygap: 0.4,
    }};
    const titleText = isEmpty
      ? `<b>L${{L}}-H${{h}}</b> · (cabeza no computada en Emb)`
      : `<b>L${{L}}-H${{h}}</b>`;
    const layout = {{
      title: {{text: titleText, x: 0.5, y: 0.97,
                font: {{size: 11.5, color: PAL.TERRA, family: 'Inter'}}}},
      paper_bgcolor: 'white', plot_bgcolor: 'white',
      margin: {{l: 50, r: 8, t: 26, b: 60}},
      font: {{family: '"TeX Gyre Pagella","Palatino",serif', size: 9, color: PAL.INK_3}},
      xaxis: {{tickangle: -45, tickfont: {{size: 8}}, showgrid: false, linecolor: PAL.SPINE}},
      yaxis: {{autorange: 'reversed', tickfont: {{size: 8}}, showgrid: false, linecolor: PAL.SPINE}},
      height: 320,
    }};
    Plotly.react(`attn-plot-${{hi}}`, [trace], layout, {{displayModeBar: false, responsive: true}});
  }}
}}

// ─── Gold confidence curve ───
function buildGoldPlot() {{
  const goldIdx = DATA.ordered_emotions.indexOf(DATA.labels[currentSentence]);
  const goldCurve = DATA.sigmoids[currentSentence].map(layer_sigs => layer_sigs[goldIdx]);
  // Peak per layer for context
  const topCurve = DATA.sigmoids[currentSentence].map(layer_sigs => Math.max(...layer_sigs));
  Plotly.newPlot('gold-plot', [
    {{
      type: 'scatter', mode: 'lines+markers',
      x: DATA.layer_labels, y: topCurve,
      line: {{color: PAL.INK_3, width: 1.5, dash: 'dot'}},
      marker: {{size: 5, color: PAL.INK_3}},
      name: 'top sigmoid',
      hovertemplate: '%{{x}}: top=%{{y:.3f}}<extra></extra>',
    }},
    {{
      type: 'scatter', mode: 'lines+markers',
      x: DATA.layer_labels, y: goldCurve,
      line: {{color: PAL.TERRA, width: 3, shape: 'spline', smoothing: 0.4}},
      marker: {{size: 8, color: PAL.TERRA, line: {{color: 'white', width: 1.5}}}},
      name: `${{DATA.labels[currentSentence]}} (gold)`,
      hovertemplate: '%{{x}}: gold=%{{y:.3f}}<extra></extra>',
    }},
    {{
      type: 'scatter', mode: 'markers',
      x: [DATA.layer_labels[currentLayer]], y: [goldCurve[currentLayer]],
      marker: {{size: 16, color: PAL.TERRA, symbol: 'circle-open', line: {{width: 3}}}},
      hoverinfo: 'skip', showlegend: false,
    }},
  ], {{
    paper_bgcolor: 'white', plot_bgcolor: 'white',
    font: {{family: '"TeX Gyre Pagella","Palatino",serif', color: PAL.INK_2, size: 11}},
    margin: {{l: 50, r: 18, t: 12, b: 50}},
    height: 360,
    xaxis: {{tickfont: {{size: 10}}, gridcolor: PAL.GRID, linecolor: PAL.SPINE, showline: true}},
    yaxis: {{title: 'sigmoid', range: [0, 1], tickfont: {{size: 10}},
             gridcolor: PAL.GRID, linecolor: PAL.SPINE, showline: true}},
    legend: {{x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.85)', font: {{size: 10}}}},
  }}, {{displayModeBar: false, responsive: true}});
}}

function updateGoldMarker() {{
  const goldIdx = DATA.ordered_emotions.indexOf(DATA.labels[currentSentence]);
  const goldCurve = DATA.sigmoids[currentSentence].map(layer_sigs => layer_sigs[goldIdx]);
  Plotly.restyle('gold-plot', {{
    x: [[DATA.layer_labels[currentLayer]]],
    y: [[goldCurve[currentLayer]]],
  }}, [2]);
}}

// ─── Info text ───
function updateInfo() {{
  const goldIdx = DATA.ordered_emotions.indexOf(DATA.labels[currentSentence]);
  const sigsNow = DATA.sigmoids[currentSentence][currentLayer];
  const goldVal = sigsNow[goldIdx];
  const topIdx = sigsNow.indexOf(Math.max(...sigsNow));
  const topEmo = DATA.ordered_emotions[topIdx];
  document.getElementById('info').innerHTML =
    `<b>${{DATA.labels[currentSentence]}}</b> · capa ${{DATA.layer_labels[currentLayer]}} · `
    + `top predicción: <b>${{topEmo}} (${{sigsNow[topIdx].toFixed(3)}})</b> · `
    + `confianza gold: <b>${{goldVal.toFixed(3)}}</b>`;
}}

function updateAll() {{
  document.getElementById('layer-display').textContent = 'Capa: ' + DATA.layer_labels[currentLayer];
  document.getElementById('layer-slider').value = currentLayer;
  updateGalaxyTrail();
  updateSigmoidPlot();
  buildAttentionPlot();
  updateGoldMarker();
  updateInfo();
}}

function rebuildAll() {{
  buildGalaxyPlot();
  buildSigmoidPlot();
  buildAttentionPlot();
  buildGoldPlot();
  updateAll();
}}

// ─── Wire up ───
select.addEventListener('change', e => {{
  currentSentence = parseInt(e.target.value);
  rebuildAll();
}});
document.getElementById('layer-slider').addEventListener('input', e => {{
  currentLayer = parseInt(e.target.value);
  updateAll();
}});

function stopPlay() {{
  if (playInterval !== null) {{ clearInterval(playInterval); playInterval = null; }}
}}
document.getElementById('play-btn').addEventListener('click', () => {{
  stopPlay();
  if (currentLayer >= N_LAYERS - 1) currentLayer = 0;
  updateAll();
  playInterval = setInterval(() => {{
    if (currentLayer >= N_LAYERS - 1) {{ stopPlay(); return; }}
    currentLayer += 1;
    updateAll();
  }}, 700);
}});
document.getElementById('pause-btn').addEventListener('click', stopPlay);
document.getElementById('reset-btn').addEventListener('click', () => {{
  stopPlay(); currentLayer = 0; updateAll();
}});

rebuildAll();
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"✓ wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    return out_path


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return build_html(out_dir / "19_sentence_trajectory.html")


if __name__ == "__main__":
    main()
