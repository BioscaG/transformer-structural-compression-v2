"""Token trajectories — every token's path through the residual stream.

For a chosen sentence, trace each TOKEN (not just CLS) through the 13
hidden states. Project to LDA-3D using the same supervised basis as the
galaxy formation, so geometry matches.

What you see:
  - ⟨CLS⟩ (red): travels far from origin toward its emotion's region —
    the aggregator's journey
  - ⟨SEP⟩ (orange): also aggregates but to a different region
  - Content tokens (blue): trajectories MUCH shorter — the residual stream
    preserves their local information, they barely move
  - Function words (the/is/and): almost stationary

This visualizes §2.2.3 (residual stream) and §2.3.2 (CLS as aggregator)
directly: the asymmetry of how different tokens move proves CLS is
special by design.
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from transformers import AutoModelForSequenceClassification

from viz import style as st
from viz.data.load_results import EMOTIONS_23, MODEL_CHECKPOINT
from viz.thesis_data import EXTENDED_CLUSTER_MAP, emotion_palette


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def _classify_token(tok: str) -> str:
    """Categorize a token: CLS / SEP / content / function."""
    if tok in ("⟨CLS⟩", "[CLS]"):
        return "CLS"
    if tok in ("⟨SEP⟩", "[SEP]"):
        return "SEP"
    if not tok:
        return "PAD"
    function_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "to", "of", "in", "on", "at", "for", "with", "and", "or",
                      "but", "i", "you", "he", "she", "it", "we", "they", "this",
                      "that", "these", "those", ".", ",", "!", "?", "'s", "n't"}
    if tok.lower() in function_words:
        return "function"
    return "content"


TYPE_COLOR = {
    "CLS": st.TERRA,
    "SEP": st.SAND,
    "content": st.BLUE,
    "function": st.INK_3,
    "PAD": "rgba(0,0,0,0)",
}


def build_trajectories_figure() -> go.Figure:
    # Load the token-level data
    data = np.load(CACHE_DIR / "token_trajectories.npz")
    hidden = data["hidden"].astype(np.float32)     # (n_sent, 13, T, 768)
    meta = json.loads((CACHE_DIR / "token_trajectories_meta.json").read_text())
    sentences = meta["sentences"]
    labels = meta["labels"]
    tokens = meta["tokens"]
    n_sent, n_layers, T, _ = hidden.shape
    print(f"Loaded {n_sent} sentences, {n_layers} layers, max {T} tokens")

    # Apply the same pooler+LDA used for galaxy formation, so the projection
    # matches geometrically.
    full_data = np.load(CACHE_DIR / "activations.npz")
    full_cls = full_data["cls_per_layer"]            # (N, 13, 768)
    full_meta = json.loads((CACHE_DIR / "meta.json").read_text())
    full_labels = full_meta["label_names"]

    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()
    bp = mdl.bert.pooler.dense.bias.detach().numpy()

    pooled_full = np.tanh(full_cls[:, -1, :] @ Wp.T + bp)
    label_to_int = {e: i for i, e in enumerate(sorted(set(full_labels)))}
    y = np.array([label_to_int[l] for l in full_labels])
    lda = LinearDiscriminantAnalysis(n_components=3).fit(pooled_full, y)

    # Project EVERY token's hidden state through pooler + LDA
    # Shape after pooler: (n_sent, n_layers, T, 768)
    pooled = np.tanh(np.einsum('sltd,pd->sltp', hidden, Wp) + bp)
    coords = np.zeros((n_sent, n_layers, T, 3), dtype=np.float32)
    for s in range(n_sent):
        for L in range(n_layers):
            coords[s, L] = lda.transform(pooled[s, L])

    # Centroids for context (the 23 emotion regions in the same projection)
    centroids = {}
    palette = emotion_palette(EMOTIONS_23)
    for emo in EMOTIONS_23:
        mask = np.array([l == emo for l in full_labels])
        if mask.any():
            centroids[emo] = lda.transform(pooled_full[mask]).mean(axis=0).tolist()

    # Layer labels
    layer_labels = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    # Build payload — round to 3 decimals to keep size reasonable
    coords_rounded = np.round(coords, 3).tolist()
    centroids_rounded = {e: [round(v, 3) for v in c] for e, c in centroids.items()}

    payload = {
        "sentences": sentences,
        "labels": labels,
        "tokens": tokens,
        "coords": coords_rounded,           # (n, L, T, 3)
        "centroids": centroids_rounded,
        "centroid_colors": {e: palette[e] for e in centroids},
        "layer_labels": layer_labels,
        "n_sentences": n_sent,
        "max_tokens": T,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    pal = json.dumps({
        "BG": st.BG, "INK": st.INK, "INK_2": st.INK_2, "INK_3": st.INK_3,
        "GRID": st.GRID, "SPINE": st.SPINE,
        "TERRA": st.TERRA, "SAND": st.SAND, "BLUE": st.BLUE,
    })

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Token trajectories · TFG Anatomía Emocional</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: "TeX Gyre Pagella","Palatino","Book Antiqua",serif;
    background: {st.BG}; color: {st.INK};
    margin: 0; padding: 16px 28px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px 0; font-weight: normal; }}
  h1 .acc {{ color: {st.TERRA}; }}
  .sub {{ color: {st.INK_3}; font-size: 13px; margin-bottom: 14px; max-width: 1100px; }}
  .controls {{
    display: flex; gap: 14px; align-items: center; margin: 12px 0;
    flex-wrap: wrap;
  }}
  select {{
    padding: 6px 10px; border: 0.5px solid {st.SPINE}; border-radius: 4px;
    background: white; font-family: inherit; font-size: 12.5px;
    color: {st.INK}; min-width: 360px;
  }}
  button {{
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 4px;
    padding: 6px 12px; font-size: 12px; color: {st.INK_2};
    cursor: pointer; font-family: inherit;
  }}
  button.primary {{ background: {st.TERRA}; color: white; border-color: {st.TERRA}; }}
  button:hover {{ background: {st.SAND}; color: {st.INK}; }}
  .layer-control {{
    display: flex; gap: 10px; align-items: center; margin: 8px 0 14px 0;
  }}
  .layer-control input[type=range] {{ flex: 1; max-width: 600px; }}
  .layer-label {{
    font-family: "Inter", monospace; font-size: 12.5px; color: {st.INK};
    min-width: 120px;
  }}
  .legend {{
    display: flex; gap: 16px; flex-wrap: wrap; font-size: 11.5px; color: {st.INK_3};
    margin-bottom: 8px;
  }}
  .legend .item {{ display: flex; align-items: center; gap: 6px; }}
  .legend .swatch {{
    width: 12px; height: 4px; display: inline-block;
  }}
  #plot {{
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 6px;
    padding: 4px;
  }}
</style>
</head>
<body>

<h1>Token <span class="acc">trajectories</span></h1>
<div class="sub">
  Las trayectorias de cada token (no solo el CLS) a través de las 13 capas.
  La proyección LDA-3D coincide con la del galaxy formation, así que los
  centroides de emoción están donde esperas. Verás que CLS viaja MUCHO
  (es el agregador), mientras que los tokens de contenido apenas se mueven
  — el residual stream preserva su información local. §2.2.3 + §2.3.2.
</div>

<div class="legend">
  <div class="item"><span class="swatch" style="background: {st.TERRA}; height: 6px"></span><b>⟨CLS⟩</b> · agregador</div>
  <div class="item"><span class="swatch" style="background: {st.SAND}; height: 6px"></span><b>⟨SEP⟩</b> · separador</div>
  <div class="item"><span class="swatch" style="background: {st.BLUE}"></span>contenido</div>
  <div class="item"><span class="swatch" style="background: {st.INK_3}"></span>palabras función</div>
  <div class="item">◆ centroides de emoción (referencia)</div>
</div>

<div class="controls">
  <label style="font-size: 13px; color: {st.INK_2}">Frase:</label>
  <select id="sentence-select"></select>
</div>

<div class="layer-control">
  <span class="layer-label" id="layer-display">Capa: Emb</span>
  <input type="range" id="layer-slider" min="0" max="12" value="12" step="1" />
  <button class="primary" id="play-btn">▶ Play</button>
  <button id="pause-btn">⏸ Pause</button>
  <button id="reset-btn">↺ Reset</button>
</div>

<div id="plot"></div>

<script>
const DATA = {payload_json};
const PAL = {pal};
const N_LAYERS = DATA.layer_labels.length;
let currentSentence = 0;
let currentLayer = N_LAYERS - 1;
let playInterval = null;

const TYPE_COLOR = {{
  "CLS": PAL.TERRA, "SEP": PAL.SAND,
  "content": PAL.BLUE, "function": PAL.INK_3,
  "PAD": "rgba(0,0,0,0)",
}};
const FUNCTION_WORDS = new Set([
  "the","a","an","is","are","was","were","be","been","to","of","in","on","at",
  "for","with","and","or","but","i","you","he","she","it","we","they","this",
  "that","these","those",".",",","!","?","'s","n't",
]);

function classifyToken(t) {{
  if (t === "⟨CLS⟩" || t === "[CLS]") return "CLS";
  if (t === "⟨SEP⟩" || t === "[SEP]") return "SEP";
  if (!t) return "PAD";
  if (FUNCTION_WORDS.has(t.toLowerCase())) return "function";
  return "content";
}}

const select = document.getElementById('sentence-select');
DATA.sentences.forEach((s, i) => {{
  const text = s.length > 80 ? s.slice(0, 77) + '…' : s;
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `[${{DATA.labels[i]}}] ${{text}}`;
  select.appendChild(opt);
}});

function buildTraces() {{
  const tokens = DATA.tokens[currentSentence];
  const sentenceCoords = DATA.coords[currentSentence];   // (L, T, 3)
  const traces = [];

  // 1) Centroid reference markers (faded)
  const cx = [], cy = [], cz = [], ctext = [], ccolor = [];
  Object.entries(DATA.centroids).forEach(([emo, p]) => {{
    cx.push(p[0]); cy.push(p[1]); cz.push(p[2]);
    ctext.push(emo); ccolor.push(DATA.centroid_colors[emo]);
  }});
  traces.push({{
    type: 'scatter3d', mode: 'markers+text',
    x: cx, y: cy, z: cz,
    marker: {{size: 6, color: ccolor, opacity: 0.35, symbol: 'diamond'}},
    text: ctext, textfont: {{size: 8, color: PAL.INK_3}},
    textposition: 'top center',
    hoverinfo: 'text',
    hovertext: ctext.map(e => `<b>${{e}}</b><br>centroide`),
    showlegend: false,
  }});

  // 2) For each token, build a trail (from L0 to currentLayer) + head marker
  for (let t = 0; t < tokens.length; t++) {{
    const tok = tokens[t];
    if (!tok) continue;     // skip pads
    const ttype = classifyToken(tok);
    const color = TYPE_COLOR[ttype];
    if (ttype === "PAD") continue;

    // Trail
    const tx = [], ty = [], tz = [];
    for (let L = 0; L <= currentLayer; L++) {{
      tx.push(sentenceCoords[L][t][0]);
      ty.push(sentenceCoords[L][t][1]);
      tz.push(sentenceCoords[L][t][2]);
    }}
    traces.push({{
      type: 'scatter3d', mode: 'lines',
      x: tx, y: ty, z: tz,
      line: {{color: color, width: ttype === "CLS" ? 5 : (ttype === "SEP" ? 4 : 2),
              opacity: ttype === "function" ? 0.4 : 0.85}},
      hoverinfo: 'skip', showlegend: false,
    }});
    // Head marker at currentLayer
    const head = sentenceCoords[currentLayer][t];
    const isSpecial = (ttype === "CLS" || ttype === "SEP");
    traces.push({{
      type: 'scatter3d', mode: 'markers+text',
      x: [head[0]], y: [head[1]], z: [head[2]],
      marker: {{
        size: isSpecial ? 13 : (ttype === "function" ? 5 : 8),
        color: color, opacity: isSpecial ? 0.95 : 0.75,
        line: {{color: PAL.INK, width: isSpecial ? 1.5 : 0.5}},
        symbol: isSpecial ? 'diamond' : 'circle',
      }},
      text: [tok], textposition: 'top center',
      textfont: {{size: isSpecial ? 12 : 9,
                  color: color, family: "serif"}},
      hovertext: [`<b>${{tok}}</b> (token ${{t}})<br>`
                  + `tipo: ${{ttype}}<br>`
                  + `capa: ${{DATA.layer_labels[currentLayer]}}`],
      hoverinfo: 'text', showlegend: false,
    }});
  }}

  return traces;
}}

const layout = {{
  title: {{text: '', font: {{size: 1}}}},
  paper_bgcolor: 'white',
  margin: {{l: 0, r: 0, t: 0, b: 0}},
  height: 720,
  font: {{family: '"TeX Gyre Pagella","Palatino",serif', color: PAL.INK_2}},
  scene: {{
    xaxis: {{title: 'LD1', showticklabels: false, backgroundcolor: PAL.BG,
             gridcolor: PAL.GRID, zerolinecolor: PAL.SPINE}},
    yaxis: {{title: 'LD2', showticklabels: false, backgroundcolor: PAL.BG,
             gridcolor: PAL.GRID, zerolinecolor: PAL.SPINE}},
    zaxis: {{title: 'LD3', showticklabels: false, backgroundcolor: PAL.BG,
             gridcolor: PAL.GRID, zerolinecolor: PAL.SPINE}},
    camera: {{eye: {{x: 1.7, y: 1.7, z: 1.0}}}},
    aspectmode: 'cube',
  }},
  showlegend: false,
}};

function render() {{
  Plotly.react('plot', buildTraces(), layout, {{displayModeBar: true, responsive: true}});
  document.getElementById('layer-display').textContent =
    'Capa: ' + DATA.layer_labels[currentLayer];
  document.getElementById('layer-slider').value = currentLayer;
}}

select.addEventListener('change', e => {{
  currentSentence = parseInt(e.target.value);
  render();
}});
document.getElementById('layer-slider').addEventListener('input', e => {{
  currentLayer = parseInt(e.target.value);
  render();
}});

function stopPlay() {{
  if (playInterval !== null) {{ clearInterval(playInterval); playInterval = null; }}
}}
document.getElementById('play-btn').addEventListener('click', () => {{
  stopPlay();
  if (currentLayer >= N_LAYERS - 1) currentLayer = 0;
  render();
  playInterval = setInterval(() => {{
    if (currentLayer >= N_LAYERS - 1) {{ stopPlay(); return; }}
    currentLayer += 1;
    render();
  }}, 700);
}});
document.getElementById('pause-btn').addEventListener('click', stopPlay);
document.getElementById('reset-btn').addEventListener('click', () => {{
  stopPlay(); currentLayer = 0; render();
}});

render();
</script>
</body>
</html>
"""
    return html


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    html = build_trajectories_figure()
    out = out_dir / "23_token_trajectories.html"
    out.write_text(html, encoding="utf-8")
    print(f"✓ wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return out


if __name__ == "__main__":
    main()
