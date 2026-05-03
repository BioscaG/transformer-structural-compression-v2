"""Decision fingerprint — pure HTML+JS implementation.

Plotly's frame system gets confused mixing dropdowns + sliders + play, so this
viz is built as a self-contained HTML page with embedded data and vanilla JS
controls. Same idea as compression_sandbox.

Each frame is one (sentence, layer) → 23-dim sigmoid vector from the
classifier head applied to the CLS at that layer. Pre-computed at build time
and embedded as JSON so the page loads offline without any model.

Controls:
  - Sentence dropdown: pick one of ~300 curated sentences
  - Layer slider: scrub from Emb to L11
  - Play / Pause / Reset: animate through the 13 layers for the current sentence
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
from transformers import AutoModelForSequenceClassification

from viz import style as st
from viz.data.load_results import EMOTIONS_23, MODEL_CHECKPOINT
from viz.thesis_data import EXTENDED_CLUSTER_MAP, emotion_palette as _emotion_palette


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def _curate_sentences(meta: dict, n_per_emotion: int = 13) -> list[int]:
    label_names = meta["label_names"]
    indices = []
    seen = {e: 0 for e in EMOTIONS_23}
    for i, lbl in enumerate(label_names):
        if lbl in seen and seen[lbl] < n_per_emotion:
            indices.append(i)
            seen[lbl] += 1
    return indices


LANG = {
    "es": {
        "frase":   "Frase",
        "capa":    "Capa",
        "layer_of": "{n} de 13",
        "of_word": "de",
        "frase_short": "Frase:",
        "gold":    "Gold",
        "top":     "Top predicción",
        "conf":    "Confianza top",
        "cofire":  "Co-firing (sigmoid &gt; 0.5)",
        "play":    "▶ Play",
        "pause":   "⏸ Pause",
        "reset":   "↺ Reset",
    },
    "en": {
        "frase":   "Sentence",
        "capa":    "Layer",
        "layer_of": "{n} of 13",
        "of_word": "of",
        "frase_short": "Sentence:",
        "gold":    "Gold",
        "top":     "Top prediction",
        "conf":    "Top confidence",
        "cofire":  "Co-firing (sigmoid &gt; 0.5)",
        "play":    "▶ Play",
        "pause":   "⏸ Pause",
        "reset":   "↺ Reset",
    },
}


def build_html(out_path: pathlib.Path, lang: str = "es") -> pathlib.Path:
    _L = LANG[lang]
    # ─── Load activations + classifier ───
    data = np.load(CACHE_DIR / "activations.npz")
    cls = data["cls_per_layer"]                      # (N, 13, 768)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    sentences = meta["sentences"]
    label_names = meta["label_names"]

    chosen = _curate_sentences(meta, n_per_emotion=13)
    chosen_cls = cls[chosen]                         # (n, 13, 768)
    chosen_sentences = [sentences[i] for i in chosen]
    chosen_labels = [label_names[i] for i in chosen]

    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    # The model applies a POOLER (Linear + tanh) to the CLS BEFORE the
    # classifier. Skipping the pooler gives garbage predictions, so we apply
    # the same transformation ourselves at every layer (logit-lens style).
    W_pool = mdl.bert.pooler.dense.weight.detach().numpy()  # (768, 768)
    b_pool = mdl.bert.pooler.dense.bias.detach().numpy()    # (768,)
    W = mdl.classifier.weight.detach().numpy()               # (23, 768)
    b = mdl.classifier.bias.detach().numpy()                 # (23,)

    # pooled[s, l, h] = tanh(CLS[s, l] @ W_pool.T + b_pool)
    pooled = np.tanh(np.einsum('slh,ph->slp', chosen_cls, W_pool) + b_pool)
    # Compute sigmoid output per (sentence, layer, emotion)
    logits = np.einsum('slh,eh->sle', pooled, W) + b
    sigmoids = 1.0 / (1.0 + np.exp(-logits))         # (n, 13, 23)

    # Order emotions by cluster for the polar wheel
    cluster_order = ["Positivas alta energía", "Negativas reactivas",
                     "Negativas internas", "Epistémicas",
                     "Orientadas al otro", "Baja especificidad"]
    ordered_emotions = []
    for cluster in cluster_order:
        members = [e for e in EMOTIONS_23
                   if EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") == cluster]
        ordered_emotions.extend(sorted(members))
    emo_to_col = {e: i for i, e in enumerate(EMOTIONS_23)}
    reorder = [emo_to_col[e] for e in ordered_emotions]
    sigmoids = sigmoids[:, :, reorder]

    palette = _emotion_palette(EMOTIONS_23)
    bar_colors = [palette[e] for e in ordered_emotions]
    cluster_per_emotion = [EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad")
                           for e in ordered_emotions]

    # ─── Truncate sigmoid array to JSON-friendly precision ───
    # Round to 3 decimal places to keep payload small (still plenty resolution)
    sigmoids_rounded = np.round(sigmoids, 3).tolist()

    # ─── Embed data + render HTML ───
    layer_labels = ["Emb"] + [f"L{i}" for i in range(12)]
    palette_json = json.dumps({
        "BG": st.BG, "INK": st.INK, "INK_2": st.INK_2, "INK_3": st.INK_3,
        "GRID": st.GRID, "SPINE": st.SPINE, "TERRA": st.TERRA, "SAGE": st.SAGE,
        "SAND": st.SAND,
    })

    payload = {
        "sentences": chosen_sentences,
        "labels": chosen_labels,
        "emotions": ordered_emotions,
        "clusters": cluster_per_emotion,
        "bar_colors": bar_colors,
        "sigmoids": sigmoids_rounded,
        "layer_labels": layer_labels,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Decision Fingerprint · TFG Anatomía Emocional</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: "TeX Gyre Pagella", "Palatino", "Book Antiqua", serif;
    background: {st.BG}; color: {st.INK};
    margin: 0; padding: 16px 28px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px 0; font-weight: normal; }}
  h1 .acc {{ color: {st.TERRA}; }}
  .sub {{ color: {st.INK_3}; font-size: 13px; margin-bottom: 16px; }}
  .grid {{
    display: grid; grid-template-columns: 1fr 320px; gap: 22px;
    align-items: start;
  }}
  .controls {{
    background: white; border: 0.5px solid {st.SPINE};
    border-radius: 6px; padding: 16px;
  }}
  .controls h2 {{
    font-size: 11px; text-transform: uppercase; letter-spacing: 1.2px;
    color: {st.INK_2}; margin: 0 0 10px 0; font-weight: 500;
    font-family: "Inter", sans-serif;
  }}
  .row {{ margin-bottom: 14px; }}
  .row .lbl {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 12.5px; color: {st.INK_2}; margin-bottom: 4px;
  }}
  .row .val {{
    font-family: "Inter", monospace; font-size: 11.5px; color: {st.INK};
  }}
  select, input[type=range] {{ width: 100%; font-family: inherit; }}
  input[type=range] {{
    -webkit-appearance: none; appearance: none;
    height: 4px; background: {st.SPINE}; border-radius: 2px; outline: none;
  }}
  input[type=range]::-webkit-slider-thumb {{
    -webkit-appearance: none; appearance: none;
    width: 14px; height: 14px; background: {st.INK_2};
    border-radius: 50%; cursor: pointer;
  }}
  input[type=range]::-moz-range-thumb {{
    width: 14px; height: 14px; background: {st.INK_2};
    border-radius: 50%; cursor: pointer; border: none;
  }}
  input[type=range]::-moz-range-track {{
    background: {st.SPINE}; height: 4px; border-radius: 2px;
  }}
  select {{
    padding: 6px 8px; border: 0.5px solid {st.SPINE};
    border-radius: 4px; background: white; color: {st.INK};
    font-size: 12.5px;
  }}
  .btnrow {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  button {{
    background: white; border: 0.5px solid {st.SPINE};
    border-radius: 4px; padding: 6px 12px; font-size: 12px;
    color: {st.INK_2}; cursor: pointer; font-family: inherit;
  }}
  button:hover {{ background: #F4F2EC; color: {st.INK}; border-color: {st.INK_3}; }}
  .info {{
    margin-top: 14px; padding: 12px;
    background: #FAFAF8; border: 0.5px solid {st.SPINE}; border-radius: 5px;
    font-size: 12.5px; line-height: 1.5;
  }}
  .info .text {{
    color: {st.INK}; font-style: italic; font-size: 13px;
    border-left: 2px solid {st.TERRA}; padding-left: 10px; margin: 6px 0;
  }}
  .info .row {{ margin-bottom: 4px; }}
  .info .row .lbl {{ color: {st.INK_3}; font-size: 11px; }}
  .info .row .val {{ color: {st.INK}; font-family: "Inter", monospace; font-size: 11.5px; }}
  .legend-tip {{
    margin-top: 10px; font-size: 10.5px; color: {st.INK_3};
    line-height: 1.5;
  }}
  .legend-tip b {{ color: {st.INK_2}; }}
  #plot {{
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 6px;
    padding: 4px;
  }}
  @media (max-width: 1000px) {{
    .grid {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<div class="grid">
  <div id="plot"></div>

  <div class="controls">
    <h2>{_L['frase']}</h2>
    <select id="sentence-select"></select>

    <h2 style="margin-top: 18px">{_L['capa']}</h2>
    <div class="row">
      <div class="lbl">
        <span id="layer-label">L11</span>
        <span class="val" id="layer-num">{_L['layer_of'].format(n=12)}</span>
      </div>
      <input type="range" id="layer-slider" min="0" max="12" step="1" value="12" />
    </div>

    <div class="btnrow">
      <button id="play-btn">{_L['play']}</button>
      <button id="pause-btn">{_L['pause']}</button>
      <button id="reset-btn">{_L['reset']}</button>
    </div>

    <div class="info">
      <div class="row"><span class="lbl">{_L['frase_short']}</span></div>
      <div class="text" id="info-text"></div>
      <div class="row"><span class="lbl">{_L['gold']}</span><span class="val" id="info-gold"></span></div>
      <div class="row"><span class="lbl">{_L['top']}</span><span class="val" id="info-top"></span></div>
      <div class="row"><span class="lbl">{_L['conf']}</span><span class="val" id="info-conf"></span></div>
      <div class="row"><span class="lbl">{_L['cofire']}</span><span class="val" id="info-cofire"></span></div>
    </div>

  </div>
</div>

<script>
const DATA = {payload_json};
const PAL = {palette_json};
const N_LAYERS = DATA.layer_labels.length;
const N_EMOTIONS = DATA.emotions.length;
let currentSentence = 0;
let currentLayer = N_LAYERS - 1;
let playInterval = null;

// Populate sentence dropdown
const select = document.getElementById('sentence-select');
DATA.sentences.forEach((s, i) => {{
  const text = s.length > 60 ? s.slice(0, 57) + '…' : s;
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `[${{DATA.labels[i]}}] ${{text}}`;
  select.appendChild(opt);
}});

// Initialize plot
function makeBarTrace() {{
  const sigmoids = DATA.sigmoids[currentSentence][currentLayer];
  const gold = DATA.labels[currentSentence];
  const lineWidths = DATA.emotions.map(e => e === gold ? 3.5 : 0.6);
  const lineColors = DATA.emotions.map(e => e === gold ? PAL.INK : 'white');
  return {{
    type: 'barpolar',
    r: sigmoids,
    theta: DATA.emotions,
    marker: {{
      color: DATA.bar_colors,
      line: {{color: lineColors, width: lineWidths}},
    }},
    opacity: 0.92,
    hovertemplate: '<b>%{{theta}}</b><br>sigmoid: %{{r:.3f}}<extra></extra>',
  }};
}}

const layout = {{
  title: {{text: '', font: {{size: 1}}}},
  paper_bgcolor: PAL.BG,
  polar: {{
    bgcolor: PAL.BG,
    radialaxis: {{
      visible: true, range: [0, 1.0],
      tickfont: {{size: 9, color: PAL.INK_3, family: 'serif'}},
      gridcolor: PAL.GRID, linecolor: PAL.SPINE,
    }},
    angularaxis: {{
      direction: 'clockwise', rotation: 90,
      tickfont: {{size: 10.5, color: PAL.INK_2, family: 'serif'}},
      gridcolor: PAL.GRID, linecolor: PAL.SPINE,
    }},
  }},
  margin: {{l: 60, r: 60, t: 30, b: 30}},
  height: 640,
  font: {{family: '"TeX Gyre Pagella", "Palatino", serif', color: PAL.INK_2}},
  showlegend: false,
}};

Plotly.newPlot('plot', [makeBarTrace()], layout, {{
  displayModeBar: false, responsive: true,
}});

function updatePlot() {{
  const sigmoids = DATA.sigmoids[currentSentence][currentLayer];
  const gold = DATA.labels[currentSentence];
  const lineWidths = DATA.emotions.map(e => e === gold ? 3.5 : 0.6);
  const lineColors = DATA.emotions.map(e => e === gold ? PAL.INK : 'white');
  Plotly.restyle('plot', {{
    r: [sigmoids],
    'marker.line.width': [lineWidths],
    'marker.line.color': [lineColors],
  }}, [0]);
  document.getElementById('layer-label').textContent = DATA.layer_labels[currentLayer];
  document.getElementById('layer-num').textContent = `${{currentLayer + 1}} {_L['of_word']} ${{N_LAYERS}}`;
  document.getElementById('layer-slider').value = currentLayer;

  // Info panel
  const sigmoids_now = DATA.sigmoids[currentSentence][currentLayer];
  const goldEmo = DATA.labels[currentSentence];
  const topIdx = sigmoids_now.indexOf(Math.max(...sigmoids_now));
  const topEmo = DATA.emotions[topIdx];
  const topVal = sigmoids_now[topIdx];
  const cofire = sigmoids_now
    .map((v, i) => ({{e: DATA.emotions[i], v}}))
    .filter(x => x.v > 0.5)
    .sort((a, b) => b.v - a.v)
    .map(x => `${{x.e}} (${{x.v.toFixed(2)}})`);

  document.getElementById('info-text').textContent = '"' + DATA.sentences[currentSentence] + '"';
  document.getElementById('info-gold').textContent = goldEmo;
  document.getElementById('info-top').textContent = topEmo + (topEmo === goldEmo ? '  ✓' : '  ✗');
  document.getElementById('info-conf').textContent = topVal.toFixed(3);
  document.getElementById('info-cofire').textContent =
    cofire.length === 0 ? '— ninguna' : cofire.length + ': ' + cofire.slice(0, 4).join(', ') + (cofire.length > 4 ? '…' : '');
}}

// Wire up controls
select.addEventListener('change', e => {{
  currentSentence = parseInt(e.target.value);
  updatePlot();
}});

document.getElementById('layer-slider').addEventListener('input', e => {{
  currentLayer = parseInt(e.target.value);
  updatePlot();
}});

function stopPlay() {{
  if (playInterval !== null) {{
    clearInterval(playInterval);
    playInterval = null;
  }}
}}

document.getElementById('play-btn').addEventListener('click', () => {{
  stopPlay();
  // Start from L0 if currently at the last layer, else from current
  if (currentLayer >= N_LAYERS - 1) currentLayer = 0;
  updatePlot();
  playInterval = setInterval(() => {{
    if (currentLayer >= N_LAYERS - 1) {{
      stopPlay();
      return;
    }}
    currentLayer += 1;
    updatePlot();
  }}, 700);
}});

document.getElementById('pause-btn').addEventListener('click', stopPlay);

document.getElementById('reset-btn').addEventListener('click', () => {{
  stopPlay();
  currentLayer = 0;
  updatePlot();
}});

// Initial render
updatePlot();
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
    return build_html(out_dir / "14_decision_fingerprint.html")


if __name__ == "__main__":
    main()
