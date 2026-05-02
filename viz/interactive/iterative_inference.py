"""Iterative inference — the U-shape of the logit lens across layers.

When you apply the trained pooler + classifier to every layer's [CLS]
(known as "logit lens"), the average per-emotion sigmoid traces a
characteristic U:

  - Early layers (Emb-L3): the pre-activation often saturates the pooler's
    tanh because L0-L3 CLS statistics differ from L12's. After saturation
    the classifier sees ±1 vectors and fires many emotions weakly →
    high "diffuse" probabilities.
  - Middle layers (L4-L9): the CLS is in transition — neither embedding-like
    nor task-aligned. tanh outputs near zero → classifier returns ~bias →
    sigmoids collapse near 0. The model is "thinking" but not deciding.
  - Late layers (L10-L11): CLS reaches the natural input space of the
    pooler+classifier (which were trained for L12). Tanh works as designed,
    one direction lights up, others stay quiet → confident decision.

This is the classic logit-lens U-shape (Nostalgebraist 2020; Belrose et
al. NeurIPS 2023, "Tuned Lens"). Documenting it on YOUR fine-tune
reinforces §5.1 (crystallization) and §5.2 (FFN-L11 bottleneck) of the
memoria from a complementary angle.
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification

from viz import style as st
from viz.data.load_results import EMOTIONS_23, MODEL_CHECKPOINT
from viz.thesis_data import EXTENDED_CLUSTER_MAP, emotion_palette as _emotion_palette


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"
CRYSTAL_CSV = (pathlib.Path(__file__).resolve().parents[2]
               / "results" / "csvs" / "notebook4"
               / "crystallization_layers.csv")


def _curate_sentences(meta: dict, n_per_emotion: int = 13) -> list[int]:
    label_names = meta["label_names"]
    indices = []
    seen = {e: 0 for e in EMOTIONS_23}
    for i, lbl in enumerate(label_names):
        if lbl in seen and seen[lbl] < n_per_emotion:
            indices.append(i)
            seen[lbl] += 1
    return indices


def build_html(out_path: pathlib.Path) -> pathlib.Path:
    # ─── Load activations + classifier ───
    data = np.load(CACHE_DIR / "activations.npz")
    cls = data["cls_per_layer"]                       # (N, 13, 768)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    sentences = meta["sentences"]
    label_names = meta["label_names"]

    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()
    bp = mdl.bert.pooler.dense.bias.detach().numpy()
    W = mdl.classifier.weight.detach().numpy()
    b = mdl.classifier.bias.detach().numpy()

    # Sigmoid output per (sentence, layer, emotion) — full 2300 dataset
    pooled = np.tanh(np.einsum('slh,ph->slp', cls, Wp) + bp)
    logits = np.einsum('slp,ep->sle', pooled, W) + b
    sigmoids = 1.0 / (1.0 + np.exp(-logits))           # (N, 13, 23)

    # ─── Aggregate U-curves across all 2300 sentences ───
    n_layers = sigmoids.shape[1]
    layer_labels = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]
    label_to_idx = {e: i for i, e in enumerate(EMOTIONS_23)}
    y = np.array([label_to_idx[l] for l in label_names])

    def _agg(mask):
        """Compute (top1, gold, sum, sum_std, n) curves over a sentence mask."""
        s = sigmoids[mask]
        y_m = y[mask]
        return dict(
            top1=s.max(axis=2).mean(axis=0).tolist(),
            gold=s[np.arange(len(y_m)), :, y_m].mean(axis=0).tolist(),
            sum=s.sum(axis=2).mean(axis=0).tolist(),
            sum_std=s.sum(axis=2).std(axis=0).tolist(),
            n=int(mask.sum()),
        )

    all_mask = np.ones(len(y), dtype=bool)
    agg_all = _agg(all_mask)
    top1_per_layer    = agg_all["top1"]
    gold_per_layer    = agg_all["gold"]
    sum_per_layer     = agg_all["sum"]
    sum_std_per_layer = agg_all["sum_std"]

    # ─── Crystallization groups (early/mid/late) ─────────────────────────
    crystal_df = pd.read_csv(CRYSTAL_CSV)
    emo_to_xlayer = dict(zip(crystal_df["emotion"],
                             crystal_df["crystallization_layer"].astype(int)))

    def _bucket(layer: int) -> str:
        if layer <= 3:   return "temprano"   # L0-L2
        if layer <= 7:   return "medio"      # L3-L6
        return "tardio"                       # L7-L11

    emo_to_group = {e: _bucket(emo_to_xlayer.get(e, 11))
                    for e in EMOTIONS_23}

    label_names_arr = np.array(label_names)
    groups_data = {}
    groups_emotions = {}
    for g in ("temprano", "medio", "tardio"):
        emos_in_g = [e for e, gg in emo_to_group.items() if gg == g]
        groups_emotions[g] = emos_in_g
        mask = np.isin(label_names_arr, emos_in_g)
        groups_data[g] = _agg(mask)

    # ─── Per-sentence curated subset for the picker ───
    chosen = _curate_sentences(meta, n_per_emotion=13)
    chosen_sentences = [sentences[i] for i in chosen]
    chosen_labels = [label_names[i] for i in chosen]
    # Sigmoid evolution per (chosen_sentence, layer, emotion)
    chosen_sigmoids = sigmoids[chosen]                 # (n_chosen, 13, 23)

    # ─── Order emotions by cluster for legend grouping ───
    cluster_order = ["Positivas alta energía", "Negativas reactivas",
                     "Negativas internas", "Epistémicas",
                     "Orientadas al otro", "Baja especificidad"]
    ordered_emotions = []
    for cluster in cluster_order:
        members = [e for e in EMOTIONS_23
                   if EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad") == cluster]
        ordered_emotions.extend(sorted(members))

    palette = _emotion_palette(EMOTIONS_23)
    cluster_for_emo = {e: EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad")
                       for e in ordered_emotions}

    # Reorder the per-sentence sigmoid columns to match `ordered_emotions`
    # so the JS that iterates over DATA.emotions[ei] reads the correct
    # column from sigs[layer][ei]. (Without this the gold-line traces
    # the wrong emotion's curve.)
    emo_to_col = {e: i for i, e in enumerate(EMOTIONS_23)}
    reorder = [emo_to_col[e] for e in ordered_emotions]
    chosen_sigmoids = chosen_sigmoids[:, :, reorder]

    # Round to 4 decimals to keep payload small
    chosen_sigmoids_r = np.round(chosen_sigmoids, 4).tolist()

    payload = {
        "layer_labels": layer_labels,
        "emotions": ordered_emotions,
        "clusters": [cluster_for_emo[e] for e in ordered_emotions],
        "palette": [palette[e] for e in ordered_emotions],
        "agg": {
            "all":      {"top1": top1_per_layer,
                         "gold": gold_per_layer,
                         "sum":  sum_per_layer,
                         "sum_std": sum_std_per_layer,
                         "n":    int(len(y))},
            "temprano": groups_data["temprano"],
            "medio":    groups_data["medio"],
            "tardio":   groups_data["tardio"],
        },
        "groups_emotions": groups_emotions,
        "chosen": {
            "sentences": chosen_sentences,
            "labels": chosen_labels,
            "sigmoids": chosen_sigmoids_r,
        },
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    pal = json.dumps({
        "BG": st.BG, "INK": st.INK, "INK_2": st.INK_2, "INK_3": st.INK_3,
        "GRID": st.GRID, "SPINE": st.SPINE,
        "TERRA": st.TERRA, "SAGE": st.SAGE, "BLUE": st.BLUE, "SAND": st.SAND,
        "PLUM": st.PLUM,
    })

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Iterative inference · TFG Anatomía Emocional</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: "TeX Gyre Pagella","Palatino","Book Antiqua",serif;
    background: {st.BG}; color: {st.INK};
    margin: 0; padding: 16px 28px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px 0; font-weight: normal; }}
  h1 .acc {{ color: {st.TERRA}; }}
  .sub {{ color: {st.INK_3}; font-size: 13px; margin-bottom: 18px; max-width: 1100px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 22px; }}
  .panel {{
    background: white; border: 0.5px solid {st.SPINE};
    border-radius: 6px; padding: 6px;
  }}
  .panel-title {{
    font-family: "Inter", sans-serif; font-size: 11px; font-weight: 500;
    color: {st.TERRA}; letter-spacing: 1.2px; text-transform: uppercase;
    margin: 8px 0 0 12px;
  }}
  .panel-sub {{
    color: {st.INK_3}; font-size: 12px; margin: 2px 0 8px 12px;
  }}
  .controls {{
    display: flex; gap: 12px; align-items: center; margin: 18px 0 8px 0;
    flex-wrap: wrap;
  }}
  .group-toggle {{
    display: flex; gap: 4px; padding: 4px 12px 8px 12px;
    flex-wrap: wrap;
  }}
  .group-toggle button {{
    background: transparent; border: 0.5px solid {st.SPINE};
    color: {st.INK_2}; font-family: "Inter", sans-serif;
    font-size: 11px; letter-spacing: 0.05em;
    padding: 5px 11px; border-radius: 2px; cursor: pointer;
    text-transform: uppercase;
    transition: all 0.18s ease;
  }}
  .group-toggle button:hover {{
    color: {st.INK}; border-color: {st.INK_3};
  }}
  .group-toggle button.active {{
    background: {st.TERRA}; color: white; border-color: {st.TERRA};
  }}
  select {{
    padding: 5px 8px; border: 0.5px solid {st.SPINE};
    border-radius: 4px; background: white; font-family: inherit;
    font-size: 12.5px; color: {st.INK}; min-width: 380px;
  }}
  .narrative {{
    background: linear-gradient(135deg, rgba(212,168,67,0.08), rgba(193,85,58,0.06));
    border-left: 2px solid {st.TERRA}; padding: 14px 18px;
    border-radius: 4px; margin: 18px 0;
    color: {st.INK}; font-size: 13.5px; line-height: 1.6;
    max-width: 1100px;
  }}
  .narrative h3 {{
    font-family: "Inter", sans-serif; font-size: 11px; letter-spacing: 1.2px;
    text-transform: uppercase; color: {st.TERRA}; margin: 0 0 6px 0; font-weight: 500;
  }}
  .legend-tip {{ font-size: 11.5px; color: {st.INK_3}; margin-top: 4px; }}
  @media (max-width: 1100px) {{ .grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Iterative inference · <span class="acc">la curva en U del logit lens</span></h1>
<div class="sub">
  Aplicamos el pooler + classifier reales del modelo a cada una de las 13
  capas ocultas (logit-lens). Lo que ves: las primeras capas explotan en
  probabilidades difusas (todas un poco), las medias se hunden a casi cero
  (transición), y las tardías cristalizan una decisión confiada. Es el
  <b>patrón U canónico</b> de Nostalgebraist (2020) y la Tuned Lens de
  Belrose et al. (NeurIPS 2023) — observable EN TU fine-tune.
</div>

<div class="grid">
  <div class="panel">
    <div class="panel-title">Agregado · sigmoid promedio por capa</div>
    <div class="panel-sub">Top-1, gold, suma de las 23. Filtra por capa de cristalización para comparar la U entre emociones tempranas y tardías.</div>
    <div class="group-toggle" id="group-toggle">
      <button data-g="all"      class="active" type="button">Todas (n=2300)</button>
      <button data-g="temprano" type="button">Tempranas</button>
      <button data-g="medio"    type="button">Medias</button>
      <button data-g="tardio"   type="button">Tardías</button>
    </div>
    <div id="agg-plot"></div>
  </div>
  <div class="panel">
    <div class="panel-title">Frase concreta</div>
    <div class="panel-sub">23 emociones × 13 capas — sigmoid completo, gold en negro</div>
    <div id="sentence-plot"></div>
  </div>
</div>

<div class="controls">
  <label style="font-size: 12.5px; color: {st.INK_2}">Frase:</label>
  <select id="sentence-select"></select>
</div>

<div class="narrative">
  <h3>¿Por qué este patrón en U?</h3>
  <b>Capas 0-3 (saturación):</b> el pooler aplica tanh, y las estadísticas del
  CLS en estas capas saturan ±1 → el clasificador recibe vectores binarios
  y dispara muchas emociones a la vez (probabilidades difusas).
  <b>Capas 4-9 (valle de transición):</b> el CLS deja el régimen de embedding
  pero todavía no llega al espacio del clasificador → tanh ≈ 0 → solo bias →
  todas las sigmoides colapsan.
  <b>Capas 10-11 (cristalización):</b> el CLS entra en su régimen natural
  (donde el pooler+classifier fueron entrenados) → un peso del clasificador
  se alinea → una emoción pega un salto, las demás se quedan bajas.
  <br><br>
  <b>Por qué importa para la memoria:</b> esto refuerza §5.1 (cristalización
  emocional) y §5.2 (FFN-L11 como cuello de botella) desde un ángulo
  complementario. El valle medio prueba que <b>la decisión emocional no
  existe</b> hasta que la pipeline pooler+classifier se activa, lo que
  ocurre exactamente al llegar a la representación de L11. Cuando la
  capa 11 se restaura en el activation patching, el F1 vuelve al 100%
  porque <b>se reactiva precisamente esta calibración</b>.
</div>

<script>
const DATA = {payload_json};
const PAL = {pal};
const N_LAYERS = DATA.layer_labels.length;
const N_EMOTIONS = DATA.emotions.length;

// ─── Aggregate plot ───
const aggLayout = {{
  title: {{text: '', font: {{size: 1}}}},
  paper_bgcolor: 'white', plot_bgcolor: 'white',
  font: {{family: '"TeX Gyre Pagella","Palatino",serif', color: PAL.INK_2, size: 11}},
  margin: {{l: 55, r: 30, t: 18, b: 50}},
  xaxis: {{
    tickmode: 'array',
    tickvals: DATA.layer_labels.map((_, i) => i),
    ticktext: DATA.layer_labels,
    gridcolor: PAL.GRID, linecolor: PAL.SPINE, showline: true,
    tickfont: {{size: 10, color: PAL.INK_3}},
  }},
  yaxis: {{
    title: {{text: 'sigmoid promedio', font: {{size: 11, color: PAL.INK_2}}}},
    gridcolor: PAL.GRID, linecolor: PAL.SPINE, showline: true,
    range: [0, 0.85],
    tickfont: {{size: 10, color: PAL.INK_3}},
  }},
  yaxis2: {{
    title: {{text: 'suma sigmoid (de 23)', font: {{size: 11, color: PAL.INK_3}}}},
    overlaying: 'y', side: 'right',
    range: [0, 7.5],
    tickfont: {{size: 10, color: PAL.INK_3}},
    gridcolor: 'rgba(0,0,0,0)',
  }},
  height: 360,
  showlegend: true,
  legend: {{x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.92)', font: {{size: 10}}}},
  shapes: [
    // Vertical bands marking phases
    {{type: 'rect', xref: 'x', yref: 'paper', x0: -0.5, x1: 3.5, y0: 0, y1: 1,
      fillcolor: 'rgba(212,168,67,0.10)', line: {{width: 0}}, layer: 'below'}},
    {{type: 'rect', xref: 'x', yref: 'paper', x0: 3.5, x1: 9.5, y0: 0, y1: 1,
      fillcolor: 'rgba(58,110,165,0.07)', line: {{width: 0}}, layer: 'below'}},
    {{type: 'rect', xref: 'x', yref: 'paper', x0: 9.5, x1: 12.5, y0: 0, y1: 1,
      fillcolor: 'rgba(193,85,58,0.10)', line: {{width: 0}}, layer: 'below'}},
  ],
  annotations: [
    {{x: 1.5, y: 0.96, xref: 'x', yref: 'paper', text: '<b>saturación</b>',
      showarrow: false, font: {{size: 10.5, color: PAL.SAND}}}},
    {{x: 6.5, y: 0.96, xref: 'x', yref: 'paper', text: '<b>valle de transición</b>',
      showarrow: false, font: {{size: 10.5, color: PAL.BLUE}}}},
    {{x: 11.0, y: 0.96, xref: 'x', yref: 'paper', text: '<b>cristalización</b>',
      showarrow: false, font: {{size: 10.5, color: PAL.TERRA}}}},
  ],
}};

function aggTracesFor(group) {{
  const d = DATA.agg[group];
  const xs = DATA.layer_labels.map((_, i) => i);
  return [
    {{
      x: xs, y: d.top1,
      type: 'scatter', mode: 'lines+markers', name: 'top-1 sigmoid',
      line: {{color: PAL.TERRA, width: 3, shape: 'spline', smoothing: 0.5}},
      marker: {{size: 8, color: PAL.TERRA, line: {{color: 'white', width: 1.5}}}},
      hovertemplate: '%{{x}}: %{{y:.3f}}<extra>top-1</extra>',
    }},
    {{
      x: xs, y: d.gold,
      type: 'scatter', mode: 'lines+markers', name: 'gold sigmoid',
      line: {{color: PAL.SAGE, width: 2.5, shape: 'spline', smoothing: 0.5}},
      marker: {{size: 7, color: PAL.SAGE, line: {{color: 'white', width: 1.5}}}},
      hovertemplate: '%{{x}}: %{{y:.3f}}<extra>gold</extra>',
    }},
    {{
      x: xs, y: d.sum,
      type: 'scatter', mode: 'lines+markers', name: 'suma de 23',
      line: {{color: PAL.INK_3, width: 1.5, dash: 'dot'}},
      marker: {{size: 5, color: PAL.INK_3}},
      yaxis: 'y2',
      hovertemplate: '%{{x}}: %{{y:.2f}}<extra>suma</extra>',
    }},
  ];
}}

let currentGroup = 'all';
Plotly.newPlot('agg-plot', aggTracesFor(currentGroup), aggLayout,
               {{displayModeBar: false, responsive: true}});

// Group toggle wiring
document.querySelectorAll('#group-toggle button').forEach(btn => {{
  btn.addEventListener('click', () => {{
    document.querySelectorAll('#group-toggle button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentGroup = btn.dataset.g;
    const traces = aggTracesFor(currentGroup);
    Plotly.react('agg-plot', traces, aggLayout,
                 {{displayModeBar: false, responsive: true}});
  }});
}});

// ─── Per-sentence plot ───
const select = document.getElementById('sentence-select');
DATA.chosen.sentences.forEach((s, i) => {{
  const text = s.length > 70 ? s.slice(0, 67) + '…' : s;
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `[${{DATA.chosen.labels[i]}}] ${{text}}`;
  select.appendChild(opt);
}});

let currentSentence = 0;

function renderSentencePlot() {{
  const sigs = DATA.chosen.sigmoids[currentSentence];   // (13, 23)
  const gold = DATA.chosen.labels[currentSentence];

  const traces = [];
  for (let ei = 0; ei < N_EMOTIONS; ei++) {{
    const emo = DATA.emotions[ei];
    const isGold = emo === gold;
    const ys = sigs.map(layer_sigs => layer_sigs[ei]);
    traces.push({{
      x: DATA.layer_labels.map((_, i) => i),
      y: ys,
      type: 'scatter', mode: 'lines',
      line: {{color: isGold ? PAL.INK : DATA.palette[ei],
              width: isGold ? 3.5 : 1.2,
              shape: 'spline', smoothing: 0.4}},
      name: emo + (isGold ? ' (gold)' : ''),
      showlegend: isGold,
      hovertemplate: emo + ': %{{y:.3f}}<extra></extra>',
      opacity: isGold ? 1.0 : 0.42,
    }});
  }}

  const layout = {{
    paper_bgcolor: 'white', plot_bgcolor: 'white',
    font: {{family: '"TeX Gyre Pagella","Palatino",serif', color: PAL.INK_2, size: 11}},
    margin: {{l: 55, r: 30, t: 18, b: 50}},
    xaxis: {{
      tickmode: 'array',
      tickvals: DATA.layer_labels.map((_, i) => i),
      ticktext: DATA.layer_labels,
      gridcolor: PAL.GRID, linecolor: PAL.SPINE, showline: true,
      tickfont: {{size: 10, color: PAL.INK_3}},
    }},
    yaxis: {{
      title: {{text: 'sigmoid', font: {{size: 11, color: PAL.INK_2}}}},
      range: [0, 1.0],
      gridcolor: PAL.GRID, linecolor: PAL.SPINE, showline: true,
      tickfont: {{size: 10, color: PAL.INK_3}},
    }},
    height: 360,
    showlegend: true,
    legend: {{x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.92)', font: {{size: 10}}}},
    shapes: [
      {{type: 'rect', xref: 'x', yref: 'paper', x0: -0.5, x1: 3.5, y0: 0, y1: 1,
        fillcolor: 'rgba(212,168,67,0.06)', line: {{width: 0}}, layer: 'below'}},
      {{type: 'rect', xref: 'x', yref: 'paper', x0: 3.5, x1: 9.5, y0: 0, y1: 1,
        fillcolor: 'rgba(58,110,165,0.04)', line: {{width: 0}}, layer: 'below'}},
      {{type: 'rect', xref: 'x', yref: 'paper', x0: 9.5, x1: 12.5, y0: 0, y1: 1,
        fillcolor: 'rgba(193,85,58,0.06)', line: {{width: 0}}, layer: 'below'}},
    ],
  }};
  Plotly.newPlot('sentence-plot', traces, layout, {{displayModeBar: false, responsive: true}});
}}

select.addEventListener('change', e => {{
  currentSentence = parseInt(e.target.value);
  renderSentencePlot();
}});

renderSentencePlot();
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
    return build_html(out_dir / "15_iterative_inference.html")


if __name__ == "__main__":
    main()
