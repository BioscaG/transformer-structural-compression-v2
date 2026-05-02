"""Emotion trading cards — 23 cards, one per emotion, deck-of-cards style.

Each card synthesizes the entire profile of one emotion in a single view:
  - Cluster color band (psychological taxonomy from §5.4.6)
  - F1 metrics (baseline / compressed / fine-tuned recovery)
  - Crystallization layer (when the probe first hits 80% of its peak)
  - Most critical attention head (and its F1 drop on ablation)
  - Number of significant FFN neurons (|d| > 2.0)
  - Mini radar with 6 functional dimensions
  - 2-3 example sentences from the test set
  - Vulnerability score (selectivity norm)

The card grid is the closest the project gets to a one-page-per-emotion
dossier — defense-grade summary you can point to when the tribunal asks
about any specific emotion.
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from viz import style as st
from viz.data.load_results import (
    EMOTIONS_23, MODEL_CHECKPOINT,
    load_probing, load_heads, load_neurons, load_informed,
    crystallization_dict, critical_head_per_emotion,
    neuron_count_per_emotion, finetune_recovery, f1_baseline_per_emotion,
)
from viz.thesis_data import EXTENDED_CLUSTER_MAP, emotion_palette


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def _build_card_data() -> dict:
    """Aggregate all per-emotion stats into a single payload."""
    probing = load_probing()
    heads = load_heads()
    neurons = load_neurons()
    informed = load_informed()

    crystals = crystallization_dict(probing)
    critical = critical_head_per_emotion(heads)
    counts = neuron_count_per_emotion(neurons)
    ft = finetune_recovery(informed)
    baseline = f1_baseline_per_emotion(informed)

    # Selectivity norm (sum |d| across catalog rows for each emotion)
    catalog = neurons.get("catalog")
    sel_norm = {}
    if catalog is not None:
        sel_norm = catalog.groupby("emotion")["abs_selectivity"].sum().to_dict()

    # Example sentences per emotion (from cached test set)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    sentences_by_emotion: dict[str, list[str]] = {}
    for s, lbl in zip(meta["sentences"], meta["label_names"]):
        if lbl not in sentences_by_emotion:
            sentences_by_emotion[lbl] = []
        if len(sentences_by_emotion[lbl]) < 6 and len(s) < 110:
            sentences_by_emotion[lbl].append(s)

    palette = emotion_palette(EMOTIONS_23)

    cards = []
    for emo in EMOTIONS_23:
        cluster = EXTENDED_CLUSTER_MAP.get(emo, "Baja especificidad")
        crystal = crystals.get(emo, {})
        cab = critical.get(emo, (None, None, 0.0))
        ft_data = ft.get(emo, {})
        sel = sel_norm.get(emo, 0.0)

        # 6-axis radar values normalized to [0, 1]
        crystal_layer = crystal.get("crystal_layer", 12)
        crystal_norm = max(0, 1 - crystal_layer / 12)         # earlier = better
        n_neurons = counts.get(emo, 0)
        # Normalize neuron count to [0,1] using reasonable max ~1000
        neurons_norm = min(n_neurons / 800, 1.0)
        f1_base = baseline.get(emo, ft_data.get("baseline", 0.5))
        f1_compr = ft_data.get("compressed", 0.0)
        f1_ft = ft_data.get("finetuned", f1_base)
        retention = (f1_compr / f1_base) if f1_base > 0 else 0
        recovery = ((f1_ft - f1_base) / f1_base) if f1_base > 0 else 0
        recovery_norm = max(0, min(1, 0.5 + recovery * 1.2))
        head_impact = abs(cab[2]) if cab[2] else 0
        robustness = 1 - min(1, head_impact / 0.32)
        sel_norm_max = max(sel_norm.values()) if sel_norm else 1
        intensity = min(sel / sel_norm_max, 1.0) if sel_norm_max else 0

        radar = {
            "cristalización": round(crystal_norm, 3),
            "intensidad neuronal": round(intensity, 3),
            "F1 baseline": round(f1_base, 3),
            "retención bajo SVD": round(retention, 3),
            "recuperación FT": round(recovery_norm, 3),
            "robustez cabeza": round(robustness, 3),
        }

        cards.append({
            "emotion": emo,
            "cluster": cluster,
            "color": palette[emo],
            "f1_baseline": round(f1_base, 3),
            "f1_compressed": round(f1_compr, 3),
            "f1_finetuned": round(f1_ft, 3),
            "delta_ft": round(f1_ft - f1_base, 3),
            "crystal_layer": crystal_layer,
            "crystal_layer_name": crystal.get("crystal_layer_name", f"L{crystal_layer-1}"),
            "max_probe_f1": round(crystal.get("max_probe_f1", f1_base), 3),
            "critical_head": (
                f"L{cab[0]}-H{cab[1]}" if cab[0] is not None else "—"
            ),
            "head_drop": round(cab[2], 3) if cab[2] else 0,
            "n_neurons": n_neurons,
            "selectivity_norm": round(sel, 1),
            "radar": radar,
            "examples": sentences_by_emotion.get(emo, [])[:4],
        })

    return {"cards": cards}


def build_html(out_path: pathlib.Path) -> pathlib.Path:
    payload = _build_card_data()
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    pal = json.dumps({
        "BG": st.BG, "INK": st.INK, "INK_2": st.INK_2, "INK_3": st.INK_3,
        "GRID": st.GRID, "SPINE": st.SPINE,
        "TERRA": st.TERRA, "SAND": st.SAND, "BLUE": st.BLUE,
    })

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Emotion trading cards · TFG</title>
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
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr));
    gap: 18px;
  }}
  .card {{
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 8px;
    padding: 14px 16px; position: relative; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  .card:hover {{ transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08); }}
  .card .band {{
    position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
  }}
  .card .head {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 4px;
  }}
  .card .name {{ font-size: 18px; color: {st.INK}; font-weight: normal; }}
  .card .cluster {{
    font-family: "Inter", sans-serif; font-size: 9px; letter-spacing: 0.6px;
    text-transform: uppercase; color: {st.INK_3}; padding: 2px 6px;
    border-radius: 3px; border: 0.5px solid currentColor;
  }}
  .stats {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 4px 8px;
    margin: 8px 0;
  }}
  .stat {{ font-size: 11.5px; }}
  .stat .lbl {{ color: {st.INK_3}; font-size: 10px; letter-spacing: 0.3px; }}
  .stat .val {{
    color: {st.INK}; font-family: "Inter", monospace; font-size: 12px;
  }}
  .stat .val.delta-pos {{ color: {st.SAGE}; }}
  .stat .val.delta-neg {{ color: {st.TERRA}; }}
  .radar-wrap {{ height: 145px; margin: 6px 0; }}
  .examples {{
    margin-top: 6px; padding-top: 6px;
    border-top: 0.5px dashed {st.GRID};
    font-size: 10.5px; color: {st.INK_2}; font-style: italic;
    line-height: 1.4;
  }}
  .examples .ex {{
    padding: 1px 0;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 100%;
  }}
  .examples .ex::before {{ content: "❝ "; color: {st.SAND}; }}
  .examples .ex::after {{ content: " ❞"; color: {st.SAND}; }}
  .legend {{
    display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 14px;
    font-size: 11.5px; color: {st.INK_2};
  }}
  .legend .item {{ display: flex; align-items: center; gap: 5px; }}
  .legend .swatch {{
    width: 11px; height: 11px; border-radius: 50%; display: inline-block;
  }}
  /* Sort/filter controls */
  .controls {{
    display: flex; gap: 12px; margin-bottom: 14px; align-items: center;
    flex-wrap: wrap;
  }}
  .controls label {{
    font-size: 11.5px; color: {st.INK_2};
  }}
  .controls select {{
    padding: 4px 8px; border: 0.5px solid {st.SPINE}; border-radius: 4px;
    background: white; font-family: inherit; font-size: 12px; color: {st.INK};
  }}
</style>
</head>
<body>

<h1>Emotion <span class="acc">trading cards</span></h1>
<div class="sub">
  23 tarjetas, una por emoción. Cada una sintetiza la huella completa: F1 antes
  y después de comprimir y reentrenar, capa de cristalización, cabeza crítica,
  neuronas dedicadas, radar funcional y frases ejemplo de tu test set.
  Defense-grade dossier.
</div>

<div class="legend">
  <div class="item"><span class="swatch" style="background:#e5b85e"></span>Positivas alta energía</div>
  <div class="item"><span class="swatch" style="background:#c64a37"></span>Negativas reactivas</div>
  <div class="item"><span class="swatch" style="background:#7b5e7b"></span>Negativas internas</div>
  <div class="item"><span class="swatch" style="background:#3a6ea5"></span>Epistémicas</div>
  <div class="item"><span class="swatch" style="background:#5a8f7b"></span>Orientadas al otro</div>
  <div class="item"><span class="swatch" style="background:#909090"></span>Baja especificidad</div>
</div>

<div class="controls">
  <label>Ordenar por:</label>
  <select id="sort-select">
    <option value="cluster">Cluster psicológico</option>
    <option value="f1">F1 baseline (descendente)</option>
    <option value="crystal">Capa de cristalización</option>
    <option value="neurons">Neuronas significativas</option>
    <option value="recovery">Mejora con fine-tuning</option>
    <option value="alpha">Alfabético</option>
  </select>
</div>

<div class="grid" id="grid"></div>

<script>
const DATA = {payload_json};
const PAL = {pal};

function fmtPct(v) {{ return (v * 100).toFixed(0) + '%'; }}

function makeCard(c) {{
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.emotion = c.emotion;

  // Color band
  const band = document.createElement('div');
  band.className = 'band';
  band.style.background = c.color;
  card.appendChild(band);

  // Header: emotion name + cluster pill
  const head = document.createElement('div');
  head.className = 'head';
  head.innerHTML = `<span class="name">${{c.emotion}}</span>`
    + `<span class="cluster" style="color:${{c.color}}">${{c.cluster}}</span>`;
  card.appendChild(head);

  // Stats grid
  const stats = document.createElement('div');
  stats.className = 'stats';
  const deltaClass = c.delta_ft >= 0 ? 'delta-pos' : 'delta-neg';
  const deltaSign = c.delta_ft >= 0 ? '+' : '';
  stats.innerHTML = `
    <div class="stat"><div class="lbl">F1 BASELINE</div>
      <div class="val">${{c.f1_baseline.toFixed(3)}}</div></div>
    <div class="stat"><div class="lbl">F1 TRAS FT</div>
      <div class="val">${{c.f1_finetuned.toFixed(3)}}
        <span class="${{deltaClass}}">(${{deltaSign}}${{c.delta_ft.toFixed(3)}})</span></div></div>
    <div class="stat"><div class="lbl">CRISTALIZACIÓN</div>
      <div class="val">${{c.crystal_layer_name}}</div></div>
    <div class="stat"><div class="lbl">CABEZA CRÍTICA</div>
      <div class="val">${{c.critical_head}} <span style="color:${{PAL.INK_3}}">(Δ${{c.head_drop.toFixed(3)}})</span></div></div>
    <div class="stat"><div class="lbl">NEURONAS SIG.</div>
      <div class="val">${{c.n_neurons}}</div></div>
    <div class="stat"><div class="lbl">SELECTIVIDAD</div>
      <div class="val">${{c.selectivity_norm.toFixed(0)}}</div></div>
  `;
  card.appendChild(stats);

  // Mini radar
  const radarWrap = document.createElement('div');
  radarWrap.className = 'radar-wrap';
  const radarId = 'radar-' + c.emotion;
  radarWrap.innerHTML = `<div id="${{radarId}}" style="width:100%;height:100%"></div>`;
  card.appendChild(radarWrap);

  // Examples
  if (c.examples.length > 0) {{
    const ex = document.createElement('div');
    ex.className = 'examples';
    ex.innerHTML = c.examples.slice(0, 3).map(s =>
      `<div class="ex">${{s.replace(/[<>"]/g, '')}}</div>`).join('');
    card.appendChild(ex);
  }}

  return {{ card, radarId, c }};
}}

function renderRadar(radarId, c) {{
  const cats = Object.keys(c.radar);
  const vals = cats.map(k => c.radar[k]);
  cats.push(cats[0]); vals.push(vals[0]);
  Plotly.newPlot(radarId, [{{
    type: 'scatterpolar',
    r: vals, theta: cats,
    fill: 'toself', fillcolor: c.color + '33',
    line: {{color: c.color, width: 2}},
    marker: {{size: 5, color: c.color}},
    hovertemplate: '%{{theta}}: %{{r:.2f}}<extra></extra>',
  }}], {{
    polar: {{
      bgcolor: 'white',
      radialaxis: {{visible: true, range: [0, 1], showticklabels: false,
                    gridcolor: PAL.GRID}},
      angularaxis: {{tickfont: {{size: 8, color: PAL.INK_3,
                     family: 'serif'}}, gridcolor: PAL.GRID}},
    }},
    margin: {{l: 28, r: 28, t: 4, b: 4}},
    paper_bgcolor: 'white',
    showlegend: false,
  }}, {{displayModeBar: false, responsive: true, staticPlot: true}});
}}

const grid = document.getElementById('grid');
const sortSel = document.getElementById('sort-select');
const cardEntries = [];

function buildAll() {{
  grid.innerHTML = '';
  cardEntries.length = 0;
  for (const c of DATA.cards) {{
    const {{ card, radarId }} = makeCard(c);
    grid.appendChild(card);
    cardEntries.push({{ card, radarId, c }});
  }}
  // Render radars after DOM is in place
  setTimeout(() => cardEntries.forEach(e => renderRadar(e.radarId, e.c)), 50);
}}

function sortAndRebuild(key) {{
  const cluster_order = ["Positivas alta energía", "Negativas reactivas",
                          "Negativas internas", "Epistémicas",
                          "Orientadas al otro", "Baja especificidad"];
  const sorted = [...DATA.cards];
  if (key === 'cluster') {{
    sorted.sort((a, b) => {{
      const ai = cluster_order.indexOf(a.cluster);
      const bi = cluster_order.indexOf(b.cluster);
      if (ai !== bi) return ai - bi;
      return b.f1_baseline - a.f1_baseline;
    }});
  }} else if (key === 'f1') {{
    sorted.sort((a, b) => b.f1_baseline - a.f1_baseline);
  }} else if (key === 'crystal') {{
    sorted.sort((a, b) => a.crystal_layer - b.crystal_layer);
  }} else if (key === 'neurons') {{
    sorted.sort((a, b) => b.n_neurons - a.n_neurons);
  }} else if (key === 'recovery') {{
    sorted.sort((a, b) => b.delta_ft - a.delta_ft);
  }} else if (key === 'alpha') {{
    sorted.sort((a, b) => a.emotion.localeCompare(b.emotion));
  }}
  DATA.cards = sorted;
  buildAll();
}}

sortSel.addEventListener('change', e => sortAndRebuild(e.target.value));

// Initial: sort by cluster
sortAndRebuild('cluster');
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
    return build_html(out_dir / "24_emotion_cards.html")


if __name__ == "__main__":
    main()
