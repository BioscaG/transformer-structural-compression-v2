"""Neuron portrait gallery — los 80 neuronas más selectivas del modelo,
cada una con sus 5 frases más activantes.

Reads:
  - viz/data/cache/neuron_activations.npz  (activations + top_indices)
  - viz/data/cache/neuron_activations_meta.json  (neurons metadata)
  - viz/data/cache/meta.json  (sentences + labels)

Each card:
  - Layer + neuron index (e.g. "L11 · N944")
  - Dominant emotion (from notebook7 catalog)
  - Selectivity score (Cohen's d)
  - Excitatory / inhibitory tag
  - Top-5 activating sentences with bars showing activation magnitude

Reveals the "language" the model speaks internally — what each neuron
"means" inside the network.
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from viz import style as st
from viz.thesis_data import EXTENDED_CLUSTER_MAP, emotion_palette
from viz.data.load_results import EMOTIONS_23


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def build_html(out_path: pathlib.Path) -> pathlib.Path:
    data = np.load(CACHE_DIR / "neuron_activations.npz")
    activations = data["activations"]      # (n_sentences, n_neurons)
    top_indices = data["top_indices"]      # (n_neurons, top_k)

    payload_meta = json.loads((CACHE_DIR / "neuron_activations_meta.json").read_text())
    neurons = payload_meta["neurons"]
    top_k = payload_meta["top_k"]

    full_meta = json.loads((CACHE_DIR / "meta.json").read_text())
    sentences = full_meta["sentences"]
    labels = full_meta["label_names"]

    palette = emotion_palette(EMOTIONS_23)

    # Build per-neuron card payload
    cards = []
    for ni, n in enumerate(neurons):
        top_idx = top_indices[ni].tolist()
        top_sent = []
        for si in top_idx:
            text = sentences[si]
            if len(text) > 110:
                text = text[:107] + "…"
            top_sent.append({
                "text": text,
                "label": labels[si],
                "activation": float(activations[si, ni]),
            })
        # Normalize activation values relative to the top one for bars
        max_act = max(abs(s["activation"]) for s in top_sent) if top_sent else 1
        for s in top_sent:
            s["bar"] = abs(s["activation"]) / max_act if max_act > 0 else 0

        cluster = EXTENDED_CLUSTER_MAP.get(n["emotion"], "Baja especificidad")
        cards.append({
            "layer": n["layer"],
            "neuron": n["neuron"],
            "emotion": n["emotion"],
            "cluster": cluster,
            "color": palette.get(n["emotion"], "#888888"),
            "selectivity": round(n["selectivity"], 2),
            "abs_selectivity": round(n["abs_selectivity"], 2),
            "direction": n["direction"],
            "top_sentences": top_sent,
        })

    # Sort by absolute selectivity desc
    cards.sort(key=lambda c: -c["abs_selectivity"])

    payload = {"cards": cards}
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Neuron portrait gallery · TFG</title>
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
    display: flex; gap: 12px; align-items: center; margin-bottom: 16px;
    flex-wrap: wrap;
  }}
  .controls select, .controls input {{
    padding: 5px 10px; border: 0.5px solid {st.SPINE}; border-radius: 4px;
    background: white; font-family: inherit; font-size: 12px; color: {st.INK};
  }}
  .controls label {{ font-size: 11.5px; color: {st.INK_2}; }}
  .grid {{
    display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr));
    gap: 16px;
  }}
  .neuron-card {{
    background: white; border: 0.5px solid {st.SPINE};
    border-radius: 8px; padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    position: relative; overflow: hidden;
    transition: transform 0.15s, box-shadow 0.15s;
  }}
  .neuron-card:hover {{
    transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08);
  }}
  .neuron-card .band {{
    position: absolute; left: 0; top: 0; bottom: 0; width: 5px;
  }}
  .neuron-card .head {{
    display: flex; justify-content: space-between; align-items: baseline;
    margin-bottom: 4px;
  }}
  .neuron-id {{
    font-family: "Inter", monospace; font-size: 14px; color: {st.INK};
    font-weight: 500;
  }}
  .selectivity {{
    font-family: "Inter", monospace; font-size: 11px; color: {st.TERRA};
    background: rgba(193,85,58,0.10); padding: 2px 8px; border-radius: 3px;
  }}
  .meta {{
    display: flex; gap: 8px; margin: 4px 0 10px 0;
    font-size: 10.5px; color: {st.INK_3}; align-items: center;
  }}
  .emo-pill {{
    color: white; padding: 2px 7px; border-radius: 3px;
    font-family: "Inter", sans-serif; font-size: 10px;
    letter-spacing: 0.3px; text-transform: lowercase;
  }}
  .direction-tag {{
    font-family: "Inter", sans-serif; font-size: 9.5px;
    padding: 2px 5px; border-radius: 3px;
    border: 0.5px solid currentColor; letter-spacing: 0.5px;
    text-transform: uppercase;
  }}
  .direction-tag.exc {{ color: {st.SAGE}; }}
  .direction-tag.inh {{ color: {st.PLUM}; }}
  .top-list {{
    margin-top: 6px; padding-top: 6px; border-top: 0.5px dashed {st.GRID};
  }}
  .top-list .lbl {{
    font-family: "Inter", sans-serif; font-size: 9.5px; color: {st.INK_3};
    letter-spacing: 0.6px; margin-bottom: 4px;
  }}
  .sentence-row {{
    margin: 4px 0; font-size: 11px; line-height: 1.4;
    display: grid; grid-template-columns: 50px 1fr;
    gap: 6px; align-items: center;
  }}
  .bar-container {{
    background: {st.GRID}; height: 6px; border-radius: 3px;
    overflow: hidden;
  }}
  .bar-fill {{ height: 100%; transition: width 0.2s; }}
  .sentence-text {{
    color: {st.INK}; font-style: italic;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    max-width: 100%;
  }}
  .sentence-text::before {{ content: "❝ "; color: {st.SAND}; font-style: normal; }}
  .sentence-text::after {{ content: " ❞"; color: {st.SAND}; font-style: normal; }}
  .gold-tag {{
    font-family: "Inter", sans-serif; font-size: 9px;
    color: {st.INK_3}; margin-right: 4px;
  }}
</style>
</head>
<body>

<h1>Neuron portrait <span class="acc">gallery</span></h1>
<div class="sub">
  Las 80 neuronas más selectivas del modelo, cada una con las 5 frases del
  test set que más la activan. Una mirada íntima al "lenguaje" interno —
  qué representa cada neurona.
</div>

<div class="controls">
  <label>Filtrar por emoción:</label>
  <select id="filter-emo">
    <option value="">— todas —</option>
  </select>
  <label>Filtrar por capa:</label>
  <select id="filter-layer">
    <option value="">— todas —</option>
  </select>
  <label>Dirección:</label>
  <select id="filter-dir">
    <option value="">— todas —</option>
    <option value="excitatory">excitatorias (+)</option>
    <option value="inhibitory">inhibitorias (−)</option>
  </select>
  <label>Ordenar:</label>
  <select id="sort-key">
    <option value="abs_selectivity">selectividad (desc)</option>
    <option value="layer">capa (asc)</option>
    <option value="emotion">emoción (alfabético)</option>
  </select>
</div>

<div class="grid" id="grid"></div>

<script>
const DATA = {payload_json};

// Populate filters
const emos = [...new Set(DATA.cards.map(c => c.emotion))].sort();
const layers = [...new Set(DATA.cards.map(c => c.layer))].sort((a,b)=>a-b);
const emoSel = document.getElementById('filter-emo');
const layerSel = document.getElementById('filter-layer');
emos.forEach(e => {{
  const o = document.createElement('option');
  o.value = e; o.textContent = e;
  emoSel.appendChild(o);
}});
layers.forEach(L => {{
  const o = document.createElement('option');
  o.value = L; o.textContent = `L${{L}}`;
  layerSel.appendChild(o);
}});

function makeCard(c) {{
  const card = document.createElement('div');
  card.className = 'neuron-card';

  const band = document.createElement('div');
  band.className = 'band';
  band.style.background = c.color;
  card.appendChild(band);

  const head = document.createElement('div');
  head.className = 'head';
  head.innerHTML = `<span class="neuron-id">L${{c.layer}} · N${{c.neuron}}</span>`
    + `<span class="selectivity">d = ${{c.selectivity > 0 ? '+' : ''}}${{c.selectivity.toFixed(2)}}</span>`;
  card.appendChild(head);

  const meta = document.createElement('div');
  meta.className = 'meta';
  const dirClass = c.direction === 'excitatory' ? 'exc' : 'inh';
  const dirSign = c.direction === 'excitatory' ? '+' : '−';
  meta.innerHTML = `<span class="emo-pill" style="background:${{c.color}}">${{c.emotion}}</span>`
    + `<span class="direction-tag ${{dirClass}}">${{dirSign}} ${{c.direction}}</span>`
    + `<span style="margin-left:auto">${{c.cluster}}</span>`;
  card.appendChild(meta);

  const list = document.createElement('div');
  list.className = 'top-list';
  list.innerHTML = '<div class="lbl">TOP-5 FRASES QUE MÁS LA ACTIVAN</div>';
  for (const s of c.top_sentences) {{
    const row = document.createElement('div');
    row.className = 'sentence-row';
    const safe = s.text.replace(/[<>"]/g, '');
    row.innerHTML = `
      <div class="bar-container">
        <div class="bar-fill" style="width:${{(s.bar*100).toFixed(0)}}%; background:${{c.color}}"></div>
      </div>
      <div class="sentence-text" title="${{safe}}">
        <span class="gold-tag">[${{s.label}}]</span>${{safe}}</div>
    `;
    list.appendChild(row);
  }}
  card.appendChild(list);

  return card;
}}

const grid = document.getElementById('grid');
function render() {{
  grid.innerHTML = '';
  const fEmo = document.getElementById('filter-emo').value;
  const fLayer = document.getElementById('filter-layer').value;
  const fDir = document.getElementById('filter-dir').value;
  const sortKey = document.getElementById('sort-key').value;

  let cards = DATA.cards.filter(c => {{
    if (fEmo && c.emotion !== fEmo) return false;
    if (fLayer && String(c.layer) !== String(fLayer)) return false;
    if (fDir && c.direction !== fDir) return false;
    return true;
  }});

  if (sortKey === 'abs_selectivity') {{
    cards.sort((a, b) => b.abs_selectivity - a.abs_selectivity);
  }} else if (sortKey === 'layer') {{
    cards.sort((a, b) => a.layer - b.layer || b.abs_selectivity - a.abs_selectivity);
  }} else if (sortKey === 'emotion') {{
    cards.sort((a, b) => a.emotion.localeCompare(b.emotion) || b.abs_selectivity - a.abs_selectivity);
  }}

  for (const c of cards) grid.appendChild(makeCard(c));
}}

['filter-emo', 'filter-layer', 'filter-dir', 'sort-key'].forEach(id => {{
  document.getElementById(id).addEventListener('change', render);
}});

render();
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
    return build_html(out_dir / "27_neuron_gallery.html")


if __name__ == "__main__":
    main()
