"""Attention atlas — see all 144 attention heads at once for a chosen sentence.

For a curated sentence, render the (layer × head) grid of token-to-token
attention matrices as small canvas heatmaps. Borders coloured by head
category (Critical Specialist red / Generalist blue / Minor sand /
Dispensable grey) from the user's notebook 6.

Click any head to open a labeled focus view with token text. The cell
intensity is the actual cached attention from `activations.npz` — first
16 tokens stored.
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from viz import style as st
from viz.data.load_results import load_heads, EMOTIONS_23


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def _curate_sentences(meta: dict, n_per_emotion: int = 6) -> list[int]:
    """Pick a balanced subset of indices."""
    label_names = meta["label_names"]
    indices = []
    seen = {e: 0 for e in EMOTIONS_23}
    for i, lbl in enumerate(label_names):
        if lbl in seen and seen[lbl] < n_per_emotion:
            indices.append(i)
            seen[lbl] += 1
    return indices


def _short(s: str, n: int = 70) -> str:
    s = s.replace('"', "'")
    return s if len(s) <= n else s[: n - 1] + "…"


def build_html(out_path: pathlib.Path) -> pathlib.Path:
    data = np.load(CACHE_DIR / "activations.npz")
    attn = data["attentions"]                  # (N, 12, 12, T, T) float16
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    sentences = meta["sentences"]
    label_names = meta["label_names"]
    tokens = meta["tokens"]                    # list of token lists per sentence

    chosen = _curate_sentences(meta, n_per_emotion=2)
    chosen_attn = attn[chosen].astype(np.float32)
    chosen_sentences = [sentences[i] for i in chosen]
    chosen_labels = [label_names[i] for i in chosen]
    chosen_tokens = [tokens[i] for i in chosen]
    n_chosen = len(chosen)
    print(f"Curated {n_chosen} sentences for atlas")

    # Head categories (from user's notebook 6) — color the cell border
    heads_data = load_heads()
    cat_grid = np.full((12, 12), "Minor Specialist", dtype=object)
    if heads_data["categories"] is not None:
        for _, row in heads_data["categories"].iterrows():
            L, h = int(row["layer"]), int(row["head"])
            cat_grid[L, h] = row["category"].rstrip("s")
    cat_color = {
        "Critical Specialist": st.TERRA,
        "Critical Generalist": st.BLUE,
        "Minor Specialist":    st.SAND,
        "Dispensable":         st.SPINE,
    }

    # Trim attention to only non-pad tokens per sentence
    # Each token list has padding tokens at the end. Find the active length.
    def active_length(toks: list[str]) -> int:
        for i, t in enumerate(toks):
            if not t or t == "[PAD]" or t == "":
                return min(i, attn.shape[-1])
        return min(len(toks), attn.shape[-1])

    # Trim and round attentions to keep the JSON payload small
    trimmed_attn = []
    trimmed_tokens = []
    for s_idx in range(n_chosen):
        T = active_length(chosen_tokens[s_idx])
        # (12, 12, T, T)
        a = chosen_attn[s_idx, :, :, :T, :T]
        trimmed_attn.append(np.round(a, 4).tolist())
        toks = chosen_tokens[s_idx][:T]
        # Pretty token labels
        toks = [t.replace("##", "") if t.startswith("##") else t for t in toks]
        toks = [{"[CLS]": "⟨CLS⟩", "[SEP]": "⟨SEP⟩"}.get(t, t) for t in toks]
        trimmed_tokens.append(toks)

    # Categories grid as 12x12 colors
    border_colors = [[cat_color[cat_grid[L, h]] for h in range(12)] for L in range(12)]
    cat_labels = [[cat_grid[L, h] for h in range(12)] for L in range(12)]

    payload = {
        "sentences": chosen_sentences,
        "labels": chosen_labels,
        "attentions": trimmed_attn,
        "tokens": trimmed_tokens,
        "border_colors": border_colors,
        "categories": cat_labels,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))

    pal = json.dumps({
        "BG": st.BG, "INK": st.INK, "INK_2": st.INK_2, "INK_3": st.INK_3,
        "GRID": st.GRID, "SPINE": st.SPINE, "TERRA": st.TERRA, "SAND": st.SAND,
        "BLUE": st.BLUE,
    })

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Attention atlas · TFG Anatomía Emocional</title>
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
    display: flex; gap: 12px; align-items: center; margin: 14px 0; flex-wrap: wrap;
  }}
  select {{
    padding: 6px 10px; border: 0.5px solid {st.SPINE}; border-radius: 4px;
    background: white; font-family: inherit; font-size: 13px;
    color: {st.INK}; min-width: 460px;
  }}
  .legend {{
    display: flex; gap: 14px; flex-wrap: wrap; font-size: 11.5px; color: {st.INK_3};
    margin-bottom: 12px;
  }}
  .legend .item {{ display: flex; align-items: center; gap: 6px; }}
  .legend .swatch {{
    width: 12px; height: 12px; border-radius: 50%;
    display: inline-block;
  }}
  .grid-wrap {{
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 6px;
    padding: 18px; overflow-x: auto;
  }}
  .grid {{
    display: grid;
    grid-template-columns: 24px repeat(12, 1fr);
    gap: 3px;
    align-items: stretch;
  }}
  .colhdr, .rowhdr {{
    font-family: "Inter", monospace; font-size: 9.5px; color: {st.INK_3};
    text-align: center;
  }}
  .rowhdr {{ display: flex; align-items: center; justify-content: flex-end; padding-right: 4px; }}
  .head-cell {{
    aspect-ratio: 1 / 1; border: 1.5px solid {st.SPINE};
    border-radius: 2px; cursor: pointer; position: relative;
    transition: transform 0.1s, box-shadow 0.1s;
  }}
  .head-cell:hover {{ transform: scale(1.06); box-shadow: 0 2px 8px rgba(0,0,0,0.2); z-index: 5; }}
  .head-cell canvas {{
    width: 100%; height: 100%; display: block; border-radius: 1px;
  }}
  .modal-bg {{
    display: none; position: fixed; inset: 0; background: rgba(26,26,26,0.62);
    align-items: center; justify-content: center; z-index: 100;
  }}
  .modal {{
    background: white; border-radius: 6px; padding: 22px; max-width: 720px;
    width: 95%; max-height: 92vh; overflow: auto;
  }}
  .modal h2 {{
    margin: 0 0 4px 0; font-size: 18px; font-weight: normal;
  }}
  .modal h2 .head-id {{ color: {st.TERRA}; font-family: "Inter", monospace; }}
  .modal .modal-sub {{ color: {st.INK_3}; font-size: 12.5px; margin-bottom: 14px; }}
  .close-btn {{
    float: right; background: none; border: none; font-size: 20px;
    color: {st.INK_3}; cursor: pointer;
  }}
  .close-btn:hover {{ color: {st.TERRA}; }}
  .narrative {{
    background: linear-gradient(135deg, rgba(212,168,67,0.08), rgba(193,85,58,0.06));
    border-left: 2px solid {st.TERRA}; padding: 12px 16px; border-radius: 4px;
    margin: 16px 0; font-size: 13px; color: {st.INK_2}; line-height: 1.6;
    max-width: 1100px;
  }}
</style>
</head>
<body>

<h1>Attention <span class="acc">atlas</span></h1>
<div class="sub">
  Las 144 cabezas de tu modelo, todas a la vez, sobre la frase elegida. Cada
  celda es un mini-mapa de calor que muestra a qué tokens atiende esa cabeza.
  Bordes coloreados por categoría funcional (notebook 6). <b>Click una celda
  para ampliar</b> con etiquetas de tokens.
</div>

<div class="legend">
  <div class="item"><span class="swatch" style="background: {st.TERRA}"></span>Critical Specialist</div>
  <div class="item"><span class="swatch" style="background: {st.BLUE}"></span>Critical Generalist</div>
  <div class="item"><span class="swatch" style="background: {st.SAND}"></span>Minor Specialist</div>
  <div class="item"><span class="swatch" style="background: {st.SPINE}"></span>Dispensable</div>
</div>

<div class="controls">
  <label style="font-size: 13px; color: {st.INK_2}">Frase:</label>
  <select id="sentence-select"></select>
</div>

<div class="grid-wrap">
  <div class="grid" id="atlas"></div>
</div>

<div class="narrative">
  La capa 11 (fila inferior) tiene <b>cero cabezas dispensables</b> según tu
  notebook 6 — todas son críticas. Mira la fila L11: bordes en rojo o azul.
  La cabeza compartida por sadness/realization según §5.3 está marcada como
  Critical Generalist. Cuando cambies de frase, mira cómo las cabezas
  tempranas atienden a tokens locales (diagonales) y las tardías concentran
  atención en [CLS] o [SEP] (rayas verticales/horizontales) — el patrón
  típico de "agregadores" que documenta tu memoria.
</div>

<div class="modal-bg" id="modal-bg" onclick="closeModal(event)">
  <div class="modal" id="modal">
    <button class="close-btn" onclick="closeModal()">×</button>
    <h2>Cabeza <span class="head-id" id="modal-head"></span></h2>
    <div class="modal-sub" id="modal-sub"></div>
    <canvas id="modal-canvas" width="600" height="600"
            style="width: 100%; max-width: 600px; border: 0.5px solid {st.SPINE}"></canvas>
    <div id="modal-tokens" style="margin-top: 10px; font-size: 12px; color: {st.INK_3};
                                   font-family: 'Inter', monospace; line-height: 1.6;"></div>
  </div>
</div>

<script>
const DATA = {payload_json};
const PAL = {pal};
let currentSentence = 0;

const select = document.getElementById('sentence-select');
DATA.sentences.forEach((s, i) => {{
  const text = s.length > 70 ? s.slice(0, 67) + '…' : s;
  const opt = document.createElement('option');
  opt.value = i;
  opt.textContent = `[${{DATA.labels[i]}}] ${{text}}`;
  select.appendChild(opt);
}});

// Build grid skeleton: 12 rows × 12 cols, with row/col headers
const atlas = document.getElementById('atlas');
function buildGrid() {{
  atlas.innerHTML = '';
  // Top-left empty
  atlas.innerHTML += '<div></div>';
  // Column headers (head index)
  for (let h = 0; h < 12; h++) {{
    atlas.innerHTML += `<div class="colhdr">H${{h}}</div>`;
  }}
  for (let L = 0; L < 12; L++) {{
    atlas.innerHTML += `<div class="rowhdr">L${{L}}</div>`;
    for (let h = 0; h < 12; h++) {{
      const cell = document.createElement('div');
      cell.className = 'head-cell';
      cell.style.borderColor = DATA.border_colors[L][h];
      cell.dataset.layer = L;
      cell.dataset.head = h;
      cell.title = `L${{L}}-H${{h}}: ${{DATA.categories[L][h]}}`;
      cell.onclick = () => openModal(L, h);
      const canvas = document.createElement('canvas');
      canvas.width = 60;
      canvas.height = 60;
      cell.appendChild(canvas);
      atlas.appendChild(cell);
    }}
  }}
}}

function drawAttention(canvas, A) {{
  // A is (T, T) array of attention values
  const T = A.length;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const cellW = canvas.width / T;
  const cellH = canvas.height / T;
  // Find max for scaling
  let maxV = 0;
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) if (A[i][j] > maxV) maxV = A[i][j];
  if (maxV === 0) maxV = 1;
  for (let i = 0; i < T; i++) {{
    for (let j = 0; j < T; j++) {{
      const v = Math.min(A[i][j] / maxV, 1);
      // sand → terra gradient
      const r = 245, g = Math.round(245 - 145 * v), bl = Math.round(225 - 165 * v);
      ctx.fillStyle = `rgb(${{Math.round(245 - 50 * v)}},${{g}},${{bl}})`;
      ctx.fillRect(j * cellW, i * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }}
  }}
}}

function renderAtlas() {{
  const A = DATA.attentions[currentSentence];   // (12, 12, T, T)
  const cells = atlas.querySelectorAll('.head-cell');
  cells.forEach(cell => {{
    const L = parseInt(cell.dataset.layer);
    const h = parseInt(cell.dataset.head);
    const canvas = cell.querySelector('canvas');
    drawAttention(canvas, A[L][h]);
  }});
}}

select.addEventListener('change', e => {{
  currentSentence = parseInt(e.target.value);
  renderAtlas();
}});

// ─── Modal ───
function openModal(L, h) {{
  document.getElementById('modal-bg').style.display = 'flex';
  document.getElementById('modal-head').textContent = `L${{L}}-H${{h}}`;
  const cat = DATA.categories[L][h];
  const sentence = DATA.sentences[currentSentence];
  document.getElementById('modal-sub').innerHTML =
    `<b style="color: ${{DATA.border_colors[L][h]}}">${{cat}}</b> · `
    + `frase: <i>"${{sentence.length > 60 ? sentence.slice(0,57)+'…' : sentence}}"</i>`;
  const canvas = document.getElementById('modal-canvas');
  // Clear and redraw at large size
  const A = DATA.attentions[currentSentence][L][h];
  const T = A.length;
  const cellW = canvas.width / T;
  const cellH = canvas.height / T;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  let maxV = 0;
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) if (A[i][j] > maxV) maxV = A[i][j];
  if (maxV === 0) maxV = 1;
  for (let i = 0; i < T; i++) {{
    for (let j = 0; j < T; j++) {{
      const v = Math.min(A[i][j] / maxV, 1);
      const r = Math.round(245 - 50 * v);
      const g = Math.round(245 - 145 * v);
      const bl = Math.round(225 - 165 * v);
      ctx.fillStyle = `rgb(${{r}},${{g}},${{bl}})`;
      ctx.fillRect(j * cellW, i * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }}
  }}
  // Draw token labels around the canvas
  const tokens = DATA.tokens[currentSentence];
  let html = '<b>Tokens (eje):</b> ';
  tokens.forEach((t, i) => {{
    html += `<span style="background: rgba(193,85,58,0.10); padding: 1px 5px; margin: 1px; border-radius: 2px">`
          + `${{i}}: ${{t}}</span> `;
  }});
  document.getElementById('modal-tokens').innerHTML = html;
}}

function closeModal(event) {{
  if (event && event.target.id !== 'modal-bg' && event.target.className !== 'close-btn') return;
  document.getElementById('modal-bg').style.display = 'none';
}}
document.querySelector('.close-btn').onclick = () => {{
  document.getElementById('modal-bg').style.display = 'none';
}};
document.addEventListener('keydown', e => {{
  if (e.key === 'Escape') document.getElementById('modal-bg').style.display = 'none';
}});

buildGrid();
renderAtlas();
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
    return build_html(out_dir / "16_attention_atlas.html")


if __name__ == "__main__":
    main()
