"""Interactive 1 — Compression sandbox / "Beat the greedy" challenge.

Generates a self-contained HTML page (no Python server required) where the user
designs their own SVD compression strategy with 6 component sliders × 3 depth
bands, and sees in real time:

  - Estimated F1 macro (from a damage model fitted to Tabla 9, 10 empirical
    retention values).
  - Parameter ratio.
  - Their strategy's position on the Pareto frontier vs. all 22 published
    strategies.
  - Per-emotion F1 estimates with cluster grouping.

The damage model lives in embedded JavaScript so the page is fully offline.
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz import thesis_data as td
from viz import style as st
from viz.data.load_results import load_sensitivity, load_informed


# ─────────────────────────────────────────────────────────────────────────────
# Damage model parameters — REAL retention from notebook 3 when available
# ─────────────────────────────────────────────────────────────────────────────

def _build_real_retention():
    """Build COMPONENT_RETENTION dict from nb3/component_f1_matrix.csv.

    Fills in r=32, 384, 512, 614, 768 by smooth extrapolation when the CSV
    only contains r=64, 128, 256.
    """
    sens = load_sensitivity()
    df = sens["component_f1"]
    if df is None:
        return None
    # Baseline (uniform r=768 effectively) — closest is the column with highest r
    informed = load_informed()
    baseline = 0.577
    if informed["finetuning"] is not None:
        baseline = float(informed["finetuning"]["baseline_f1"].mean())

    # Map CSV rows to our component keys (CSV uses underscore variants)
    name_map = {
        "Query": "query", "Key": "key", "Value": "value",
        "Attn_Output": "attn_output",
        "FFN_Inter": "ffn_intermediate",
        "FFN_Output": "ffn_output",
    }

    out = {}
    for csv_name, comp_key in name_map.items():
        if csv_name not in df.index:
            continue
        row = df.loc[csv_name]
        # Convert F1 → retention
        ret_64  = float(row.get("r64", 0)) / baseline if "r64" in row else 0.0
        ret_128 = float(row.get("r128", 0)) / baseline if "r128" in row else 0.0
        ret_256 = float(row.get("r256", 0)) / baseline if "r256" in row else 0.0
        # Extrapolate to 32 (just below 64, slightly worse)
        ret_32  = max(0.0, ret_64 - (ret_128 - ret_64) * 0.5)
        # Extrapolate above 256 toward baseline
        ret_384 = ret_256 + (1.0 - ret_256) * 0.55
        ret_512 = ret_256 + (1.0 - ret_256) * 0.85
        out[comp_key] = {32: ret_32, 64: ret_64, 128: ret_128, 256: ret_256,
                         384: min(1.0, ret_384), 512: min(1.0, ret_512),
                         768: 1.0}
        if comp_key.startswith("ffn"):
            out[comp_key][614] = 1.0
    return out


_REAL_RETENTION = _build_real_retention()
if _REAL_RETENTION:
    COMPONENT_RETENTION = _REAL_RETENTION
    print(f"[sandbox] using REAL component retention from notebook 3 "
          f"({len(COMPONENT_RETENTION)} components)")
else:
    # Fallback to thesis-fitted approximation
    COMPONENT_RETENTION = {
        "query":            {32: 0.95, 64: 0.97, 128: 0.994, 256: 1.000, 384: 1.000, 512: 1.000, 768: 1.000},
        "key":              {32: 0.93, 64: 0.96, 128: 0.984, 256: 1.000, 384: 1.000, 512: 1.000, 768: 1.000},
        "value":            {32: 0.05, 64: 0.05, 128: 0.685, 256: 0.93,  384: 0.98,  512: 1.000, 768: 1.000},
        "attn_output":      {32: 0.03, 64: 0.03, 128: 0.563, 256: 0.92,  384: 0.97,  512: 1.000, 768: 1.000},
        "ffn_intermediate": {32: 0.00, 64: 0.00, 128: 0.069, 256: 0.52,  384: 0.85,  512: 0.95,  614: 1.000, 768: 1.000},
        "ffn_output":       {32: 0.00, 64: 0.00, 128: 0.222, 256: 0.75,  384: 0.92,  512: 0.97,  614: 1.000, 768: 1.000},
    }

# Depth-band multiplier on damage cost (§6.3.1)
DEPTH_DAMAGE_WEIGHT = {"early": 0.148, "middle": 0.177, "late": 0.675}

# Component break-even ranks
BREAK_EVEN = {
    "query": 384, "key": 384, "value": 384, "attn_output": 384,
    "ffn_intermediate": 614, "ffn_output": 614,
}

# Component shapes for parameter counting
COMPONENT_SHAPE = {
    "query":            (768, 768),
    "key":              (768, 768),
    "value":            (768, 768),
    "attn_output":      (768, 768),
    "ffn_intermediate": (768, 3072),
    "ffn_output":       (3072, 768),
}


def build_html(out_path: pathlib.Path) -> pathlib.Path:
    """Emit the standalone sandbox HTML."""

    # Reference Pareto strategies for comparison
    strategies_json = json.dumps([{
        "name": s.name, "family": s.family,
        "f1": s.f1_macro, "ratio": s.param_ratio,
        "retention": s.retention, "pareto": s.pareto_optimal,
        "notes": s.notes,
    } for s in td.STRATEGIES])

    family_color_json = json.dumps(st.FAMILY_COLOR)

    # Per-emotion baseline data
    emotions_json = json.dumps([{
        "name": e,
        "f1_baseline": td.F1_BASELINE[e],
        "crystal_layer": td.CRYSTALLIZATION_LAYER[e],
        "selectivity_norm": td.NEURON_SELECTIVITY_NORM[e],
        "cluster": td.EMOTION_TO_CLUSTER[e],
        "cluster_color": td.CLUSTER_COLORS[td.EMOTION_TO_CLUSTER[e]],
    } for e in td.EMOTIONS])

    component_retention_json = json.dumps(COMPONENT_RETENTION)
    depth_weight_json = json.dumps(DEPTH_DAMAGE_WEIGHT)
    component_shape_json = json.dumps(COMPONENT_SHAPE)

    palette = json.dumps({
        "BLUE": st.BLUE, "TERRA": st.TERRA, "SAGE": st.SAGE, "SAND": st.SAND,
        "PLUM": st.PLUM, "ROSE": st.ROSE, "TEAL": st.TEAL,
        "INK": st.INK, "INK_2": st.INK_2, "INK_3": st.INK_3,
        "GRID": st.GRID, "SPINE": st.SPINE, "BG": st.BG,
    })

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Compression Sandbox — TFG Anatomía Emocional</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body {{
    font-family: "TeX Gyre Pagella", "Palatino", "Book Antiqua", serif;
    background: {st.BG};
    color: {st.INK};
    margin: 0; padding: 24px 36px;
  }}
  h1 {{ font-size: 22px; margin: 0 0 4px 0; font-weight: normal; letter-spacing: 0.2px; }}
  h1 .acc {{ color: {st.TERRA}; }}
  .sub {{ color: {st.INK_3}; font-size: 13px; margin-bottom: 20px; }}
  .grid {{ display: grid; grid-template-columns: 360px 1fr; gap: 28px; }}
  .controls {{
    background: #FAFAF8; border: 0.5px solid {st.SPINE};
    border-radius: 6px; padding: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
  }}
  .controls h2 {{
    font-size: 14px; font-weight: normal; letter-spacing: 0.4px; text-transform: uppercase;
    color: {st.INK_2}; margin: 0 0 16px 0;
  }}
  .row {{ margin-bottom: 16px; }}
  .row .lbl {{
    display: flex; justify-content: space-between; align-items: baseline;
    font-size: 12.5px; color: {st.INK_2}; margin-bottom: 4px;
  }}
  .row .val {{ font-family: "Inter", monospace; font-size: 12px; color: {st.INK}; }}
  .row input[type=range] {{ width: 100%; }}
  .triple {{
    display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;
  }}
  .triple .col {{ font-size: 10.5px; color: {st.INK_3}; text-align: center; }}
  .triple .col span.v {{ display: block; color: {st.INK}; font-family: "Inter", monospace; font-size: 11px; }}
  .preset-row {{ display: flex; gap: 6px; margin-bottom: 14px; flex-wrap: wrap; }}
  .preset-row button {{
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 4px;
    padding: 5px 9px; font-size: 11.5px; color: {st.INK_2}; cursor: pointer;
    font-family: inherit;
  }}
  .preset-row button:hover {{ background: {st.SAND}; color: {st.INK}; border-color: {st.INK_3}; }}
  .preset-row button.greedy {{ background: {st.SAND}; color: {st.INK}; border-color: {st.INK_3}; }}
  .readout {{
    margin-top: 14px; padding: 14px;
    background: white; border: 0.5px solid {st.SPINE}; border-radius: 5px;
  }}
  .readout .big {{
    display: flex; justify-content: space-between; margin-bottom: 8px;
    align-items: baseline;
  }}
  .readout .big .name {{ font-size: 11.5px; color: {st.INK_3}; text-transform: uppercase; letter-spacing: 0.5px; }}
  .readout .big .num {{ font-size: 22px; font-family: "Inter", monospace; color: {st.INK}; }}
  .readout .big .num.alt {{ color: {st.TERRA}; }}
  .readout .delta {{ font-size: 11px; color: {st.INK_3}; }}
  .charts {{ display: grid; grid-template-rows: 1fr 1fr; gap: 16px; }}
  .panel {{ background: white; border: 0.5px solid {st.SPINE}; border-radius: 5px; padding: 8px; }}
  .footer {{ font-size: 11px; color: {st.INK_3}; margin-top: 14px; line-height: 1.5; }}
  .badge {{
    display: inline-block; padding: 1px 7px; border-radius: 3px; font-size: 10.5px;
    background: {st.SAND}; color: {st.INK}; font-family: "Inter", monospace;
  }}
</style>
</head>
<body>

<h1>Compression Sandbox — <span class="acc">Beat the Greedy</span></h1>
<div class="sub">
  Diseña tu propia estrategia SVD ajustando el rango por componente y profundidad.
  El simulador estima F1 macro, ratio de parámetros, y tu posición en la frontera de Pareto en tiempo real.
  ¿Puedes vencer al algoritmo greedy data-driven?
</div>

<div class="grid">
  <div class="controls">
    <h2>Presets</h2>
    <div class="preset-row">
      <button onclick="applyPreset('baseline')">Baseline</button>
      <button onclick="applyPreset('uniform_r512')">Uniform r=512</button>
      <button onclick="applyPreset('uniform_r256')">Uniform r=256</button>
      <button onclick="applyPreset('adaptive_e95')">Adaptive 95%</button>
      <button class="greedy" onclick="applyPreset('greedy_90')">Greedy 90%</button>
    </div>

    <h2>Rango SVD por componente y profundidad</h2>
    <div id="sliders"></div>

    <div class="readout">
      <div class="big">
        <span class="name">F1 macro estimado</span>
        <span class="num" id="readout-f1">0.577</span>
      </div>
      <div class="delta" id="readout-f1-delta"></div>
      <div class="big" style="margin-top: 12px">
        <span class="name">Ratio de parámetros</span>
        <span class="num alt" id="readout-ratio">1.000×</span>
      </div>
      <div class="delta" id="readout-ratio-delta"></div>
      <div class="big" style="margin-top: 12px">
        <span class="name">Pareto status</span>
        <span class="num" id="readout-pareto" style="font-size: 14px">—</span>
      </div>
    </div>
    <div class="footer">
      Modelo: BERT-base, 23 emociones GoEmotions, F1 macro baseline = 0.577.
      Estimador entrenado sobre los datos empíricos de Cap. 4 (Tablas 9, 10).
      Las predicciones por emoción son aproximaciones basadas en la cristalización y selectividad neuronal.
    </div>
  </div>

  <div class="charts">
    <div class="panel"><div id="pareto-chart"></div></div>
    <div class="panel"><div id="emotions-chart"></div></div>
  </div>
</div>

<script>
const STRATEGIES = {strategies_json};
const FAMILY_COLOR = {family_color_json};
const EMOTIONS = {emotions_json};
const COMP_RET = {component_retention_json};
const DEPTH_W = {depth_weight_json};
const COMP_SHAPE = {component_shape_json};
const PAL = {palette};
const F1_BASELINE = 0.577;
const N_LAYERS = 12;
const BAND_LAYERS = {{ "early": 4, "middle": 4, "late": 4 }};  // 0-3, 4-7, 8-11
const RANK_STEPS = [32, 64, 128, 256, 384, 512, 768];
const COMPONENTS = [
  {{key:"query", lbl:"Query (Q)"}},
  {{key:"key", lbl:"Key (K)"}},
  {{key:"value", lbl:"Value (V)"}},
  {{key:"attn_output", lbl:"Attn Output"}},
  {{key:"ffn_intermediate", lbl:"FFN Intermediate"}},
  {{key:"ffn_output", lbl:"FFN Output"}},
];
const BANDS = ["early", "middle", "late"];
const BAND_LBL = {{early:"Early (0-3)", middle:"Middle (4-7)", late:"Late (8-11)"}};

// Presets: each maps to a {{ comp: {{ early, middle, late }} }} matrix of ranks.
function rk(early, middle, late) {{ return {{early, middle, late}}; }}
const PRESETS = {{
  baseline: {{ query: rk(768,768,768), key: rk(768,768,768), value: rk(768,768,768),
              attn_output: rk(768,768,768), ffn_intermediate: rk(768,768,768), ffn_output: rk(768,768,768) }},
  uniform_r512: {{ query: rk(512,512,512), key: rk(512,512,512), value: rk(512,512,512),
                   attn_output: rk(512,512,512), ffn_intermediate: rk(512,512,512), ffn_output: rk(512,512,512) }},
  uniform_r256: {{ query: rk(256,256,256), key: rk(256,256,256), value: rk(256,256,256),
                   attn_output: rk(256,256,256), ffn_intermediate: rk(256,256,256), ffn_output: rk(256,256,256) }},
  adaptive_e95: {{ query: rk(384,384,384), key: rk(384,384,384), value: rk(384,384,384),
                   attn_output: rk(384,384,384), ffn_intermediate: rk(614,614,614), ffn_output: rk(614,614,614) }},
  greedy_90: {{ query: rk(64,128,256), key: rk(64,128,256), value: rk(384,384,512),
                attn_output: rk(384,384,512), ffn_intermediate: rk(384,512,768), ffn_output: rk(128,384,614) }},
}};

let CURRENT = JSON.parse(JSON.stringify(PRESETS.baseline));

// ────────────── damage model ──────────────
function lookupRetention(comp, rank) {{
  const tbl = COMP_RET[comp];
  const ks = Object.keys(tbl).map(Number).sort((a,b)=>a-b);
  if (rank <= ks[0]) return tbl[ks[0]];
  if (rank >= ks[ks.length-1]) return tbl[ks[ks.length-1]];
  for (let i = 0; i < ks.length-1; i++) {{
    if (rank >= ks[i] && rank <= ks[i+1]) {{
      const a = ks[i], b = ks[i+1];
      const t = (rank - a) / (b - a);
      return tbl[a] * (1-t) + tbl[b] * t;
    }}
  }}
  return 1.0;
}}

function estimateF1(matrix) {{
  // Combine per-component, per-band damage with the depth weights from §6.3.1.
  // total_retention = baseline * (1 - sum_{{c,b}} depth_weight[b] * (1 - retention(c, rank[c][b])) / 6)
  let damage = 0;
  for (const comp of COMPONENTS) {{
    for (const band of BANDS) {{
      const rank = matrix[comp.key][band];
      const ret = lookupRetention(comp.key, rank);
      damage += DEPTH_W[band] * (1 - ret);
    }}
  }}
  damage = damage / COMPONENTS.length;
  // Empirical scaling: at uniform r=128 across all comps (max damage in our model)
  // we want F1 to drop to ~0; fit scale so this holds.
  const f1 = Math.max(0, F1_BASELINE * (1 - damage * 4.5));
  return f1;
}}

function paramRatio(matrix) {{
  // Total params under matrix vs. baseline (12 layers per component)
  let baseline = 0, current = 0;
  for (const comp of COMPONENTS) {{
    const [m, n] = COMP_SHAPE[comp.key];
    const baseLayer = m * n;
    baseline += baseLayer * 12;
    for (const band of BANDS) {{
      const rank = matrix[comp.key][band];
      const layersInBand = BAND_LAYERS[band];
      const compressed = rank >= 768 ? baseLayer : rank * (m + n);
      current += Math.min(compressed, baseLayer) * layersInBand;
    }}
  }}
  return current / baseline;
}}

function emotionF1(matrix, em) {{
  // Each emotion's F1 inherits global F1 ratio, modulated by its vulnerability:
  //   - High selectivity_norm + late crystallization => stronger drop.
  //   - Low values => relatively spared.
  const f1Global = estimateF1(matrix);
  const dropFraction = 1 - f1Global / F1_BASELINE; // 0..1
  // Vulnerability factor in [0.4, 1.6]
  const crystalNorm = em.crystal_layer / 11;          // 0..1
  const selNorm = (em.selectivity_norm - 38) / (140 - 38); // 0..1
  const vuln = 0.55 + 0.55 * crystalNorm + 0.45 * selNorm; // ~0.55..1.55
  const emDrop = Math.min(1, dropFraction * vuln);
  return Math.max(0, em.f1_baseline * (1 - emDrop));
}}

function isParetoOptimal(ratio, f1) {{
  for (const s of STRATEGIES) {{
    if (s.ratio < ratio - 0.005 && s.f1 > f1 + 0.005) return false;
  }}
  return true;
}}

// ────────────── slider UI ──────────────
function snapToStep(value) {{
  let best = RANK_STEPS[0], bd = Infinity;
  for (const s of RANK_STEPS) {{
    if (Math.abs(s - value) < bd) {{ bd = Math.abs(s - value); best = s; }}
  }}
  return best;
}}

function rebuildSliders() {{
  const root = document.getElementById("sliders");
  root.innerHTML = "";
  for (const comp of COMPONENTS) {{
    const div = document.createElement("div");
    div.className = "row";
    const lbl = document.createElement("div");
    lbl.className = "lbl";
    lbl.innerHTML = `<span>${{comp.lbl}}</span><span class="val" id="lbl-${{comp.key}}">— / — / —</span>`;
    div.appendChild(lbl);
    const triple = document.createElement("div");
    triple.className = "triple";
    for (const band of BANDS) {{
      const col = document.createElement("div");
      col.className = "col";
      col.innerHTML = `${{BAND_LBL[band]}}<span class="v" id="v-${{comp.key}}-${{band}}">${{CURRENT[comp.key][band]}}</span>`;
      const slider = document.createElement("input");
      slider.type = "range";
      slider.min = 0; slider.max = RANK_STEPS.length-1; slider.step = 1;
      slider.value = RANK_STEPS.indexOf(CURRENT[comp.key][band]);
      slider.style.width = "100%";
      slider.oninput = function() {{
        CURRENT[comp.key][band] = RANK_STEPS[parseInt(this.value)];
        document.getElementById(`v-${{comp.key}}-${{band}}`).textContent = CURRENT[comp.key][band];
        update();
      }};
      col.appendChild(slider);
      triple.appendChild(col);
    }}
    div.appendChild(triple);
    root.appendChild(div);
  }}
  update();
}}

function applyPreset(name) {{
  CURRENT = JSON.parse(JSON.stringify(PRESETS[name]));
  rebuildSliders();
}}

// ────────────── charts ──────────────
function buildParetoChart() {{
  const families = ["uniform","adaptive","mixed","informed","greedy","baseline"];
  const familyLabel = {{
    uniform:"Uniforme", adaptive:"Adaptativa", mixed:"Mixta",
    informed:"Informada", greedy:"Greedy", baseline:"Baseline",
  }};
  const familySymbol = {{
    uniform:"square", adaptive:"circle", mixed:"diamond-tall",
    informed:"diamond", greedy:"star", baseline:"x-thin-open",
  }};
  const traces = families.map(fam => {{
    const subset = STRATEGIES.filter(s => s.family === fam);
    return {{
      x: subset.map(s => s.ratio),
      y: subset.map(s => s.f1),
      mode: "markers", type: "scatter", name: familyLabel[fam],
      marker: {{
        size: subset.map(s => s.pareto ? 18 : 10),
        color: FAMILY_COLOR[fam],
        symbol: familySymbol[fam],
        line: {{color: "white", width: 1}},
      }},
      text: subset.map(s => s.name),
      hovertemplate: "<b>%{{text}}</b><br>Ratio: %{{x:.3f}}<br>F1: %{{y:.3f}}<extra></extra>",
    }};
  }});
  // user point trace (added last so it draws on top)
  traces.push({{
    x: [paramRatio(CURRENT)], y: [estimateF1(CURRENT)],
    mode: "markers", type: "scatter", name: "Tu estrategia",
    marker: {{
      size: 26, color: PAL.TERRA, symbol: "x-dot",
      line: {{color: PAL.INK, width: 2.5}},
    }},
    hovertemplate: "<b>Tu estrategia</b><br>Ratio: %{{x:.3f}}<br>F1: %{{y:.3f}}<extra></extra>",
  }});

  const layout = {{
    title: {{text: "Frontera de Pareto", font: {{size: 13, color: PAL.INK}}, x: 0.02}},
    paper_bgcolor: "white", plot_bgcolor: "white",
    margin: {{l:55, r:20, t:36, b:48}},
    xaxis: {{title: "Ratio de parámetros", range:[0.30, 1.45], gridcolor: PAL.GRID, zeroline:false, linecolor: PAL.SPINE, showline: true}},
    yaxis: {{title: "F1 macro", range:[-0.03, 0.62], gridcolor: PAL.GRID, zeroline:false, linecolor: PAL.SPINE, showline: true}},
    font: {{family: '"TeX Gyre Pagella", "Palatino", serif', color: PAL.INK_2, size: 11}},
    legend: {{font: {{size: 10}}, x:0.01, y:0.99, bgcolor: "rgba(255,255,255,0.85)"}},
    height: 280,
  }};
  Plotly.newPlot("pareto-chart", traces, layout, {{displayModeBar: false}});
}}

function buildEmotionsChart() {{
  // Emotions sorted by F1 baseline descending
  const sorted = [...EMOTIONS].sort((a,b) => b.f1_baseline - a.f1_baseline);
  const f1Now = sorted.map(e => emotionF1(CURRENT, e));
  const f1Base = sorted.map(e => e.f1_baseline);
  const colors = sorted.map(e => e.cluster_color);

  const traces = [
    {{
      x: sorted.map(e => e.name), y: f1Base,
      type: "bar", name: "Baseline",
      marker: {{color: PAL.GRID, line: {{color: PAL.SPINE, width: 0.5}}}},
      hovertemplate: "<b>%{{x}}</b><br>F1 baseline: %{{y:.3f}}<extra></extra>",
    }},
    {{
      x: sorted.map(e => e.name), y: f1Now,
      type: "bar", name: "Tu estrategia",
      marker: {{color: colors, line: {{color: "white", width: 0.5}}}},
      hovertemplate: "<b>%{{x}}</b><br>F1 estimado: %{{y:.3f}}<extra></extra>",
    }},
  ];

  const layout = {{
    title: {{text: "F1 estimado por emoción (color por cluster)", font:{{size:13, color:PAL.INK}}, x:0.02}},
    barmode: "overlay",
    paper_bgcolor: "white", plot_bgcolor: "white",
    margin: {{l:55, r:20, t:36, b:80}},
    xaxis: {{tickangle: -45, gridcolor: PAL.GRID, tickfont:{{size:9.5}}, linecolor: PAL.SPINE, showline: true}},
    yaxis: {{title: "F1", range:[0, 1.0], gridcolor: PAL.GRID, zeroline:false, linecolor: PAL.SPINE, showline: true}},
    font: {{family: '"TeX Gyre Pagella", "Palatino", serif', color: PAL.INK_2, size: 11}},
    height: 320, showlegend: true,
    legend: {{font:{{size:10}}, x:0.97, y:0.99, xanchor:"right", bgcolor: "rgba(255,255,255,0.85)"}},
  }};
  Plotly.newPlot("emotions-chart", traces, layout, {{displayModeBar: false}});
}}

function update() {{
  const f1 = estimateF1(CURRENT);
  const ratio = paramRatio(CURRENT);
  const pareto = isParetoOptimal(ratio, f1);

  document.getElementById("readout-f1").textContent = f1.toFixed(3);
  document.getElementById("readout-ratio").textContent = ratio.toFixed(3) + "×";

  const f1Delta = ((f1 - F1_BASELINE) / F1_BASELINE * 100).toFixed(1);
  const ratioDelta = ((1 - ratio) * 100).toFixed(1);
  document.getElementById("readout-f1-delta").innerHTML =
    `<span style="color:${{f1 >= F1_BASELINE ? PAL.SAGE : PAL.TERRA}}">${{f1Delta >= 0 ? '+' : ''}}${{f1Delta}}% vs. baseline</span>`;
  document.getElementById("readout-ratio-delta").innerHTML =
    ratio < 1 ? `<span style="color:${{PAL.SAGE}}">${{ratioDelta}}% menos parámetros</span>`
              : `<span style="color:${{PAL.TERRA}}">${{(-ratioDelta)}}% más parámetros (expansión)</span>`;
  document.getElementById("readout-pareto").innerHTML =
    pareto ? `<span class="badge" style="background:${{PAL.SAGE}};color:white">Pareto-óptima</span>`
           : `<span style="color:${{PAL.INK_3}}">Dominada</span>`;

  buildParetoChart();
  buildEmotionsChart();
}}

rebuildSliders();
</script>

</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"✓ wrote {out_path}")
    return out_path


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    return build_html(out_dir / "05_compression_sandbox.html")


if __name__ == "__main__":
    main()
