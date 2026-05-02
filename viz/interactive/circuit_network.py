"""Custom signature visualization — Emotion ↔ Head ↔ Neuron circuit network.

Hand-built in D3.js (no Plotly). Force-directed network where:

  - 23 emotion nodes (left column) form a vertical ladder.
  - 144 attention head nodes (center) sized by importance.
  - 6 cluster nodes (right) where each emotion's neurons aggregate.

Edges:
  - Emotion → critical head (per Tabla 19).
  - Head → cluster (the cluster of the emotions it serves).
  - Emotion → cluster (its psychological group).

The shared L11-H6 link (sadness, realization) is rendered with a special
double-edge style to make the §5.3.5 finding visually obvious.

Click an emotion to highlight its full pathway. Click a head to see which
emotions depend on it. The two emotions that share L11-H6 are the visual
punchline of the chart.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz import thesis_data as td
from viz import style as st
from viz.data.load_results import (load_heads, load_neurons, load_informed,
                                    critical_head_per_emotion,
                                    neuron_count_per_emotion,
                                    f1_baseline_per_emotion,
                                    EMOTIONS_23)


def build_graph_data() -> dict:
    """Construct nodes and edges for the D3 force simulation."""
    nodes = []
    links = []

    # REAL data
    heads_data = load_heads()
    neurons_data = load_neurons()
    informed_data = load_informed()
    real_critical = critical_head_per_emotion(heads_data) if heads_data["top_heads"] is not None else td.CRITICAL_HEAD_PER_EMOTION
    real_counts = neuron_count_per_emotion(neurons_data) if neurons_data["significant_counts"] is not None else td.NEURON_COUNT_PER_EMOTION
    real_baseline = f1_baseline_per_emotion(informed_data) if informed_data["finetuning"] is not None else td.F1_BASELINE

    # Emotion nodes (23, ordered by REAL F1 baseline desc)
    sorted_emotions = sorted(EMOTIONS_23, key=lambda e: -real_baseline.get(e, 0))
    for i, emo in enumerate(sorted_emotions):
        cluster = td.EMOTION_TO_CLUSTER[emo]
        nodes.append({
            "id": f"emo:{emo}",
            "kind": "emotion",
            "label": emo,
            "cluster": cluster,
            "color": td.CLUSTER_COLORS[cluster],
            "f1_baseline": real_baseline.get(emo, td.F1_BASELINE.get(emo, 0)),
            "neuron_count": real_counts.get(emo, 0),
            "selectivity_norm": td.NEURON_SELECTIVITY_NORM.get(emo, 50),
            "crystal_layer": td.CRYSTALLIZATION_LAYER[emo],
            "rank_y": i,
        })

    # Cluster nodes (6)
    cluster_idx = 0
    for cluster, members in td.CLUSTER_DEFS.items():
        nodes.append({
            "id": f"cls:{cluster}",
            "kind": "cluster",
            "label": cluster,
            "color": td.CLUSTER_COLORS[cluster],
            "size": sum(real_counts.get(e, 0) for e in members),
            "n_members": len(members),
            "rank_y": cluster_idx,
        })
        cluster_idx += 1
        # Emotion → cluster edges
        for emo in members:
            links.append({
                "source": f"emo:{emo}", "target": f"cls:{cluster}",
                "kind": "membership", "weight": 0.4,
                "color": td.CLUSTER_COLORS[cluster] + "66",  # semi-transparent
            })

    # Head nodes — only those that are critical for at least one emotion (REAL)
    head_ids: dict[str, list[str]] = {}  # head_id -> list of emotions
    for emo, (layer, h, df1) in real_critical.items():
        hid = f"head:L{layer}-H{h}"
        head_ids.setdefault(hid, []).append(emo)

    for hid, emos in head_ids.items():
        # Compute aggregate impact (REAL)
        total_impact = sum(abs(real_critical[e][2]) for e in emos)
        is_shared = len(emos) > 1
        layer = int(hid.split("L")[1].split("-")[0])
        head = int(hid.split("H")[1])
        nodes.append({
            "id": hid,
            "kind": "head",
            "label": hid.replace("head:", ""),
            "layer": layer, "head": head,
            "color": st.TERRA if is_shared else st.BLUE if layer >= 8 else st.SAGE,
            "shared": is_shared,
            "emotions": emos,
            "total_impact": total_impact,
            "size": 12 + total_impact * 60,
        })
        # Emotion → head edges (REAL)
        for emo in emos:
            df1_emo = real_critical[emo][2]
            links.append({
                "source": f"emo:{emo}", "target": hid,
                "kind": "critical",
                "weight": 1.5 + abs(df1_emo) * 8,
                "df1": df1_emo,
                "shared": is_shared,
                "color": st.TERRA if is_shared else st.INK_2,
            })

    # Stats
    stats = {
        "n_emotions": 23,
        "n_clusters": 6,
        "n_critical_heads": len(head_ids),
        "n_shared_heads": sum(1 for v in head_ids.values() if len(v) > 1),
        "shared_pairs": [v for v in head_ids.values() if len(v) > 1],
    }

    return {"nodes": nodes, "links": links, "stats": stats}


def build_html(out_path: pathlib.Path) -> pathlib.Path:
    graph = build_graph_data()
    data_json = json.dumps(graph)

    palette = json.dumps({
        "BLUE": st.BLUE, "TERRA": st.TERRA, "SAGE": st.SAGE, "SAND": st.SAND,
        "PLUM": st.PLUM, "INK": st.INK, "INK_2": st.INK_2, "INK_3": st.INK_3,
        "GRID": st.GRID, "SPINE": st.SPINE, "BG": st.BG,
    })

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Circuit Network — Anatomía Emocional</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  body {{
    font-family: "TeX Gyre Pagella", "Palatino", "Book Antiqua", serif;
    background: #FAFAF6; color: #1A1A1A; margin: 0; padding: 0;
  }}
  header {{
    padding: 24px 32px 8px 32px;
  }}
  header h1 {{
    font-size: 24px; font-weight: normal; margin: 0 0 4px 0;
  }}
  header h1 .acc {{ color: #C1553A; }}
  header .sub {{ color: #8A8A85; font-size: 13.5px; margin-bottom: 6px; }}
  header .stats {{
    color: #4A4A4A; font-size: 12.5px; font-style: italic;
    border-top: 0.5px solid #C8C7C1; padding-top: 8px; margin-top: 6px;
  }}
  .layout {{
    display: grid; grid-template-columns: 1fr 280px; gap: 0;
    height: calc(100vh - 130px);
  }}
  #graph {{ background: #FAFAF6; }}
  #panel {{
    background: white; border-left: 0.5px solid #C8C7C1;
    padding: 22px 22px; overflow-y: auto;
  }}
  #panel h2 {{
    font-size: 11.5px; text-transform: uppercase; letter-spacing: 1.2px;
    color: #4A4A4A; font-weight: normal; margin: 18px 0 8px 0;
  }}
  #panel h2:first-child {{ margin-top: 0; }}
  #panel .big {{ font-size: 21px; color: #1A1A1A; margin-bottom: 4px; font-family: serif; }}
  #panel .meta {{ font-size: 12px; color: #8A8A85; margin-bottom: 14px; line-height: 1.4; }}
  #panel .row {{
    display: flex; justify-content: space-between; padding: 5px 0;
    border-bottom: 0.5px dashed #EBEBEB; font-size: 12.5px;
  }}
  #panel .row .lbl {{ color: #8A8A85; }}
  #panel .row .val {{ font-family: "Inter", monospace; color: #1A1A1A; font-size: 11.5px; }}
  #panel .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 3px;
    font-size: 10.5px; font-family: "Inter", sans-serif; color: white;
  }}
  .hint {{
    font-size: 11.5px; color: #8A8A85; padding: 10px 14px;
    background: rgba(193,85,58,0.06); border-left: 2px solid #C1553A;
    margin: 14px 0; line-height: 1.5;
  }}
  .legend {{
    display: flex; gap: 14px; padding: 0 32px 14px 32px; flex-wrap: wrap;
    font-size: 11.5px; color: #4A4A4A;
  }}
  .legend .item {{ display: flex; align-items: center; gap: 5px; }}
  .legend .swatch {{
    width: 11px; height: 11px; border-radius: 50%; display: inline-block;
  }}
  .node-emotion text {{ font-size: 11px; fill: #1A1A1A; pointer-events: none; }}
  .node-cluster text {{ font-size: 11.5px; fill: #1A1A1A; font-style: italic; pointer-events: none; }}
  .node-head text {{ font-size: 9.5px; fill: #4A4A4A; font-family: "Inter", monospace; pointer-events: none; }}
  .link {{ stroke-opacity: 0.5; fill: none; }}
  .link.critical {{ stroke-dasharray: none; }}
  .link.membership {{ stroke-dasharray: 3 3; }}
  .link.shared {{ stroke-width: 2.4 !important; }}
  circle {{ cursor: pointer; transition: stroke-width 0.15s, opacity 0.15s; }}
  circle:hover {{ stroke-width: 2.5 !important; }}
  .dimmed {{ opacity: 0.18; }}
  .highlighted {{ opacity: 1; stroke-width: 2.5 !important; }}
  .highlighted-link {{ stroke-opacity: 1 !important; stroke-width: 3 !important; }}
</style>
</head>
<body>

<header>
  <h1>Circuit network — <span class="acc">emoción · cabeza · cluster</span></h1>
  <div class="sub">Cap. 5 · Mapa de los circuitos atencionales que cada emoción reserva, y la cabeza que las unifica.</div>
  <div class="stats" id="stats"></div>
</header>

<div class="legend">
  <div class="item"><span class="swatch" style="background:#3A6EA5"></span>Cabeza crítica capa 8-11</div>
  <div class="item"><span class="swatch" style="background:#5A8F7B"></span>Cabeza crítica capa 0-7</div>
  <div class="item"><span class="swatch" style="background:#C1553A"></span>Cabeza compartida (varias emociones)</div>
  <div class="item">— enlaces continuos: dependencia crítica</div>
  <div class="item">- - enlaces discontinuos: pertenencia a cluster</div>
</div>

<div class="layout">
  <svg id="graph" width="100%" height="100%"></svg>
  <div id="panel">
    <div class="hint">
      <b>Haz click</b> en cualquier nodo. Los nodos rojos son las dos cabezas
      compartidas — el "circuito común" de §5.3.5.
    </div>
    <div id="detail"></div>
  </div>
</div>

<script>
const DATA = {data_json};
const PAL = {palette};

// Stats header
document.getElementById('stats').textContent =
  `${{DATA.stats.n_emotions}} emociones · ${{DATA.stats.n_clusters}} clusters · `
  + `${{DATA.stats.n_critical_heads}} cabezas críticas distintas · `
  + `${{DATA.stats.n_shared_heads}} cabeza(s) compartida(s) por más de una emoción.`;

const svg = d3.select("#graph");
const W = svg.node().getBoundingClientRect().width;
const H = svg.node().getBoundingClientRect().height;

// Define column targets for each kind
const COL = {{
  emotion: 0.18 * W,
  head: 0.55 * W,
  cluster: 0.86 * W,
}};

// Initial positions
DATA.nodes.forEach(n => {{
  if (n.kind === "emotion") {{
    n.fx = COL.emotion; n.fy = 60 + n.rank_y * (H - 100) / 22;
  }} else if (n.kind === "cluster") {{
    n.fx = COL.cluster; n.fy = 80 + n.rank_y * (H - 120) / 5;
  }} else {{
    // heads — let force layout place them in the middle column
    n.x = COL.head + (Math.random() - 0.5) * 200;
    n.y = H / 2 + (Math.random() - 0.5) * (H - 100);
  }}
}});

// Force simulation — heads only
const sim = d3.forceSimulation(DATA.nodes)
  .force("link", d3.forceLink(DATA.links).id(d => d.id).distance(d => {{
    if (d.kind === "membership") return 200;
    return 90 + (1 - (d.weight || 1) / 8) * 60;
  }}).strength(d => d.kind === "membership" ? 0.15 : 0.55))
  .force("charge", d3.forceManyBody().strength(d => {{
    if (d.kind === "head") return -160;
    return -40;
  }}))
  .force("collide", d3.forceCollide().radius(d =>
      d.kind === "emotion" ? 16 : d.kind === "cluster" ? 24 : (d.size || 12) * 0.6 + 6))
  .force("x", d3.forceX(d => COL[d.kind] || W/2).strength(d => d.kind === "head" ? 0.18 : 0))
  .force("y", d3.forceY(H/2).strength(d => d.kind === "head" ? 0.04 : 0));

// Curved links via path
const linkPath = d => {{
  const sx = d.source.x, sy = d.source.y, tx = d.target.x, ty = d.target.y;
  const dx = tx - sx, dy = ty - sy;
  const dr = Math.sqrt(dx*dx + dy*dy) * 1.6;
  return `M${{sx}},${{sy}}A${{dr}},${{dr}} 0 0,1 ${{tx}},${{ty}}`;
}};

const linkSel = svg.append("g")
  .selectAll("path")
  .data(DATA.links)
  .join("path")
  .attr("class", d => `link ${{d.kind}}${{d.shared ? " shared" : ""}}`)
  .attr("stroke", d => d.color)
  .attr("stroke-width", d => d.weight || 1);

// Node groups
const nodeSel = svg.append("g")
  .selectAll("g")
  .data(DATA.nodes)
  .join("g")
  .attr("class", d => `node-${{d.kind}}`)
  .on("click", (event, d) => {{
    showDetail(d);
    highlightConnected(d);
    event.stopPropagation();
  }});

// Background click clears highlight
svg.on("click", () => {{
  nodeSel.classed("dimmed", false).classed("highlighted", false);
  linkSel.classed("dimmed", false).classed("highlighted-link", false);
  showDetail(null);
}});

nodeSel.append("circle")
  .attr("r", d => {{
    if (d.kind === "emotion") return 8 + d.f1_baseline * 7;
    if (d.kind === "cluster") return 14 + Math.sqrt(d.size) * 0.45;
    return Math.max(6, Math.min(22, d.size * 0.45));
  }})
  .attr("fill", d => d.color)
  .attr("stroke", "white")
  .attr("stroke-width", 1.4)
  .attr("opacity", 0.92);

nodeSel.append("text")
  .attr("dx", d => {{
    if (d.kind === "emotion") return -12;
    if (d.kind === "cluster") return 18;
    return 10;
  }})
  .attr("dy", 4)
  .attr("text-anchor", d => d.kind === "emotion" ? "end" : "start")
  .text(d => d.kind === "head" ? d.label.replace("L", "L").replace("H", "H") : d.label);

sim.on("tick", () => {{
  linkSel.attr("d", linkPath);
  nodeSel.attr("transform", d => `translate(${{d.x}},${{d.y}})`);
}});

function highlightConnected(node) {{
  if (!node) {{
    nodeSel.classed("dimmed", false).classed("highlighted", false);
    linkSel.classed("dimmed", false).classed("highlighted-link", false);
    return;
  }}
  const connectedIds = new Set([node.id]);
  DATA.links.forEach(l => {{
    if (l.source.id === node.id) connectedIds.add(l.target.id);
    if (l.target.id === node.id) connectedIds.add(l.source.id);
  }});
  nodeSel
    .classed("dimmed", n => !connectedIds.has(n.id))
    .classed("highlighted", n => n.id === node.id);
  linkSel
    .classed("dimmed", l => l.source.id !== node.id && l.target.id !== node.id)
    .classed("highlighted-link", l => l.source.id === node.id || l.target.id === node.id);
}}

function showDetail(node) {{
  const detail = document.getElementById('detail');
  if (!node) {{
    detail.innerHTML = `
      <h2>Cómo leer este grafo</h2>
      <div class="meta">
        Cada emoción (izquierda) se conecta a su <b>cabeza crítica</b> según
        Tabla 19 de la memoria. La línea termina en una cabeza (centro) y se
        ramifica al cluster psicológico (derecha). Las cabezas en rojo son
        compartidas — múltiples emociones dependen de la misma circuiteria.
      </div>
      <h2>Hallazgo central</h2>
      <div class="meta">
        <b>L11-H6 sirve a la vez a sadness y realization</b>: un patrón atencional
        de "expectativa frustrada" que ambas emociones comparten. Es el primer
        circuito reutilizado documentado en este dataset.
      </div>
      <h2>Estadísticas</h2>
      <div class="row"><span class="lbl">Emociones</span><span class="val">${{DATA.stats.n_emotions}}</span></div>
      <div class="row"><span class="lbl">Cabezas críticas</span><span class="val">${{DATA.stats.n_critical_heads}}</span></div>
      <div class="row"><span class="lbl">Cabezas compartidas</span><span class="val">${{DATA.stats.n_shared_heads}}</span></div>
    `;
    return;
  }}

  if (node.kind === "emotion") {{
    detail.innerHTML = `
      <h2>Emoción</h2>
      <div class="big">${{node.label}}</div>
      <div class="meta">
        <span class="badge" style="background:${{node.color}}">${{node.cluster}}</span>
      </div>
      <h2>Geografía interna</h2>
      <div class="row"><span class="lbl">F1 baseline</span><span class="val">${{node.f1_baseline.toFixed(3)}}</span></div>
      <div class="row"><span class="lbl">Cristaliza en</span><span class="val">L${{node.crystal_layer}}</span></div>
      <div class="row"><span class="lbl">Norma selectividad</span><span class="val">${{node.selectivity_norm}}</span></div>
      <div class="row"><span class="lbl">Neuronas significativas</span><span class="val">${{node.neuron_count}}</span></div>
      <h2>Vulnerabilidad</h2>
      <div class="meta">
        Esta emoción depende críticamente de su cabeza marcada. Si la cabeza es
        compartida (roja), está más protegida; si es única, una sola ablación
        la destruye.
      </div>
    `;
  }} else if (node.kind === "head") {{
    const emoLines = node.emotions.map(e => {{
      const df1 = DATA.nodes.find(n => n.id === `emo:${{e}}`);
      const link = DATA.links.find(l => l.source.id === `emo:${{e}}` && l.target.id === node.id);
      return `<div class="row"><span class="lbl">${{e}}</span><span class="val">ΔF1 ${{(link.df1).toFixed(3)}}</span></div>`;
    }}).join("");
    detail.innerHTML = `
      <h2>Cabeza de atención</h2>
      <div class="big">${{node.label}}</div>
      <div class="meta">
        Capa ${{node.layer}}, índice de cabeza ${{node.head}}
        ${{node.shared ? '<br><span class="badge" style="background:#C1553A">cabeza compartida</span>' : ''}}
      </div>
      <h2>Emociones que dependen</h2>
      ${{emoLines}}
      <h2>Impacto agregado</h2>
      <div class="row"><span class="lbl">Suma |ΔF1|</span><span class="val">${{node.total_impact.toFixed(3)}}</span></div>
      ${{node.shared ? `<div class="hint" style="margin-top:14px">
        Esta cabeza activa un patrón atencional que ${{node.emotions.length}} emociones
        diferentes <b>reutilizan</b>. Es el ejemplo del paper sobre interpretabilidad:
        circuiteria compartida emergente sin diseño explícito.
      </div>` : ''}}
    `;
  }} else if (node.kind === "cluster") {{
    const members = Object.entries({{
      "Positivas alta energía":  ["admiration", "amusement", "excitement", "gratitude", "joy", "love"],
      "Negativas reactivas":     ["anger", "annoyance", "disappointment", "disapproval", "disgust"],
      "Negativas internas":      ["embarrassment", "fear", "remorse", "sadness"],
      "Epistémicas":             ["confusion", "curiosity", "surprise"],
      "Orientadas al otro":      ["caring", "desire", "optimism"],
      "Baja especificidad":      ["approval", "realization"],
    }}).find(([k, _]) => k === node.label)[1];
    const emoLines = members.map(e => {{
      const en = DATA.nodes.find(n => n.id === `emo:${{e}}`);
      return `<div class="row"><span class="lbl">${{e}}</span><span class="val">F1 ${{en.f1_baseline.toFixed(2)}}</span></div>`;
    }}).join("");
    detail.innerHTML = `
      <h2>Cluster emergente</h2>
      <div class="big">${{node.label}}</div>
      <div class="meta">
        <span class="badge" style="background:${{node.color}}">${{members.length}} emociones</span>
        <br><br>
        Estos seis grupos surgieron del clustering jerárquico sobre los vectores
        de selectividad neuronal — sin imponer la taxonomía a priori.
      </div>
      <h2>Miembros</h2>
      ${{emoLines}}
      <h2>Neuronas totales</h2>
      <div class="row"><span class="lbl">Significativas (|d|>2)</span><span class="val">${{node.size}}</span></div>
    `;
  }}
}}

// Initialize panel with intro
showDetail(null);
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
    return build_html(out_dir / "09_circuit_network.html")


if __name__ == "__main__":
    main()
