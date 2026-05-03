"""Greedy algorithm replay — watch the compression strategy build up.

Shows the user's greedy algorithm making compression decisions one level at
a time, from baseline → greedy_95 → greedy_90 → … → greedy_50. At each
step a 12×6 grid (layer × component) lights up the cells that just got
compressed, with their assigned ranks. Two side panels track F1 retention
and parameter ratio across the path.

Reveals the §6.3 narrative empirically: the algorithm starts with "free"
moves (Q/K and FFN-output in early layers) and only later begins paying
F1 cost as it reaches the late-layer or FFN-intermediate components.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import style as st


CSVS = pathlib.Path(__file__).resolve().parents[2] / "results" / "csvs" / "notebook9"


COMPONENTS = ["query", "key", "value", "attn_output", "ffn_intermediate", "ffn_output"]
COMPONENT_LABEL = {
    "query": "Q", "key": "K", "value": "V",
    "attn_output": "Attn-O", "ffn_intermediate": "FFN-i", "ffn_output": "FFN-o",
}


def parse_layer_name(name: str) -> tuple[int, str]:
    """bert.encoder.layer.7.attention.self.query → (7, 'query')."""
    parts = name.split(".")
    layer = int(parts[3])
    if "self.query" in name: comp = "query"
    elif "self.key" in name: comp = "key"
    elif "self.value" in name: comp = "value"
    elif "attention.output.dense" in name: comp = "attn_output"
    elif "intermediate.dense" in name: comp = "ffn_intermediate"
    elif name.endswith("output.dense"): comp = "ffn_output"
    else: comp = "unknown"
    return layer, comp


LANG = {
    "es": {
        "cbar":      "rango log<sub>2</sub><br>(rojo=más compresión)",
        "heatmap_h": "<b>L%{y}-%{x}</b><br>rank: %{text}<extra></extra>",
        "ratio":     "Ratio params",
        "ratio_h":   "%{x}: %{y:.3f}× params<extra></extra>",
        "f1":        "F1 retention",
        "f1_h":      "%{x}: %{y:.1%} F1<extra></extra>",
        "panel_mat": "Matriz de compresión · 12 capas × 6 componentes",
        "panel_traj": "Trayectoria F1 vs ratio",
        "stage_pref": "Etapa: ",
        "play":      "▶ Play",
        "pause":     "⏸ Pause",
        "reset":     "↺ Reset",
        "axis_y":    "ratio / retention",
        "step_pref": "Step ",
        "captions": {
            "baseline":  "Modelo base, ningún componente comprimido. F1 = 0.469, ratio = 1.00× (todo a rango 768).",
            "greedy_95": "Greedy 95% — 24 movimientos baratos. Solo Q y K en algunas capas. Casi sin coste de F1: el algoritmo está aprovechando que Q y K son inmunes a compresión agresiva (§4.3).",
            "greedy_90": "Greedy 90% — 28 movimientos. El algoritmo añade FFN-output en capas tempranas. Sigue siendo prácticamente gratis. La banda right side del grid empieza a iluminarse.",
            "greedy_85": "Greedy 85% — 32 movimientos. Más V y FFN-output. F1 sigue alto, ratio bajando. Aún sin tocar capas tardías intactas.",
            "greedy_80": "Greedy 80% — 36 movimientos. Llega al primer FFN-intermediate. Empieza a notarse la pérdida de F1, pero todavía controlada.",
            "greedy_75": "Greedy 75% — 40 movimientos. La parte FFN del grid se llena. El algoritmo evita SISTEMÁTICAMENTE las capas 8-11.",
            "greedy_70": "Greedy 70% — 52 movimientos. El algoritmo entra en zona agresiva. Las capas tardías (8-11) siguen relativamente intactas — el cuello de botella se respeta.",
            "greedy_60": "Greedy 60% — 64 movimientos. Casi todo comprimido excepto las capas tardías clave. F1 cae notablemente.",
            "greedy_50": "Greedy 50% — 64 movimientos. Compresión máxima del greedy. Las capas tardías de FFN-int sobreviven hasta el final. F1 cerca del colapso pero ratio drástico.",
        },
    },
    "en": {
        "cbar":      "rank log<sub>2</sub><br>(red = more compression)",
        "heatmap_h": "<b>L%{y}-%{x}</b><br>rank: %{text}<extra></extra>",
        "ratio":     "Param ratio",
        "ratio_h":   "%{x}: %{y:.3f}× params<extra></extra>",
        "f1":        "F1 retention",
        "f1_h":      "%{x}: %{y:.1%} F1<extra></extra>",
        "panel_mat": "Compression matrix · 12 layers × 6 components",
        "panel_traj": "F1 vs ratio trajectory",
        "stage_pref": "Stage: ",
        "play":      "▶ Play",
        "pause":     "⏸ Pause",
        "reset":     "↺ Reset",
        "axis_y":    "ratio / retention",
        "step_pref": "Step ",
        "captions": {
            "baseline":  "Baseline model, no component compressed. F1 = 0.469, ratio = 1.00× (everything at rank 768).",
            "greedy_95": "Greedy 95% — 24 cheap moves. Just Q and K on some layers. Almost no F1 cost: the algorithm exploits Q/K's immunity to aggressive compression (§4.3).",
            "greedy_90": "Greedy 90% — 28 moves. The algorithm adds FFN-output on early layers. Still almost free. The right-side band of the grid starts lighting up.",
            "greedy_85": "Greedy 85% — 32 moves. More V and FFN-output. F1 still high, ratio dropping. Late layers still untouched.",
            "greedy_80": "Greedy 80% — 36 moves. First FFN-intermediate hit. F1 loss starts to show but stays controlled.",
            "greedy_75": "Greedy 75% — 40 moves. The FFN part of the grid fills up. The algorithm SYSTEMATICALLY avoids layers 8–11.",
            "greedy_70": "Greedy 70% — 52 moves. The algorithm enters aggressive territory. Late layers (8–11) stay relatively intact — the bottleneck is respected.",
            "greedy_60": "Greedy 60% — 64 moves. Almost everything compressed except the key late layers. F1 drops noticeably.",
            "greedy_50": "Greedy 50% — 64 moves. Greedy's max compression. The late FFN-int layers survive to the end. F1 near collapse, ratio drastic.",
        },
    },
}


def build_replay_figure(lang: str = "es") -> go.Figure:
    _L = LANG[lang]
    levels = [50, 60, 70, 75, 80, 85, 90, 95]
    levels_path = list(reversed(levels))   # build from least to most compressed: 95 → 50

    # Load each greedy level's rank assignments and convert to (layer, component) → rank
    rank_grids = {}
    for lvl in levels_path:
        df = pd.read_csv(CSVS / f"greedy_{lvl}_ranks.csv")
        grid = {}
        for _, row in df.iterrows():
            layer, comp = parse_layer_name(row["layer_name"])
            grid[(layer, comp)] = int(row["rank"])
        rank_grids[lvl] = grid

    # Compression comparison for F1 + ratio
    cmp = pd.read_csv(CSVS / "compression_comparison.csv")
    greedy_metrics = {}
    for lvl in levels_path:
        row = cmp[cmp["strategy"] == f"greedy_{lvl}"]
        if not row.empty:
            r = row.iloc[0]
            greedy_metrics[lvl] = {
                "f1": float(r["macro_f1"]),
                "ratio": float(r["compression_ratio"]),
                "retention": float(r["f1_retention"]),
            }
    baseline_row = cmp[cmp["strategy"].str.contains("baseline", case=False, na=False)]
    if baseline_row.empty:
        # Use the highest f1 entry as baseline
        baseline_f1 = cmp["macro_f1"].max()
    else:
        baseline_f1 = float(baseline_row.iloc[0]["macro_f1"])

    # Build 12×6 rank matrix per level. Cells without an entry mean "full rank
    # (not compressed)" — represented as 768.
    n_layers = 12
    n_comp = len(COMPONENTS)

    def rank_matrix(grid: dict) -> np.ndarray:
        M = np.full((n_layers, n_comp), 768, dtype=np.int32)
        for (L, c), r in grid.items():
            ci = COMPONENTS.index(c)
            M[L, ci] = r
        return M

    matrices = {lvl: rank_matrix(rank_grids[lvl]) for lvl in levels_path}
    # Add a baseline (no compression)
    matrices["baseline"] = np.full((n_layers, n_comp), 768, dtype=np.int32)
    path_keys = ["baseline"] + levels_path

    # Score each cell: 1 = full rank, 0 = heavily compressed (rank 32)
    def matrix_to_score(M: np.ndarray) -> np.ndarray:
        # rank 768 → 1, rank 32 → 0; logarithmic so visual progression is smooth
        return np.clip(np.log2(M.astype(np.float32)) / np.log2(768), 0, 1)

    # ─── Build figure ───
    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        specs=[[{"type": "heatmap"}, {"type": "xy"}]],
        horizontal_spacing=0.10,
    )

    # Path summary: for each step, the F1 retention and parameter ratio.
    path_labels = ["baseline"] + [f"greedy_{lvl}" for lvl in levels_path]
    path_ratios = [1.0] + [greedy_metrics.get(lvl, {}).get("ratio", 1.0) for lvl in levels_path]
    path_retentions = [1.0] + [greedy_metrics.get(lvl, {}).get("retention", 0)
                                for lvl in levels_path]

    # Initial heatmap — baseline (all full rank)
    init_score = matrix_to_score(matrices["baseline"])
    init_text = [[("768" if matrices["baseline"][L, ci] == 768
                   else str(matrices["baseline"][L, ci]))
                  for ci in range(n_comp)] for L in range(n_layers)]

    fig.add_trace(go.Heatmap(
        z=init_score, x=[COMPONENT_LABEL[c] for c in COMPONENTS], y=[f"L{L}" for L in range(n_layers)],
        colorscale=[[0, st.TERRA], [0.4, st.SAND], [0.7, st.SAGE_L], [1, "#F5F1E5"]],
        zmin=0, zmax=1, showscale=True,
        colorbar=dict(thickness=14, len=0.8, x=0.55,
                      title=dict(text=_L["cbar"],
                                 font=dict(size=11)),
                      tickmode="array",
                      tickvals=[0, np.log2(64)/np.log2(768), np.log2(256)/np.log2(768), 1.0],
                      ticktext=["32", "64", "256", "768"]),
        text=init_text, texttemplate="%{text}",
        textfont=dict(size=11, color=st.INK, family="serif"),
        xgap=2, ygap=2,
        hovertemplate=_L["heatmap_h"],
    ), row=1, col=1)

    # Path trajectories: ratio + retention vs step
    fig.add_trace(go.Scatter(
        x=path_labels, y=path_ratios, mode="lines+markers", name=_L["ratio"],
        line=dict(color=st.BLUE, width=2.5, shape="hv"),
        marker=dict(size=8, color=st.BLUE, line=dict(color="white", width=1.5)),
        hovertemplate=_L["ratio_h"],
    ), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=path_labels, y=path_retentions, mode="lines+markers", name=_L["f1"],
        line=dict(color=st.TERRA, width=2.5, shape="hv"),
        marker=dict(size=8, color=st.TERRA, line=dict(color="white", width=1.5)),
        hovertemplate=_L["f1_h"],
    ), row=1, col=2)
    # Vertical marker for current step
    fig.add_shape(type="line", x0=0, x1=0, y0=0, y1=1.05,
                  line=dict(color=st.INK_3, width=1.5, dash="dot"),
                  xref="x2", yref="y2")

    # Frames per step
    frames = []
    captions = _L["captions"]

    for step_i, key in enumerate(path_keys):
        M = matrices[key]
        score = matrix_to_score(M)
        text = [[str(M[L, ci]) for ci in range(n_comp)] for L in range(n_layers)]
        cap = captions.get(key if isinstance(key, str) else f"greedy_{key}",
                           f"{_L['step_pref']}{key}")

        # Update vertical marker on path chart
        marker_x = step_i

        frames.append(go.Frame(
            data=[
                go.Heatmap(
                    z=score, x=[COMPONENT_LABEL[c] for c in COMPONENTS],
                    y=[f"L{L}" for L in range(n_layers)],
                    colorscale=[[0, st.TERRA], [0.4, st.SAND], [0.7, st.SAGE_L], [1, "#F5F1E5"]],
                    zmin=0, zmax=1, text=text, texttemplate="%{text}",
                    textfont=dict(size=11, color=st.INK, family="serif"),
                    xgap=2, ygap=2,
                    hovertemplate=_L["heatmap_h"],
                    showscale=True,
                    colorbar=dict(thickness=14, len=0.8, x=0.55,
                                  title=dict(text=_L["cbar"],
                                             font=dict(size=11)),
                                  tickmode="array",
                                  tickvals=[0, np.log2(64)/np.log2(768), np.log2(256)/np.log2(768), 1.0],
                                  ticktext=["32", "64", "256", "768"]),
                ),
                go.Scatter(x=path_labels, y=path_ratios, mode="lines+markers",
                           line=dict(color=st.BLUE, width=2.5, shape="hv"),
                           marker=dict(size=8, color=st.BLUE, line=dict(color="white", width=1.5))),
                go.Scatter(x=path_labels, y=path_retentions, mode="lines+markers",
                           line=dict(color=st.TERRA, width=2.5, shape="hv"),
                           marker=dict(size=8, color=st.TERRA, line=dict(color="white", width=1.5))),
            ],
            traces=[0, 1, 2],
            name=key if isinstance(key, str) else f"greedy_{key}",
            layout=dict(
                shapes=[dict(type="line",
                             x0=path_labels[step_i], x1=path_labels[step_i],
                             y0=0, y1=1.05,
                             line=dict(color=st.INK_3, width=1.5, dash="dot"),
                             xref="x2", yref="y2")],
                annotations=[
                    dict(text=f"<b>{path_labels[step_i]}</b> · {cap}",
                         x=0.5, y=-0.16, xref="paper", yref="paper", showarrow=False,
                         align="center", xanchor="center",
                         font=dict(size=11.5, color=st.INK_2, family="serif"),
                         bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                         borderwidth=0.5, borderpad=10),
                ],
            ),
        ))

    fig.frames = frames

    steps = [dict(
        method="animate",
        args=[[k if isinstance(k, str) else f"greedy_{k}"],
              dict(mode="immediate", frame=dict(duration=0, redraw=True),
                   transition=dict(duration=400, easing="cubic-in-out"))],
        label=path_labels[i],
    ) for i, k in enumerate(path_keys)]

    fig.update_layout(
        **st.thesis_layout(
            title=("Greedy algorithm replay · cómo construye la compresión informada"
                   "<br><sub>El algoritmo añade decisiones por orden de eficiencia "
                   "(parámetros ahorrados / coste F1). Las celdas más rojas son las "
                   "más comprimidas. Datos reales de notebook 9.</sub>"),
            height=720, width=1380,
        ),
        annotations=[
            dict(text=_L["panel_mat"],
                 x=0.31, y=1.11, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12.5, color=st.INK_2, family="serif")),
            dict(text=_L["panel_traj"],
                 x=0.85, y=1.11, xref="paper", yref="paper", showarrow=False,
                 font=dict(size=12.5, color=st.INK_2, family="serif")),
            dict(text=f"<b>{path_labels[0]}</b> · {captions['baseline']}",
                 x=0.5, y=-0.16, xref="paper", yref="paper", showarrow=False,
                 align="center", xanchor="center",
                 font=dict(size=11.5, color=st.INK_2, family="serif"),
                 bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                 borderwidth=0.5, borderpad=10),
        ],
        sliders=[dict(
            active=0, x=0.05, y=-0.04, len=0.85,
            currentvalue=dict(prefix=_L["stage_pref"],
                              font=dict(size=14, color=st.INK, family="serif"),
                              xanchor="left"),
            steps=steps,
            pad=dict(t=20, b=10),
            tickcolor=st.INK_3,
            font=dict(size=10, color=st.INK_3),
        )],
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.05, y=-0.18, xanchor="left", yanchor="top",
            buttons=[
                dict(label=_L["play"], method="animate",
                     args=[None, dict(frame=dict(duration=1100, redraw=True),
                                      transition=dict(duration=500, easing="cubic-in-out"),
                                      fromcurrent=True, mode="immediate")]),
                dict(label=_L["pause"], method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
                dict(label=_L["reset"], method="animate",
                     args=[[path_labels[0]], dict(mode="immediate",
                                                   frame=dict(duration=0, redraw=True))]),
            ],
            font=dict(size=11, color=st.INK_2),
            bgcolor="white", bordercolor=st.SPINE,
        )],
        legend=dict(x=0.97, y=0.02, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=st.SPINE,
                    borderwidth=0.5, font=dict(size=10)),
    )

    fig.update_xaxes(autorange="reversed", side="top",
                     tickfont=dict(size=11, color=st.INK_2, family="serif"),
                     showgrid=False, linecolor=st.SPINE, row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=10, color=st.INK_2, family="serif"),
                     showgrid=False, linecolor=st.SPINE,
                     autorange="reversed", row=1, col=1)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10, color=st.INK_3),
                     showgrid=False, linecolor=st.SPINE, row=1, col=2)
    fig.update_yaxes(title=dict(text=_L["axis_y"], font=dict(size=11, color=st.INK_2)),
                     range=[0, 1.1], gridcolor=st.GRID, linecolor=st.SPINE,
                     tickfont=dict(size=10, color=st.INK_3),
                     tickformat=".0%", row=1, col=2)

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_replay_figure()
    out = out_dir / "18_greedy_replay.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
