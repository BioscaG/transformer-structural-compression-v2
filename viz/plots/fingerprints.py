"""Hero plot 5 — The emotional landscape.

Two-panel composite:
  Left: 2D emotion landscape, where each emotion sits at (crystallization_layer,
  selectivity_norm). Bubble size = F1 baseline; color = cluster. Quadrants
  reveal the lexical/contextual × intense/diffuse partition.

  Right: per-emotion 6-axis radar fingerprint. Switch between emotions with the
  dropdown. Each radar shows: cristalización, intensidad neuronal, F1 baseline,
  retención bajo greedy 90%, recuperación tras fine-tuning, impacto cabeza
  crítica.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import thesis_data as td
from viz import style as st
from viz.data.load_results import (load_heads, load_neurons, load_informed,
                                    load_probing,
                                    critical_head_per_emotion,
                                    neuron_count_per_emotion,
                                    crystallization_dict,
                                    f1_baseline_per_emotion,
                                    finetune_recovery,
                                    EMOTIONS_23)


# Lazy-loaded module-level cache
_REAL = {}


def _real_data():
    if "loaded" in _REAL:
        return _REAL
    heads = load_heads()
    neurons = load_neurons()
    informed = load_informed()
    probing = load_probing()
    _REAL["critical"] = critical_head_per_emotion(heads) if heads["top_heads"] is not None else td.CRITICAL_HEAD_PER_EMOTION
    _REAL["counts"] = neuron_count_per_emotion(neurons) if neurons["significant_counts"] is not None else td.NEURON_COUNT_PER_EMOTION
    _REAL["baseline"] = f1_baseline_per_emotion(informed) if informed["finetuning"] is not None else td.F1_BASELINE
    _REAL["finetune"] = finetune_recovery(informed) if informed["finetuning"] is not None else None
    _REAL["crystal"] = crystallization_dict(probing) if probing["crystallization"] is not None else None
    # Selectivity norm: aggregate from neuron_catalog if available
    cat = neurons.get("catalog")
    if cat is not None:
        _REAL["selnorm"] = cat.groupby("emotion")["abs_selectivity"].sum().to_dict()
    else:
        _REAL["selnorm"] = td.NEURON_SELECTIVITY_NORM
    _REAL["loaded"] = True
    return _REAL


RADAR_AXES = [
    "Cristalización<br>temprana",
    "Intensidad<br>neuronal",
    "F1 baseline",
    "Retención<br>greedy 90%",
    "Recuperación<br>fine-tuning",
    "Robustez<br>cabeza crítica",
]


def _normalize_emotion_metrics(emotion: str) -> list[float]:
    """6 metrics in [0, 1], the radar axes — REAL values when available."""
    R = _real_data()

    if R["crystal"] and emotion in R["crystal"]:
        crystal = R["crystal"][emotion]["crystal_layer"]
    else:
        crystal = td.CRYSTALLIZATION_LAYER[emotion]
    crystal_norm = 1 - crystal / 12  # 1 = early, 0 = late

    sel_vals = list(R["selnorm"].values())
    sel_min, sel_max = min(sel_vals), max(sel_vals)
    intensity = (R["selnorm"].get(emotion, sel_min) - sel_min) / max(sel_max - sel_min, 1e-6)

    f1_b = R["baseline"].get(emotion, td.F1_BASELINE[emotion])

    if R["finetune"] and emotion in R["finetune"]:
        base = R["finetune"][emotion]["baseline"]
        comp = R["finetune"][emotion]["compressed"]
        ft = R["finetune"][emotion]["finetuned"]
    else:
        base, comp, ft = td.FINETUNE_RECOVERY[emotion]
    retention = comp / base if base > 0 else 0
    recovery = (ft - base) / base if base > 0 else 0
    recovery_norm = max(0, min(1, 0.5 + recovery * 1.2))

    head_impact = abs(R["critical"][emotion][2]) if emotion in R["critical"] else \
                  abs(td.CRITICAL_HEAD_PER_EMOTION[emotion][2])
    robustness = 1 - min(1, head_impact / 0.32)

    return [crystal_norm, intensity, f1_b, retention, recovery_norm, robustness]


def build_landscape_figure() -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.62, 0.38],
        specs=[[{"type": "xy"}, {"type": "polar"}]],
        subplot_titles=(
            "Paisaje emocional — geografía de las 23 emociones",
            "Huella de la emoción seleccionada",
        ),
        horizontal_spacing=0.10,
    )

    # ───── 2D landscape ─────
    R = _real_data()
    for cluster, members in td.CLUSTER_DEFS.items():
        x = [R["crystal"][e]["crystal_layer"] if R["crystal"] and e in R["crystal"]
             else td.CRYSTALLIZATION_LAYER[e] for e in members]
        y = [R["selnorm"].get(e, 50) for e in members]
        sizes = [R["baseline"].get(e, td.F1_BASELINE[e]) * 60 + 12 for e in members]
        names = members

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers+text",
            text=names, textposition="top center",
            textfont=dict(size=9.8, color=st.INK_2, family="serif"),
            marker=dict(
                size=sizes, color=td.CLUSTER_COLORS[cluster],
                line=dict(color="white", width=1.5),
                opacity=0.85,
            ),
            customdata=[[
                R["baseline"].get(e, td.F1_BASELINE[e]),
                R["counts"].get(e, 0),
                R["critical"].get(e, td.CRITICAL_HEAD_PER_EMOTION[e])[0],
                R["critical"].get(e, td.CRITICAL_HEAD_PER_EMOTION[e])[1],
                R["critical"].get(e, td.CRITICAL_HEAD_PER_EMOTION[e])[2],
                cluster,
            ] for e in members],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Cluster: %{customdata[5]}<br>"
                "Cristaliza en: L%{x}<br>"
                "Norma selectividad: %{y}<br>"
                "F1 baseline: %{customdata[0]:.3f}<br>"
                "Neuronas significativas: %{customdata[1]}<br>"
                "Cabeza crítica: L%{customdata[2]}-H%{customdata[3]} (ΔF1 %{customdata[4]:.3f})"
                "<extra></extra>"
            ),
            name=cluster, showlegend=True,
        ), row=1, col=1)

    # Quadrant annotations — adapt y-thresholds to the real selnorm scale
    sel_max = max(R["selnorm"].values()) if R["selnorm"] else 160
    sel_y_mid = sel_max * 0.5
    fig.add_shape(type="line", x0=5.5, x1=5.5, y0=0, y1=sel_max * 1.05,
                  line=dict(color=st.SPINE, width=0.6, dash="dot"), row=1, col=1)
    fig.add_shape(type="line", x0=-0.5, x1=12.5, y0=sel_y_mid, y1=sel_y_mid,
                  line=dict(color=st.SPINE, width=0.6, dash="dot"), row=1, col=1)

    sel_top = sel_max * 0.92
    sel_bot = sel_max * 0.18
    quad_annotations = [
        (1.0,  sel_top, "Cristalización temprana<br>+ codificación intensa", "right"),
        (11.5, sel_top, "Cristalización tardía<br>+ codificación intensa", "left"),
        (1.0,  sel_bot, "Cristalización temprana<br>+ codificación difusa", "right"),
        (11.5, sel_bot, "Cristalización tardía<br>+ codificación difusa", "left"),
    ]
    for x, y, txt, anchor in quad_annotations:
        fig.add_annotation(x=x, y=y, xref="x1", yref="y1", text=txt,
                           showarrow=False, font=dict(size=9.5, color=st.INK_3),
                           xanchor=anchor)

    # Highlight any pair of emotions sharing a critical head (REAL detection)
    head_to_emotions: dict[tuple, list] = {}
    for e in td.EMOTIONS:
        lh = (R["critical"].get(e, td.CRITICAL_HEAD_PER_EMOTION[e])[0],
              R["critical"].get(e, td.CRITICAL_HEAD_PER_EMOTION[e])[1])
        head_to_emotions.setdefault(lh, []).append(e)
    shared = [(lh, emos) for lh, emos in head_to_emotions.items() if len(emos) > 1]
    for lh, emos in shared[:3]:  # at most 3 connectors to avoid clutter
        e1, e2 = emos[0], emos[1]
        x1 = R["crystal"][e1]["crystal_layer"] if R["crystal"] and e1 in R["crystal"] else td.CRYSTALLIZATION_LAYER[e1]
        x2 = R["crystal"][e2]["crystal_layer"] if R["crystal"] and e2 in R["crystal"] else td.CRYSTALLIZATION_LAYER[e2]
        y1 = R["selnorm"].get(e1, 50)
        y2 = R["selnorm"].get(e2, 50)
        fig.add_shape(type="line", x0=x1, x1=x2, y0=y1, y1=y2,
                      line=dict(color=st.PLUM, width=1.0, dash="dashdot"),
                      row=1, col=1)
        fig.add_annotation(
            x=(x1 + x2) / 2, y=(y1 + y2) / 2 - 2, xref="x1", yref="y1",
            text=f"comparten<br>L{lh[0]}-H{lh[1]}", showarrow=False,
            font=dict(size=8.5, color=st.PLUM),
        )

    # ───── Radar (single, populated via dropdown) ─────
    default_emotion = "embarrassment"  # the dramatic one
    radar_values = _normalize_emotion_metrics(default_emotion) + [_normalize_emotion_metrics(default_emotion)[0]]

    fig.add_trace(go.Scatterpolar(
        r=radar_values,
        theta=RADAR_AXES + [RADAR_AXES[0]],
        fill="toself", fillcolor="rgba(193, 85, 58, 0.25)",
        line=dict(color=st.TERRA, width=2.5),
        name=default_emotion,
        marker=dict(size=8, color=st.TERRA, line=dict(color="white", width=1.5)),
        showlegend=False,
    ), row=1, col=2)

    # Build the dropdown menu — one entry per emotion
    buttons = []
    for emo in td.EMOTIONS:
        v = _normalize_emotion_metrics(emo) + [_normalize_emotion_metrics(emo)[0]]
        cluster = td.EMOTION_TO_CLUSTER[emo]
        ccolor = td.CLUSTER_COLORS[cluster]
        # rgba fill from hex (alpha 0.25)
        r, g, b = int(ccolor[1:3], 16), int(ccolor[3:5], 16), int(ccolor[5:7], 16)
        fillrgba = f"rgba({r},{g},{b},0.30)"
        buttons.append(dict(
            label=f"{emo} — {cluster}",
            method="restyle",
            args=[
                {"r": [v], "fillcolor": [fillrgba],
                 "line.color": [ccolor], "marker.color": [ccolor], "name": [emo]},
                [len(td.CLUSTER_DEFS)],  # the radar trace index (after the 6 cluster scatters)
            ],
        ))

    fig.update_layout(
        **st.thesis_layout(
            title="Cap. 5 — El paisaje emocional de BERT-base",
            height=680, width=1320,
        ),
        legend=dict(
            x=0.02, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)", bordercolor=st.SPINE, borderwidth=0.5,
            font=dict(size=10.5),
        ),
        polar=dict(
            bgcolor=st.BG,
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=True,
                            tickfont=dict(size=8.5, color=st.INK_3),
                            gridcolor=st.GRID),
            angularaxis=dict(tickfont=dict(size=10, color=st.INK_2, family="serif"),
                             gridcolor=st.GRID),
        ),
        updatemenus=[dict(
            buttons=buttons, direction="down",
            x=0.85, y=1.07, xanchor="left", yanchor="top",
            bgcolor="white", bordercolor=st.SPINE, borderwidth=0.5,
            font=dict(size=10.5, family="serif"),
            showactive=True,
        )],
        annotations=list(fig["layout"]["annotations"]) + [
            dict(x=0.85, y=1.10, xref="paper", yref="paper",
                 text="<b>Selecciona emoción ▶</b>", showarrow=False,
                 font=dict(size=10.5, color=st.INK_2), xanchor="right"),
        ],
    )
    fig.update_xaxes(
        title=dict(text="Capa de cristalización (probing)",
                   font=dict(size=12, color=st.INK_2)),
        tickmode="linear", dtick=1, range=[-0.5, 11.5],
        gridcolor=st.GRID, zeroline=False,
        showline=True, linecolor=st.SPINE, ticks="outside", tickcolor=st.INK_3,
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text="Selectividad neuronal agregada (sum |Cohen's d|)",
                   font=dict(size=12, color=st.INK_2)),
        range=[0, sel_max * 1.10],
        gridcolor=st.GRID, zeroline=False,
        showline=True, linecolor=st.SPINE, ticks="outside", tickcolor=st.INK_3,
        row=1, col=1,
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_landscape_figure()
    out = out_dir / "06_emotional_landscape.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
