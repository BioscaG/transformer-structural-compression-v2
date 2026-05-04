"""Activation patching by component (§5.3 of the thesis).

Heatmap of restoration score by layer (rows) × component
(attention vs FFN, columns). Story: only L11 lights up — and
within L11, the FFN is what does the work. Restoring ONLY the
FFN of layer 11 (≈1/3 of that layer's parameters) recovers 100%
of F1 from a totally collapsed model. Restoring only the
attention of L11 gets 63.3%. Everything earlier is essentially
zero. This challenges the attention-centric narrative of "Attention
Is All You Need" — for emotion classification on this fine-tune,
the bottleneck is the FFN of the last layer.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viz import style as st


CSV_PATH = (pathlib.Path(__file__).resolve().parents[2]
            / "results" / "csvs" / "notebook5"
            / "activation_patching_per_component.csv")


LANG = {
    "es": {
        "title":      "Patching por componente: dónde reside la suficiencia",
        "ylabel":     "Capa restaurada",
        "xlabel":     "Componente restaurado",
        "x_attn":     "Atención",
        "x_ffn":      "FFN",
        "ann_l11_ffn": "<b>FFN de L11 sola → 100 %</b><br>Un sub-componente, no toda la capa.",
        "ann_l11_at": "Atn. de L11 sola → 63 %",
        "ann_below": "Capas 0–7 omitidas: restauran 0 % al ser restauradas individualmente",
        "cbar":       "Restauración (%)",
    },
    "en": {
        "title":      "Patching by component: where sufficiency lives",
        "ylabel":     "Restored layer",
        "xlabel":     "Restored component",
        "x_attn":     "Attention",
        "x_ffn":      "FFN",
        "ann_l11_ffn": "<b>L11 FFN alone → 100 %</b><br>A sub-component, not the whole layer.",
        "ann_l11_at": "L11 Attn alone → 63 %",
        "ann_below": "Layers 0–7 omitted: each restores 0 % when patched individually",
        "cbar":       "Restoration (%)",
    },
}


def build_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    df = pd.read_csv(CSV_PATH)

    # Aggregate per (layer, component) — mean restoration over 23 emotions
    summary = df.groupby(["layer", "component"])["restoration_score"].mean().reset_index()

    layers_to_show = [8, 9, 10, 11]
    components = ["attention", "ffn"]
    z = np.zeros((len(layers_to_show), len(components)))
    for i, layer in enumerate(layers_to_show):
        for j, comp in enumerate(components):
            row = summary[(summary["layer"] == layer) &
                          (summary["component"] == comp)]
            if not row.empty:
                z[i, j] = row["restoration_score"].iloc[0] * 100

    # Display top-down (L11 on top)
    z_disp = z[::-1]
    y_labels = [f"L{l}" for l in layers_to_show[::-1]]
    x_labels = [L["x_attn"], L["x_ffn"]]

    # Custom colorscale: light to terra
    colorscale = [
        [0.0,  st.BG],
        [0.10, "#F2EBE5"],
        [0.30, "#E8C4B0"],
        [0.60, "#D98A76"],
        [1.0,  st.TERRA],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z_disp,
        x=x_labels, y=y_labels,
        colorscale=colorscale,
        zmin=0, zmax=100,
        text=[[f"{v:.1f} %" for v in row] for row in z_disp],
        texttemplate="%{text}",
        textfont=dict(size=14, color=st.INK,
                      family='"Geist Mono", monospace'),
        hovertemplate=(
            "<b>%{y} · %{x}</b><br>"
            "Restauración: %{z:.1f} %"
            "<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text=L["cbar"],
                       font=dict(size=11, color=st.INK_2)),
            tickfont=dict(size=10, color=st.INK_3),
            thickness=14, len=0.7, x=1.02,
            ticksuffix=" %",
        ),
        xgap=2, ygap=2,
    ))

    # Annotation on the L11 FFN cell — paper-coords arrow to data-coords cell
    fig.add_annotation(
        x=L["x_ffn"], y="L11",
        text=L["ann_l11_ffn"],
        showarrow=False,
        xshift=140, yshift=0,
        font=dict(size=11, color=st.TERRA),
        align="left",
    )

    fig.add_annotation(
        x=0.5, xref="paper", y=-0.18, yref="paper",
        text=f"<i>{L['ann_below']}</i>",
        showarrow=False,
        font=dict(size=10.5, color=st.INK_3),
        align="center",
    )

    layout = st.thesis_layout(height=520)
    fig.update_layout(**layout)
    fig.update_layout(
        margin=dict(l=80, r=200, t=70, b=80),
    )
    fig.update_xaxes(
        title=dict(text=L["xlabel"], font=dict(size=13, color=st.INK_2)),
        showgrid=False, showline=False, zeroline=False,
        ticks="", tickfont=dict(size=12, color=st.INK_2),
    )
    fig.update_yaxes(
        title=dict(text=L["ylabel"], font=dict(size=13, color=st.INK_2)),
        showgrid=False, showline=False, zeroline=False,
        ticks="", tickfont=dict(size=12, color=st.INK_2),
        autorange="reversed" if False else None,
    )
    return fig


if __name__ == "__main__":
    fig = build_figure(lang="es")
    out = pathlib.Path(__file__).resolve().parents[2] / "viz" / "output" / "patching_components.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"✓ {out}")
