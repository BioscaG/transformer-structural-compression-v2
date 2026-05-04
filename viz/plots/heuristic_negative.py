"""The informative negative result of §6.2 — heuristics aren't enough.

Pareto frontier scatter showing all 21 evaluated strategies. The story
this figure tells: three heuristic informed strategies built on
qualitative interpretability rules collapse exactly onto blind
baselines:

  - informed_aggressive  ≡ uniform_r256 (same point on the plane).
  - informed_moderate    ≡ uniform_r512 (same point).
  - informed_light       requires ratio 1.285 — actually MORE
    parameters than the original.

None lie on the Pareto frontier. The greedy data-driven family
dominates 8 of 9 optimal points. Negative result that bounds the
applied value of qualitative interpretability: knowing WHAT to measure
isn't enough — you need to measure HOW MUCH.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go

from viz import style as st


CSV_PATH = (pathlib.Path(__file__).resolve().parents[2]
            / "results" / "csvs" / "notebook9"
            / "compression_comparison.csv")


FAMILY_LABEL_ES = {
    "uniform":  "Uniforme (ciega)",
    "adaptive": "Adaptativa (ciega)",
    "informed": "Heurística informada",
    "greedy":   "Greedy (data-driven)",
}
FAMILY_LABEL_EN = {
    "uniform":  "Uniform (blind)",
    "adaptive": "Adaptive (blind)",
    "informed": "Informed heuristic",
    "greedy":   "Greedy (data-driven)",
}


LANG = {
    "es": {
        "title":    "Frontera de Pareto: 21 estrategias evaluadas",
        "ylabel":   "F1 macro retenido",
        "xlabel":   "Ratio de parámetros (1.0 = baseline)",
        "ann_aggr": "informed_aggressive<br>≡ uniform_r256",
        "ann_mod":  "informed_moderate<br>≡ uniform_r512",
        "ann_lite": "informed_light<br>1.28× parámetros",
        "ann_g80":  "greedy_80: 87 % F1<br>con 80 % de los parámetros",
        "ann_pareto": "Greedy domina 8 de 9 puntos Pareto-óptimos",
        "fam_label": FAMILY_LABEL_ES,
    },
    "en": {
        "title":    "Pareto frontier: 21 evaluated strategies",
        "ylabel":   "F1 macro retained",
        "xlabel":   "Parameter ratio (1.0 = baseline)",
        "ann_aggr": "informed_aggressive<br>≡ uniform_r256",
        "ann_mod":  "informed_moderate<br>≡ uniform_r512",
        "ann_lite": "informed_light<br>1.28× parameters",
        "ann_g80":  "greedy_80: 87 % F1<br>with 80 % of parameters",
        "ann_pareto": "Greedy dominates 8 of 9 Pareto-optimal points",
        "fam_label": FAMILY_LABEL_EN,
    },
}


FAMILY_COLOR = {
    "uniform":  st.BLUE,
    "adaptive": st.TERRA,
    "informed": st.SAGE,
    "greedy":   st.SAND,
}
FAMILY_SYMBOL = {
    "uniform":  "square",
    "adaptive": "diamond",
    "informed": "x",
    "greedy":   "circle",
}


def build_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    df = pd.read_csv(CSV_PATH)

    # Add baseline as a reference point
    fig = go.Figure()

    # One trace per family (legend entries)
    for fam in ["uniform", "adaptive", "informed", "greedy"]:
        sub = df[df["type"] == fam]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["compression_ratio"], y=sub["f1_retention"],
            mode="markers",
            name=L["fam_label"][fam],
            marker=dict(
                color=FAMILY_COLOR[fam],
                size=14 if fam in ("informed", "greedy") else 12,
                symbol=FAMILY_SYMBOL[fam],
                line=dict(color=st.INK, width=0.6),
                opacity=0.95,
            ),
            customdata=sub[["strategy", "macro_f1", "params"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "F1 macro: %{customdata[1]:.3f}<br>"
                "Retención: %{y:.1%}<br>"
                "Ratio params: %{x:.3f}<br>"
                "Params: %{customdata[2]:,.0f}"
                "<extra></extra>"
            ),
        ))

    # Reference line at baseline
    fig.add_vline(x=1.0, line=dict(color=st.INK_3, width=0.5, dash="dot"))
    fig.add_hline(y=1.0, line=dict(color=st.INK_3, width=0.5, dash="dot"))

    # Annotations: highlight key collisions
    fig.add_annotation(
        x=0.612, y=0.043, ax=0.45, ay=0.20,
        xref="x", yref="y", axref="x", ayref="y",
        text=L["ann_aggr"],
        showarrow=True, arrowhead=2, arrowwidth=0.6, arrowsize=0.7,
        arrowcolor=st.SAGE,
        font=dict(size=10.5, color=st.SAGE),
        align="left",
    )
    fig.add_annotation(
        x=1.0, y=0.804, ax=0.85, ay=0.65,
        xref="x", yref="y", axref="x", ayref="y",
        text=L["ann_mod"],
        showarrow=True, arrowhead=2, arrowwidth=0.6, arrowsize=0.7,
        arrowcolor=st.SAGE,
        font=dict(size=10.5, color=st.SAGE),
        align="left",
    )
    fig.add_annotation(
        x=1.285, y=0.972, ax=1.20, ay=0.55,
        xref="x", yref="y", axref="x", ayref="y",
        text=L["ann_lite"],
        showarrow=True, arrowhead=2, arrowwidth=0.6, arrowsize=0.7,
        arrowcolor=st.SAGE,
        font=dict(size=10.5, color=st.SAGE),
        align="left",
    )
    fig.add_annotation(
        x=0.799, y=0.869, ax=0.55, ay=0.95,
        xref="x", yref="y", axref="x", ayref="y",
        text=L["ann_g80"],
        showarrow=True, arrowhead=2, arrowwidth=0.6, arrowsize=0.7,
        arrowcolor=st.SAND,
        font=dict(size=10.5, color=st.SAND),
        align="left",
    )

    fig.add_annotation(
        x=0.5, xref="paper", y=1.07, yref="paper",
        text=L["ann_pareto"],
        showarrow=False,
        font=dict(size=11.5, color=st.INK_2),
        align="center",
    )

    layout = st.thesis_layout(height=560)
    fig.update_layout(**layout)
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.20,
            xanchor="center", x=0.5,
            font=dict(size=11, color=st.INK_2),
        ),
    )
    st.style_axes(fig, xtitle=L["xlabel"], ytitle=L["ylabel"])
    fig.update_xaxes(range=[0.25, 1.35])
    fig.update_yaxes(range=[-0.05, 1.10], tickformat=".0%")
    return fig


if __name__ == "__main__":
    fig = build_figure(lang="es")
    out = pathlib.Path(__file__).resolve().parents[2] / "viz" / "output" / "heuristic_negative.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"✓ {out}")
