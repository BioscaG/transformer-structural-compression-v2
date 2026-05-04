"""Component sensitivity to SVD compression (§4.3 of the thesis).

Bar chart showing F1 retention by component type at three rank levels.
Highlights the radical asymmetry between Q/K (immune) and FFN
(fragile): Query at rank 128 keeps 99.4% of F1 while FFN Intermediate
collapses to 6.9% — a 14× gap that becomes 72× when normalised by
parameters eliminated.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go

from viz import style as st


CSV_PATH = (pathlib.Path(__file__).resolve().parents[2]
            / "results" / "csvs" / "notebook3" / "component_sensitivity.csv")


# Components in display order, with regime grouping
COMPONENTS = [
    ("Query",        "query",            "Inmune",  "Immune"),
    ("Key",          "key",              "Inmune",  "Immune"),
    ("Value",        "value",            "Acantilado", "Cliff"),
    ("Attn_Output",  "attention_output", "Acantilado", "Cliff"),
    ("FFN_Output",   "ffn_output",       "Frágil",  "Fragile"),
    ("FFN_Inter",    "intermediate",     "Frágil",  "Fragile"),
]
DISPLAY_NAMES = {
    "Query":       "Q",
    "Key":         "K",
    "Value":       "V",
    "Attn_Output": "Attn-O",
    "FFN_Output":  "FFN-out",
    "FFN_Inter":   "FFN-int",
}

REGIME_COLOR = {
    "Inmune":     st.SAGE,
    "Immune":     st.SAGE,
    "Acantilado": st.SAND,
    "Cliff":      st.SAND,
    "Frágil":     st.TERRA,
    "Fragile":    st.TERRA,
}

RANK_OPACITY = {256: 1.0, 128: 0.62, 64: 0.32}


LANG = {
    "es": {
        "title":        "Sensibilidad por componente (rango uniforme)",
        "ylabel":       "Retención de F1 macro (%)",
        "xlabel":       "",
        "rank_fmt":     "rango {r}",
        "regimes":      ["Inmune", "Acantilado", "Frágil"],
        "annotation_q": "Q a r=128 conserva 99.4 % de F1",
        "annotation_f": "FFN-int a r=128 colapsa a 6.9 %",
        "annotation_g": "Asimetría: 14× (absoluta) · 72× (norm. por parámetros)",
    },
    "en": {
        "title":        "Component sensitivity (uniform rank)",
        "ylabel":       "F1 macro retention (%)",
        "xlabel":       "",
        "rank_fmt":     "rank {r}",
        "regimes":      ["Immune", "Cliff", "Fragile"],
        "annotation_q": "Q at r=128 retains 99.4 % F1",
        "annotation_f": "FFN-int at r=128 collapses to 6.9 %",
        "annotation_g": "Asymmetry: 14× (absolute) · 72× (per-parameter)",
    },
}


def build_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    df = pd.read_csv(CSV_PATH)

    fig = go.Figure()

    # One trace per rank, grouped bars per component
    rank_order = [256, 128, 64]
    x_labels = [DISPLAY_NAMES[c] for c, _, _, _ in COMPONENTS]

    for r in rank_order:
        ys = []
        cs = []
        custom = []
        for comp_name, key, regime_es, regime_en in COMPONENTS:
            row = df[(df["component"] == comp_name) & (df["rank"] == r)]
            if row.empty:
                ys.append(0)
                cs.append("#cccccc")
                custom.append([0, 0])
                continue
            ret = float(row["f1_retention_pct"].iloc[0])
            f1  = float(row["f1_macro"].iloc[0])
            regime = regime_es if lang == "es" else regime_en
            ys.append(ret)
            cs.append(REGIME_COLOR[regime])
            custom.append([f1, regime])

        fig.add_trace(go.Bar(
            x=x_labels, y=ys,
            name=L["rank_fmt"].format(r=r),
            marker=dict(
                color=cs,
                opacity=RANK_OPACITY[r],
                line=dict(color=st.INK, width=0.6),
            ),
            customdata=custom,
            hovertemplate=(
                "<b>%{x}</b><br>"
                f"{L['rank_fmt'].format(r=r)}<br>"
                "F1 macro: %{customdata[0]:.3f}<br>"
                "Retención: %{y:.1f} %<br>"
                "Régimen: %{customdata[1]}"
                "<extra></extra>"
            ),
        ))

    # Reference line at 100%
    fig.add_hline(y=100, line=dict(color=st.INK_3, width=0.5, dash="dot"))

    # Annotations: highlight the asymmetry story
    fig.add_annotation(
        x="Q", y=99.4, yshift=18,
        text=L["annotation_q"],
        showarrow=False,
        font=dict(size=10.5, color=st.SAGE),
        align="center",
    )
    fig.add_annotation(
        x="FFN-int", y=6.9, yshift=24,
        text=L["annotation_f"],
        showarrow=False,
        font=dict(size=10.5, color=st.TERRA),
        align="center",
    )
    fig.add_annotation(
        x=0.5, xref="paper", y=1.06, yref="paper",
        text=L["annotation_g"],
        showarrow=False,
        font=dict(size=11, color=st.INK_2),
        align="center",
    )

    layout = st.thesis_layout(height=520)
    fig.update_layout(**layout)
    fig.update_layout(
        barmode="group",
        bargap=0.18,
        bargroupgap=0.05,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.18,
            xanchor="center", x=0.5,
            font=dict(size=11, color=st.INK_2),
        ),
    )
    st.style_axes(fig, xtitle=L["xlabel"], ytitle=L["ylabel"])
    fig.update_yaxes(range=[0, 110], dtick=20)
    fig.update_xaxes(showgrid=False)
    return fig


if __name__ == "__main__":
    fig = build_figure(lang="es")
    out = pathlib.Path(__file__).resolve().parents[2] / "viz" / "output" / "component_sensitivity.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"✓ {out}")
