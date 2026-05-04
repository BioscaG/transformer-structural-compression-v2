"""Per-emotion fine-tuning recovery (§6.4 of the thesis).

Three F1 columns per emotion: baseline (uncompressed), compressed
(greedy_90, 86.4 % params, F1 drop), and fine-tuned (3 epochs after
compression). The compressed model after 3 epochs of FT reaches F1
0.591 — slightly above the uncompressed baseline (0.577) — with 13.6 %
fewer parameters. The biggest relative gain is on the underrepresented
emotion *embarrassment*: 0.267 → 0.509 (+90 % relative). Pattern
suggests SVD compression acts as implicit regularisation that helps
minority classes.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go

from viz import style as st


CSV_PATH = (pathlib.Path(__file__).resolve().parents[2]
            / "results" / "csvs" / "notebook9" / "finetuning_recovery.csv")


LANG = {
    "es": {
        "title":         "Recuperación por fine-tuning tras compresión greedy_90",
        "ylabel":        "F1 por emoción",
        "xlabel":        "",
        "trace_base":    "Baseline (sin comprimir)",
        "trace_comp":    "Comprimido (greedy_90, 86.4 % params)",
        "trace_ft":      "+3 épocas de fine-tuning",
        "ann_emb":       "<b>embarrassment</b><br>+90 % relativo<br>(0.267 → 0.509)",
        "summary":       "F1 macro: 0.577 → 0.539 → 0.591 · params 100 % → 86.4 %",
    },
    "en": {
        "title":         "Fine-tuning recovery after greedy_90 compression",
        "ylabel":        "F1 per emotion",
        "xlabel":        "",
        "trace_base":    "Baseline (uncompressed)",
        "trace_comp":    "Compressed (greedy_90, 86.4 % params)",
        "trace_ft":      "+3 epochs of fine-tuning",
        "ann_emb":       "<b>embarrassment</b><br>+90 % relative<br>(0.267 → 0.509)",
        "summary":       "F1 macro: 0.577 → 0.539 → 0.591 · params 100 % → 86.4 %",
    },
}


def build_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    df = pd.read_csv(CSV_PATH)

    # Order: by absolute FT recovery descending → makes the embarrassment
    # spike pop. Tie-break by baseline F1 descending.
    df = df.sort_values(by=["ft_recovery", "baseline_f1"],
                        ascending=[False, False]).reset_index(drop=True)
    emotions = df["emotion"].tolist()

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=emotions, y=df["baseline_f1"],
        name=L["trace_base"],
        marker=dict(color=st.INK_3, line=dict(color=st.INK, width=0.4),
                    opacity=0.55),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{L['trace_base']}: %{{y:.3f}}"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=emotions, y=df["compressed_f1"],
        name=L["trace_comp"],
        marker=dict(color=st.TERRA, line=dict(color=st.INK, width=0.4),
                    opacity=0.85),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{L['trace_comp']}: %{{y:.3f}}"
            "<extra></extra>"
        ),
    ))
    fig.add_trace(go.Bar(
        x=emotions, y=df["finetuned_f1"],
        name=L["trace_ft"],
        marker=dict(color=st.SAGE, line=dict(color=st.INK, width=0.4)),
        hovertemplate=(
            "<b>%{x}</b><br>"
            f"{L['trace_ft']}: %{{y:.3f}}"
            "<extra></extra>"
        ),
    ))

    # Highlight embarrassment story
    if "embarrassment" in emotions:
        fig.add_annotation(
            x="embarrassment", y=0.509, yshift=20,
            text=L["ann_emb"],
            showarrow=True, arrowhead=2, arrowsize=0.7,
            arrowwidth=0.6, arrowcolor=st.SAGE,
            font=dict(size=10.5, color=st.SAGE),
            align="center",
        )

    fig.add_annotation(
        x=0.5, xref="paper", y=1.07, yref="paper",
        text=L["summary"],
        showarrow=False,
        font=dict(size=11, color=st.INK_2),
        align="center",
    )

    layout = st.thesis_layout(height=520)
    fig.update_layout(**layout)
    fig.update_layout(
        barmode="group",
        bargap=0.20,
        bargroupgap=0.04,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.30,
            xanchor="center", x=0.5,
            font=dict(size=11, color=st.INK_2),
        ),
    )
    st.style_axes(fig, xtitle=L["xlabel"], ytitle=L["ylabel"])
    fig.update_yaxes(range=[0, 1.0], dtick=0.2)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=10, color=st.INK_3),
                     showgrid=False)
    return fig


if __name__ == "__main__":
    fig = build_figure(lang="es")
    out = pathlib.Path(__file__).resolve().parents[2] / "viz" / "output" / "finetuning_recovery.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"✓ {out}")
