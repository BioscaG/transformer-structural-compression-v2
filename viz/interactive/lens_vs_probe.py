"""Logit lens vs probing: lo que el modelo SABE vs lo que SABE LEER.

Triple-panel stacked vertically — one per crystallization group
(temprano, medio, tardío). Each panel shows two curves on the same
[0, 1] axis:

  - Probe F1 (red): F1 of a fresh linear classifier trained on the [CLS]
    of that layer. Tells you how much emotion info is linearly extractable
    from that layer's representation. Source: notebook 4.

  - Gold sigmoid via logit lens (green): apply the model's REAL pooler +
    classifier (trained on L11) to the [CLS] of each layer. Tells you
    what the trained head would predict at that layer.

The probe rises monotonically (information accumulates with depth).
The logit lens traces a U (the trained head only operates correctly on
L11-style activations). The gap between them at any given layer is the
info that EXISTS but the model can't USE.

The vertical stack shows the contrast at three speeds: emotions that
crystallize early have an early-rising probe but the lens still drags;
emotions that crystallize late show both curves rising late.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoModelForSequenceClassification

from viz import style as st
from viz.data.load_results import EMOTIONS_23, MODEL_CHECKPOINT


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE_CSV = PROJECT_ROOT / "results" / "csvs" / "notebook4" / "probe_results.csv"
CRYSTAL_CSV = PROJECT_ROOT / "results" / "csvs" / "notebook4" / "crystallization_layers.csv"


def _bucket(layer: int) -> str:
    if layer <= 3:   return "temprano"
    if layer <= 7:   return "medio"
    return "tardio"


GROUP_LABEL = {
    "temprano": "Tempranas · cristalizan en L0–L2",
    "medio":    "Medias · cristalizan en L3–L6",
    "tardio":   "Tardías · cristalizan en L7–L11",
}


def build_figure() -> go.Figure:
    # ─── Logit lens gold sigmoid per layer per emotion ────────────────────
    data = np.load(CACHE_DIR / "activations.npz")
    cls = data["cls_per_layer"]                     # (N, 13, 768)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    label_names = meta["label_names"]

    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()
    bp = mdl.bert.pooler.dense.bias.detach().numpy()
    W = mdl.classifier.weight.detach().numpy()
    b = mdl.classifier.bias.detach().numpy()

    pooled   = np.tanh(np.einsum('slh,ph->slp', cls, Wp) + bp)
    logits   = np.einsum('slp,ep->sle', pooled, W) + b
    sigmoids = 1.0 / (1.0 + np.exp(-logits))        # (N, 13, 23)

    label_to_idx = {e: i for i, e in enumerate(EMOTIONS_23)}
    y = np.array([label_to_idx[l] for l in label_names])

    gold_sig = sigmoids[np.arange(len(y)), :, y]    # (N, 13)

    # ─── Probe F1 per layer per emotion ──────────────────────────────────
    probe_df = pd.read_csv(PROBE_CSV)
    layer_cols = ["Emb"] + [f"L{i}" for i in range(12)]
    probe_df = probe_df.set_index("emotion")[layer_cols]   # (23, 13)
    n_layers = len(layer_cols)

    # ─── Crystallization groups ──────────────────────────────────────────
    crystal_df = pd.read_csv(CRYSTAL_CSV)
    emo_to_xlayer = dict(zip(crystal_df["emotion"],
                             crystal_df["crystallization_layer"].astype(int)))
    emo_to_group = {e: _bucket(emo_to_xlayer.get(e, 11)) for e in EMOTIONS_23}

    label_arr = np.array(label_names)

    def _curves(mask, emos):
        n = int(mask.sum())
        if n == 0:
            return [0] * n_layers, [0] * n_layers, 0
        gold = gold_sig[mask].mean(axis=0).tolist()
        probe_subset = probe_df.loc[probe_df.index.isin(emos)]
        probe = probe_subset.mean(axis=0).tolist()
        return probe, gold, n

    groups = []
    for g in ("temprano", "medio", "tardio"):
        emos = [e for e, gg in emo_to_group.items() if gg == g]
        mask = np.isin(label_arr, emos)
        probe, gold, n = _curves(mask, emos)
        groups.append({"key": g, "label": GROUP_LABEL[g],
                       "probe": probe, "gold": gold,
                       "n": n, "emos": emos})

    # ─── Build figure: 3 stacked panels, shared x-axis ───────────────────
    layer_labels = layer_cols
    layer_x = list(range(n_layers))

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=[g["label"] + f"  ·  {len(g['emos'])} emociones · "
                                     f"{g['n']} frases"
                        for g in groups],
    )

    for row_i, g in enumerate(groups, start=1):
        # Phase shading bands. With row/col specified, Plotly resolves
        # the axis references automatically.
        for x0, x1, fc in [
            (-0.5, 3.5, "rgba(212,168,67,0.08)"),
            ( 3.5, 9.5, "rgba(58,110,165,0.05)"),
            ( 9.5,12.5, "rgba(193,85,58,0.07)"),
        ]:
            fig.add_shape(type="rect",
                          x0=x0, x1=x1, y0=0, y1=1,
                          yref="y domain",
                          fillcolor=fc, line=dict(width=0),
                          layer="below", row=row_i, col=1)

        # Probe (red)
        fig.add_trace(go.Scatter(
            x=layer_x, y=g["probe"],
            mode="lines+markers",
            name="probe F1",
            legendgroup="probe",
            showlegend=(row_i == 1),
            line=dict(color=st.TERRA, width=2.6, shape="spline", smoothing=0.5),
            marker=dict(size=7, color=st.TERRA,
                        line=dict(color="white", width=1.4)),
            hovertemplate="%{x}: F1 = %{y:.3f}<extra>probe</extra>",
        ), row=row_i, col=1)

        # Gold sigmoid (green)
        fig.add_trace(go.Scatter(
            x=layer_x, y=g["gold"],
            mode="lines+markers",
            name="gold sigmoid (logit lens)",
            legendgroup="lens",
            showlegend=(row_i == 1),
            line=dict(color=st.SAGE, width=2.6, shape="spline", smoothing=0.5),
            marker=dict(size=7, color=st.SAGE,
                        line=dict(color="white", width=1.4)),
            hovertemplate="%{x}: σ = %{y:.3f}<extra>lens</extra>",
        ), row=row_i, col=1)

        # Filled gap between curves to make the "info que no se usa" visible
        gap_top = [max(p, l) for p, l in zip(g["probe"], g["gold"])]
        gap_bot = [min(p, l) for p, l in zip(g["probe"], g["gold"])]
        fig.add_trace(go.Scatter(
            x=layer_x + layer_x[::-1],
            y=gap_top + gap_bot[::-1],
            fill="toself",
            fillcolor="rgba(122,122,118,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=(row_i == 1),
            name="info no leída",
            legendgroup="gap",
        ), row=row_i, col=1)

    # ─── Layout ──────────────────────────────────────────────────────────
    fig.update_layout(
        **st.thesis_layout(
            title=("Lo que el modelo <b>sabe</b> vs lo que <b>sabe leer</b>"
                   "<br><sub>Probe lineal (rojo) y logit lens (verde) "
                   "comparados sobre las mismas frases. La banda gris es "
                   "info que el probe extrae pero el classifier no usa. "
                   "Tres paneles por capa de cristalización: el patrón se "
                   "desplaza, el contraste persiste.</sub>"),
            height=940, width=1380,
        ),
        legend=dict(x=0.99, y=1.05, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.92)",
                    bordercolor=st.SPINE, borderwidth=0.5,
                    font=dict(size=11), orientation="h"),
    )

    # Axes per row
    for r in range(1, 4):
        fig.update_xaxes(
            tickmode="array", tickvals=layer_x, ticktext=layer_labels,
            gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
            tickfont=dict(size=10, color=st.INK_3),
            row=r, col=1,
        )
        fig.update_yaxes(
            range=[0, 1.0],
            gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
            tickfont=dict(size=10, color=st.INK_3),
            row=r, col=1,
        )
    # Only the bottom panel gets the x-axis title
    fig.update_xaxes(title=dict(text="capa", font=dict(size=12, color=st.INK_2)),
                     row=3, col=1)
    # Only the middle panel labels the y-axis
    fig.update_yaxes(title=dict(text="probe F1  /  gold sigmoid",
                                font=dict(size=12, color=st.INK_2)),
                     row=2, col=1)

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    out = out_dir / "30_lens_vs_probe.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
