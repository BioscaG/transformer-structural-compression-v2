"""3D BERT architecture viewer — the iconic project image.

Renders the user's BERT-base architecture as a 3D scene:
  - 12 encoder layers stacked vertically
  - Each layer has 12 attention head spheres + an FFN block beside them
  - Sphere COLOR = head category (Critical Specialist red, etc. from notebook 6)
  - Sphere SIZE = head importance
  - FFN block intensity = relative weight change after fine-tuning
  - Hover any component for full details

Click + drag to rotate. The full transformer reduced to one rotatable
visual — the "MRI of BERT".
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go

from viz import style as st
from viz.data.load_results import load_heads


def _category_color(cat: str) -> str:
    cat = cat.rstrip("s") if isinstance(cat, str) else "Minor Specialist"
    return {
        "Critical Specialist": st.TERRA,
        "Critical Generalist": st.BLUE,
        "Minor Specialist":    st.SAND,
        "Dispensable":         st.SPINE,
    }.get(cat, st.SPINE)


LANG = {
    "es": {
        "head_hover":   "<b>L{L}-H{h}</b><br>categoría: {cat}<br>importancia: {imp:.3f}",
        "ffn":          "FFN L{L}",
        "residual":     "residual stream",
        "cls_hover":    "entrada [CLS]",
        "classifier":   "classifier (23 emociones)",
        "classifier_h": "pooler + Linear(768→23)<br>la decisión final",
        "ffn_block":    "FFN block",
        "residual_lbl": "Residual stream",
        "axis_layer":   "capa",
        "categories": {
            "Critical Specialist": "Critical Specialist",
            "Critical Generalist": "Critical Generalist",
            "Minor Specialist":    "Minor Specialist",
            "Dispensable":         "Dispensable",
        },
    },
    "en": {
        "head_hover":   "<b>L{L}-H{h}</b><br>category: {cat}<br>importance: {imp:.3f}",
        "ffn":          "FFN L{L}",
        "residual":     "residual stream",
        "cls_hover":    "[CLS] entry",
        "classifier":   "classifier (23 emotions)",
        "classifier_h": "pooler + Linear(768→23)<br>the final decision",
        "ffn_block":    "FFN block",
        "residual_lbl": "Residual stream",
        "axis_layer":   "layer",
        "categories": {
            "Critical Specialist": "Critical Specialist",
            "Critical Generalist": "Critical Generalist",
            "Minor Specialist":    "Minor Specialist",
            "Dispensable":         "Dispensable",
        },
    },
}


def build_architecture_figure(lang: str = "es") -> go.Figure:
    L_ = LANG[lang]
    heads_data = load_heads()
    cat_grid = np.full((12, 12), "Minor Specialist", dtype=object)
    importance_grid = np.zeros((12, 12), dtype=np.float32)
    if heads_data["categories"] is not None:
        for _, row in heads_data["categories"].iterrows():
            L, h = int(row["layer"]), int(row["head"])
            cat_grid[L, h] = row["category"].rstrip("s")
            importance_grid[L, h] = float(row.get("total_importance", 0))

    n_layers = 12
    n_heads = 12

    fig = go.Figure()

    # ─── Layer base plates (translucent disks) ───
    theta = np.linspace(0, 2 * np.pi, 80)
    for L in range(n_layers):
        z = float(L)
        # Plate as a closed loop
        r_plate = 4.5
        fig.add_trace(go.Scatter3d(
            x=r_plate * np.cos(theta), y=r_plate * np.sin(theta),
            z=[z] * len(theta),
            mode="lines",
            line=dict(color=st.SPINE, width=2),
            hoverinfo="skip", showlegend=False,
        ))
        # Layer label
        fig.add_trace(go.Scatter3d(
            x=[r_plate + 0.4], y=[0], z=[z],
            mode="text", text=[f"L{L}"],
            textfont=dict(size=12, color=st.INK_2, family="serif"),
            hoverinfo="skip", showlegend=False,
        ))

    # ─── Attention heads as spheres around each layer ───
    head_x, head_y, head_z = [], [], []
    head_size, head_color = [], []
    head_hover, head_text = [], []
    head_cats = []
    for L in range(n_layers):
        for h in range(n_heads):
            angle = 2 * np.pi * h / n_heads
            r = 3.0
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = float(L)
            cat = cat_grid[L, h]
            head_x.append(x)
            head_y.append(y)
            head_z.append(z)
            imp = importance_grid[L, h]
            # Size proportional to importance, with floor for visibility
            size = 8 + imp * 35
            head_size.append(size)
            head_color.append(_category_color(cat))
            head_cats.append(cat)
            head_hover.append(L_["head_hover"].format(L=L, h=h, cat=cat, imp=imp))
            head_text.append(f"H{h}")

    fig.add_trace(go.Scatter3d(
        x=head_x, y=head_y, z=head_z,
        mode="markers",
        marker=dict(size=head_size, color=head_color, opacity=0.92,
                    line=dict(color=st.INK, width=0.5),
                    sizemode="diameter"),
        hovertext=head_hover, hoverinfo="text",
        name="cabezas de atención",
        showlegend=False,
    ))

    # ─── FFN blocks: cylinders at the centre of each layer ───
    # Approximate cylinder with a closed loop + line up
    ffn_color = st.TEAL
    for L in range(n_layers):
        z = float(L)
        r_ffn = 1.0
        fig.add_trace(go.Scatter3d(
            x=r_ffn * np.cos(theta), y=r_ffn * np.sin(theta),
            z=[z] * len(theta),
            mode="lines",
            line=dict(color=ffn_color, width=4),
            hovertext=[f"FFN L{L}"] * len(theta),
            hoverinfo="text",
            name=f"FFN L{L}", showlegend=False,
        ))

    # ─── Residual stream (vertical line through centre) ───
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[-0.5, n_layers - 0.5],
        mode="lines",
        line=dict(color=st.INK_3, width=3, dash="dot"),
        hovertext=[L_["residual"]], hoverinfo="text",
        name=L_["residual"], showlegend=False,
    ))

    # ─── [CLS] token entry point ───
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-0.7],
        mode="markers+text",
        marker=dict(size=14, color=st.TERRA, symbol="diamond",
                    line=dict(color=st.INK, width=1.5)),
        text=["⟨CLS⟩"], textposition="bottom center",
        textfont=dict(size=12, color=st.TERRA, family="serif"),
        hovertext=[L_["cls_hover"]], hoverinfo="text",
        showlegend=False,
    ))

    # ─── Classifier above L11 ───
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[n_layers + 0.3],
        mode="markers+text",
        marker=dict(size=18, color=st.SAGE, symbol="square",
                    line=dict(color=st.INK, width=1.5)),
        text=[L_["classifier"]], textposition="top center",
        textfont=dict(size=12, color=st.SAGE, family="serif"),
        hovertext=[L_["classifier_h"]],
        hoverinfo="text",
        showlegend=False,
    ))

    # ─── Legend (manual via dummy traces) ───
    legend_items = [
        (L_["categories"]["Critical Specialist"], st.TERRA),
        (L_["categories"]["Critical Generalist"], st.BLUE),
        (L_["categories"]["Minor Specialist"],    st.SAND),
        (L_["categories"]["Dispensable"],         st.SPINE),
        (L_["ffn_block"],                          st.TEAL),
        (L_["residual_lbl"],                       st.INK_3),
    ]
    for label, col in legend_items:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers",
            marker=dict(size=12, color=col),
            name=label, showlegend=True,
        ))

    fig.update_layout(
        **st.thesis_layout(
            title=("BERT-base · arquitectura completa de tu modelo en 3D"
                   "<br><sub>12 capas de encoder. Cada una: 12 cabezas de atención "
                   "(esferas) + bloque FFN (anillo turquesa). Color de cabeza = "
                   "categoría de §5.3. Tamaño = importancia. Datos reales de tu "
                   "<code>head_categories.csv</code>. Rota libremente.</sub>"),
            height=820, width=1100,
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(title=L_["axis_layer"],
                       tickmode="array",
                       tickvals=list(range(n_layers)),
                       ticktext=[f"L{L}" for L in range(n_layers)],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.4)),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=2),
            bgcolor=st.BG,
        ),
        legend=dict(x=1.02, y=0.5, xanchor="left", yanchor="middle",
                    bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                    borderwidth=0.5, font=dict(size=11)),
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_architecture_figure()
    out = out_dir / "26_bert_architecture.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
