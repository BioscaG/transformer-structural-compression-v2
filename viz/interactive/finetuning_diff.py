"""Pre vs post fine-tuning · ¿qué cambió en los pesos?

Loads `bert-base-uncased` (the pretrained checkpoint) AND the user's
`23emo-final` and computes, for each of the 72 weight matrices in the
encoder, the relative change:

    rel_change(W) = ||W_finetuned - W_pretrained||_F / ||W_pretrained||_F

Renders as a 12×6 heatmap (layers × components) showing where fine-tuning
concentrated its changes. Empirical proof of §5.5's "two-phase
architecture": late layers should change MUCH more than early layers,
because gradient flow weakens with depth and pretrained representations
are already fine in early layers.

Companion view: a 3D bar chart of the same data, plus a marginal summary
showing total change per layer and per component.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import AutoModelForSequenceClassification, AutoModel

from viz import style as st
from viz.data.load_results import MODEL_CHECKPOINT


COMPONENTS = ["query", "key", "value", "attn_output",
              "ffn_intermediate", "ffn_output"]
COMPONENT_LABEL = {
    "query": "Q", "key": "K", "value": "V",
    "attn_output": "Attn-O", "ffn_intermediate": "FFN-i", "ffn_output": "FFN-o",
}
COMPONENT_PATH = {
    "query":            "encoder.layer.{L}.attention.self.query",
    "key":              "encoder.layer.{L}.attention.self.key",
    "value":            "encoder.layer.{L}.attention.self.value",
    "attn_output":      "encoder.layer.{L}.attention.output.dense",
    "ffn_intermediate": "encoder.layer.{L}.intermediate.dense",
    "ffn_output":       "encoder.layer.{L}.output.dense",
}


def _get_layer_weight(model, path: str, prefix: str = "") -> np.ndarray:
    """Resolve a dotted attribute path to its weight tensor."""
    obj = model
    full = (prefix + "." + path) if prefix else path
    for part in full.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj.weight.detach().float().numpy()


def compute_diff_matrix() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rel_change [12, 6], frob_pre [12, 6], frob_diff [12, 6])."""
    print("Loading bert-base-uncased (pretrained)...")
    pre = AutoModel.from_pretrained("bert-base-uncased")
    print("Loading user's 23emo-final (fine-tuned)...")
    ft = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))

    # The fine-tuned model wraps BertModel as `bert`
    ft_bert = ft.bert
    pre_bert = pre

    n_layers = 12
    rel_change = np.zeros((n_layers, len(COMPONENTS)), dtype=np.float32)
    frob_pre = np.zeros_like(rel_change)
    frob_diff = np.zeros_like(rel_change)

    for L in range(n_layers):
        for ci, comp in enumerate(COMPONENTS):
            path = COMPONENT_PATH[comp].format(L=L)
            W_pre = _get_layer_weight(pre_bert, path)
            W_ft = _get_layer_weight(ft_bert, path)
            d = W_ft - W_pre
            f_pre = np.linalg.norm(W_pre)
            f_diff = np.linalg.norm(d)
            rel_change[L, ci] = f_diff / f_pre if f_pre > 0 else 0
            frob_pre[L, ci] = f_pre
            frob_diff[L, ci] = f_diff

    return rel_change, frob_pre, frob_diff


LANG = {
    "es": {
        "panel_main":  "Cambio relativo por matriz (ratio Frobenius)",
        "panel_3d":    "Mismo dato en 3D",
        "panel_layer": "Cambio total por capa",
        "panel_comp":  "Cambio total por componente",
        "main_h":      "<b>L%{y}-%{x}</b><br>cambio relativo: %{z:.4f} (%{text})<extra></extra>",
        "surface_h":   "L%{y} %{x}<br>cambio: %{z:.4f}<extra></extra>",
        "bar_layer_h": "<b>L%{x}</b><br>cambio medio: %{y:.4f}<extra></extra>",
        "bar_comp_h":  "<b>%{x}</b><br>cambio medio: %{y:.4f}<extra></extra>",
        "axis_change": "cambio medio",
        "axis_comp":   "componente",
        "axis_layer":  "capa",
    },
    "en": {
        "panel_main":  "Relative change per matrix (Frobenius ratio)",
        "panel_3d":    "Same data in 3D",
        "panel_layer": "Total change per layer",
        "panel_comp":  "Total change per component",
        "main_h":      "<b>L%{y}-%{x}</b><br>relative change: %{z:.4f} (%{text})<extra></extra>",
        "surface_h":   "L%{y} %{x}<br>change: %{z:.4f}<extra></extra>",
        "bar_layer_h": "<b>L%{x}</b><br>mean change: %{y:.4f}<extra></extra>",
        "bar_comp_h":  "<b>%{x}</b><br>mean change: %{y:.4f}<extra></extra>",
        "axis_change": "mean change",
        "axis_comp":   "component",
        "axis_layer":  "layer",
    },
}


def build_diff_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    rel_change, frob_pre, frob_diff = compute_diff_matrix()
    n_layers, n_comp = rel_change.shape

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.62, 0.38],
        row_heights=[0.78, 0.22],
        specs=[[{"type": "heatmap"}, {"type": "scene"}],
               [{"type": "xy"},      {"type": "xy"}]],
        subplot_titles=(L["panel_main"], L["panel_3d"],
                        L["panel_layer"], L["panel_comp"]),
        horizontal_spacing=0.10, vertical_spacing=0.18,
    )

    # ─── Main heatmap: layers × components ───
    text = [[f"{rel_change[L, ci]*100:.1f}%" for ci in range(n_comp)]
            for L in range(n_layers)]
    fig.add_trace(go.Heatmap(
        z=rel_change, x=[COMPONENT_LABEL[c] for c in COMPONENTS],
        y=[f"L{L}" for L in range(n_layers)],
        colorscale=[[0, "#FFFFFF"], [0.30, st.SAND_L], [0.60, st.SAND],
                    [0.85, st.TERRA], [1.0, "#7A2A18"]],
        zmin=0, zmax=float(rel_change.max()),
        text=text, texttemplate="%{text}",
        textfont=dict(size=11, color=st.INK, family="serif"),
        hovertemplate=L["main_h"],
        xgap=1, ygap=1, showscale=True,
        colorbar=dict(thickness=14, len=0.65, x=0.55,
                      title=dict(text="‖ΔW‖/‖W₀‖", font=dict(size=11)),
                      tickfont=dict(size=10, color=st.INK_3),
                      tickformat=".0%"),
    ), row=1, col=1)

    # ─── 3D mesh of same data ───
    component_color = {
        "query": st.BLUE, "key": st.BLUE_L, "value": st.TERRA,
        "attn_output": st.ROSE, "ffn_intermediate": st.TEAL, "ffn_output": st.TEAL_L,
    }
    fig.add_trace(go.Surface(
        z=rel_change, x=list(range(n_comp)), y=list(range(n_layers)),
        colorscale=[[0, "#FFFFFF"], [0.30, st.SAND_L], [0.60, st.SAND],
                    [0.85, st.TERRA], [1.0, "#7A2A18"]],
        cmin=0, cmax=float(rel_change.max()),
        showscale=False,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True,
                             highlightcolor="white")),
        lighting=dict(ambient=0.6, diffuse=0.85, specular=0.2),
        hovertemplate=L["surface_h"],
        opacity=0.95,
    ), row=1, col=2)

    # ─── Per-layer total change ───
    per_layer = rel_change.mean(axis=1)
    fig.add_trace(go.Bar(
        x=[f"L{L}" for L in range(n_layers)], y=per_layer,
        marker=dict(color=per_layer,
                    colorscale=[[0, st.SAND_L], [1.0, st.TERRA]],
                    line=dict(color="white", width=0.5)),
        hovertemplate=L["bar_layer_h"],
        showlegend=False,
    ), row=2, col=1)

    # ─── Per-component total change ───
    per_comp = rel_change.mean(axis=0)
    fig.add_trace(go.Bar(
        x=[COMPONENT_LABEL[c] for c in COMPONENTS], y=per_comp,
        marker=dict(color=[component_color[c] for c in COMPONENTS],
                    line=dict(color="white", width=0.5)),
        hovertemplate=L["bar_comp_h"],
        showlegend=False,
    ), row=2, col=2)

    # Layout
    fig.update_layout(
        **st.thesis_layout(
            title=("Pre vs Post fine-tuning · ¿qué cambió en los pesos?"
                   "<br><sub>Cambio relativo Frobenius entre <b>bert-base-uncased</b> "
                   "y tu <b>23emo-final</b>. Las capas tardías deberían cambiar más "
                   "que las tempranas (predicción de §5.5).</sub>"),
            height=860, width=1280,
        ),
        scene=dict(
            xaxis=dict(title=L["axis_comp"],
                       tickmode="array",
                       tickvals=list(range(n_comp)),
                       ticktext=[COMPONENT_LABEL[c] for c in COMPONENTS],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       color=st.INK_2),
            yaxis=dict(title=L["axis_layer"],
                       tickmode="array",
                       tickvals=list(range(n_layers)),
                       ticktext=[f"L{L}" for L in range(n_layers)],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       color=st.INK_2),
            zaxis=dict(title="‖ΔW‖/‖W₀‖",
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       color=st.INK_2, tickformat=".0%"),
            camera=dict(eye=dict(x=1.7, y=-1.6, z=1.1)),
            aspectmode="manual", aspectratio=dict(x=1.0, y=1.4, z=0.7),
        ),
    )
    fig.update_xaxes(autorange="reversed",
                     tickfont=dict(size=11, color=st.INK_2, family="serif"),
                     row=1, col=1)
    fig.update_yaxes(autorange="reversed",
                     tickfont=dict(size=10, color=st.INK_2, family="serif"),
                     row=1, col=1)
    fig.update_xaxes(tickfont=dict(size=10, color=st.INK_3), gridcolor=st.GRID,
                     showline=True, linecolor=st.SPINE, row=2, col=1)
    fig.update_yaxes(title=dict(text=L["axis_change"], font=dict(size=10, color=st.INK_2)),
                     tickformat=".0%", gridcolor=st.GRID,
                     tickfont=dict(size=10, color=st.INK_3),
                     showline=True, linecolor=st.SPINE, row=2, col=1)
    fig.update_xaxes(tickfont=dict(size=10, color=st.INK_3), gridcolor=st.GRID,
                     showline=True, linecolor=st.SPINE, row=2, col=2)
    fig.update_yaxes(title=dict(text=L["axis_change"], font=dict(size=10, color=st.INK_2)),
                     tickformat=".0%", gridcolor=st.GRID,
                     tickfont=dict(size=10, color=st.INK_3),
                     showline=True, linecolor=st.SPINE, row=2, col=2)

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_diff_figure()
    out = out_dir / "25_finetuning_diff.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
