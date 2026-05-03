"""Spectral landscape — the asymmetry of §4.1 as 3D topography.

72 weight matrices laid out as rows. The X-axis is the singular value index
(0..767), the Z-axis is σᵢ / σ₁ (normalized magnitude). Rows are grouped by
component type so visually you see:

  - Q, K matrices: sharp PEAKS — 1-2 huge singular values dominating, then
    plummet. Rank effective ~395.
  - V, Attn-Output: intermediate decay
  - FFN-Intermediate, FFN-Output: smooth PLATEAUS — every singular value
    contributes meaningfully. Rank effective ~620.

The asymmetry of §4.1 visible as a 3D landscape you can rotate.

Reads from `viz/data/cache/spectra.json` (real SVD of the user's checkpoint).
"""

from __future__ import annotations

import json
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go

from viz import style as st


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


LANG = {
    "es": {
        "k95":         "k95 (95% energía)",
        "spectrum_h":  "matriz: %{customdata}<br>σ index: %{x}<br>σᵢ/σ₁: %{z:.3f}<extra></extra>",
        "axis_x":      "índice de valor singular",
        "axis_y":      "matriz (12 capas × 6 componentes)",
    },
    "en": {
        "k95":         "k95 (95% energy)",
        "spectrum_h":  "matrix: %{customdata}<br>σ index: %{x}<br>σᵢ/σ₁: %{z:.3f}<extra></extra>",
        "axis_x":      "singular-value index",
        "axis_y":      "matrix (12 layers × 6 components)",
    },
}


def build_landscape_figure(lang: str = "es") -> go.Figure:
    _L = LANG[lang]
    spectra = json.loads((CACHE_DIR / "spectra.json").read_text())

    components = ["query", "key", "value", "attn_output",
                  "ffn_intermediate", "ffn_output"]
    comp_label = {"query": "Q", "key": "K", "value": "V",
                  "attn_output": "Attn-O", "ffn_intermediate": "FFN-i",
                  "ffn_output": "FFN-o"}
    comp_color = {
        "query": st.BLUE, "key": st.BLUE_L, "value": st.TERRA,
        "attn_output": st.ROSE, "ffn_intermediate": st.TEAL, "ffn_output": st.TEAL_L,
    }

    n_layers = 12
    top_k = 64        # number of singular values to show per matrix

    # Build matrix order: first all Q (12 rows), then K, V, Attn-O, FFN-i, FFN-o
    # Total 72 rows.
    matrix_order = []
    for cname in components:
        for L in range(n_layers):
            matrix_order.append((L, cname))

    # Z grid: rows = matrices, cols = singular value index
    z = np.zeros((len(matrix_order), top_k), dtype=np.float32)
    row_labels = []
    row_colors = []
    k95_per_row = []
    for ri, (L, cname) in enumerate(matrix_order):
        key = f"L{L}.{cname}"
        sp = spectra[key]
        spectrum = sp["spectrum"][:top_k]
        z[ri, : len(spectrum)] = spectrum
        row_labels.append(f"L{L} {comp_label[cname]}")
        row_colors.append(comp_color[cname])
        k95_per_row.append(sp["k95"])

    x_axis = np.arange(top_k)
    y_axis = np.arange(len(matrix_order))

    fig = go.Figure()

    # Main surface
    fig.add_trace(go.Surface(
        z=z, x=x_axis, y=y_axis,
        colorscale=[
            [0.0, "#FFFFFF"], [0.05, "#F5E1C8"], [0.20, st.SAND_L],
            [0.40, st.SAND], [0.65, st.TERRA], [1.0, "#7A2A18"],
        ],
        cmin=0, cmax=1.0, showscale=True,
        colorbar=dict(thickness=14, len=0.7, x=1.0,
                      title=dict(text="σᵢ / σ₁", font=dict(size=12)),
                      tickfont=dict(size=10, color=st.INK_3)),
        contours=dict(
            z=dict(show=True, usecolormap=True, project_z=False,
                   highlightcolor="white", width=1),
        ),
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.15,
                      roughness=0.7, fresnel=0.2),
        lightposition=dict(x=80, y=20, z=120),
        hovertemplate=_L["spectrum_h"],
        customdata=np.array([[lab] * top_k for lab in row_labels]),
        opacity=0.96,
        name="spectrum",
    ))

    # Markers at the k95 of each matrix — visualize rank effective
    fig.add_trace(go.Scatter3d(
        x=k95_per_row, y=y_axis, z=[0.05] * len(matrix_order),
        mode="markers",
        marker=dict(size=4, color=row_colors, symbol="diamond",
                    line=dict(color=st.INK, width=0.5)),
        hovertext=[f"<b>{row_labels[i]}</b><br>k95 = {k95_per_row[i]}"
                   for i in range(len(matrix_order))],
        hoverinfo="text",
        name=_L["k95"],
        showlegend=True,
    ))

    # Component group separators on the y-axis
    annotations = []
    for ci, cname in enumerate(components):
        y_mid = ci * n_layers + n_layers / 2 - 0.5
        annotations.append(dict(
            text=f"<b>{comp_label[cname]}</b>",
            x=top_k * 1.05, y=y_mid, z=0,
            xanchor="left",
            font=dict(size=12, color=comp_color[cname]),
            showarrow=False,
        ))

    fig.update_layout(
        **st.thesis_layout(
            title=("Spectral landscape · la asimetría de §4.1 como topografía"
                   "<br><sub>72 matrices del modelo. Cada fila es una matriz, eje X = índice de σ, "
                   "eje Z = σᵢ/σ₁. Q/K son picos abruptos (espectro concentrado), FFN son mesetas "
                   "(espectro plano). Diamantes = k95 (rango efectivo). SVD real de tu checkpoint.</sub>"),
            height=820, width=1280,
        ),
        scene=dict(
            xaxis=dict(title=_L["axis_x"], range=[0, top_k - 1],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2),
            yaxis=dict(title=_L["axis_y"],
                       tickmode="array",
                       tickvals=[ci * n_layers + n_layers / 2 for ci in range(len(components))],
                       ticktext=[comp_label[c] for c in components],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2),
            zaxis=dict(title="σᵢ / σ₁", range=[0, 1.05],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2),
            camera=dict(eye=dict(x=1.6, y=-2.0, z=1.0)),
            aspectmode="manual", aspectratio=dict(x=1.4, y=1.6, z=0.8),
            annotations=annotations,
        ),
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=st.SPINE,
                    borderwidth=0.5, font=dict(size=11)),
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_landscape_figure()
    out = out_dir / "21_spectral_landscape.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
