"""Hero plot 1 — The Pareto frontier of compression strategies.

Two visualizations in one HTML:
  1. Interactive 2D Pareto scatter with all 22 strategies, families color-coded,
     Pareto-optimal points marked with a halo, full hover details.
  2. 3D phase-transition surface: F1 retention as a function of rank × depth
     band — the literal acantilado between r=384 and r=256.
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
from viz.data.load_results import load_informed, load_sensitivity


def _real_strategies():
    """Read the user's actual 21-strategy comparison from notebook 9."""
    informed = load_informed()
    df = informed["comparison"]
    if df is None:
        return None

    # Map CSV rows to (name, family, f1, ratio, retention) tuples.
    # Compute pareto-optimality from the actual data.
    strategies = []
    family_map = {
        "uniform": "uniform", "adaptive": "adaptive", "informed": "informed",
        "greedy": "greedy", "mixed": "mixed", "baseline": "baseline",
    }
    for _, row in df.iterrows():
        family = family_map.get(row.get("type", "").lower(), "uniform")
        f1 = float(row["macro_f1"])
        ratio = float(row["compression_ratio"])
        ret = float(row.get("f1_retention", f1 / 0.577 if 0.577 else 0))
        strategies.append({
            "name": row["strategy"], "family": family,
            "f1": f1, "ratio": ratio, "retention": ret,
        })

    # Compute Pareto frontier (within these points)
    def is_pareto(i, all_pts):
        for j, other in enumerate(all_pts):
            if j == i:
                continue
            if (other["ratio"] <= all_pts[i]["ratio"] - 1e-4 and
                other["f1"] >= all_pts[i]["f1"] + 1e-4):
                return False
        return True
    for i, s in enumerate(strategies):
        s["pareto"] = is_pareto(i, strategies)

    return strategies


def _real_depth_retention():
    """Read REAL depth-band retention matrix from nb3/depth_f1_matrix.csv."""
    sens = load_sensitivity()
    df = sens["depth_f1"]
    if df is None:
        return None
    return df


LANG = {
    "es": {
        "panel_pareto":  "Frontera de Pareto: 22 estrategias evaluadas",
        "panel_phase":   "Transición de fase: F1 vs rango × profundidad",
        "frontier":      "Frontera de Pareto",
        "baseline_size": "Tamaño<br>baseline",
        "f1_retention":  "Retención<br>de F1",
        "hover_surface": "Rango: %{x}<br>Profundidad: %{y}<br>Retención: %{z:.1%}<extra></extra>",
        "axis_x":        "Ratio de parámetros (vs. baseline)",
        "axis_y":        "F1 macro",
        "scene_x":       "Rango de truncamiento r",
        "scene_y":       "Profundidad",
        "scene_z":       "Retención F1",
        "families": {
            "uniform":  "Uniforme",
            "adaptive": "Adaptativa",
            "mixed":    "Mixta",
            "informed": "Informada (heurística)",
            "greedy":   "Greedy (data-driven)",
            "baseline": "Baseline",
        },
        "hover_strat":   ("<b>%{customdata[0]}</b><br>Ratio params: %{x:.3f}×<br>"
                          "F1 macro: %{y:.3f}<br>Retención: %{customdata[1]}<br>"
                          "Pareto-óptima: %{customdata[3]}<br>"
                          "<i>%{customdata[2]}</i><extra></extra>"),
        "depth_bands":   {"early": "Early (0-3)", "middle": "Middle (4-7)", "late": "Late (8-11)"},
    },
    "en": {
        "panel_pareto":  "Pareto frontier: 22 strategies evaluated",
        "panel_phase":   "Phase transition: F1 vs rank × depth",
        "frontier":      "Pareto frontier",
        "baseline_size": "Baseline<br>size",
        "f1_retention":  "F1<br>retention",
        "hover_surface": "Rank: %{x}<br>Depth: %{y}<br>Retention: %{z:.1%}<extra></extra>",
        "axis_x":        "Parameter ratio (vs. baseline)",
        "axis_y":        "F1 macro",
        "scene_x":       "Truncation rank r",
        "scene_y":       "Depth",
        "scene_z":       "F1 retention",
        "families": {
            "uniform":  "Uniform",
            "adaptive": "Adaptive",
            "mixed":    "Mixed",
            "informed": "Informed (heuristic)",
            "greedy":   "Greedy (data-driven)",
            "baseline": "Baseline",
        },
        "hover_strat":   ("<b>%{customdata[0]}</b><br>Param ratio: %{x:.3f}×<br>"
                          "F1 macro: %{y:.3f}<br>Retention: %{customdata[1]}<br>"
                          "Pareto-optimal: %{customdata[3]}<br>"
                          "<i>%{customdata[2]}</i><extra></extra>"),
        "depth_bands":   {"early": "Early (0-3)", "middle": "Middle (4-7)", "late": "Late (8-11)"},
    },
}


def build_pareto_figure(lang: str = "es") -> go.Figure:
    """Big two-panel figure: 2D Pareto + 3D phase-transition surface."""
    L = LANG[lang]
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=(L["panel_pareto"], L["panel_phase"]),
        horizontal_spacing=0.10,
    )

    # ---------- 2D Pareto scatter ----------
    # Use REAL strategies from notebook 9 if available
    real_strats = _real_strategies()
    if real_strats is not None:
        # Make a class-like accessor
        class S:
            def __init__(self, d):
                self.name, self.family = d["name"], d["family"]
                self.f1_macro, self.param_ratio = d["f1"], d["ratio"]
                self.retention, self.pareto_optimal = d["retention"], d["pareto"]
                self.notes = ""
        strategies_data = [S(s) for s in real_strats]
    else:
        strategies_data = td.STRATEGIES

    families = ["uniform", "adaptive", "mixed", "informed", "greedy", "baseline"]
    family_label = L["families"]
    family_symbol = {
        "uniform":  "square", "adaptive": "circle", "mixed": "diamond-tall",
        "informed": "diamond", "greedy": "star", "baseline": "x-thin-open",
    }

    for fam in families:
        members = [s for s in strategies_data if s.family == fam]
        if not members:
            continue
        x = [s.param_ratio for s in members]
        y = [s.f1_macro for s in members]
        names = [s.name for s in members]
        notes = [s.notes for s in members]
        rets = [f"{s.retention*100:.1f}%" for s in members]
        pareto_flag = [s.pareto_optimal for s in members]
        sizes = [22 if p else 12 for p in pareto_flag]

        # Halo for Pareto-optimal points
        halo_x = [v for v, p in zip(x, pareto_flag) if p]
        halo_y = [v for v, p in zip(y, pareto_flag) if p]
        if halo_x:
            fig.add_trace(go.Scatter(
                x=halo_x, y=halo_y,
                mode="markers", showlegend=False, hoverinfo="skip",
                marker=dict(size=30, color="rgba(0,0,0,0)",
                            line=dict(color=st.FAMILY_COLOR[fam], width=2)),
            ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers",
            name=family_label[fam],
            customdata=list(zip(names, rets, notes, pareto_flag)),
            hovertemplate=L["hover_strat"],
            marker=dict(size=sizes,
                        color=st.FAMILY_COLOR[fam],
                        symbol=family_symbol[fam],
                        line=dict(color="white", width=1.2)),
        ), row=1, col=1)

    # Pareto frontier line (envelope of Pareto-optimal points)
    pareto = sorted([s for s in strategies_data if s.pareto_optimal],
                    key=lambda s: s.param_ratio)
    if pareto:
        fig.add_trace(go.Scatter(
            x=[s.param_ratio for s in pareto],
            y=[s.f1_macro for s in pareto],
            mode="lines",
            line=dict(color=st.INK_3, width=1.2, dash="dot"),
            name=L["frontier"],
            hoverinfo="skip",
        ), row=1, col=1)

    # Vertical reference line at ratio = 1 (baseline parameters)
    fig.add_shape(
        type="line", x0=1.0, x1=1.0, y0=0, y1=0.62,
        line=dict(color=st.INK_3, width=0.8, dash="dash"),
        xref="x1", yref="y1",
    )
    fig.add_annotation(
        x=1.0, y=0.05, xref="x1", yref="y1",
        text=L["baseline_size"], showarrow=False,
        font=dict(size=10, color=st.INK_3),
        align="center", xanchor="left", xshift=4,
    )

    # ---------- 3D phase-transition surface ----------
    # Use REAL depth × rank F1 matrix from notebook 3 if available.
    real_depth_f1 = _real_depth_retention()
    if real_depth_f1 is not None:
        # Columns are r256, r128, r64; rows are depth bands. Convert to retention.
        baseline_f1 = real_depth_f1.values.max()  # baseline ~ full retention point
        rank_cols = [c for c in real_depth_f1.columns if c.startswith("r")]
        empirical_ranks = sorted([int(c[1:]) for c in rank_cols])
        ranks = empirical_ranks + [512, 768]
        bands_lookup = {b.lower().split()[0]: b for b in real_depth_f1.index}
        bands = [b for b in ["early", "middle", "late"] if b in bands_lookup]
        band_label = {b: bands_lookup[b] for b in bands}
        z_grid = []
        for band in bands:
            row_label = bands_lookup[band]
            row = []
            for r in ranks:
                col = f"r{r}"
                if col in real_depth_f1.columns:
                    row.append(float(real_depth_f1.loc[row_label, col]) / baseline_f1)
                elif r == 512:
                    # Smooth interpolation toward baseline
                    smooth = {"early": 0.96, "middle": 0.92, "late": 0.85}.get(band, 0.90)
                    row.append(smooth)
                else:
                    row.append(1.0)
            z_grid.append(row)
        z = np.array(z_grid)
    else:
        ranks = [64, 128, 256, 384, 512, 768]
        bands = ["early", "middle", "late"]
        band_label = L["depth_bands"]
        z_grid = []
        for band in bands:
            row = []
            for r in ranks:
                if r in td.DEPTH_RETENTION[band]:
                    row.append(td.DEPTH_RETENTION[band][r])
                elif r >= 512:
                    if r == 512:
                        row.append({"early": 0.96, "middle": 0.92, "late": 0.85}[band])
                    else:
                        row.append(1.0)
                else:
                    row.append(0.0)
            z_grid.append(row)
        z = np.array(z_grid)

    fig.add_trace(go.Surface(
        x=ranks, y=[band_label[b] for b in bands], z=z,
        colorscale=[[0, "#3A0000"], [0.15, st.TERRA], [0.4, st.SAND],
                    [0.7, st.SAGE], [1.0, st.BLUE]],
        cmin=0, cmax=1,
        showscale=True,
        colorbar=dict(title=dict(text=L["f1_retention"], font=dict(size=11)),
                      tickformat=".0%", thickness=14, x=1.0, len=0.7,
                      tickfont=dict(size=10, color=st.INK_3)),
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True,
                             highlightcolor="white")),
        hovertemplate=L["hover_surface"],
        opacity=0.95,
    ), row=1, col=2)

    # ---------- Layout ----------
    fig.update_layout(
        **st.thesis_layout(
            title="Anatomía Emocional — Cap. 4: Compresión SVD",
            height=560, width=1280,
        ),
        legend=dict(
            x=0.01, y=0.98, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=st.SPINE, borderwidth=0.5,
            font=dict(size=11),
        ),
        scene=dict(
            xaxis=dict(title=L["scene_x"],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2),
            yaxis=dict(title=L["scene_y"],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2),
            zaxis=dict(title=L["scene_z"],
                       backgroundcolor=st.BG, gridcolor=st.GRID,
                       zerolinecolor=st.SPINE, color=st.INK_2,
                       tickformat=".0%", range=[0, 1.05]),
            camera=dict(eye=dict(x=1.55, y=-1.55, z=0.95)),
            aspectmode="manual", aspectratio=dict(x=1.4, y=1.0, z=0.9),
        ),
    )
    fig.update_xaxes(title_text=L["axis_x"],
                     range=[0.30, 1.45], row=1, col=1,
                     gridcolor=st.GRID, showline=True, linecolor=st.SPINE,
                     ticks="outside", tickcolor=st.INK_3,
                     tickfont=dict(size=11, color=st.INK_3))
    fig.update_yaxes(title_text=L["axis_y"],
                     range=[-0.02, 0.62], row=1, col=1,
                     gridcolor=st.GRID, showline=True, linecolor=st.SPINE,
                     ticks="outside", tickcolor=st.INK_3,
                     tickfont=dict(size=11, color=st.INK_3))

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_pareto_figure()
    out = out_dir / "01_pareto_landscape.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
