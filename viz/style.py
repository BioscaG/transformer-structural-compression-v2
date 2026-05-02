"""Plotly styling that matches plots/tfg_plot_style.py for visual coherence
with the matplotlib figures already in the memoria."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio


# Palette ported from plots/tfg_plot_style.py
INK       = "#1A1A1A"
INK_2     = "#4A4A4A"
INK_3     = "#8A8A85"
SPINE     = "#C8C7C1"
GRID      = "#EBEBEB"
BG        = "#FFFFFF"

BLUE      = "#3A6EA5"
BLUE_L    = "#6A9CC9"
TERRA     = "#C1553A"
TERRA_L   = "#D98A76"
SAGE      = "#5A8F7B"
SAGE_L    = "#8CB8A4"
SAND      = "#D4A843"
SAND_L    = "#E5C87A"
PLUM      = "#7B5E7B"
TEAL      = "#2A8F8F"
TEAL_L    = "#5BB5B5"
ROSE      = "#B5555B"

FAMILY_COLOR = {
    "uniform":  BLUE,
    "adaptive": TERRA,
    "informed": SAGE,
    "greedy":   SAND,
    "mixed":    PLUM,
    "baseline": INK,
}

COMPONENT_COLOR = {
    "query": BLUE, "key": BLUE_L, "value": TERRA,
    "attn_output": ROSE, "ffn_intermediate": TEAL, "ffn_output": TEAL_L,
}

DEPTH_COLOR = {
    "early": SAGE, "middle": SAND, "late": TERRA,
}


def thesis_layout(title: str | None = None, height: int = 520, width: int | None = None) -> dict:
    """Return a Plotly layout dict matching the thesis matplotlib aesthetic."""
    layout = dict(
        font=dict(family='"TeX Gyre Pagella", "Palatino", "Book Antiqua", "DejaVu Serif", serif',
                  size=13, color=INK),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        title=dict(text=title or "", x=0.02, y=0.96, xanchor="left",
                   font=dict(size=16, color=INK)) if title else None,
        margin=dict(l=70, r=40, t=70 if title else 40, b=60),
        hoverlabel=dict(
            bgcolor="white", bordercolor=SPINE,
            font=dict(family='"Inter", "Helvetica", sans-serif', size=12, color=INK),
        ),
        height=height,
    )
    if width is not None:
        layout["width"] = width
    return layout


def style_axes(fig: go.Figure, xtitle: str = "", ytitle: str = "") -> go.Figure:
    fig.update_xaxes(
        title=dict(text=xtitle, font=dict(size=13, color=INK_2)),
        showgrid=True, gridcolor=GRID, gridwidth=0.5,
        zeroline=False, showline=True, linecolor=SPINE, linewidth=0.5,
        ticks="outside", tickcolor=INK_3, tickfont=dict(size=11, color=INK_3),
    )
    fig.update_yaxes(
        title=dict(text=ytitle, font=dict(size=13, color=INK_2)),
        showgrid=True, gridcolor=GRID, gridwidth=0.5,
        zeroline=False, showline=True, linecolor=SPINE, linewidth=0.5,
        ticks="outside", tickcolor=INK_3, tickfont=dict(size=11, color=INK_3),
    )
    return fig


def register_template():
    pio.templates["tfg"] = go.layout.Template(layout=thesis_layout(height=520))
    pio.templates.default = "tfg"


# Apply on import
register_template()
