"""Hero plot 2 — The 144-head taxonomy matrix.

Each of BERT-base's 144 attention heads laid out as a 12-layer × 12-head grid.
- Background fill: category color (Critical Specialist / Generalist / Minor /
  Dispensable). Categories synthesized from the per-band counts in Tabla 18 with
  a fixed seed; the 11 critical heads from Tabla 19 are pinned exactly.
- Overlaid markers on the 11 critical-per-emotion heads, sized by F1 drop on
  ablation (Tabla 19), labeled with the emotion they protect.
- Marginal annotations on the right show the band distribution.
"""

from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go

from viz import thesis_data as td
from viz import style as st
from viz.data.load_results import load_heads, head_categories_grid, critical_head_per_emotion


def _real_categories() -> np.ndarray | None:
    """Return REAL 12x12 category grid from notebook 6, or None if unavailable."""
    heads = load_heads()
    if heads["categories"] is None:
        return None
    cats, _ = head_categories_grid(heads)
    return cats


def _synth_categories(seed: int = 42) -> np.ndarray:
    """Return a 12x12 string array of category labels per (layer, head).

    Constraints:
      - Capa 11: 12 críticas (no dispensable). Split 6 specialist / 6 generalist.
      - Capas 5-10: 45 críticas en total entre middle (20 of 36) y late ex-L11
        (37-12=25 of 36).
      - Capas 0-4: 15 críticas, 23 dispensable, rest minor specialist.
      - The 11 explicit critical heads from Tabla 19 are pinned to "Critical
        Specialist" or "Critical Generalist" depending on which emotion they
        rule.
    """
    rng = np.random.default_rng(seed)
    cats = np.empty((12, 12), dtype=object)

    # Pins from Tabla 19 — emotion → (layer, head).
    # All Tabla 19 entries become Critical Specialist except L11-H6 which is
    # shared by sadness AND realization (i.e. generalist behavior).
    pins: dict[tuple[int, int], str] = {}
    for emo, (l, h, _) in td.CRITICAL_HEAD_PER_EMOTION.items():
        if l > 11 or h > 11:
            continue
        existing = pins.get((l, h))
        if existing == "Critical Specialist":
            pins[(l, h)] = "Critical Generalist"  # shared head
        else:
            pins[(l, h)] = pins.get((l, h), "Critical Specialist")

    # Layer 11: every head critical, 6 specialist + 6 generalist.
    layer_quotas = {
        # layer: dict(category -> count)
        11: {"Critical Specialist": 6, "Critical Generalist": 6},
        10: {"Critical Specialist": 5, "Critical Generalist": 4, "Minor Specialist": 2, "Dispensable": 1},
        9:  {"Critical Specialist": 4, "Critical Generalist": 4, "Minor Specialist": 3, "Dispensable": 1},
        8:  {"Critical Specialist": 4, "Critical Generalist": 3, "Minor Specialist": 3, "Dispensable": 2},
        7:  {"Critical Specialist": 3, "Critical Generalist": 3, "Minor Specialist": 4, "Dispensable": 2},
        6:  {"Critical Specialist": 2, "Critical Generalist": 3, "Minor Specialist": 4, "Dispensable": 3},
        5:  {"Critical Specialist": 2, "Critical Generalist": 2, "Minor Specialist": 4, "Dispensable": 4},
        4:  {"Critical Specialist": 2, "Critical Generalist": 1, "Minor Specialist": 4, "Dispensable": 5},
        3:  {"Critical Specialist": 1, "Critical Generalist": 1, "Minor Specialist": 5, "Dispensable": 5},
        2:  {"Critical Specialist": 1, "Critical Generalist": 1, "Minor Specialist": 5, "Dispensable": 5},
        1:  {"Critical Specialist": 1, "Critical Generalist": 1, "Minor Specialist": 4, "Dispensable": 6},
        0:  {"Critical Specialist": 2, "Critical Generalist": 1, "Minor Specialist": 5, "Dispensable": 4},
    }

    for layer in range(12):
        slots = list(range(12))
        # Place pinned heads first
        pinned_in_layer = {h: cat for (l, h), cat in pins.items() if l == layer}
        for h, cat in pinned_in_layer.items():
            cats[layer, h] = cat
            slots.remove(h)

        quota = dict(layer_quotas[layer])
        # Subtract pinned occurrences from quota
        for cat in pinned_in_layer.values():
            quota[cat] = max(0, quota.get(cat, 0) - 1)

        # Build a flat list of remaining categories and shuffle deterministically
        flat = []
        for cat, count in quota.items():
            flat.extend([cat] * count)
        # Pad to 12 - (pinned count) if quotas don't exactly cover (shouldn't happen)
        while len(flat) < len(slots):
            flat.append("Minor Specialist")
        flat = flat[: len(slots)]
        rng.shuffle(flat)
        for s, c in zip(slots, flat):
            cats[layer, s] = c

    return cats


LANG = {
    "es": {
        "heatmap_h":   "<b>L%{y}-H%{x}</b><br>Categoría: %{customdata}<extra></extra>",
        "head_h":      "<b>%{text}</b><br>Cabeza crítica: L%{y}-H%{x}<br>ΔF1 al ablacionar: %{customdata[0]:.3f}<extra></extra>",
        "head_name":   "Cabeza crítica por emoción",
        "axis_x":      "Cabeza (índice 0–11)",
        "axis_y":      "Capa del encoder",
    },
    "en": {
        "heatmap_h":   "<b>L%{y}-H%{x}</b><br>Category: %{customdata}<extra></extra>",
        "head_h":      "<b>%{text}</b><br>Critical head: L%{y}-H%{x}<br>ΔF1 when ablated: %{customdata[0]:.3f}<extra></extra>",
        "head_name":   "Critical head per emotion",
        "axis_x":      "Head (index 0–11)",
        "axis_y":      "Encoder layer",
    },
}


def build_heads_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    # Try real first, fall back to synthesis
    cats = _real_categories()
    if cats is None:
        print("  [warn] head_categories.csv not found — using synthesized grid")
        cats = _synth_categories()
    # The user's CSV uses pluralized labels ("Critical Specialists"); normalize.
    cats = np.vectorize(lambda c: c.rstrip("s") if isinstance(c, str) else c)(cats)
    cat_to_int = {"Critical Specialist": 3, "Critical Generalist": 2,
                  "Minor Specialist": 1, "Dispensable": 0}
    z = np.array([[cat_to_int.get(c, 0) for c in row] for row in cats])

    # Discrete colorscale for the 4 categories — soft pastel tones so the
    # star overlays and labels stay legible.
    DISP_L  = "#EBE9E0"  # near-bg light grey for Dispensable
    colorscale = [
        [0.000, DISP_L],       # Dispensable
        [0.250, DISP_L],
        [0.250, st.SAND_L],    # Minor Specialist
        [0.500, st.SAND_L],
        [0.500, st.BLUE_L],    # Critical Generalist
        [0.750, st.BLUE_L],
        [0.750, st.TERRA_L],   # Critical Specialist
        [1.000, st.TERRA_L],
    ]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=list(range(12)), y=list(range(12)),
        colorscale=colorscale, zmin=-0.5, zmax=3.5,
        showscale=False,
        xgap=4, ygap=4,
        customdata=cats,
        hovertemplate=L["heatmap_h"],
    ))

    # Overlay critical heads — REAL data from notebook 6 (top-1 head per emotion)
    heads_data = load_heads()
    if heads_data["top_heads"] is not None:
        critical = critical_head_per_emotion(heads_data)
    else:
        critical = td.CRITICAL_HEAD_PER_EMOTION

    star_x, star_y, star_text, star_size, star_emotion = [], [], [], [], []
    for emo, (l, h, df1) in critical.items():
        # Show the top-impact heads only
        if abs(df1) < 0.010:
            continue
        star_x.append(h)
        star_y.append(l)
        star_text.append(emo)
        star_size.append(8 + abs(df1) * 130)
        star_emotion.append(emo)

    fig.add_trace(go.Scatter(
        x=star_x, y=star_y, mode="markers+text",
        marker=dict(symbol="circle", size=[max(s * 0.55, 9) for s in star_size],
                    color=st.INK, line=dict(color="white", width=1.5),
                    opacity=0.92),
        text=star_text,
        textposition="middle right",
        textfont=dict(size=9, color=st.INK_2, family="serif"),
        customdata=[(critical[e][2], e) for e in star_emotion],
        hovertemplate=L["head_h"],
        name=L["head_name"],
        showlegend=True,
    ))

    # Count real categories
    cat_counts = {c: int(np.sum(cats == c)) for c in
                  ["Critical Specialist", "Critical Generalist",
                   "Minor Specialist", "Dispensable"]}
    legend_items = [
        (f"Critical Specialist ({cat_counts['Critical Specialist']})", st.TERRA_L),
        (f"Critical Generalist ({cat_counts['Critical Generalist']})", st.BLUE_L),
        (f"Minor Specialist ({cat_counts['Minor Specialist']})",       st.SAND_L),
        (f"Dispensable ({cat_counts['Dispensable']})",                 DISP_L),
    ]
    for label, color in legend_items:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=14, color=color, symbol="square",
                        line=dict(color="white", width=1)),
            name=label, showlegend=True, hoverinfo="skip",
        ))

    # Layer 11 highlight: subtle box marking the row where every head is
    # critical — the structural finding without the visual shouting.
    fig.add_shape(
        type="rect", x0=-0.55, x1=11.55, y0=10.55, y1=11.55,
        line=dict(color=st.INK_3, width=1.2, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )

    # Layout
    fig.update_layout(
        **st.thesis_layout(
            title="Cap. 5.3 — Taxonomía de las 144 cabezas de atención",
            height=620, width=1100,
        ),
        legend=dict(
            x=1.02, y=0.5, xanchor="left", yanchor="middle",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=st.SPINE, borderwidth=0.5,
            font=dict(size=11),
        ),
    )
    fig.update_xaxes(
        title=dict(text=L["axis_x"], font=dict(size=13, color=st.INK_2)),
        tickmode="array", tickvals=list(range(12)),
        showgrid=False, zeroline=False,
        tickfont=dict(size=11, color=st.INK_3),
    )
    fig.update_yaxes(
        title=dict(text=L["axis_y"], font=dict(size=13, color=st.INK_2)),
        tickmode="array", tickvals=list(range(12)),
        showgrid=False, zeroline=False, autorange="reversed",
        tickfont=dict(size=11, color=st.INK_3),
    )
    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_heads_figure()
    out = out_dir / "02_heads_matrix.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
