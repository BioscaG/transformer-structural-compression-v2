"""Export web figures as static PNGs ready to drop into the TFG PDF.

Same Plotly code as the web (single source of truth), rendered with
thesis-ready parameters: larger fonts (compensating LaTeX scaling),
6.3 in width (matches \\textwidth on A4 with 2.5 cm margins), 300 DPI.

The output lands directly in `latex_figures/figures/` with the same
naming convention as the notebook-generated figures
(`cap{N}_{name}_es.png`), so in LaTeX you just write:

    \\includegraphics[width=\\textwidth]{latex_figures/figures/cap5_lens_vs_probe_es.png}

…regardless of whether the figure was authored in matplotlib or in
Plotly. No mental tax about origin.

Usage:
    .viz_venv/bin/python viz/site/build_pdf_figures.py            # all
    .viz_venv/bin/python viz/site/build_pdf_figures.py NAME ...   # subset
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import plotly.graph_objects as go

from viz.site._site_mode import (
    FONT_STACK, PAGE_BG_PLOT, PAGE_INK, PAGE_INK_2, PAGE_INK_3,
    PAGE_LINE, PAGE_LINE_2,
)
from viz.site.build_figures import PLOTLY_FIGS
from viz.interactive import galaxy_formation


# galaxy_formation has 2D and 3D variants — both available for PDF use.
EXTRA_FIGS = [
    ("galaxy_formation_3d", lambda: galaxy_formation.build_galaxy_figure(dim=3)),
    ("galaxy_formation_2d", lambda: galaxy_formation.build_galaxy_figure(dim=2)),
]


# ── PDF render parameters ────────────────────────────────────────────────
# Aggressively bumped relative to tfg_plot_style.py + LaTeX scaling so
# that — even when included with `[width=0.7\textwidth]` or similar —
# the text stays clearly readable. Notebook effective sizes after LaTeX
# scaling are ~11/9/8 pt (title/label/tick); these are bigger than that
# so the web-origin figures don't look weaker.
PDF_TITLE  = 19
PDF_LABEL  = 15
PDF_TICK   = 13
PDF_LEGEND = 13
PDF_ANN    = 12

# 6.3 in matches \textwidth on A4 with 2.5 cm margins. 300 DPI is the
# standard for print quality.
PDF_WIDTH_IN  = 6.3
PDF_HEIGHT_IN = 4.0           # default for FIG_FULL-ish ratios
PDF_DPI       = 300

# Per-figure height overrides (in inches) for those whose default
# 4.0 ratio doesn't fit. Tune as needed when you preview the output.
HEIGHT_OVERRIDES = {
    "spectral_flowers":     6.0,
    "neuron_gallery":       6.0,
    "attention_atlas":      6.0,
    "emotion_cards":        6.0,
    "lens_vs_probe":        5.5,    # 3 stacked panels
    "internal_compression": 5.0,    # curves + heatmap
    "galaxy_formation_3d":  5.0,
    "galaxy_formation_2d":  4.5,
}


# Map each figure to its TFG chapter so the output filename matches the
# notebook convention: cap{N}_{name}_{lang}.png.
FIGURE_CHAPTER = {
    "bert_architecture":     2,
    "lexical_to_semantic":   5,
    "internal_compression":  4,
    # "lens_vs_probe": authored in latex_figures/generate_cap5_figures.ipynb
    # (matplotlib version preferred — kept out of web→PDF overwrite path).
    "spectral_flowers":      4,
    "spectral_landscape":    4,
    "pareto_3d":             4,
    "compression_decay":     4,
    "crystallization":       5,
    "galaxy_formation_3d":   5,
    "galaxy_formation_2d":   5,
    "heads_matrix":          5,
    "probe_constellations":  5,
    "lesion_theater":        5,
    "sunburst":              5,
    "emotional_landscape":   5,
    "confusion_evolution":   5,
    "confusion_volume":      5,
    "greedy_replay":         6,
    "finetuning_diff":       4,
}


OUT_DIR = (pathlib.Path(__file__).resolve().parents[2]
           / "latex_figures" / "figures")


def apply_pdf_mode(fig: go.Figure) -> go.Figure:
    """Bump fonts + clean chrome for print-grade rendering."""
    fig.layout.template = None

    fig.update_layout(
        title=dict(text=""),
        paper_bgcolor=PAGE_BG_PLOT,
        plot_bgcolor=PAGE_BG_PLOT,
        font=dict(family=FONT_STACK, size=PDF_LABEL, color=PAGE_INK_2),
        margin=dict(l=70, r=30, t=24, b=60),
        autosize=False,
        legend_bgcolor="rgba(255,255,255,0.92)",
        legend_bordercolor=PAGE_LINE_2,
        legend_borderwidth=0.5,
        legend_font=dict(size=PDF_LEGEND, color=PAGE_INK_2,
                         family=FONT_STACK),
    )

    # Subplot titles
    if fig.layout.annotations:
        new_anns = []
        for ann in fig.layout.annotations:
            d = ann.to_plotly_json() if hasattr(ann, "to_plotly_json") else dict(ann)
            yref = d.get("yref", "")
            y = d.get("y", 0)
            has_bg = bool(d.get("bgcolor")) and d.get("bgcolor") not in (
                "rgba(0,0,0,0)", "rgba(255,255,255,0)")
            # Drop bottom caption boxes
            if yref.endswith("paper") and y is not None and y < -0.04 and has_bg:
                continue
            is_subplot_title = (
                d.get("xref", "").endswith("paper")
                and yref.endswith("paper")
                and d.get("yanchor") == "bottom"
            )
            if is_subplot_title:
                d["font"] = dict(family=FONT_STACK, size=PDF_TITLE,
                                 color=PAGE_INK_2)
            elif "font" not in d:
                d["font"] = dict(family=FONT_STACK, size=PDF_ANN,
                                 color=PAGE_INK_2)
            new_anns.append(d)
        fig.layout.annotations = tuple(new_anns)

    # Frames carry their own annotations; clean those too.
    if fig.frames:
        new_frames = []
        for fr in fig.frames:
            d = fr.to_plotly_json()
            layout = d.get("layout", {})
            if layout.get("annotations"):
                clean = []
                for ann in layout["annotations"]:
                    yref = ann.get("yref", "")
                    y = ann.get("y", 0)
                    has_bg = bool(ann.get("bgcolor")) and ann.get("bgcolor") not in (
                        "rgba(0,0,0,0)", "rgba(255,255,255,0)")
                    if yref.endswith("paper") and y is not None and y < -0.04 and has_bg:
                        continue
                    clean.append(ann)
                layout["annotations"] = clean
                d["layout"] = layout
            new_frames.append(d)
        fig.frames = tuple(new_frames)

    fig.update_xaxes(
        gridcolor=PAGE_LINE_2,
        zerolinecolor=PAGE_LINE,
        linecolor=PAGE_LINE,
        tickcolor=PAGE_INK_3,
        tickfont=dict(family=FONT_STACK, size=PDF_TICK, color=PAGE_INK_3),
        title_font=dict(family=FONT_STACK, size=PDF_LABEL, color=PAGE_INK_2),
        showline=True, linewidth=0.5,
        ticks="outside", ticklen=4, tickwidth=0.5,
    )
    fig.update_yaxes(
        gridcolor=PAGE_LINE_2,
        zerolinecolor=PAGE_LINE,
        linecolor=PAGE_LINE,
        tickcolor=PAGE_INK_3,
        tickfont=dict(family=FONT_STACK, size=PDF_TICK, color=PAGE_INK_3),
        title_font=dict(family=FONT_STACK, size=PDF_LABEL, color=PAGE_INK_2),
        showline=True, linewidth=0.5,
        ticks="outside", ticklen=4, tickwidth=0.5,
    )

    # 3D scenes
    try:
        scene = fig.layout.scene
        if scene and (scene.xaxis or scene.yaxis or scene.zaxis):
            fig.update_layout(scene=dict(
                xaxis=dict(backgroundcolor=PAGE_BG_PLOT,
                           gridcolor=PAGE_LINE_2, color=PAGE_INK_2),
                yaxis=dict(backgroundcolor=PAGE_BG_PLOT,
                           gridcolor=PAGE_LINE_2, color=PAGE_INK_2),
                zaxis=dict(backgroundcolor=PAGE_BG_PLOT,
                           gridcolor=PAGE_LINE_2, color=PAGE_INK_2),
                bgcolor=PAGE_BG_PLOT,
            ))
    except Exception:
        pass

    # Hide sliders / Play+Pause buttons (no animation in static PNG)
    fig.layout.sliders = ()
    fig.layout.updatemenus = ()

    return fig


def export_one(name: str, builder, lang: str = "es") -> pathlib.Path | None:
    if name not in FIGURE_CHAPTER:
        # Authored elsewhere (e.g. matplotlib notebook); skip web export.
        print(f"  – {name}: skipped (not in FIGURE_CHAPTER)")
        return None
    fig = builder()
    apply_pdf_mode(fig)

    h_in = HEIGHT_OVERRIDES.get(name, PDF_HEIGHT_IN)
    width_px  = int(PDF_WIDTH_IN * PDF_DPI)
    height_px = int(h_in * PDF_DPI)

    chapter = FIGURE_CHAPTER[name]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / f"cap{chapter}_{name}_{lang}.png"
    fig.write_image(str(out_png), format="png",
                    width=width_px, height=height_px, scale=1)
    kb = out_png.stat().st_size / 1024
    print(f"  ✓ {out_png.name} ({width_px}×{height_px}px, {kb:.0f} KB)")
    return out_png


def main(only: list[str] | None = None) -> None:
    print("─" * 60)
    print("Building PDF-ready figures")
    print("─" * 60)
    print(f"Output: {OUT_DIR}")
    print()

    all_figs = list(PLOTLY_FIGS) + EXTRA_FIGS
    targets = all_figs
    if only:
        targets = [(n, b) for n, b in all_figs if n in only]
        if not targets:
            print(f"No figures match: {only}")
            return

    for name, builder in targets:
        try:
            export_one(name, builder)
        except Exception as exc:
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(only=args if args else None)
