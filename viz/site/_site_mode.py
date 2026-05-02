"""Site-mode rendering: convert any figure (Plotly or custom HTML+JS)
into an embeddable artifact for the editorial site.

Two paths:

  - For Plotly figures (build_X_figure -> go.Figure): full retheme via
    apply_site_mode + write_site_figure. Strips title, transparent
    paper, Geist typography, hairline grids.

  - For custom HTML+JS figures (build_html writes a self-contained
    page): post-process via inject_site_mode_into_html. Injects body
    transparency + a postMessage height-reporter so the parent page
    can size the iframe.

Both paths produce HTML files that:
  - Are transparent at the body level (white card shows through).
  - Report their rendered height to the parent via postMessage('fig-height').
  - Load Geist + Fraunces from Google Fonts.
"""

from __future__ import annotations

import pathlib
import re
import textwrap

import plotly.graph_objects as go


# ── Page palette (in sync with viz/site/index.html) ──────────────────────
PAGE_BG       = "#F7F6F2"
PAGE_BG_PLOT  = "#FFFFFF"
PAGE_INK      = "#141413"
PAGE_INK_2    = "#4A4A47"
PAGE_INK_3    = "#8A8A82"
PAGE_LINE     = "#D8D7D1"
PAGE_LINE_2   = "#E8E7E2"
PAGE_ACCENT   = "#1F4E6C"

# Match the thesis matplotlib style (viz/style.py): Pagella for everything.
# Pagella is the GUST e-foundry's free Palatino clone (used by LaTeX). When
# unavailable, the browser/Plotly falls through to Palatino → Book Antiqua →
# DejaVu Serif. All four are visually near-identical.
FONT_STACK = ('"TeX Gyre Pagella", "Palatino", "Palatino Linotype", '
              '"Book Antiqua", "Iowan Old Style", "DejaVu Serif", serif')
FONT_MONO  = '"Geist Mono", "JetBrains Mono", "SF Mono", Menlo, monospace'

# Sizes matching latex_figures/tfg_plot_style.py + small bump for screen
# legibility (Plotly renders a bit smaller than matplotlib at equal nominal
# size). Source values: TITLE 13, LABEL 10.5, TICK 9, LEGEND 8.5, ANN 8.
SZ_TITLE      = 13
SZ_LABEL      = 11
SZ_TICK       = 10
SZ_LEGEND     = 10
SZ_ANN        = 10

GOOGLE_FONTS_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com" />'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Fraunces:opsz,wght,SOFT@9..144,300..900,30..100&'
    'family=Geist:wght@300..700&'
    'family=Geist+Mono:wght@300..600&display=swap" rel="stylesheet" />'
)

# Heights are author-declared in viz/site/sections.py (FIG_HEIGHTS).
# No runtime measurement → no feedback loops, predictable layout.
POSTMESSAGE_SCRIPT = ""

BODY_RESET_CSS = """
<style id="site-mode-reset">
  html, body {
    margin: 0 !important;
    padding: 0 !important;
    height: auto !important;
    background: transparent !important;
    overflow-x: hidden !important;
  }
  body {
    font-family: """ + FONT_STACK + """ !important;
    color: """ + PAGE_INK_2 + """ !important;
  }
  /* Make scrollbars unobtrusive if any appear */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-thumb { background: """ + PAGE_LINE + """; border-radius: 3px; }
</style>
"""


# ── Plotly path ──────────────────────────────────────────────────────────

def apply_site_mode(fig: go.Figure) -> go.Figure:
    """Retheme a Plotly figure for the editorial site.

    - Drop redundant title (page handles it).
    - Transparent paper, white plot area.
    - Geist typography, ink-2 default text color.
    - Hairline grids matching the page line-2.
    - Width is responsive; height preserved from original layout
      (figures are designed for specific aspect ratios).
    """
    fig.layout.template = None  # drop the bound thesis template

    fig.update_layout(
        title=dict(text=""),
        # White paper (not transparent) so PNG exports come out with a
        # solid white background. Visually identical on the page because
        # the card behind the iframe is also #FFFFFF.
        paper_bgcolor=PAGE_BG_PLOT,
        plot_bgcolor=PAGE_BG_PLOT,
        font=dict(family=FONT_STACK, size=SZ_LABEL, color=PAGE_INK_2),
        margin=dict(l=64, r=28, t=24, b=56),
        autosize=True,
        width=None,
        hoverlabel=dict(
            bgcolor=PAGE_BG_PLOT,
            bordercolor=PAGE_LINE,
            font=dict(family=FONT_STACK, size=SZ_LABEL, color=PAGE_INK),
        ),
        legend_bgcolor="rgba(255,255,255,0.92)",
        legend_bordercolor=PAGE_LINE_2,
        legend_borderwidth=0.5,
        legend_font=dict(size=SZ_LEGEND, color=PAGE_INK_2, family=FONT_STACK),
    )

    # Annotations cleanup:
    #   - Subplot titles → quiet monospace, ink-3.
    #   - Bottom caption boxes (those with bgcolor and below the chart)
    #     are dropped because they collide with the slider area.
    def _clean_annotations(anns):
        out = []
        for ann in anns:
            d = ann.to_plotly_json() if hasattr(ann, "to_plotly_json") else dict(ann)
            yref = d.get("yref", "")
            y = d.get("y", 0)
            has_bg = bool(d.get("bgcolor")) and d.get("bgcolor") not in (
                "rgba(0,0,0,0)", "rgba(255,255,255,0)")
            # Drop bottom caption boxes
            if yref.endswith("paper") and y is not None and y < -0.04 and has_bg:
                continue
            # Style subplot titles
            is_subplot_title = (
                d.get("xref", "").endswith("paper")
                and yref.endswith("paper")
                and d.get("yanchor") == "bottom"
            )
            if is_subplot_title:
                d["font"] = dict(family=FONT_STACK, size=SZ_TITLE,
                                 color=PAGE_INK_2)
            out.append(d)
        return out

    if fig.layout.annotations:
        fig.layout.annotations = tuple(_clean_annotations(fig.layout.annotations))

    # Frames carry their own annotations (per animation step) — clean those too.
    if fig.frames:
        new_frames = []
        for fr in fig.frames:
            d = fr.to_plotly_json()
            layout = d.get("layout", {})
            if layout.get("annotations"):
                layout["annotations"] = _clean_annotations(layout["annotations"])
                d["layout"] = layout
            new_frames.append(d)
        fig.frames = tuple(new_frames)

    fig.update_xaxes(
        gridcolor=PAGE_LINE_2,
        zerolinecolor=PAGE_LINE,
        linecolor=PAGE_LINE,
        tickcolor=PAGE_INK_3,
        tickfont=dict(family=FONT_STACK, size=SZ_TICK, color=PAGE_INK_3),
        title_font=dict(family=FONT_STACK, size=SZ_LABEL, color=PAGE_INK_2),
        showline=True, linewidth=0.5,
        ticks="outside", ticklen=3, tickwidth=0.4,
    )
    fig.update_yaxes(
        gridcolor=PAGE_LINE_2,
        zerolinecolor=PAGE_LINE,
        linecolor=PAGE_LINE,
        tickcolor=PAGE_INK_3,
        tickfont=dict(family=FONT_STACK, size=SZ_TICK, color=PAGE_INK_3),
        title_font=dict(family=FONT_STACK, size=SZ_LABEL, color=PAGE_INK_2),
        showline=True, linewidth=0.5,
        ticks="outside", ticklen=3, tickwidth=0.4,
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

    # ── Sliders + Play/Pause buttons ─────────────────────────────────────
    # Plotly's defaults clash with the editorial aesthetic. Also: the
    # slider's `currentvalue` text routinely collides with chart-level
    # annotations because both end up at the bottom margin. We hide it
    # everywhere — the step tick labels already convey the same info.
    if fig.layout.sliders:
        new_sliders = []
        for sl in fig.layout.sliders:
            d = sl.to_plotly_json()
            d["bgcolor"]         = PAGE_LINE_2
            d["bordercolor"]     = PAGE_LINE
            d["borderwidth"]     = 0
            d["activebgcolor"]   = PAGE_ACCENT
            d["tickcolor"]       = PAGE_INK_3
            d["ticklen"]         = 4
            d["minorticklen"]    = 0
            d["font"] = dict(family=FONT_STACK, size=SZ_ANN,
                             color=PAGE_INK_3)
            d["currentvalue"] = dict(visible=False)
            d["pad"] = dict(t=12, b=8, l=12, r=12)
            d["x"]   = 0.06
            d["len"] = 0.88
            d["xanchor"] = "left"
            d["yanchor"] = "top"
            new_sliders.append(d)
        fig.layout.sliders = tuple(new_sliders)

    if fig.layout.updatemenus:
        new_menus = []
        for um in fig.layout.updatemenus:
            d = um.to_plotly_json()
            d["bgcolor"]       = "rgba(0,0,0,0)"
            d["bordercolor"]   = PAGE_LINE
            d["borderwidth"]   = 0.5
            d["font"] = dict(family=FONT_STACK, size=SZ_ANN,
                             color=PAGE_INK_2)
            d["pad"] = dict(t=8, b=8, l=10, r=10)
            new_menus.append(d)
        fig.layout.updatemenus = tuple(new_menus)

    return fig


def write_site_figure(fig: go.Figure, out_path: pathlib.Path) -> pathlib.Path:
    """Render a Plotly figure for the site (full retheme + postMessage)."""
    apply_site_mode(fig)

    fig_html = fig.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={
            "displayModeBar": "hover",
            "displaylogo": False,
            "responsive": True,
            "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
            "toImageButtonOptions": {
                "format": "png",
                "filename": "figura",
                "height": 1200,
                "width": 1800,
                "scale": 3,
            },
        },
        div_id="site-fig",
    )

    template = textwrap.dedent("""\
    <!DOCTYPE html>
    <html lang="es">
    <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Figura</title>
    __FONTS__
    <style>
      html, body {
        margin: 0; padding: 0;
        height: 100%; width: 100%;
        background: transparent;
        font-family: __FONT__;
        color: __INK2__;
        overflow: hidden;
      }
      #site-fig { width: 100%; }
      .modebar { background: transparent !important; }
      .modebar-btn path { fill: __INK3__ !important; }
      .modebar-btn:hover path { fill: __ACCENT__ !important; }
      .gtitle { display: none !important; }
    </style>
    </head>
    <body>
    __FIG__
    __SCRIPT__
    </body>
    </html>
    """)

    html = (template
            .replace("__FONTS__", GOOGLE_FONTS_LINK)
            .replace("__FONT__", FONT_STACK)
            .replace("__INK2__", PAGE_INK_2)
            .replace("__INK3__", PAGE_INK_3)
            .replace("__ACCENT__", PAGE_ACCENT)
            .replace("__FIG__", fig_html)
            .replace("__SCRIPT__", POSTMESSAGE_SCRIPT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ── HTML+JS custom path ───────────────────────────────────────────────────

def inject_site_mode_into_html(src: pathlib.Path,
                               dst: pathlib.Path) -> pathlib.Path:
    """Read a self-contained HTML viz, inject body-reset CSS + postMessage
    height reporter, and write the result to dst.

    The original visual content is preserved — we only add a CSS reset
    that makes the body transparent (so the editorial card behind it
    shows through), and a script that reports the body's rendered height
    so the parent iframe can auto-size.
    """
    html = src.read_text(encoding="utf-8")

    # Inject Google Fonts link in <head>
    if "fonts.googleapis.com/css2?family=Geist" not in html:
        html = re.sub(r"</head>",
                      GOOGLE_FONTS_LINK + "\n</head>",
                      html, count=1)

    # Inject body-reset CSS at end of <head>
    html = re.sub(r"</head>",
                  BODY_RESET_CSS + "\n</head>",
                  html, count=1)

    # Inject postMessage script before </body>
    html = re.sub(r"</body>",
                  POSTMESSAGE_SCRIPT + "\n</body>",
                  html, count=1)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(html, encoding="utf-8")
    return dst
