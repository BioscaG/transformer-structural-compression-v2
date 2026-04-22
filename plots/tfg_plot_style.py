"""
tfg_plot_style.py — Unified plotting style for TFG thesis
"Anatomía Emocional de un Modelo Transformer"

v3 — Elevated style + dual language (es/en) + LaTeX-compatible typography.

Usage:
    import tfg_plot_style as sty
    sty.apply()            # default: castellano
    sty.apply(lang="en")   # english labels

    # In any plot:
    ax.set_xlabel(sty.L["f1_retention"])   # auto-translated
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from cycler import cycler

# ═════════════════════════════════════════════════════════════
# LANGUAGE SYSTEM
# ═════════════════════════════════════════════════════════════

LABELS_ES = {
    # Axes
    "f1_retention":       "Retención de F1",
    "f1_macro":           "F1 macro",
    "rank":               "Rango de truncamiento $r$",
    "rank_assigned":      "Rango asignado",
    "param_ratio":        "Ratio de parámetros (vs. baseline)",
    "layer":              "Capa",
    "component":          "Componente",
    "emotion":            "Emoción",
    "singular_index":     "Índice del valor singular",
    "singular_magnitude": "Magnitud normalizada",
    "energy_threshold":   "Umbral de energía",

    # Titles
    "t_anatomy":          "Rango mínimo $k_{95}$ por capa y componente",
    "t_spectral":         "Decaimiento espectral por tipo de componente",
    "t_uniform":          "Compresión uniforme: transición de fase",
    "t_emotion_vuln":     "F1 por emoción y nivel de compresión uniforme",
    "t_comp_sens":        "Sensibilidad por componente a rango $r{=}128$",
    "t_depth_sens":       "Sensibilidad por grupo de profundidad",
    "t_pareto":           "Frontera de Pareto: estrategias de compresión",
    "t_adaptive_ranks":   "Distribución de rangos por umbral de energía",

    # Annotations
    "collapse_zone":      "Zona de\ncolapso",
    "phase_transition":   "Transición\nde fase",
    "baseline_params":    "Baseline\nparams",
    "expansion_zone":     "Expansión\n(no compresión)",
    "break_even_attn":    "Break-even Attention (384)",
    "break_even_ffn":     "Break-even FFN (614)",
    "self_attention":     "Self-Attention",
    "feed_forward":       "Feed-Forward",

    # Legend / strategy names
    "uniform":            "Uniforme",
    "adaptive":           "Adaptativa",
    "informed":           "Informada (v1)",
    "greedy":             "Greedy",

    # Depth groups
    "early":              "Early\n(0–3)",
    "middle":             "Middle\n(4–7)",
    "late":               "Late\n(8–11)",
}

LABELS_EN = {
    # Axes
    "f1_retention":       "F1 Retention",
    "f1_macro":           "Macro F1",
    "rank":               "Truncation rank $r$",
    "rank_assigned":      "Assigned rank",
    "param_ratio":        "Parameter ratio (vs. baseline)",
    "layer":              "Layer",
    "component":          "Component",
    "emotion":            "Emotion",
    "singular_index":     "Singular value index",
    "singular_magnitude": "Normalized magnitude",
    "energy_threshold":   "Energy threshold",

    # Titles
    "t_anatomy":          "Minimum rank $k_{95}$ by layer and component",
    "t_spectral":         "Spectral decay by component type",
    "t_uniform":          "Uniform compression: phase transition",
    "t_emotion_vuln":     "F1 by emotion and uniform compression level",
    "t_comp_sens":        "Component sensitivity at rank $r{=}128$",
    "t_depth_sens":       "Sensitivity by depth group",
    "t_pareto":           "Pareto frontier: compression strategies",
    "t_adaptive_ranks":   "Rank distribution by energy threshold",

    # Annotations
    "collapse_zone":      "Collapse\nzone",
    "phase_transition":   "Phase\ntransition",
    "baseline_params":    "Baseline\nparams",
    "expansion_zone":     "Expansion\n(no compression)",
    "break_even_attn":    "Break-even Attention (384)",
    "break_even_ffn":     "Break-even FFN (614)",
    "self_attention":     "Self-Attention",
    "feed_forward":       "Feed-Forward",

    # Legend / strategy names
    "uniform":            "Uniform",
    "adaptive":           "Adaptive",
    "informed":           "Informed (v1)",
    "greedy":             "Greedy",

    # Depth groups
    "early":              "Early\n(0–3)",
    "middle":             "Middle\n(4–7)",
    "late":               "Late\n(8–11)",
}

# Active label dict — set by apply()
L = LABELS_ES


# ═════════════════════════════════════════════════════════════
# COLOR SYSTEM
# ═════════════════════════════════════════════════════════════

BG        = "#FAFAF7"
BG_FIGURE = "#FAFAF7"
GRID      = "#E5E4DF"

INK       = "#1A1A1A"
INK_2     = "#4A4A4A"
INK_3     = "#8A8A85"
SPINE     = "#C8C7C1"

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

COLORS = {
    "uniform": BLUE, "adaptive": TERRA, "informed": SAGE,
    "greedy": SAND, "baseline": INK,
    "query": BLUE, "key": BLUE_L, "value": TERRA,
    "attn_output": ROSE, "ffn_inter": TEAL, "ffn_output": TEAL_L,
    "early": SAGE, "middle": SAND, "late": TERRA,
    "positive_high": SAND, "negative_react": TERRA,
    "negative_intern": ROSE, "epistemic": BLUE,
    "other_oriented": SAGE, "low_spec": "#B0AFA8",
    "accent": TERRA, "highlight": SAND,
    "grid": GRID, "text": INK, "text_2": INK_2, "text_3": INK_3,
    "bg": BG, "spine": SPINE,
}

CMAP_DIVERGING  = "RdYlBu_r"
CMAP_SEQUENTIAL = "YlOrRd"
CMAP_COOL       = "Blues"

STRATEGY_MARKERS = {"uniform": "s", "adaptive": "o", "informed": "D", "greedy": "^"}


# ═════════════════════════════════════════════════════════════
# TYPOGRAPHY & SIZES
# ═════════════════════════════════════════════════════════════

TITLE_SIZE      = 13
TITLE_WEIGHT    = "semibold"
LABEL_SIZE      = 10.5
TICK_SIZE       = 9
LEGEND_SIZE     = 8.5
ANNOTATION_SIZE = 8
SMALL_SIZE      = 7

FIG_FULL   = (7.2, 4.5)
FIG_WIDE   = (7.2, 3.5)
FIG_HALF   = (3.4, 3.0)
FIG_SQUARE = (5.5, 5.5)
FIG_TALL   = (7.2, 6.0)

DPI         = 300
DPI_PREVIEW = 150


# ═════════════════════════════════════════════════════════════
# APPLY
# ═════════════════════════════════════════════════════════════

def apply(lang="es", use_latex=False):
    """
    Apply the thesis style globally.

    Parameters
    ----------
    lang : "es" or "en"
        Language for axis labels, titles, and annotations.
    use_latex : bool
        If True, use LaTeX rendering with newpxtext (requires LaTeX installed).
        If False, use TeX Gyre Pagella (Palatino clone, recommended).
    """
    global L
    L = LABELS_ES if lang == "es" else LABELS_EN

    if use_latex:
        font_cfg = {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["TeX Gyre Pagella"],
            "text.latex.preamble": r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{amsmath}",
            "font.size": LABEL_SIZE,
            "text.color": INK,
        }
    else:
        # TeX Gyre Pagella = free Palatino clone, matches newpxtext in LaTeX
        font_cfg = {
            "text.usetex": False,
            "font.family": "serif",
            "font.serif": ["TeX Gyre Pagella", "Palatino", "Palatino Linotype",
                           "Book Antiqua", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": LABEL_SIZE,
            "text.color": INK,
        }

    plt.rcParams.update({
        **font_cfg,

        # Figure
        "figure.facecolor": BG_FIGURE, "figure.edgecolor": "none",
        "figure.dpi": DPI_PREVIEW,
        "savefig.dpi": DPI, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2, "savefig.facecolor": BG_FIGURE,

        # Axes
        "axes.facecolor": BG, "axes.edgecolor": SPINE, "axes.linewidth": 0.5,
        "axes.titlesize": TITLE_SIZE, "axes.titleweight": TITLE_WEIGHT,
        "axes.titlepad": 14,
        "axes.labelsize": LABEL_SIZE, "axes.labelcolor": INK_2, "axes.labelpad": 8,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.prop_cycle": cycler("color", [BLUE, TERRA, SAGE, SAND, TEAL, PLUM, ROSE, BLUE_L]),

        # Grid
        "grid.color": GRID, "grid.linewidth": 0.4, "grid.alpha": 0.8,

        # Ticks
        "xtick.labelsize": TICK_SIZE, "ytick.labelsize": TICK_SIZE,
        "xtick.color": INK_3, "ytick.color": INK_3,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.4, "ytick.major.width": 0.4,

        # Legend
        "legend.fontsize": LEGEND_SIZE, "legend.frameon": True,
        "legend.framealpha": 0.92, "legend.edgecolor": SPINE,
        "legend.fancybox": False, "legend.borderpad": 0.6,

        # Lines & patches
        "lines.linewidth": 1.8, "lines.markersize": 6, "lines.markeredgewidth": 0.8,
        "patch.edgecolor": BG, "patch.linewidth": 0.6,
    })

    print(f"✓ TFG style applied — lang={lang}, latex={use_latex}")


# ═════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════

def format_pct(ax, axis="y", decimals=0):
    fmt = mticker.PercentFormatter(1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)

def annotate_point(ax, x, y, label, offset=(8, 8), fontsize=None, color=None):
    ax.annotate(label, (x, y), textcoords="offset points", xytext=offset,
                fontsize=fontsize or ANNOTATION_SIZE, color=color or INK_3,
                arrowprops=dict(arrowstyle="-", color=SPINE, lw=0.5))

def save(fig, name, chapter=4, fmt="png", lang=None):
    """Save figure. If lang is given, appends suffix: cap4_name_en.png"""
    import os
    os.makedirs("figures", exist_ok=True)
    suffix = f"_{lang}" if lang else ""
    path = f"figures/cap{chapter}_{name}{suffix}.{fmt}"
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")
    return path

def despine(ax, left=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)

def strategy_color(stype):
    return COLORS.get(stype, INK_3)

def strategy_marker(stype):
    return STRATEGY_MARKERS.get(stype, "o")

def component_color(comp):
    c = comp.lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "query": BLUE, "q": BLUE, "key": BLUE_L, "k": BLUE_L,
        "value": TERRA, "v": TERRA,
        "attn_output": ROSE, "attn_out": ROSE, "attention_output": ROSE,
        "ffn_inter": TEAL, "ffn_intermediate": TEAL, "intermediate": TEAL,
        "ffn_output": TEAL_L, "ffn_out": TEAL_L,
    }
    return mapping.get(c, INK_3)

def depth_color(group):
    g = group.lower().strip()
    if "early" in g or "tempran" in g: return SAGE
    elif "mid" in g or "media" in g: return SAND
    elif "late" in g or "tard" in g: return TERRA
    return INK_3


# ═════════════════════════════════════════════════════════════
# EMOTION ORDERING
# ═════════════════════════════════════════════════════════════

EMOTION_ORDER = [
    "gratitude", "amusement", "love", "admiration", "curiosity",
    "fear", "remorse", "joy", "sadness", "optimism",
    "caring", "anger", "surprise", "excitement", "desire",
    "annoyance", "approval", "confusion", "disapproval", "disgust",
    "disappointment", "realization", "embarrassment",
]

EMOTION_DISPLAY = {e: e.capitalize() for e in EMOTION_ORDER}
EMOTION_DISPLAY.update({
    "disappointment": "Disappoint.",
    "embarrassment": "Embarrass.",
    "disapproval": "Disapprov.",
})


# ═════════════════════════════════════════════════════════════
# DUAL-LANGUAGE FIGURE GENERATOR
# ═════════════════════════════════════════════════════════════

def generate_both(plot_func, name, chapter=4, **kwargs):
    """
    Run a plotting function twice (es + en) and save both versions.

    Usage:
        def plot_transition(ax):
            ax.set_title(sty.L["t_uniform"])
            ax.set_xlabel(sty.L["rank"])
            ax.set_ylabel(sty.L["f1_retention"])
            ...

        sty.generate_both(plot_transition, "uniform_transition")
        # → figures/cap4_uniform_transition_es.png
        # → figures/cap4_uniform_transition_en.png
    """
    for lang in ["es", "en"]:
        apply(lang=lang)
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", FIG_WIDE))
        plot_func(ax, **{k: v for k, v in kwargs.items() if k != "figsize"})
        plt.tight_layout()
        save(fig, name, chapter=chapter, lang=lang)


if __name__ == "__main__":
    apply()
    print(f"  Canvas: {BG}  |  Ink: {INK}  |  Accent: {TERRA}")
    print(f"  {len(COLORS)} named colors  |  {len(EMOTION_ORDER)} emotions")
    print(f"  Labels: {len(L)} keys in active language")
    print(f"\n  Example: sty.L['t_uniform'] → '{L['t_uniform']}'")
    apply(lang="en")
    print(f"  Example: sty.L['t_uniform'] → '{L['t_uniform']}'")
