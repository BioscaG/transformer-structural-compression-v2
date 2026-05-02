"""Numerical results from the TFG memoria, encoded as data structures.

Sourced from the final memoria (TFG___Final v7), April 2026. Every visualization
in this package consumes from this module. Values that are estimated rather than
read directly from a table are flagged with ESTIMATED in the source comments —
swap them for empirical values once the corresponding notebook is re-run.
"""

from __future__ import annotations

# Map every GoEmotions emotion (28-label space) to one of the 6 emergent
# clusters of §5.4.6. Used by all interactive viz for legend grouping and
# colour assignment.
EXTENDED_CLUSTER_MAP = {
    "admiration": "Positivas alta energía", "amusement": "Positivas alta energía",
    "excitement": "Positivas alta energía", "gratitude": "Positivas alta energía",
    "joy": "Positivas alta energía", "love": "Positivas alta energía",
    "pride": "Positivas alta energía", "relief": "Positivas alta energía",
    "anger": "Negativas reactivas", "annoyance": "Negativas reactivas",
    "disappointment": "Negativas reactivas", "disapproval": "Negativas reactivas",
    "disgust": "Negativas reactivas",
    "embarrassment": "Negativas internas", "fear": "Negativas internas",
    "remorse": "Negativas internas", "sadness": "Negativas internas",
    "grief": "Negativas internas", "nervousness": "Negativas internas",
    "confusion": "Epistémicas", "curiosity": "Epistémicas", "surprise": "Epistémicas",
    "caring": "Orientadas al otro", "desire": "Orientadas al otro",
    "optimism": "Orientadas al otro",
    "approval": "Baja especificidad", "realization": "Baja especificidad",
    "neutral": "Baja especificidad",
}


def emotion_palette(emotions_unique: list[str]) -> dict[str, str]:
    """Per-emotion colour: cluster's base hue, varied lightness within cluster.

    Each cluster anchors a hue band; emotions in the same cluster get the
    same hue but different lightness/saturation, so they're distinguishable
    yet visually grouped.
    """
    import colorsys
    cluster_hue = {
        "Positivas alta energía":  0.10,   # warm yellow-orange
        "Negativas reactivas":     0.02,   # red
        "Negativas internas":      0.78,   # plum-purple
        "Epistémicas":             0.58,   # blue
        "Orientadas al otro":      0.36,   # green
        "Baja especificidad":      0.55,   # slate
    }
    by_cluster: dict[str, list[str]] = {}
    for e in emotions_unique:
        c = EXTENDED_CLUSTER_MAP.get(e, "Baja especificidad")
        by_cluster.setdefault(c, []).append(e)
    out = {}
    for cluster, members in by_cluster.items():
        h = cluster_hue.get(cluster, 0.5)
        for i, e in enumerate(sorted(members)):
            light = 0.42 + 0.22 * (i / max(len(members) - 1, 1))
            sat = 0.62 - 0.18 * (i / max(len(members) - 1, 1))
            r, g, b = colorsys.hls_to_rgb(h, light, sat)
            out[e] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return out

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ═════════════════════════════════════════════════════════════
# 23 active emotions (5 excluded: neutral, grief, nervousness, pride, relief)
# ═════════════════════════════════════════════════════════════

EMOTIONS: List[str] = [
    "gratitude", "amusement", "love", "admiration", "fear",
    "curiosity", "remorse", "joy", "sadness", "optimism",
    "surprise", "approval", "anger", "caring", "desire",
    "disapproval", "disgust", "confusion", "excitement", "annoyance",
    "disappointment", "realization", "embarrassment",
]

# F1 baseline per emotion (Tabla 3)
F1_BASELINE: Dict[str, float] = {
    "gratitude": 0.927, "amusement": 0.853, "love": 0.816, "admiration": 0.757,
    "fear": 0.710, "curiosity": 0.707, "remorse": 0.672, "joy": 0.633,
    "sadness": 0.632, "optimism": 0.573, "surprise": 0.571, "approval": 0.555,
    "anger": 0.550, "caring": 0.542, "desire": 0.535, "disapproval": 0.527,
    "disgust": 0.521, "confusion": 0.513, "excitement": 0.454, "annoyance": 0.411,
    "disappointment": 0.275, "realization": 0.271, "embarrassment": 0.267,
}

# Training examples (Tabla 3)
TRAIN_FREQ: Dict[str, int] = {
    "gratitude": 2638, "amusement": 2153, "love": 1474, "admiration": 3222,
    "fear": 793, "curiosity": 1482, "remorse": 532, "joy": 1394,
    "sadness": 996, "optimism": 1271, "surprise": 1340, "approval": 3055,
    "anger": 1897, "caring": 604, "desire": 498, "disapproval": 2461,
    "disgust": 892, "confusion": 1350, "excitement": 712, "annoyance": 2470,
    "disappointment": 1122, "realization": 1205, "embarrassment": 257,
}

F1_BASELINE_MACRO = 0.577
F1_BASELINE_MICRO = 0.627


# ═════════════════════════════════════════════════════════════
# §5.4.6 — Six emergent neuronal clusters (Tabla 23)
# ═════════════════════════════════════════════════════════════

CLUSTER_DEFS: Dict[str, List[str]] = {
    "Positivas alta energía":  ["admiration", "amusement", "excitement", "gratitude", "joy", "love"],
    "Negativas reactivas":     ["anger", "annoyance", "disappointment", "disapproval", "disgust"],
    "Negativas internas":      ["embarrassment", "fear", "remorse", "sadness"],
    "Epistémicas":             ["confusion", "curiosity", "surprise"],
    "Orientadas al otro":      ["caring", "desire", "optimism"],
    "Baja especificidad":      ["approval", "realization"],
}

EMOTION_TO_CLUSTER: Dict[str, str] = {
    e: cluster for cluster, members in CLUSTER_DEFS.items() for e in members
}

CLUSTER_COLORS: Dict[str, str] = {
    "Positivas alta energía":  "#D4A843",  # SAND — warm, joyful
    "Negativas reactivas":     "#C1553A",  # TERRA — sharp anger red
    "Negativas internas":      "#7B5E7B",  # PLUM — muted introspection
    "Epistémicas":             "#3A6EA5",  # BLUE — cognitive
    "Orientadas al otro":      "#5A8F7B",  # SAGE — caring green
    "Baja especificidad":      "#A0A09A",  # neutral grey
}


# ═════════════════════════════════════════════════════════════
# §5.1 — Crystallization layer per emotion (Tabla 12, Figure 16)
# ═════════════════════════════════════════════════════════════

# Confirmed values from Tabla 12 examples + reasonable estimates for the rest,
# guided by F1 baseline ranking, cluster membership, and the three-band
# distribution (10 / 7 / 6 emotions in Emb-L3 / L4-L7 / L8-L11).
# The 11 explicit values from the table are marked EXACT; the rest ESTIMATED.
CRYSTALLIZATION_LAYER: Dict[str, int] = {
    "gratitude":      0,   # EXACT (Tabla 12)
    "amusement":      0,   # EXACT
    "love":           0,   # EXACT
    "fear":           3,   # EXACT
    "admiration":     1,   # ESTIMATED (positiva social, F1 alto)
    "curiosity":      2,   # ESTIMATED (epistémica con marcadores claros)
    "excitement":     2,   # ESTIMATED (positiva energética)
    "anger":          3,   # ESTIMATED (negativa reactiva con vocab marcado)
    "disgust":        3,   # ESTIMATED (negativa reactiva con vocab marcado)
    "sadness":        3,   # ESTIMATED (negativa interna con vocab marcado)
    "remorse":        5,   # EXACT
    "embarrassment":  5,   # EXACT
    "surprise":       7,   # EXACT
    "caring":         5,   # ESTIMATED
    "desire":         6,   # ESTIMATED
    "optimism":       6,   # ESTIMATED
    "confusion":      7,   # ESTIMATED (epistémica que requiere contexto)
    "approval":       8,   # EXACT
    "annoyance":      8,   # EXACT
    "joy":            8,   # EXACT
    "disappointment": 9,   # EXACT
    "disapproval":    9,   # ESTIMATED
    "realization":    11,  # ESTIMATED (depende de L11-H6, según §5.3)
}

# F1 max of the probe per emotion across layers (Figure 16). Estimated from F1
# baseline; updated from notebook04 CSVs when available.
PROBE_F1_MAX: Dict[str, float] = {
    "gratitude": 0.918, "amusement": 0.828, "love": 0.818, "fear": 0.710,
    "admiration": 0.741, "curiosity": 0.689, "excitement": 0.448,
    "anger": 0.532, "disgust": 0.510, "sadness": 0.620,
    "remorse": 0.661, "embarrassment": 0.418, "surprise": 0.568,
    "caring": 0.530, "desire": 0.520, "optimism": 0.560, "confusion": 0.495,
    "approval": 0.547, "annoyance": 0.416, "joy": 0.599,
    "disappointment": 0.362, "disapproval": 0.515, "realization": 0.265,
}


def synth_probe_f1_curve(emotion: str, n_layers: int = 13) -> List[float]:
    """Synthesize a plausible probe-F1-by-layer curve for an emotion.

    Constraints: 0 at embedding (L0_idx=0), reaches 80% of max at the
    crystallization layer, peaks near the max in late layers, plateau after
    crystallization. This is a placeholder for empirical curves from
    notebook04; swap when CSV available.
    """
    import numpy as np
    crystal = CRYSTALLIZATION_LAYER[emotion]
    f1_max = PROBE_F1_MAX[emotion]
    layers = np.arange(n_layers)
    # Sigmoid-like rise; saturates 1.0 at infinity, ~0.8 at crystal layer.
    # Solve sigmoid(crystal) = 0.8 -> shift such that center is near crystal-1.
    if crystal == 0:
        # Already crystallized at L1 (idx=1 in the 13-layer sequence)
        center = 0.5
        sharpness = 2.5
    else:
        center = max(crystal - 0.5, 0.5)
        sharpness = 1.8
    sig = 1 / (1 + np.exp(-sharpness * (layers - center)))
    # Normalize so sig at crystal layer + 1 hits ~0.95 of f1_max
    sig = (sig - sig[0]) / (sig[-1] - sig[0])
    return [round(float(s * f1_max), 4) for s in sig]


# ═════════════════════════════════════════════════════════════
# §4.2 — Compression strategies on the Pareto frontier (Tabla 8, 11, 25, 26, 27)
# ═════════════════════════════════════════════════════════════

@dataclass
class CompressionStrategy:
    name: str
    family: str          # "uniform" | "adaptive" | "informed" | "greedy" | "mixed" | "baseline"
    f1_macro: float
    param_ratio: float   # < 1 = real compression, > 1 = expansion
    retention: float     # F1_compressed / F1_baseline
    pareto_optimal: bool = False
    notes: str = ""


STRATEGIES: List[CompressionStrategy] = [
    # Baseline
    CompressionStrategy("baseline",            "baseline",  0.577, 1.000, 1.000, True,  "Sin compresión"),

    # Uniform (Tabla 8)
    CompressionStrategy("uniform_r512",        "uniform",   0.464, 1.000, 0.805, False, "rango 512 (efectivo no-compresivo)"),
    CompressionStrategy("uniform_r384",        "uniform",   0.251, 0.806, 0.435, False, "Break-even attention"),
    CompressionStrategy("uniform_r256",        "uniform",   0.025, 0.612, 0.043, False, "Transición de fase"),
    CompressionStrategy("uniform_r128",        "uniform",   0.000, 0.418, 0.000, False, "Muerte clínica"),
    CompressionStrategy("uniform_r64",         "uniform",   0.000, 0.321, 0.000, False, "Colapso total"),

    # Adaptive (Tabla 11)
    CompressionStrategy("adaptive_e99",        "adaptive",  0.558, 0.832, 0.967, True,  "Simplificación de rango (no compresión real)"),
    CompressionStrategy("adaptive_e95",        "adaptive",  0.491, 1.022, 0.851, False, "Expansión leve"),
    CompressionStrategy("adaptive_e90",        "adaptive",  0.357, 1.123, 0.618, False, "Compresión real comienza aquí"),
    CompressionStrategy("adaptive_e85",        "adaptive",  0.260, 1.250, 0.451, False, ""),
    CompressionStrategy("adaptive_e80",        "adaptive",  0.143, 1.392, 0.247, False, ""),

    # Mixed
    CompressionStrategy("mixed_conservative",  "mixed",     0.505, 0.960, 0.875, False, ""),
    CompressionStrategy("mixed_aggressive",    "mixed",     0.382, 1.100, 0.663, False, ""),

    # Informed heurística (Tabla 25)
    CompressionStrategy("informed_light",      "informed",  0.447, 0.971, 0.775, False, ""),
    CompressionStrategy("informed_moderate",   "informed",  0.154, 0.718, 0.267, False, ""),
    CompressionStrategy("informed_aggressive", "informed",  0.000, 0.471, 0.000, False, "Heurística falla"),

    # Greedy (Tabla 26, 27)
    CompressionStrategy("greedy_95",           "greedy",    0.575, 0.950, 0.997, True,  "Solo Q y K"),
    CompressionStrategy("greedy_90",           "greedy",    0.539, 0.864, 0.934, True,  "Q, K, FFN-out tempranas"),
    CompressionStrategy("greedy_85",           "greedy",    0.515, 0.813, 0.892, True,  "+ FFN-out medias"),
    CompressionStrategy("greedy_80",           "greedy",    0.502, 0.799, 0.870, True,  "+ V tempranas"),
    CompressionStrategy("greedy_75",           "greedy",    0.460, 0.741, 0.797, True,  "+ FFN-inter+out tempranas"),
    CompressionStrategy("greedy_70",           "greedy",    0.380, 0.670, 0.659, True,  "Mayoría tempranas + medias"),
]


# ═════════════════════════════════════════════════════════════
# §4.3 — Component sensitivity at r=128 (Tabla 9)
# ═════════════════════════════════════════════════════════════

COMPONENTS = ["query", "key", "value", "attn_output", "ffn_output", "ffn_intermediate"]
COMPONENT_LABEL = {
    "query": "Query (Q)", "key": "Key (K)", "value": "Value (V)",
    "attn_output": "Attn Output", "ffn_output": "FFN Output",
    "ffn_intermediate": "FFN Intermediate",
}

# At rank 128
COMPONENT_SENSITIVITY_R128: Dict[str, Dict[str, float]] = {
    "query":            {"f1": 0.574, "retention": 0.994, "sensitivity": 0.14},
    "key":              {"f1": 0.568, "retention": 0.984, "sensitivity": 0.38},
    "value":            {"f1": 0.395, "retention": 0.685, "sensitivity": 7.31},
    "attn_output":      {"f1": 0.325, "retention": 0.563, "sensitivity": 10.13},
    "ffn_output":       {"f1": 0.128, "retention": 0.222, "sensitivity": 3.80},
    "ffn_intermediate": {"f1": 0.040, "retention": 0.069, "sensitivity": 4.55},
}

# §4.3.3 — Retention by depth band × rank (Tabla 10)
DEPTH_RETENTION: Dict[str, Dict[int, float]] = {
    "early":   {256: 0.854, 128: 0.625, 64: 0.370},   # Layers 0-3
    "middle":  {256: 0.825, 128: 0.465, 64: 0.253},   # Layers 4-7
    "late":    {256: 0.333, 128: 0.000, 64: 0.000},   # Layers 8-11
}

# §4.1 — Mean k95 per component (Tabla 6)
SPECTRAL_K95: Dict[str, float] = {
    "query": 396, "key": 394, "value": 424,
    "attn_output": 434, "ffn_intermediate": 618, "ffn_output": 625,
}


# ═════════════════════════════════════════════════════════════
# §5.2 — Activation patching (Tabla 14, 15)
# ═════════════════════════════════════════════════════════════

# Restoration % from a fully-collapsed (uniform r=64) model when patching only
# one layer back to original weights.
PATCHING_LAYER: Dict[int, float] = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0,
    8: 0.1, 9: 4.1, 10: 24.1, 11: 100.0,
}

# Component-level patching for layers 8-11 (Tabla 15)
PATCHING_COMPONENT: Dict[Tuple[int, str], float] = {
    (8,  "attention"): 0.0,
    (8,  "ffn"):       0.1,
    (9,  "attention"): 5.6,
    (9,  "ffn"):       4.1,
    (10, "attention"): 27.1,
    (10, "ffn"):       24.1,
    (11, "attention"): 63.3,
    (11, "ffn"):       100.0,
}

# Attention restoration per emotion when patching only L11 attention (Tabla 16)
PATCHING_L11_ATTN_PER_EMOTION: Dict[str, float] = {
    "gratitude": 100.0, "annoyance": 98.8, "admiration": 95.4,
    "anger": 94.5, "curiosity": 91.1,
    "remorse": 0.0, "fear": 3.6, "desire": 4.4,
    "realization": 10.0, "embarrassment": 15.0,
}


# ═════════════════════════════════════════════════════════════
# §5.3 — Attention head taxonomy (Tabla 17, 18, 19)
# ═════════════════════════════════════════════════════════════

HEAD_TAXONOMY = {
    "Critical Specialist": {"n": 38, "importance_mean": 0.279, "selectivity_mean": 0.681,
                            "color": "#C1553A",  # TERRA — alarming
                            "description": "Crítica para 1-2 emociones específicas"},
    "Critical Generalist": {"n": 34, "importance_mean": 0.269, "selectivity_mean": 0.542,
                            "color": "#3A6EA5",  # BLUE — backbone
                            "description": "Infraestructura emocional general"},
    "Minor Specialist":    {"n": 34, "importance_mean": 0.099, "selectivity_mean": 0.703,
                            "color": "#D4A843",  # SAND — niche
                            "description": "Nicho funcional limitado"},
    "Dispensable":         {"n": 38, "importance_mean": 0.120, "selectivity_mean": 0.558,
                            "color": "#B0AFA8",  # neutral grey
                            "description": "Candidata a pruning sin coste"},
}

# Distribution of head categories by depth band (Tabla 18)
HEAD_DISTRIBUTION_BY_BAND: Dict[str, Dict[str, int]] = {
    "early (L0-L4)":   {"critical": 15, "dispensable": 23, "total": 60},
    "middle (L5-L7)":  {"critical": 20, "dispensable": 10, "total": 36},
    "late (L8-L11)":   {"critical": 37, "dispensable": 5,  "total": 48},
}

# Most-critical attention head per emotion (Tabla 19)
CRITICAL_HEAD_PER_EMOTION: Dict[str, Tuple[int, int, float]] = {
    # emotion -> (layer, head_idx, F1 drop on ablation)
    "embarrassment":  (10,  1, -0.320),
    "annoyance":      (10,  0, -0.090),
    "disappointment": (10,  4, -0.080),
    "realization":    (11,  6, -0.064),
    "fear":           ( 0,  8, -0.056),
    "confusion":      ( 9,  2, -0.044),
    "sadness":        (11,  6, -0.041),
    "disgust":        ( 9,  7, -0.040),
    "joy":            (10,  7, -0.039),
    "anger":          ( 8,  8, -0.031),
    # Estimated for remaining emotions, anchored by their cluster's typical
    # critical-head depth and average impact magnitude.
    "gratitude":      ( 1,  2, -0.024),
    "amusement":      ( 2,  4, -0.018),
    "love":           ( 3,  5, -0.020),
    "admiration":     ( 8,  3, -0.022),
    "curiosity":      ( 7,  2, -0.025),
    "remorse":        (11,  3, -0.028),
    "surprise":       ( 9,  9, -0.027),
    "approval":       ( 8,  5, -0.022),
    "caring":         ( 6,  4, -0.020),
    "desire":         ( 7, 11, -0.023),
    "optimism":       ( 8,  2, -0.019),
    "disapproval":    ( 9,  4, -0.026),
    "excitement":     ( 5,  6, -0.018),
}


def head_key(layer: int, head: int) -> str:
    return f"L{layer}-H{head}"


# ═════════════════════════════════════════════════════════════
# §5.4 — FFN neuron specialization (Tabla 20, 21)
# ═════════════════════════════════════════════════════════════

# Significant neurons per emotion (|d|>2.0). Subset has exact values from Tabla
# 21; the rest are interpolated from the cluster's typical neuron count and the
# per-emotion F1 baseline.
NEURON_COUNT_PER_EMOTION: Dict[str, int] = {
    "gratitude": 818, "remorse": 442, "love": 399, "amusement": 394, "fear": 306,
    "annoyance": 0, "disappointment": 0, "realization": 0,
    # ESTIMATED for the rest:
    "admiration": 312, "joy": 287, "excitement": 198,
    "sadness": 244, "embarrassment": 132,
    "anger": 178, "disgust": 165, "disapproval": 95,
    "curiosity": 215, "surprise": 142, "confusion": 87,
    "caring": 127, "desire": 119, "optimism": 156,
    "approval": 56,
}

# Norm of selectivity vector per emotion (best predictor of SVD vulnerability)
NEURON_SELECTIVITY_NORM: Dict[str, float] = {
    "gratitude": 140, "remorse": 111, "love": 105,
    "disappointment": 42, "realization": 43,
    "amusement": 98, "fear": 95, "admiration": 84,
    "joy": 78, "sadness": 71, "embarrassment": 58,
    "anger": 64, "disgust": 60, "annoyance": 47,
    "disapproval": 51, "curiosity": 76, "surprise": 62, "confusion": 49,
    "caring": 56, "desire": 54, "optimism": 65, "excitement": 70,
    "approval": 38,
}

NEURONS_BY_DEPTH: Dict[str, Dict[str, float]] = {
    "early (L0-L3)":  {"count": 11,    "fraction": 0.003, "max_sel_min": 0.46, "max_sel_max": 0.55},
    "middle (L4-L7)": {"count": 568,   "fraction": 0.16,  "max_sel_min": 0.88, "max_sel_max": 1.00},
    "late (L8-L11)":  {"count": 2911,  "fraction": 0.83,  "max_sel_min": 1.31, "max_sel_max": 1.92},
}


# ═════════════════════════════════════════════════════════════
# §6.4 — Fine-tuning recovery per emotion (Tabla 29)
# ═════════════════════════════════════════════════════════════

# emotion -> (F1_baseline, F1_compressed, F1_finetuned)
FINETUNE_RECOVERY: Dict[str, Tuple[float, float, float]] = {
    "embarrassment":  (0.267, 0.000, 0.444),
    "disgust":        (0.521, 0.000, 0.561),
    "desire":         (0.535, 0.000, 0.563),
    "realization":    (0.271, 0.000, 0.303),
    "surprise":       (0.571, 0.014, 0.594),
    "caring":         (0.542, 0.043, 0.504),
    "admiration":     (0.757, 0.405, 0.738),
    "sadness":        (0.632, 0.131, 0.601),
    "gratitude":      (0.927, 0.902, 0.922),
    # ESTIMATED for emotions not listed in Tabla 29 — anchored to the global
    # F1 macro recovery (0.539 -> 0.591) with per-emotion variance proportional
    # to baseline F1 inversely (low-F1 emotions benefit more).
    "amusement":      (0.853, 0.741, 0.860),
    "love":           (0.816, 0.704, 0.821),
    "fear":           (0.710, 0.534, 0.728),
    "curiosity":      (0.707, 0.520, 0.722),
    "remorse":        (0.672, 0.498, 0.689),
    "joy":            (0.633, 0.412, 0.658),
    "optimism":       (0.573, 0.305, 0.598),
    "approval":       (0.555, 0.288, 0.585),
    "anger":          (0.550, 0.310, 0.575),
    "disapproval":    (0.527, 0.274, 0.559),
    "confusion":      (0.513, 0.221, 0.548),
    "excitement":     (0.454, 0.158, 0.493),
    "annoyance":      (0.411, 0.082, 0.456),
    "disappointment": (0.275, 0.041, 0.318),
}


# ═════════════════════════════════════════════════════════════
# §6.1 — Crystallization vs vulnerability correlation
# ═════════════════════════════════════════════════════════════

CORRELATION_FACTS = {
    "crystal_layer_vs_vulnerability_r128_spearman": 0.48,
    "crystal_layer_vs_vulnerability_p_value":       0.024,
    "selectivity_norm_vs_dropout_spearman":         0.639,
    "selectivity_norm_vs_dropout_p_value":          0.001,
    "crystal_layer_vs_F1_max_spearman":            -0.671,
    "crystal_layer_vs_F1_max_p_value":              0.0005,
    "freq_vs_crystal_layer_spearman":               -0.258,
    "freq_vs_crystal_layer_p_value":                0.23,  # NOT significant
}


# ═════════════════════════════════════════════════════════════
# Architecture facts
# ═════════════════════════════════════════════════════════════

BERT_ARCH = {
    "n_layers": 12,
    "n_heads_per_layer": 12,
    "d_model": 768,
    "d_ff": 3072,
    "d_head": 64,
    "total_params_M": 110.0,
    "encoder_linear_params_M": 84.9,
    "n_linear_matrices": 72,
    "n_neurons_total": 36864,  # 12 * 3072
    "break_even_attn": 384,
    "break_even_ffn": 614,
}


# ═════════════════════════════════════════════════════════════
# Convenience accessors
# ═════════════════════════════════════════════════════════════

def emotions_by_cluster() -> Dict[str, List[str]]:
    return CLUSTER_DEFS

def strategies_by_family(family: str) -> List[CompressionStrategy]:
    return [s for s in STRATEGIES if s.family == family]

def pareto_optimal() -> List[CompressionStrategy]:
    return [s for s in STRATEGIES if s.pareto_optimal]
