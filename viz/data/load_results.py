"""Central loader for the empirical CSVs from the notebooks.

Single entry point: `load_all()` returns a dict bundle that every
visualization consumes in place of the (partially synthetic)
`viz/thesis_data.py`. Falls back gracefully when individual files are
missing.

Project layout consumed:
    results/
    ├── csvs/
    │   ├── notebook2/  spectral analysis
    │   ├── notebook3/  compression sensitivity
    │   ├── notebook4/  probing
    │   ├── notebook5/  activation patching
    │   ├── notebook6/  head analysis
    │   ├── notebook7/  neuron analysis
    │   ├── notebook8/  emotional map (lesion study)
    │   └── notebook9/  informed compression + greedy
    ├── checkpoints/
    │   └── 23emo-final/   ← THE fine-tuned BERT used by extractors
    └── figures/           (PNGs from notebooks; not consumed here)
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results"
CSVS = RESULTS / "csvs"
MODEL_CHECKPOINT = RESULTS / "checkpoints" / "23emo-final"


# 23 active emotions, in the order saved by the user's notebook
EMOTIONS_23 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "joy", "love",
    "optimism", "realization", "remorse", "sadness", "surprise",
]


def _maybe_read(path: pathlib.Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [skip] {path.relative_to(PROJECT_ROOT)} not found")
        return None
    return pd.read_csv(path, **kwargs)


def _emotion_names_from_checkpoint() -> list[str]:
    p = MODEL_CHECKPOINT / "emotion_names.json"
    if p.exists():
        return json.loads(p.read_text())["emotion_names"]
    return EMOTIONS_23


# ────────────────────────────────────────────────────────────────────────────
# Notebook 2: spectral analysis
# ────────────────────────────────────────────────────────────────────────────

def load_spectral() -> dict[str, Any]:
    nb2 = CSVS / "notebook2"
    return {
        "spectral_per_layer": _maybe_read(nb2 / "spectral_analysis_results.csv"),
        "rank_matrix_k95":    _maybe_read(nb2 / "rank_matrix_k95.csv", index_col=0),
        "rank_matrix_k90":    _maybe_read(nb2 / "rank_matrix_k90.csv", index_col=0),
        "rank_matrix_k99":    _maybe_read(nb2 / "rank_matrix_k99.csv", index_col=0),
        "singular_values":    _maybe_read(nb2 / "singular_values_by_component.csv"),
        "uniform_results":    _maybe_read(nb2 / "uniform_compression_results.csv"),
        "adaptive_results":   _maybe_read(nb2 / "adaptive_compression_results.csv"),
        "mixed_results":      _maybe_read(nb2 / "mixed_compression_results.csv"),
        "uniform_emotion_f1": _maybe_read(nb2 / "uniform_emotion_f1.csv"),
        "per_emotion":        _maybe_read(nb2 / "per_emotion_results.csv"),
        "qkv_overlap":        _maybe_read(nb2 / "qkv_subspace_overlap.csv"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Notebook 3: compression sensitivity
# ────────────────────────────────────────────────────────────────────────────

def load_sensitivity() -> dict[str, Any]:
    nb3 = CSVS / "notebook3"
    return {
        "component":       _maybe_read(nb3 / "component_sensitivity.csv"),
        "depth":           _maybe_read(nb3 / "depth_sensitivity.csv"),
        "component_f1":    _maybe_read(nb3 / "component_f1_matrix.csv", index_col=0),
        "depth_f1":        _maybe_read(nb3 / "depth_f1_matrix.csv", index_col=0),
        "component_sil":   _maybe_read(nb3 / "component_silhouette_matrix.csv", index_col=0),
        "depth_sil":       _maybe_read(nb3 / "depth_silhouette_matrix.csv", index_col=0),
        "per_emotion":     _maybe_read(nb3 / "per_emotion_results.csv"),
        "top_emotions":    _maybe_read(nb3 / "top_emotions.csv"),
        "master":          _maybe_read(nb3 / "sensitivity_master.csv"),
        "tsne_component":  _maybe_read(nb3 / "tsne_component.csv"),
        "tsne_depth":      _maybe_read(nb3 / "tsne_depth.csv"),
        "baseline":        _maybe_read(nb3 / "baseline_info.csv"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Notebook 4: probing
# ────────────────────────────────────────────────────────────────────────────

def load_probing() -> dict[str, Any]:
    nb4 = CSVS / "notebook4"
    return {
        "probe_results_long": _maybe_read(nb4 / "probe_results_long.csv"),
        "probe_results_wide": _maybe_read(nb4 / "probe_results.csv", index_col=0),
        "crystallization":    _maybe_read(nb4 / "crystallization_layers.csv"),
        "info_gain":          _maybe_read(nb4 / "layer_information_gain.csv"),
        "frequency":          _maybe_read(nb4 / "emotion_frequency.csv"),
        "correlations":       _maybe_read(nb4 / "probing_correlations.csv"),
        "summary":            _maybe_read(nb4 / "probing_summary.csv"),
        "vs_lesion":          _maybe_read(nb4 / "probe_vs_lesion.csv"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Notebook 5: activation patching
# ────────────────────────────────────────────────────────────────────────────

def load_patching() -> dict[str, Any]:
    nb5 = CSVS / "notebook5"
    return {
        "per_layer":         _maybe_read(nb5 / "activation_patching_per_layer.csv"),
        "per_component":     _maybe_read(nb5 / "activation_patching_per_component.csv"),
        "summary":           _maybe_read(nb5 / "activation_patching_summary.csv"),
        "mean_restoration":  _maybe_read(nb5 / "mean_restoration_per_layer.csv"),
        "critical_layer":    _maybe_read(nb5 / "critical_layer_per_emotion.csv"),
        "baseline_compr":    _maybe_read(nb5 / "baseline_vs_compressed.csv"),
        "master":            _maybe_read(nb5 / "patching_master_summary.csv"),
        "vs_lesion":         _maybe_read(nb5 / "patching_vs_lesion.csv"),
        "restoration_matrix": _maybe_read(nb5 / "restoration_matrix.csv", index_col=0),
    }


# ────────────────────────────────────────────────────────────────────────────
# Notebook 6: head analysis
# ────────────────────────────────────────────────────────────────────────────

def load_heads() -> dict[str, Any]:
    nb6 = CSVS / "notebook6"
    importance_npy = nb6 / "head_importance_matrix.npy"
    return {
        "ablation_long":     _maybe_read(nb6 / "head_ablation_long.csv"),
        "ablation_results":  _maybe_read(nb6 / "head_ablation_results.csv"),
        "categories":        _maybe_read(nb6 / "head_categories.csv"),
        "top_heads":         _maybe_read(nb6 / "top_heads_per_emotion.csv"),
        "head_top_emotions": _maybe_read(nb6 / "head_top_emotions.csv"),
        "redundancy":        _maybe_read(nb6 / "head_redundancy_pairs.csv"),
        "correlation":       _maybe_read(nb6 / "head_correlation_matrix.csv", index_col=0),
        "importance_matrix": _maybe_read(nb6 / "head_importance_matrix.csv", index_col=0),
        "importance_npy":    np.load(importance_npy) if importance_npy.exists() else None,
        "layer_importance":  _maybe_read(nb6 / "layer_attention_importance.csv"),
        "layer_redundancy":  _maybe_read(nb6 / "layer_redundancy_matrix.csv", index_col=0),
        "summary":           _maybe_read(nb6 / "head_analysis_summary.csv"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Notebook 7: neuron analysis
# ────────────────────────────────────────────────────────────────────────────

def load_neurons() -> dict[str, Any]:
    nb7 = CSVS / "notebook7"
    return {
        "catalog":           _maybe_read(nb7 / "neuron_catalog.csv"),
        "significant_counts": _maybe_read(nb7 / "neuron_significant_counts.csv"),
        "clusters":          _maybe_read(nb7 / "neuron_emotion_clusters.csv"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Notebook 8: emotional map / lesion study
# ────────────────────────────────────────────────────────────────────────────

def load_lesion() -> dict[str, Any]:
    nb8 = CSVS / "notebook8"
    return {
        "per_layer":     _maybe_read(nb8 / "lesion_per_layer.csv"),
        "per_component": _maybe_read(nb8 / "lesion_per_component.csv"),
        "summary":       _maybe_read(nb8 / "lesion_summary.csv"),
    }


# ────────────────────────────────────────────────────────────────────────────
# Notebook 9: informed / greedy compression
# ────────────────────────────────────────────────────────────────────────────

def load_informed() -> dict[str, Any]:
    nb9 = CSVS / "notebook9"
    out: dict[str, Any] = {
        "comparison":      _maybe_read(nb9 / "compression_comparison.csv"),
        "finetuning":      _maybe_read(nb9 / "finetuning_recovery.csv"),
    }
    # All greedy/informed rank assignments
    rank_files = {}
    for f in nb9.glob("*_ranks.csv"):
        key = f.stem.replace("_ranks", "")
        rank_files[key] = pd.read_csv(f)
    out["rank_assignments"] = rank_files
    return out


# ────────────────────────────────────────────────────────────────────────────
# All-in-one loader
# ────────────────────────────────────────────────────────────────────────────

def load_all() -> dict[str, Any]:
    print("Loading user CSVs from results/ ...")
    bundle = {
        "emotions":       _emotion_names_from_checkpoint(),
        "spectral":       load_spectral(),
        "sensitivity":    load_sensitivity(),
        "probing":        load_probing(),
        "patching":       load_patching(),
        "heads":          load_heads(),
        "neurons":        load_neurons(),
        "lesion":         load_lesion(),
        "informed":       load_informed(),
        "model_path":     str(MODEL_CHECKPOINT),
    }
    n_tables = sum(
        1 for top in ["spectral", "sensitivity", "probing", "patching",
                      "heads", "neurons", "lesion", "informed"]
        for v in bundle[top].values()
        if v is not None and not (isinstance(v, dict) and len(v) == 0)
    )
    print(f"  ✓ {n_tables} tables loaded")
    return bundle


# ────────────────────────────────────────────────────────────────────────────
# Helpers used directly by visualizations
# ────────────────────────────────────────────────────────────────────────────

def probe_f1_matrix(probing_data: dict) -> tuple[np.ndarray, list[str], list[str]]:
    """Return (f1[n_emotions, n_layers=13], emotion_names, layer_names) from
    the long-format probe results."""
    df = probing_data["probe_results_long"]
    if df is None:
        raise RuntimeError("probe_results_long.csv not found")
    emos = df["emotion"].drop_duplicates().tolist()
    layers = df["layer_name"].drop_duplicates().tolist()
    pivoted = df.pivot(index="emotion", columns="layer_idx", values="probe_f1")
    pivoted = pivoted.reindex(EMOTIONS_23)
    return pivoted.values.astype(np.float32), pivoted.index.tolist(), layers


def crystallization_dict(probing_data: dict) -> dict[str, dict]:
    """Return {emotion: {crystal_layer, max_f1, argmax, frequency}}."""
    df = probing_data["crystallization"]
    if df is None:
        raise RuntimeError("crystallization_layers.csv not found")
    return {
        row["emotion"]: {
            "crystal_layer": int(row["crystallization_layer"]) if not pd.isna(row["crystallization_layer"]) else 12,
            "crystal_layer_name": row["crystallization_layer_name"],
            "max_probe_f1":  float(row["max_probe_f1"]),
            "argmax":        int(row["argmax_layer"]) if not pd.isna(row["argmax_layer"]) else 12,
            "frequency":     int(row["train_frequency"]) if not pd.isna(row["train_frequency"]) else 0,
        }
        for _, row in df.iterrows()
    }


def head_categories_grid(heads_data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (categories[12, 12], importance[12, 12]) — REAL categories
    from the user's notebook 6, replacing the synthesized grid."""
    df = heads_data["categories"]
    if df is None:
        raise RuntimeError("head_categories.csv not found")
    cats = np.empty((12, 12), dtype=object)
    importance = np.zeros((12, 12), dtype=np.float32)
    for _, row in df.iterrows():
        L, h = int(row["layer"]), int(row["head"])
        cats[L, h] = row["category"]
        importance[L, h] = float(row["total_importance"])
    return cats, importance


def critical_head_per_emotion(heads_data: dict) -> dict[str, tuple[int, int, float]]:
    """{emotion: (layer, head, f1_drop)} for the TOP-1 head per emotion."""
    df = heads_data["top_heads"]
    if df is None:
        raise RuntimeError("top_heads_per_emotion.csv not found")
    top = df[df["rank"] == 1]
    return {
        row["emotion"]: (int(row["layer"]), int(row["head"]), -float(row["f1_drop"]))
        for _, row in top.iterrows()
    }


def patching_per_layer_emotion(patching_data: dict) -> np.ndarray:
    """Return (12, n_emotions) restoration_score matrix from per_layer patching."""
    df = patching_data["per_layer"]
    if df is None:
        raise RuntimeError("activation_patching_per_layer.csv not found")
    pivot = df.pivot(index="layer", columns="emotion", values="restoration_score")
    pivot = pivot.reindex(columns=EMOTIONS_23)
    pivot = pivot.reindex(index=range(12))
    return pivot.values.astype(np.float32)


def patching_per_layer_f1(patching_data: dict) -> np.ndarray:
    """Return (12, n_emotions) F1 patched matrix."""
    df = patching_data["per_layer"]
    if df is None:
        return None
    pivot = df.pivot(index="layer", columns="emotion", values="patched_f1")
    pivot = pivot.reindex(columns=EMOTIONS_23)
    pivot = pivot.reindex(index=range(12))
    return pivot.values.astype(np.float32)


def neuron_count_per_emotion(neurons_data: dict) -> dict[str, int]:
    df = neurons_data["significant_counts"]
    if df is None:
        return {}
    return {row["emotion"]: int(row["total_significant"]) for _, row in df.iterrows()}


def emotion_clusters(neurons_data: dict) -> dict[str, int]:
    df = neurons_data["clusters"]
    if df is None:
        return {}
    return {row["emotion"]: int(row["cluster"]) for _, row in df.iterrows()}


def finetune_recovery(informed_data: dict) -> dict[str, dict]:
    df = informed_data["finetuning"]
    if df is None:
        return {}
    return {
        row["emotion"]: {
            "baseline":    float(row["baseline_f1"]),
            "compressed":  float(row["compressed_f1"]),
            "finetuned":   float(row["finetuned_f1"]),
            "recovery":    float(row["ft_recovery"]),
        }
        for _, row in df.iterrows()
    }


def f1_baseline_per_emotion(informed_data: dict) -> dict[str, float]:
    df = informed_data["finetuning"]
    if df is None:
        return {}
    return {row["emotion"]: float(row["baseline_f1"]) for _, row in df.iterrows()}


def all_strategies(informed_data: dict) -> pd.DataFrame:
    """Master Pareto table — REAL F1 + ratio for every strategy the user ran."""
    return informed_data["comparison"]


if __name__ == "__main__":
    bundle = load_all()
    print("\nSummary:")
    print(f"  Emotions: {len(bundle['emotions'])}")
    print(f"  Model:    {bundle['model_path']}")
    if bundle["probing"]["probe_results_long"] is not None:
        f1m, _, _ = probe_f1_matrix(bundle["probing"])
        print(f"  Probe F1 matrix: {f1m.shape}")
    if bundle["heads"]["categories"] is not None:
        print(f"  Head categories: {len(bundle['heads']['categories'])} rows")
    if bundle["informed"]["comparison"] is not None:
        print(f"  Strategies:      {len(bundle['informed']['comparison'])}")
