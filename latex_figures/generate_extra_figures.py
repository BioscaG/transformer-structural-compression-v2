"""Genera figuras del TFG que no están en los notebooks generate_cap{4,5}.

Estas figuras consumen el cache local en `viz/data/cache/` para no tener
que volver a ejecutar el modelo cada vez. Si el cache no existe, ejecuta
los extractores en `viz/extractors/`.

Uso:

    # Todas las figuras (es + en):
    python latex_figures/generate_extra_figures.py

    # Una sola, los dos idiomas:
    python latex_figures/generate_extra_figures.py iterative_u
    python latex_figures/generate_extra_figures.py internal_compression

    # Lista de las disponibles:
    python latex_figures/generate_extra_figures.py --list

Las figuras se guardan en `latex_figures/figures/cap{N}_{name}_{lang}.png`,
mismo naming convention que los notebooks `generate_cap{4,5}.ipynb`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
HERE         = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from transformers import AutoModelForSequenceClassification

import tfg_plot_style as sty


# ─── Paths ────────────────────────────────────────────────────────────────
CACHE_DIR        = PROJECT_ROOT / "viz" / "data" / "cache"
MODEL_CHECKPOINT = PROJECT_ROOT / "results" / "checkpoints" / "23emo-final"
NB04_DIR         = PROJECT_ROOT / "results" / "csvs" / "notebook4"
FIGURES_DIR      = HERE / "figures"

EMOTIONS_23 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "joy",
    "love", "optimism", "realization", "remorse", "sadness", "surprise",
]
LAYER_LABELS = ["Emb"] + [f"L{i}" for i in range(12)]

# Same as viz/thesis_data.py CLUSTER_DEFS — replicated here so the script
# stays independent from viz/.
CLUSTER_DEFS = {
    "Positivas alta energía": [
        "admiration", "amusement", "excitement", "gratitude", "joy", "love",
    ],
    "Negativas reactivas": [
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
    ],
    "Negativas internas": [
        "embarrassment", "fear", "remorse", "sadness",
    ],
    "Epistémicas": [
        "confusion", "curiosity", "surprise",
    ],
    "Orientadas al otro": [
        "caring", "desire", "optimism",
    ],
    "Baja especificidad": [
        "approval", "realization",
    ],
}
EMO_TO_CLUSTER = {e: c for c, ems in CLUSTER_DEFS.items() for e in ems}
CLUSTER_COLORS = {
    "Positivas alta energía": sty.SAND,
    "Negativas reactivas":    sty.TERRA,
    "Negativas internas":     sty.ROSE,
    "Epistémicas":            sty.BLUE,
    "Orientadas al otro":     sty.SAGE,
    "Baja especificidad":     sty.PLUM,
}


# ─── Data loaders (caches) ────────────────────────────────────────────────

def _load_logit_lens_data():
    """Logit lens sigmoids over 2300 cached test sentences.

    Returns dict with: top1_per_layer, gold_per_layer, sum_per_layer,
    gold_sig (per sentence), label_arr, n_sentences, sigmoids.
    """
    print("[data] loading activations + classifier...")
    data = np.load(CACHE_DIR / "activations.npz")
    cls = data["cls_per_layer"]                            # (N, 13, 768)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    label_names = meta["label_names"]

    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()
    bp = mdl.bert.pooler.dense.bias.detach().numpy()
    W  = mdl.classifier.weight.detach().numpy()
    b  = mdl.classifier.bias.detach().numpy()

    pooled   = np.tanh(np.einsum("slh,ph->slp", cls, Wp) + bp)
    logits   = np.einsum("slp,ep->sle", pooled, W) + b
    sigmoids = 1.0 / (1.0 + np.exp(-logits))               # (N, 13, 23)

    label_to_idx = {e: i for i, e in enumerate(EMOTIONS_23)}
    y = np.array([label_to_idx[l] for l in label_names])
    gold_sig = sigmoids[np.arange(len(y)), :, y]           # (N, 13)

    return dict(
        sigmoids=sigmoids,
        gold_sig=gold_sig,
        label_arr=np.array(label_names),
        y=y,
        n=len(y),
        top1_per_layer=sigmoids.max(axis=2).mean(axis=0),
        gold_per_layer=sigmoids[np.arange(len(y)), :, y].mean(axis=0),
        sum_per_layer=sigmoids.sum(axis=2).mean(axis=0),
    )


def _load_token_trajectory_data():
    """Hidden states per token per layer for 46 cached sentences.

    Returns: norm_cls, norm_cnt, eff_rank, k95, cum_energy.
    """
    print("[data] loading token trajectories...")
    d = np.load(CACHE_DIR / "token_trajectories.npz")
    H = d["hidden"].astype(np.float32)                     # (N, L, T, 768)
    meta = json.loads((CACHE_DIR / "token_trajectories_meta.json").read_text())
    tokens = meta["tokens"]
    n_sent, n_layers, T, _ = H.shape

    SPECIAL = {"⟨CLS⟩", "⟨SEP⟩", "[CLS]", "[SEP]", "[PAD]", ""}
    cls_m = np.zeros((n_sent, T), dtype=bool)
    cnt_m = np.zeros((n_sent, T), dtype=bool)
    for si, toks in enumerate(tokens):
        for ti, tok in enumerate(toks):
            if ti >= T: break
            if tok in ("⟨CLS⟩", "[CLS]"): cls_m[si, ti] = True
            elif tok not in SPECIAL: cnt_m[si, ti] = True

    cls_idx = cls_m.nonzero()
    cnt_idx = cnt_m.nonzero()
    norm_cls = np.zeros(n_layers)
    norm_cnt = np.zeros(n_layers)
    eff_rank = np.zeros(n_layers)
    k95 = np.zeros(n_layers, dtype=np.int32)
    K_VIEW = 256
    cum_energy = np.zeros((n_layers, K_VIEW), dtype=np.float32)

    for L in range(n_layers):
        norm_cls[L] = float(np.linalg.norm(H[cls_idx[0], L, cls_idx[1]], axis=-1).mean())
        norm_cnt[L] = float(np.linalg.norm(H[cnt_idx[0], L, cnt_idx[1]], axis=-1).mean())
        pooled = H[cnt_idx[0], L, cnt_idx[1]]
        pooled_c = pooled - pooled.mean(axis=0, keepdims=True)
        s = np.linalg.svd(pooled_c, compute_uv=False)
        var = s ** 2
        p = var / var.sum()
        p_safe = np.clip(p, 1e-12, None)
        eff_rank[L] = float(np.exp(-np.sum(p_safe * np.log(p_safe))))
        cum = np.cumsum(p)
        k95[L] = int(np.argmax(cum >= 0.95)) + 1
        cum_energy[L, :K_VIEW] = cum[:K_VIEW]

    return dict(
        norm_cls=norm_cls, norm_cnt=norm_cnt,
        eff_rank=eff_rank, k95=k95,
        cum_energy=cum_energy, K_VIEW=K_VIEW,
    )


# ─── Figure generators ────────────────────────────────────────────────────

def gen_iterative_u(lang: str = "es", *, data=None) -> pathlib.Path:
    """Cap. 5: U-curve agregada del logit lens."""
    if data is None:
        data = _load_logit_lens_data()

    sty.apply(lang=lang)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    L = {
        "es": dict(top1="top-1 sigmoid", gold="gold sigmoid",
                   sum="suma de 23",
                   sat="saturación", valley="valle de transición",
                   crys="cristalización",
                   ylabel="sigmoid promedio",
                   ylabel2="suma sigmoid (de 23)",
                   xlabel=sty.L["layer"]),
        "en": dict(top1="top-1 sigmoid", gold="gold sigmoid",
                   sum="sum of 23",
                   sat="saturation", valley="transition valley",
                   crys="crystallization",
                   ylabel="mean sigmoid",
                   ylabel2="sum of sigmoids (of 23)",
                   xlabel="Layer"),
    }[lang]

    layer_x = np.arange(13)
    PHASE_Y = 0.97

    ax.axvspan(-0.5, 3.5, alpha=0.10, color=sty.SAND, zorder=0)
    ax.axvspan(3.5, 9.5,  alpha=0.06, color=sty.BLUE, zorder=0)
    ax.axvspan(9.5, 12.5, alpha=0.10, color=sty.TERRA, zorder=0)

    for x, txt, col in [(1.5, L["sat"], sty.SAND),
                         (6.5, L["valley"], sty.BLUE),
                         (11.0, L["crys"], sty.TERRA)]:
        ax.text(x, PHASE_Y, txt, ha="center", color=col,
                fontsize=sty.ANNOTATION_SIZE, style="italic",
                fontweight="bold", zorder=4,
                transform=ax.get_xaxis_transform())

    ax.plot(layer_x, data["top1_per_layer"], color=sty.TERRA, marker="s",
            markersize=6, lw=2.2, markeredgecolor="white",
            markeredgewidth=0.9, label=L["top1"], zorder=5)
    ax.plot(layer_x, data["gold_per_layer"], color=sty.SAGE, marker="o",
            markersize=6, lw=2.0, markeredgecolor="white",
            markeredgewidth=0.9, label=L["gold"], zorder=5)

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(layer_x)
    ax.set_xticklabels(LAYER_LABELS)
    ax.set_xlabel(L["xlabel"])
    ax.set_ylabel(L["ylabel"])
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(layer_x, data["sum_per_layer"], color=sty.INK_3, lw=1.2, ls=":",
             marker="^", markersize=4.5, markeredgecolor="white",
             markeredgewidth=0.6, label=L["sum"], zorder=4)
    ax2.set_ylim(0, 8.5)
    ax2.set_ylabel(L["ylabel2"], color=sty.INK_3, fontsize=sty.LABEL_SIZE)
    ax2.tick_params(axis="y", colors=sty.INK_3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(sty.SPINE)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=sty.LEGEND_SIZE, framealpha=0.92,
               frameon=True, edgecolor=sty.SPINE)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, "iterative_u", chapter=5, lang=lang)


def gen_lens_vs_probe(lang: str = "es", *, data=None) -> pathlib.Path:
    """Cap. 5: Probe F1 vs gold sigmoid del logit lens, en 3 paneles
    por capa de cristalización."""
    if data is None:
        data = _load_logit_lens_data()

    probe_df = pd.read_csv(NB04_DIR / "probe_results.csv")
    crystal_df = pd.read_csv(NB04_DIR / "crystallization_layers.csv")

    layer_cols = LAYER_LABELS
    probe_df_indexed = probe_df.set_index("emotion")[layer_cols]

    emo_to_xlayer = dict(zip(crystal_df["emotion"],
                              crystal_df["crystallization_layer"].astype(int)))

    def _bucket(layer):
        if layer <= 3: return "early"
        if layer <= 7: return "middle"
        return "late"

    emo_to_group = {e: _bucket(emo_to_xlayer.get(e, 11)) for e in EMOTIONS_23}
    label_arr = data["label_arr"]
    gold_sig = data["gold_sig"]

    groups = {}
    for g in ("early", "middle", "late"):
        emos = sorted([e for e, gg in emo_to_group.items() if gg == g])
        mask = np.isin(label_arr, emos)
        probe_subset = probe_df_indexed.loc[probe_df_indexed.index.isin(emos)]
        groups[g] = dict(emos=emos, n=int(mask.sum()),
                         probe=probe_subset.mean(axis=0).values,
                         gold=gold_sig[mask].mean(axis=0))

    sty.apply(lang=lang)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=False)

    titles_es = {
        "early":  "Tempranas (cristalizan en L0–L2)",
        "middle": "Medias (cristalizan en L3–L6)",
        "late":   "Tardías (cristalizan en L7–L11)",
    }
    titles_en = {
        "early":  "Early (crystallize at L0–L2)",
        "middle": "Middle (crystallize at L3–L6)",
        "late":   "Late (crystallize at L7–L11)",
    }
    titles = titles_es if lang == "es" else titles_en
    n_lbl    = "frases" if lang == "es" else "sentences"
    emo_lbl  = "emociones" if lang == "es" else "emotions"
    gap_lbl  = ("Info no leída por el classifier" if lang == "es"
                else "Info unread by the classifier")
    capa_lbl = sty.L["layer"]
    layer_x  = np.arange(13)

    for ax, g in zip(axes, ("early", "middle", "late")):
        d = groups[g]
        ax.axvspan(-0.5, 3.5, alpha=0.06, color=sty.SAND, zorder=0)
        ax.axvspan(3.5, 9.5,  alpha=0.04, color=sty.BLUE, zorder=0)
        ax.axvspan(9.5, 12.5, alpha=0.06, color=sty.TERRA, zorder=0)
        gap_top = np.maximum(d["probe"], d["gold"])
        gap_bot = np.minimum(d["probe"], d["gold"])
        ax.fill_between(layer_x, gap_bot, gap_top, color=sty.INK_3,
                        alpha=0.12, zorder=1,
                        label=gap_lbl if g == "early" else None)
        ax.plot(layer_x, d["probe"], color=sty.TERRA, marker="s", markersize=5,
                lw=1.8, markeredgecolor="white", markeredgewidth=0.8,
                label="Probe F1" if g == "early" else None, zorder=3)
        ax.plot(layer_x, d["gold"], color=sty.SAGE, marker="o", markersize=5,
                lw=1.8, markeredgecolor="white", markeredgewidth=0.8,
                label="Gold sigmoid (logit lens)" if g == "early" else None,
                zorder=3)
        # In-panel marker (NOT a chart title — the figure caption holds the
        # full description; this is just so the reader knows which panel
        # corresponds to which crystallization group).
        label = (f"{titles[g]}  ·  {len(d['emos'])} {emo_lbl} · "
                 f"{d['n']} {n_lbl}")
        ax.text(0.012, 0.94, label, transform=ax.transAxes,
                fontsize=sty.SMALL_SIZE, color=sty.INK_3,
                style="italic", va="top", ha="left", zorder=10,
                bbox=dict(facecolor="white", alpha=0.85,
                          edgecolor="none", pad=2))
        ax.set_ylim(-0.02, 1.02)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(layer_x)
        ax.set_xticklabels([])
        ax.set_xlim(-0.5, 12.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=sty.LEGEND_SIZE, framealpha=0.92,
               frameon=True, edgecolor=sty.SPINE)

    axes[1].set_ylabel("F1 / sigmoid", fontsize=sty.LABEL_SIZE,
                       color=sty.INK_2)
    axes[-1].set_xticklabels(layer_cols, fontsize=sty.TICK_SIZE)
    axes[-1].set_xlabel(capa_lbl, fontsize=sty.LABEL_SIZE, color=sty.INK_2)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _save(fig, "lens_vs_probe", chapter=5, lang=lang)


def gen_internal_compression(lang: str = "es", *, data=None) -> pathlib.Path:
    """Cap. 4: el modelo se comprime a sí mismo (norma vs rango efectivo)."""
    if data is None:
        data = _load_token_trajectory_data()

    sty.apply(lang=lang)
    fig = plt.figure(figsize=(7.2, 6.5))
    gs = GridSpec(2, 1, height_ratios=[1.0, 0.85], hspace=0.50)

    L = {
        "es": dict(
            xlabel=sty.L["layer"],
            ylabel_norm=r"$\|h\|_2$ (norma euclídea)",
            ylabel_rank="rango efectivo / k95 (dim)",
            top1="norma media (tokens contenido)", cls="norma del [CLS]",
            erank="rango efectivo (entropía espectral)",
            k95_lbl="k95 (95% energía)",
            xlabel2="índice del valor singular (1 = más fuerte)",
            cbar="energía acum.",
            phase_lex="fase léxica", phase_mix="fase de mezcla",
            phase_col="colapso espectral",
        ),
        "en": dict(
            xlabel="Layer", ylabel_norm=r"$\|h\|_2$ (Euclidean norm)",
            ylabel_rank="effective rank / k95 (dim)",
            top1="mean norm (content tokens)", cls="[CLS] norm",
            erank="effective rank (spectral entropy)",
            k95_lbl="k95 (95% energy)",
            xlabel2="singular value index (1 = strongest)",
            cbar="cum. energy",
            phase_lex="lexical", phase_mix="mixing", phase_col="collapse",
        ),
    }[lang]

    layer_x = np.arange(13)
    n_layers = 13

    ax = fig.add_subplot(gs[0])
    ax.axvspan(-0.5, 4.5, alpha=0.10, color=sty.BLUE,  zorder=0)
    ax.axvspan( 4.5, 8.5, alpha=0.06, color=sty.SAND,  zorder=0)
    ax.axvspan( 8.5,12.5, alpha=0.10, color=sty.TERRA, zorder=0)
    for x, txt, col in [(2.0, L["phase_lex"], sty.BLUE),
                         (6.5, L["phase_mix"], sty.SAND),
                         (10.5, L["phase_col"], sty.TERRA)]:
        ax.text(x, 0.96, txt, ha="center", color=col,
                fontsize=sty.ANNOTATION_SIZE, style="italic",
                fontweight="bold",
                transform=ax.get_xaxis_transform(), zorder=4)

    ax.plot(layer_x, data["norm_cnt"], color=sty.BLUE, marker="s",
            markersize=6, lw=2.0, markeredgecolor="white",
            markeredgewidth=0.8, label=L["top1"], zorder=5)
    ax.plot(layer_x, data["norm_cls"], color=sty.BLUE_L, marker="o",
            markersize=4.5, lw=1.4, ls=":", markeredgecolor="white",
            markeredgewidth=0.6, label=L["cls"], zorder=4)
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(0, max(data["norm_cnt"].max(), data["norm_cls"].max()) * 1.15)
    ax.set_xticks(layer_x)
    ax.set_xticklabels(LAYER_LABELS, fontsize=sty.TICK_SIZE)
    ax.set_xlabel(L["xlabel"])
    ax.set_ylabel(L["ylabel_norm"], color=sty.BLUE)
    ax.tick_params(axis="y", colors=sty.BLUE)
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(layer_x, data["eff_rank"], color=sty.TERRA, marker="D",
             markersize=6, lw=2.0, markeredgecolor="white",
             markeredgewidth=0.8, label=L["erank"], zorder=5)
    ax2.plot(layer_x, data["k95"], color=sty.TERRA_L, marker="^",
             markersize=5, lw=1.4, ls="--", markeredgecolor="white",
             markeredgewidth=0.6, label=L["k95_lbl"], zorder=4)
    ax2.set_ylim(0, max(data["k95"].max(), data["eff_rank"].max()) * 1.15)
    ax2.set_ylabel(L["ylabel_rank"], color=sty.TERRA)
    ax2.tick_params(axis="y", colors=sty.TERRA)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(sty.TERRA)
    ax2.grid(False)

    # Legend ABOVE the panel (out of data area — eff_rank curve drops
    # dramatically at L11 and would cross any in-panel legend).
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2,
              loc="lower center", bbox_to_anchor=(0.5, 1.02),
              ncol=2, fontsize=sty.LEGEND_SIZE, framealpha=0.92,
              frameon=True, edgecolor=sty.SPINE)

    ax_h = fig.add_subplot(gs[1])
    K = data["K_VIEW"]
    im = ax_h.imshow(data["cum_energy"], aspect="auto", origin="upper",
                     cmap="YlOrRd", vmin=0, vmax=1,
                     extent=[0.5, K + 0.5, n_layers - 0.5, -0.5])
    ax_h.plot(data["k95"], layer_x, color=sty.INK, lw=1.4, ls=":",
              marker="o", markersize=4.5,
              markeredgecolor="white", markeredgewidth=0.8, label="k95")
    ax_h.set_yticks(layer_x)
    ax_h.set_yticklabels(LAYER_LABELS, fontsize=sty.TICK_SIZE)
    ax_h.set_xlabel(L["xlabel2"])
    ax_h.set_xlim(0.5, K + 0.5)
    ax_h.grid(False)
    ax_h.legend(loc="lower right", fontsize=sty.LEGEND_SIZE, framealpha=0.92)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02)
    cbar.set_label(L["cbar"], fontsize=sty.SMALL_SIZE, color=sty.INK_3)
    cbar.ax.tick_params(labelsize=sty.SMALL_SIZE, colors=sty.INK_3)
    cbar.ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    return _save(fig, "internal_compression", chapter=4, lang=lang)


# ─── finetuning_diff ──────────────────────────────────────────────────────

COMPONENTS = ["query", "key", "value", "attn_output",
              "ffn_intermediate", "ffn_output"]
COMPONENT_LABEL = {
    "query": "Q", "key": "K", "value": "V",
    "attn_output": "Attn-O",
    "ffn_intermediate": "FFN-i",
    "ffn_output": "FFN-o",
}
COMPONENT_PATH = {
    "query":            "encoder.layer.{L}.attention.self.query",
    "key":              "encoder.layer.{L}.attention.self.key",
    "value":            "encoder.layer.{L}.attention.self.value",
    "attn_output":      "encoder.layer.{L}.attention.output.dense",
    "ffn_intermediate": "encoder.layer.{L}.intermediate.dense",
    "ffn_output":       "encoder.layer.{L}.output.dense",
}


def _load_finetuning_diff_data():
    """Frobenius diff between bert-base-uncased and 23emo-final per matrix."""
    print("[data] loading bert-base-uncased + 23emo-final and computing diff...")
    from transformers import AutoModel
    pre = AutoModel.from_pretrained("bert-base-uncased")
    ft = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    ft_bert = ft.bert
    pre_bert = pre

    def get_w(model, path):
        obj = model
        for part in path.split("."):
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        return obj.weight.detach().float().numpy()

    n_layers = 12
    rel_change = np.zeros((n_layers, len(COMPONENTS)), dtype=np.float32)
    for L in range(n_layers):
        for ci, comp in enumerate(COMPONENTS):
            path = COMPONENT_PATH[comp].format(L=L)
            W_pre = get_w(pre_bert, path)
            W_ft  = get_w(ft_bert,  path)
            d = W_ft - W_pre
            f_pre = np.linalg.norm(W_pre)
            f_diff = np.linalg.norm(d)
            rel_change[L, ci] = f_diff / f_pre if f_pre > 0 else 0
    return rel_change


def gen_finetuning_diff(lang: str = "es", *, data=None) -> pathlib.Path:
    """Cap. 5 §5.5: ¿qué cambió el fine-tuning? Heatmap 12×6."""
    if data is None:
        data = _load_finetuning_diff_data()
    rel_change = data
    n_layers, n_comp = rel_change.shape

    L_es = dict(xlabel="Componente", ylabel="Capa", cbar=r"$\|\Delta W\| / \|W_0\|$")
    L_en = dict(xlabel="Component", ylabel="Layer", cbar=r"$\|\Delta W\| / \|W_0\|$")
    L = L_es if lang == "es" else L_en

    sty.apply(lang=lang)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    im = ax.imshow(rel_change, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=float(rel_change.max()))

    # Annotate each cell with the percentage
    for L_idx in range(n_layers):
        for ci in range(n_comp):
            v = rel_change[L_idx, ci]
            color = "white" if v > rel_change.max() * 0.55 else sty.INK
            ax.text(ci, L_idx, f"{v*100:.1f}%", ha="center", va="center",
                    color=color, fontsize=sty.SMALL_SIZE)

    ax.set_xticks(range(n_comp))
    ax.set_xticklabels([COMPONENT_LABEL[c] for c in COMPONENTS],
                       fontsize=sty.LABEL_SIZE)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{i}" for i in range(n_layers)],
                       fontsize=sty.TICK_SIZE)
    ax.set_xlabel(L["xlabel"])
    ax.set_ylabel(L["ylabel"])
    ax.invert_yaxis()           # L0 arriba, L11 abajo
    ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(L["cbar"], fontsize=sty.LABEL_SIZE, color=sty.INK_2)
    cbar.ax.tick_params(labelsize=sty.SMALL_SIZE, colors=sty.INK_3)
    cbar.ax.yaxis.set_major_formatter(
        plt.matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    fig.tight_layout()
    return _save(fig, "finetuning_diff", chapter=5, lang=lang)


# ─── emotional_landscape ──────────────────────────────────────────────────

def _load_emotional_landscape_data():
    """Crystallization layer + selectivity norm per emotion."""
    print("[data] loading emotional landscape (CSV only)...")
    crystal = pd.read_csv(NB04_DIR / "crystallization_layers.csv")
    neuron_cat = pd.read_csv(PROJECT_ROOT / "results" / "csvs"
                             / "notebook7" / "neuron_catalog.csv")

    # Selectivity norm per emotion = sqrt(sum of |selectivity|^2 across top-K
    # neurons). Use top 50 absolute selectivities per emotion.
    sel = (neuron_cat.groupby("emotion")
           .apply(lambda g: float(np.linalg.norm(
               g["abs_selectivity"].nlargest(50).values)))
           .rename("sel_norm"))

    df = crystal.merge(sel, on="emotion")
    df["cluster"] = df["emotion"].map(EMO_TO_CLUSTER).fillna("Baja especificidad")
    return df


def gen_emotional_landscape(lang: str = "es", *, data=None) -> pathlib.Path:
    """Cap. 6/8: las 23 emociones en el plano (cristalización × selectividad)."""
    if data is None:
        data = _load_emotional_landscape_data()
    df = data

    L_es = dict(xlabel="Capa de cristalización", ylabel="Norma de selectividad")
    L_en = dict(xlabel="Crystallization layer", ylabel="Selectivity norm")
    L = L_es if lang == "es" else L_en

    sty.apply(lang=lang)
    fig, ax = plt.subplots(figsize=(7.2, 5.5))

    # Phase shading (matching the rest of cap 5)
    ax.axvspan(-0.5, 3.5, alpha=0.06, color=sty.BLUE,  zorder=0)
    ax.axvspan( 3.5, 7.5, alpha=0.04, color=sty.SAND,  zorder=0)
    ax.axvspan( 7.5,12.5, alpha=0.06, color=sty.TERRA, zorder=0)

    for cluster, group in df.groupby("cluster"):
        ax.scatter(group["crystallization_layer"], group["sel_norm"],
                   s=120, c=CLUSTER_COLORS[cluster],
                   edgecolor="white", linewidth=1.2, alpha=0.92,
                   label=cluster, zorder=4)

    # Label each emotion
    for _, row in df.iterrows():
        ax.annotate(row["emotion"],
                    (row["crystallization_layer"], row["sel_norm"]),
                    xytext=(7, 4), textcoords="offset points",
                    fontsize=sty.SMALL_SIZE, color=sty.INK_2,
                    style="italic", zorder=5)

    ax.set_xlabel(L["xlabel"])
    ax.set_ylabel(L["ylabel"])
    ax.set_xlim(-0.5, df["crystallization_layer"].max() + 0.5)
    ax.set_xticks(range(int(df["crystallization_layer"].min()),
                        int(df["crystallization_layer"].max()) + 1))
    ax.grid(axis="both", alpha=0.3)

    # Legend at top, outside
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, fontsize=sty.LEGEND_SIZE, framealpha=0.92,
              frameon=True, edgecolor=sty.SPINE)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save(fig, "emotional_landscape", chapter=6, lang=lang)


# ─── galaxia_evolution (3 panels: L0, L6, L11) ───────────────────────────

def _load_galaxia_data():
    """t-SNE-2D projection of 2300 sentences in Emb, L5, L11.

    t-SNE per layer (independent fit). Captures local cluster structure
    non-linearly, ideal for "emergent clusters" visuals. Random state fixed
    for reproducibility. Note: absolute positions are meaningless across
    panels (t-SNE distorts global structure), only RELATIVE clustering is
    informative — exactly what we want here."""
    print("[data] loading activations + fitting PCA→t-SNE per layer...")
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    data = np.load(CACHE_DIR / "activations.npz")
    cls = data["cls_per_layer"]                          # (N, 13, 768)
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    label_names = meta["label_names"]

    mdl = AutoModelForSequenceClassification.from_pretrained(str(MODEL_CHECKPOINT))
    Wp = mdl.bert.pooler.dense.weight.detach().numpy()
    bp = mdl.bert.pooler.dense.bias.detach().numpy()

    pooled = np.tanh(np.einsum("slh,ph->slp", cls, Wp) + bp)   # (N, 13, 768)

    panels = {}
    PCA_DIM = 50  # standard preprocessing for t-SNE on high-dim data
    # 4 panels for evenly-spaced emergence story:
    #   Emb (input) → L4 (early-mid) → L8 (late-mid) → L11 (final)
    for layer_idx, layer_name in [(0, "Emb"), (5, "L4"), (9, "L8"), (12, "L11")]:
        print(f"  PCA→t-SNE on {layer_name}...")
        # PCA to reduce noise + computational cost
        X = pooled[:, layer_idx, :]
        pca = PCA(n_components=PCA_DIM, random_state=42)
        X_pca = pca.fit_transform(X)
        # t-SNE on the PCA-reduced space
        tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                    init="pca", learning_rate="auto", max_iter=1000)
        coords = tsne.fit_transform(X_pca)
        panels[layer_name] = coords
    return dict(
        panels=panels,
        labels=np.array(label_names),
        emotions=EMOTIONS_23,
    )


def gen_galaxia_evolution(lang: str = "es", *, data=None) -> pathlib.Path:
    """Cap. 5: cristalización geométrica de las emociones — 3 paneles."""
    if data is None:
        data = _load_galaxia_data()
    panels = data["panels"]
    labels = data["labels"]
    emotions = data["emotions"]

    # No axis labels: t-SNE axes are arbitrary (no physical meaning), so
    # standard practice is to omit them. Layer names below each panel
    # serve as identification; method explanation goes in the caption.
    L_es = dict(
        panel_es=lambda name: f"Capa {name}" if name != "Emb" else "Embedding",
    )
    L_en = dict(
        panel_es=lambda name: f"Layer {name}" if name != "Emb" else "Embedding",
    )
    L = L_es if lang == "es" else L_en

    sty.apply(lang=lang)
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.8))

    panel_order = ["Emb", "L4", "L8", "L11"]

    emo_colors = {
        e: CLUSTER_COLORS[EMO_TO_CLUSTER.get(e, "Baja especificidad")]
        for e in emotions
    }

    for ax, panel_key in zip(axes, panel_order):
        coords = panels[panel_key]
        # Points
        for emo in emotions:
            mask = labels == emo
            if not mask.any():
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=8, c=emo_colors[emo], alpha=0.50,
                       edgecolor="none", zorder=2)

        # Centroids
        for emo in emotions:
            mask = labels == emo
            if not mask.any():
                continue
            cx, cy = coords[mask].mean(axis=0)
            ax.scatter([cx], [cy], s=55, c=emo_colors[emo],
                       marker="D", edgecolor=sty.INK, linewidth=1.0,
                       zorder=4)

        # Each panel scales to its own data — pad uniformly
        x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
        y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
        pad_x = (x_max - x_min) * 0.08
        pad_y = (y_max - y_min) * 0.08
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(y_min - pad_y, y_max + pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(L["panel_es"](panel_key), fontsize=sty.LABEL_SIZE,
                      color=sty.INK_2)
        ax.grid(False)

    # Cluster legend at top
    handles = []
    for cluster, color in CLUSTER_COLORS.items():
        h = ax.scatter([], [], s=60, c=color, edgecolor="white",
                       linewidth=0.8, label=cluster)
        handles.append(h)
    fig.legend(handles=handles,
               loc="lower center", bbox_to_anchor=(0.5, 1.0),
               ncol=3, fontsize=sty.LEGEND_SIZE, framealpha=0.92,
               frameon=True, edgecolor=sty.SPINE)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return _save(fig, "galaxia_evolution", chapter=5, lang=lang)


# ─── Helpers ──────────────────────────────────────────────────────────────

def _save(fig, name: str, chapter: int, lang: str) -> pathlib.Path:
    """Save fig as cap{chapter}_{name}_{lang}.png in latex_figures/figures/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out = FIGURES_DIR / f"cap{chapter}_{name}_{lang}.png"
    fig.savefig(str(out))
    plt.close(fig)
    print(f"  ✓ {out.name}")
    return out


# ─── Registry ─────────────────────────────────────────────────────────────

# name → (generator, data_loader_key)
# data_loader_key reuses data across figures that need the same source.
GENERATORS = {
    "iterative_u":          (gen_iterative_u,          "logit_lens"),
    "lens_vs_probe":        (gen_lens_vs_probe,        "logit_lens"),
    "internal_compression": (gen_internal_compression, "token_trajectory"),
    "finetuning_diff":      (gen_finetuning_diff,      "ft_diff"),
    "emotional_landscape":  (gen_emotional_landscape,  "landscape"),
    "galaxia_evolution":    (gen_galaxia_evolution,    "galaxia"),
}

DATA_LOADERS = {
    "logit_lens":       _load_logit_lens_data,
    "token_trajectory": _load_token_trajectory_data,
    "ft_diff":          _load_finetuning_diff_data,
    "landscape":        _load_emotional_landscape_data,
    "galaxia":          _load_galaxia_data,
}


# ─── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Genera figuras del TFG que viven fuera de los notebooks "
                    "generate_cap{4,5}.ipynb.")
    parser.add_argument("names", nargs="*",
                        help="Nombres de las figuras a generar. "
                             "Vacío = todas.")
    parser.add_argument("--list", action="store_true",
                        help="Lista las figuras disponibles y sale.")
    parser.add_argument("--lang", choices=["es", "en", "both"], default="both",
                        help="Idioma. 'both' (default) genera los dos.")
    args = parser.parse_args()

    if args.list:
        print("Figuras disponibles:")
        for name in GENERATORS:
            print(f"  - {name}")
        return

    targets = list(args.names) if args.names else list(GENERATORS.keys())
    unknown = [n for n in targets if n not in GENERATORS]
    if unknown:
        print(f"Desconocidas: {unknown}")
        print(f"Disponibles: {list(GENERATORS.keys())}")
        sys.exit(1)

    langs = ["es", "en"] if args.lang == "both" else [args.lang]

    # Cargar cada loader sólo una vez aunque se usen varias figuras
    loaded = {}

    print("─" * 60)
    print(f"Generando {len(targets)} figura(s) × {len(langs)} idioma(s)")
    print("─" * 60)

    for name in targets:
        gen, loader_key = GENERATORS[name]
        if loader_key not in loaded:
            loaded[loader_key] = DATA_LOADERS[loader_key]()
        data = loaded[loader_key]
        print(f"\n[{name}]")
        for lang in langs:
            gen(lang=lang, data=data)

    print("\n" + "─" * 60)
    print(f"✓ Done. Outputs in {FIGURES_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
