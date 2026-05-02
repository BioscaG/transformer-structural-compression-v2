"""Compute CLS embeddings under multiple SVD compression levels.

For the same 588 sentences, run the model with uniform SVD compression at
ranks {768, 512, 384, 256, 128, 64} and store the per-layer CLS coordinates
projected onto a fixed PCA basis (fitted on baseline L12).

Output: viz/data/cache/compression_decay.npz with shape
(n_ranks, n_layers=13, n_sentences, 3) — small enough for direct embed.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.compression import apply_svd_compression


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAME = str(PROJECT_ROOT / "results" / "checkpoints" / "23emo-final")
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"

RANKS = [768, 512, 384, 256, 128, 64]


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_rank_dict(rank: int) -> dict[str, int]:
    """Uniform compression: same rank for all 72 matrices (or no-op if 768)."""
    if rank >= 768:
        return {}
    config = {}
    for L in range(12):
        for comp in ["attention.self.query", "attention.self.key",
                     "attention.self.value", "attention.output.dense",
                     "intermediate.dense", "output.dense"]:
            config[f"bert.encoder.layer.{L}.{comp}"] = rank
    return config


@torch.no_grad()
def forward_all(model, tokenizer, texts: list[str], device: str, batch_size: int = 32) -> np.ndarray:
    """Run forward on all texts, return (N, 13, 768) CLS array."""
    all_cls = []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i+batch_size], padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(device)
        out = model(**enc)
        # (n_layers, B, T, H) -> select [CLS] at position 0
        hs = torch.stack(out.hidden_states, dim=0)  # (13, B, T, 768)
        cls = hs[:, :, 0, :].permute(1, 0, 2)        # (B, 13, 768)
        all_cls.append(cls.cpu().numpy())
    return np.concatenate(all_cls, axis=0).astype(np.float32)


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    # Load test sentences from earlier extraction
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    texts = meta["sentences"]
    label_names = meta["label_names"]
    print(f"Sentences: {len(texts)} (across 28 emotions)")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Pass 1: baseline forward — also used to fit the global LDA
    print("\n[1/N] Forward on baseline (rank=768)")
    baseline = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, output_hidden_states=True, output_attentions=False,
    ).eval().to(device)
    Wp = baseline.bert.pooler.dense.weight.detach().cpu().numpy()
    bp = baseline.bert.pooler.dense.bias.detach().cpu().numpy()
    t0 = time.time()
    cls_baseline = forward_all(baseline, tokenizer, texts, device)   # (N, 13, 768)
    print(f"  baseline forward: {time.time()-t0:.1f}s")

    # Apply pooler to baseline L12 then fit LDA — gives the "emotion-discriminative
    # axes" that show class structure most cleanly. Compressed runs project onto
    # the SAME axes so degradation appears as cluster blurring (real drift).
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    pooled_baseline_l12 = np.tanh(cls_baseline[:, 12, :] @ Wp.T + bp)
    label_to_int = {e: i for i, e in enumerate(sorted(set(label_names)))}
    y = np.array([label_to_int[l] for l in label_names])
    lda = LinearDiscriminantAnalysis(n_components=3)
    lda.fit(pooled_baseline_l12, y)
    print(f"  LDA class variance explained: {lda.explained_variance_ratio_.sum():.2%}")

    def project(cls_arr: np.ndarray) -> np.ndarray:
        """Apply pooler then LDA to (N, L, 768) → (L, N, 3)."""
        pooled = np.tanh(cls_arr @ Wp.T + bp)   # (N, L, 768)
        out = np.zeros((cls_arr.shape[1], cls_arr.shape[0], 3), dtype=np.float32)
        for L in range(cls_arr.shape[1]):
            out[L] = lda.transform(pooled[:, L, :])
        return out

    coords_per_rank = np.zeros((len(RANKS), 13, len(texts), 3), dtype=np.float32)
    coords_per_rank[0] = project(cls_baseline)

    del baseline
    if device == "mps":
        try: torch.mps.empty_cache()
        except: pass

    # Pass 2..N: compressed forward at each rank
    for ri, rank in enumerate(RANKS[1:], start=1):
        print(f"\n[{ri+1}/{len(RANKS)}] Compressing to rank={rank} ...")
        t0 = time.time()
        cpu_baseline = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, output_hidden_states=True, output_attentions=False,
        ).eval()
        rank_dict = build_rank_dict(rank)
        compressed = apply_svd_compression(cpu_baseline, rank=rank_dict, inplace=True)
        compressed.to(device)
        compressed.eval()
        print(f"  build+move: {time.time()-t0:.1f}s")

        t0 = time.time()
        cls_comp = forward_all(compressed, tokenizer, texts, device)
        print(f"  forward: {time.time()-t0:.1f}s")

        coords_per_rank[ri] = project(cls_comp)

        del compressed, cpu_baseline
        if device == "mps":
            try: torch.mps.empty_cache()
            except: pass

    # Save
    out = CACHE_DIR / "compression_decay.npz"
    np.savez_compressed(
        out,
        coords=coords_per_rank,         # (n_ranks, 13, N, 3)
        ranks=np.array(RANKS),
        labels=np.array(label_names),
    )
    print(f"\n✓ saved {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print(f"  shape: {coords_per_rank.shape}")


if __name__ == "__main__":
    main()
