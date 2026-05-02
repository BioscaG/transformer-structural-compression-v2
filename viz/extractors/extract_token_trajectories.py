"""Extract per-token hidden states for ~30 curated sentences across 13 layers.

The standard activations cache only stores the [CLS] token (position 0).
This script keeps the FULL token sequence so we can show how each token
moves through the residual stream — not just CLS. Used by the
token_trajectories visualization.

Output: viz/data/cache/token_trajectories.npz
  - hidden_per_layer: (n_sentences, 13, max_T, 768)
  - token_strs: per-sentence token list
  - sentences, labels: meta

Default: ~30 sentences (1-2 per emotion). Storage: 30 × 13 × 16 × 768 × 4 ≈ 19 MB.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAME = str(PROJECT_ROOT / "results" / "checkpoints" / "23emo-final")
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"

EMOTIONS_23 = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "joy", "love",
    "optimism", "realization", "remorse", "sadness", "surprise",
]


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(n_per_emotion: int = 2, max_len: int = 16):
    """Extract token-level hidden states for n_per_emotion sentences per gold."""
    device = get_device()
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, output_hidden_states=True, output_attentions=False,
    ).to(device).eval()

    # Curated sample: pick `n_per_emotion` short sentences per gold (single-label)
    print("Sampling go_emotions ...")
    ds = load_dataset("go_emotions", "raw", split="train", streaming=False)
    excluded = {"neutral", "grief", "nervousness", "pride", "relief"}
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(ds))

    selected = []
    counts = {e: 0 for e in EMOTIONS_23}
    for idx in indices:
        if all(c >= n_per_emotion for c in counts.values()):
            break
        ex = ds[int(idx)]
        if any(ex.get(e, 0) == 1 for e in excluded):
            continue
        active = [e for e in EMOTIONS_23 if ex.get(e, 0) == 1]
        if len(active) != 1:
            continue
        emo = active[0]
        if counts[emo] >= n_per_emotion:
            continue
        # Prefer shorter sentences for cleaner visuals
        text = ex["text"]
        if len(text.split()) > 12:
            continue
        counts[emo] += 1
        selected.append((emo, text))

    print(f"Selected {len(selected)} sentences (single-label, ≤12 words)")

    # Forward each sentence keeping ALL tokens
    n = len(selected)
    n_layers = 13
    hidden_per_layer = np.zeros((n, n_layers, max_len, 768), dtype=np.float16)
    token_strs = []

    for i, (emo, text) in enumerate(selected):
        enc = tok(text, return_tensors="pt", padding="max_length",
                  truncation=True, max_length=max_len).to(device)
        with torch.no_grad():
            out = mdl(**enc)
        hs = torch.stack(out.hidden_states, dim=0).squeeze(1)   # (13, T, 768)
        hidden_per_layer[i] = hs.cpu().numpy().astype(np.float16)
        toks = [tok.convert_ids_to_tokens(int(t)) for t in enc["input_ids"][0]]
        token_strs.append(toks)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}")

    # Pretty-format tokens
    pretty = []
    for toks in token_strs:
        out_toks = []
        for t in toks:
            if t == "[PAD]":
                out_toks.append("")
            elif t == "[CLS]":
                out_toks.append("⟨CLS⟩")
            elif t == "[SEP]":
                out_toks.append("⟨SEP⟩")
            elif t.startswith("##"):
                out_toks.append(t[2:])
            else:
                out_toks.append(t)
        pretty.append(out_toks)

    np.savez_compressed(
        CACHE_DIR / "token_trajectories.npz",
        hidden=hidden_per_layer,
    )
    meta = {
        "sentences": [s for _, s in selected],
        "labels": [e for e, _ in selected],
        "tokens": pretty,
        "max_len": max_len,
    }
    (CACHE_DIR / "token_trajectories_meta.json").write_text(json.dumps(meta, indent=2))

    sz = (CACHE_DIR / "token_trajectories.npz").stat().st_size / 1e6
    print(f"\n✓ saved token_trajectories.npz ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
