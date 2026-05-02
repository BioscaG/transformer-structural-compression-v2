"""Extract real BERT-base-uncased GoEmotions activations and cache them.

Runs ONE forward pass per sentence through the fine-tuned model exposing all
hidden states and attentions, caches the data needed for downstream
visualizations:

  - Per-layer [CLS] embeddings for N sentences  (for galaxy + probing)
  - Per-layer attention weights for the top-K critical heads
  - Pre-trained logistic regression probes (one per layer, 28 emotions)
  - Final classifier output for ground-truth probability comparison

Usage:
  .viz_venv/bin/python viz/data/extract_real.py                # default 600 sentences
  .viz_venv/bin/python viz/data/extract_real.py --n-sentences 1500
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Use the user's actual fine-tuned 23-emotion checkpoint.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAME = str(PROJECT_ROOT / "results" / "checkpoints" / "23emo-final")
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


# 23 active emotions matching the user's checkpoint (5 excluded: neutral, grief,
# nervousness, pride, relief).
GOEMOTIONS_28 = [
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


def load_model_and_tokenizer():
    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    mdl = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, output_hidden_states=True, output_attentions=True,
    )
    mdl.eval()
    return mdl, tok


@torch.no_grad()
def forward_batch(model, tokenizer, texts: list[str], device: str, max_len: int = 64):
    """Single batched forward pass. Returns dicts of arrays.

    Returns
    -------
    hidden : (n_texts, n_layers=13, hidden_size=768) array of CLS embeddings
    logits : (n_texts, 28) array of final classifier logits
    attentions : (n_texts, n_layers=12, n_heads=12, seq, seq) attention tensor (only stored truncated to seq=16)
    """
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_len,
                    return_tensors="pt").to(device)
    out = model(**enc)
    hidden = torch.stack(out.hidden_states, dim=1)        # (B, 13, T, 768)
    cls = hidden[:, :, 0, :]                              # (B, 13, 768)
    attn = torch.stack(out.attentions, dim=1)             # (B, 12, 12, T, T)
    # Truncate attention to first 16 positions to keep cache small
    T = min(16, attn.shape[-1])
    attn = attn[:, :, :, :T, :T]
    return {
        "cls": cls.cpu().numpy(),
        "logits": out.logits.cpu().numpy(),
        "attention": attn.cpu().numpy(),
        "input_ids": enc["input_ids"].cpu().numpy(),
        "attention_mask": enc["attention_mask"].cpu().numpy(),
    }


def extract_embeddings(n_sentences: int, batch_size: int = 16) -> dict:
    """Run BERT on N test-set sentences and return a dict suitable for caching.

    Per-layer CLS for each sentence + per-sentence labels + final logits.
    """
    device = get_device()
    print(f"Device: {device}")

    model, tok = load_model_and_tokenizer()
    model.to(device)

    print("Loading go_emotions test split ...")
    ds = load_dataset("go_emotions", "raw", split="train", streaming=False)
    # Balanced sampling: n_per_emotion across the 23 active emotions.
    # Critically, we filter to SINGLE-LABEL sentences — frases con UNA sola
    # emoción activa en el gold. This avoids the "between centroids" mush that
    # multi-label sentences cause in the projection.
    rng = np.random.default_rng(42)
    n_per_emotion = max(1, n_sentences // len(GOEMOTIONS_28))
    excluded = {"neutral", "grief", "nervousness", "pride", "relief"}

    selected = []
    counts = {e: 0 for e in GOEMOTIONS_28}
    skipped_multi = 0
    indices = rng.permutation(len(ds))
    for idx in indices:
        if len(selected) >= n_sentences:
            break
        ex = ds[int(idx)]
        # Skip if any excluded emotion is present
        if any(ex.get(e, 0) == 1 for e in excluded):
            continue
        # Find ALL active emotions in the gold
        active = [e for e in GOEMOTIONS_28 if ex.get(e, 0) == 1]
        # SINGLE-LABEL FILTER: skip sentences with more than one emotion
        if len(active) != 1:
            skipped_multi += 1
            continue
        emo = active[0]
        if counts[emo] >= n_per_emotion:
            continue
        counts[emo] += 1
        selected.append({"text": ex["text"], "label": GOEMOTIONS_28.index(emo),
                         "label_name": emo, "id": ex.get("id", "")})

    print(f"Selected {len(selected)} single-label sentences "
          f"(skipped {skipped_multi} multi-label)")

    print(f"Selected {len(selected)} sentences across {sum(1 for c in counts.values() if c > 0)} emotions")

    # Forward in batches
    all_cls = []
    all_logits = []
    all_attn = []
    all_input_ids = []

    t0 = time.time()
    for i in range(0, len(selected), batch_size):
        batch = selected[i : i + batch_size]
        texts = [b["text"] for b in batch]
        out = forward_batch(model, tok, texts, device)
        all_cls.append(out["cls"])
        all_logits.append(out["logits"])
        all_attn.append(out["attention"])
        all_input_ids.append(out["input_ids"])
        print(f"  batch {i // batch_size + 1} / {(len(selected) + batch_size - 1) // batch_size} "
              f"({time.time() - t0:.1f}s)")

    # Concatenate (handle variable seq lengths via padding to max in each batch)
    max_seq = max(a.shape[-1] for a in all_attn)
    max_input = max(ids.shape[-1] for ids in all_input_ids)

    def pad_attn(a, target):
        if a.shape[-1] == target:
            return a
        pad = target - a.shape[-1]
        return np.pad(a, ((0, 0), (0, 0), (0, 0), (0, pad), (0, pad)),
                      constant_values=0)

    def pad_ids(a, target):
        if a.shape[-1] == target:
            return a
        return np.pad(a, ((0, 0), (0, target - a.shape[-1])), constant_values=0)

    cls_arr = np.concatenate(all_cls, axis=0).astype(np.float32)              # (N, 13, 768)
    logits_arr = np.concatenate(all_logits, axis=0).astype(np.float32)        # (N, 28)
    attn_arr = np.concatenate([pad_attn(a, max_seq) for a in all_attn], axis=0).astype(np.float16)
    ids_arr = np.concatenate([pad_ids(ids, max_input) for ids in all_input_ids], axis=0)

    # Token strings for attention overlay (decoded from input_ids)
    token_strs = []
    for ids in ids_arr:
        toks = [tok.convert_ids_to_tokens(int(t)) if int(t) > 0 else "" for t in ids]
        token_strs.append(toks)

    return {
        "model_name": MODEL_NAME,
        "labels": GOEMOTIONS_28,
        "sentences": [b["text"] for b in selected],
        "label_indices": [b["label"] for b in selected],
        "label_names": [b["label_name"] for b in selected],
        "cls_per_layer": cls_arr,                # (N, 13, 768)
        "final_logits": logits_arr,              # (N, 28)
        "attentions": attn_arr,                  # (N, 12, 12, T, T) float16
        "tokens": token_strs,                    # list of lists
    }


def train_layer_probes(cache: dict) -> dict:
    """Train one logistic regression probe per layer over the multi-label
    target. Used by the sentence inspector to show 'what emotion would a
    linear probe read from this layer'."""
    from sklearn.linear_model import LogisticRegression

    cls = cache["cls_per_layer"]                   # (N, 13, 768)
    labels = np.array(cache["label_indices"])      # (N,)
    n_layers = cls.shape[1]
    n_classes = len(cache["labels"])

    # One probe per (layer, emotion) — trained on whether-emotion-is-the-label
    # = a 28-class softmax probe per layer.
    print(f"Training {n_layers} probes (one per layer, multinomial)...")
    probes = []
    for L in range(n_layers):
        clf = LogisticRegression(C=1.0, max_iter=400)
        try:
            clf.fit(cls[:, L, :], labels)
            probes.append({"coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
                           "classes": clf.classes_.tolist()})
            print(f"  layer {L}: {clf.score(cls[:, L, :], labels):.3f} train accuracy")
        except Exception as e:
            print(f"  layer {L} failed: {e}")
            probes.append(None)
    return probes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-sentences", type=int, default=2300)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cache = extract_embeddings(args.n_sentences, args.batch_size)

    # Save: split into separate files because some are large
    np.savez_compressed(CACHE_DIR / "activations.npz",
                        cls_per_layer=cache["cls_per_layer"],
                        final_logits=cache["final_logits"],
                        attentions=cache["attentions"])

    meta = {
        "model_name": cache["model_name"],
        "labels": cache["labels"],
        "sentences": cache["sentences"],
        "label_indices": cache["label_indices"],
        "label_names": cache["label_names"],
        "tokens": cache["tokens"],
        "n_sentences": len(cache["sentences"]),
    }
    (CACHE_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\n✓ Cached to {CACHE_DIR}")
    print(f"  activations.npz: {(CACHE_DIR / 'activations.npz').stat().st_size / 1e6:.1f} MB")
    print(f"  meta.json:       {(CACHE_DIR / 'meta.json').stat().st_size / 1e3:.1f} KB")


if __name__ == "__main__":
    main()
