"""Extract per-neuron activations for the top neurons in notebook 7's catalog.

For each (layer, neuron_idx) listed in `notebook7/neuron_catalog.csv`, hook
the FFN intermediate output at that layer and capture the neuron's
activation across all 2300 cached test sentences. From that we can find
the top-K sentences that most strongly activate each neuron.

Output: viz/data/cache/neuron_activations.npz
  - top_neurons: (n_neurons, 3) array of (layer, idx, abs_sel)
  - activations: (n_sentences, n_neurons) post-GELU activations
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL_NAME = str(PROJECT_ROOT / "results" / "checkpoints" / "23emo-final")
CSVS = PROJECT_ROOT / "results" / "csvs"
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main(top_n: int = 80):
    device = get_device()
    print(f"Device: {device}")

    # Load top neurons from notebook 7 catalog
    cat = pd.read_csv(CSVS / "notebook7" / "neuron_catalog.csv")
    cat = cat.sort_values("abs_selectivity", ascending=False)
    # Take top_n unique (layer, neuron) pairs
    seen = set()
    neurons = []
    for _, row in cat.iterrows():
        key = (int(row["layer"]), int(row["neuron"]))
        if key in seen:
            continue
        seen.add(key)
        neurons.append({
            "layer": int(row["layer"]),
            "neuron": int(row["neuron"]),
            "emotion": row["emotion"],
            "selectivity": float(row["selectivity"]),
            "abs_selectivity": float(row["abs_selectivity"]),
            "direction": row["direction"],
        })
        if len(neurons) >= top_n:
            break
    print(f"Selected top {len(neurons)} unique neurons")

    # Group by layer for efficient hooking
    by_layer: dict[int, list[int]] = {}
    for n in neurons:
        by_layer.setdefault(n["layer"], []).append(n["neuron"])

    # Load cached sentences
    meta = json.loads((CACHE_DIR / "meta.json").read_text())
    sentences = meta["sentences"]
    label_names = meta["label_names"]
    print(f"Will run on {len(sentences)} cached sentences")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()

    # Hook: capture intermediate (post-GELU) activations
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inp, out):
            captured[layer_idx] = out.detach()
        return hook

    handles = []
    for L in by_layer:
        target = mdl.bert.encoder.layer[L].intermediate
        handles.append(target.register_forward_hook(make_hook(L)))

    # Run forward in batches
    activations = np.zeros((len(sentences), len(neurons)), dtype=np.float32)
    batch_size = 32
    print("Forward passes...")
    with torch.no_grad():
        for bi in range(0, len(sentences), batch_size):
            batch = sentences[bi: bi + batch_size]
            enc = tok(batch, padding=True, truncation=True, max_length=64,
                      return_tensors="pt").to(device)
            captured.clear()
            mdl(**enc)
            # For each neuron, get its mean activation over sequence positions
            for ni, n in enumerate(neurons):
                act = captured[n["layer"]]    # (B, T, 3072)
                # Mean over non-pad positions
                mask = enc["attention_mask"].unsqueeze(-1).float()
                vals = (act[:, :, n["neuron"]] * mask.squeeze(-1)).sum(dim=1) / mask.sum(dim=(1, 2))
                activations[bi: bi + len(batch), ni] = vals.cpu().numpy()
            if (bi // batch_size) % 5 == 0:
                print(f"  batch {bi // batch_size + 1}/{(len(sentences) + batch_size - 1) // batch_size}")

    for h in handles:
        h.remove()

    # Find top-K activating sentences per neuron
    print("\nTop-K computation...")
    top_k = 5
    top_indices = np.zeros((len(neurons), top_k), dtype=np.int32)
    for ni in range(len(neurons)):
        # For excitatory neurons: top by largest activation
        # For inhibitory: top by most negative activation
        signed = activations[:, ni]
        if neurons[ni]["direction"] == "inhibitory":
            top_indices[ni] = np.argsort(signed)[:top_k]
        else:
            top_indices[ni] = np.argsort(-signed)[:top_k]

    # Save
    np.savez_compressed(
        CACHE_DIR / "neuron_activations.npz",
        activations=activations,
        top_indices=top_indices,
    )
    payload = {
        "neurons": neurons,
        "top_k": top_k,
    }
    (CACHE_DIR / "neuron_activations_meta.json").write_text(json.dumps(payload, indent=2))

    sz_npz = (CACHE_DIR / "neuron_activations.npz").stat().st_size / 1e6
    print(f"\n✓ saved neuron_activations.npz ({sz_npz:.1f} MB)")
    print(f"  shape: ({len(sentences)} sentences, {len(neurons)} neurons)")


if __name__ == "__main__":
    main()
