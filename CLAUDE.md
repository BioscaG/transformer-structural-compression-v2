# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TFG (Thesis) project: **"Compresión Selectiva y Análisis Estructural de Modelos basados en Arquitectura Transformer mediante SVD"**.

Two main goals:
1. **Mechanistic interpretability**: Understand WHERE and HOW BERT processes each of 28 emotions (probing, activation patching, head ablation, neuron analysis)
2. **Informed compression**: Use that understanding to compress BERT via SVD in a way that protects emotion-critical components

Model: BERT-base-uncased fine-tuned on GoEmotions (28 emotion labels, multi-label classification).

## Setup

```bash
pip install -r requirements.txt
```

## Development Workflow

Development is notebook-driven. Run notebooks in order:

### Act 1: Compression Analysis (foundation)
1. `notebooks/01_finetuning.ipynb` — Fine-tune BERT baseline on GoEmotions
2. `notebooks/02_spectral_analysis.ipynb` — SVD spectral anatomy of BERT's 72 weight matrices, QKV subspace overlap, adaptive compression, Pareto frontier
3. `notebooks/03_compression_sensitivity.ipynb` — Exhaustive sensitivity analysis by component type and layer depth, t-SNE + silhouette scores

### Act 2: Mechanistic Interpretability (core contribution)
4. `notebooks/04_probing.ipynb` — Layer-wise probing classifiers: where each emotion crystallizes
5. `notebooks/05_activation_patching.ipynb` — Causal localization: patching activations to find where emotions live
6. `notebooks/06_head_analysis.ipynb` — Attention head ablation: which of 144 heads detect which emotions
7. `notebooks/07_neuron_analysis.ipynb` — FFN neuron specialization: emotion-specific neurons and their connection to SVD

### Act 3: Synthesis
8. `notebooks/08_emotional_map.ipynb` — Lesion study + emotional genealogy + survival curves
9. `notebooks/09_informed_compression.ipynb` — Informed vs blind compression, fine-tuning recovery, inference benchmarks, final Pareto

Old notebooks archived in `notebooks/archive/`.

Notebooks are compatible with both local execution and Google Colab Pro. Model checkpoints save to `results/bert-goemotions-final/`.

## Architecture

Four modules in `src/`:

- **`src/data/dataset.py`** — Loads GoEmotions dataset, tokenizes with BERT tokenizer, converts labels to multi-hot vectors. Key constants: `NUM_LABELS=28`, `MAX_LENGTH=128`, `MODEL_NAME="bert-base-uncased"`.

- **`src/models/classifier.py`** — Loads BERT with multi-label classification head (BCEWithLogitsLoss via `problem_type="multi_label_classification"`).

- **`src/compression/svd.py`** — Core module (~270 lines). `SVDLinear` replaces `nn.Linear` with two low-rank layers (Vh_k → U_k·diag(S_k)). Key functions:
  - `apply_svd_compression(model, rank, layer_names)` — Main entry point. Supports uniform rank (int) or per-layer ranks (dict).
  - `get_target_layer_names(model)` / `filter_layer_names(names, component, layers)` — Select layers by component type ("query", "key", "value", "attention", "ffn", etc.) and layer index (0-11).
  - `compute_singular_value_energy(model)` / `compute_adaptive_ranks(energy_info, threshold)` — Energy-based adaptive rank selection.

- **`src/utils/metrics.py`** — `compute_metrics()` for Hugging Face Trainer: sigmoid → threshold 0.5 → F1 macro/micro/per-emotion.

## Key Design Decisions

- SVD decomposition is computed in float32 for numerical stability, then cast back to original dtype
- Compression operates on a deep copy by default (`inplace=False`)
- Classifier head and pooler are excluded from compression (only `bert.encoder` layers targeted)
- Multi-hot encoding with BCEWithLogitsLoss for multi-label classification
- Probing uses LogisticRegression on CLS token hidden states from each layer
- Activation patching uses PyTorch forward hooks to inject baseline activations into compressed models
- Head ablation zeros out head-specific slices (h*64 : (h+1)*64) in self-attention output

## Python API

```python
from src.data import load_goemotions
from src.models import load_bert_classifier
from src.compression import apply_svd_compression, compute_singular_value_energy
from src.utils import compute_metrics
```

## Known Issues
- `filter_layer_names` with component="ffn_output" also matches attention.output.dense. Workaround: filter out names containing "attention" when using ffn_output.
