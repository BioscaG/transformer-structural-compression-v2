# `viz/` — Interactive companion to the TFG memoria

Live, real-data interactive companion to *Anatomía Emocional de un Modelo
Transformer*. Combines a Distill-style scrollytelling website (8 Plotly
figures + 1 custom D3.js network) with a live Gradio app that runs the actual
fine-tuned BERT model and applies SVD compression in real time.

## Quick start

```bash
# One-time setup
python3 -m venv .viz_venv
.viz_venv/bin/pip install plotly numpy scipy scikit-learn pandas matplotlib \
                          kaleido torch transformers gradio datasets

# Step 1 — extract real activations from BERT-base-uncased-GoEmotions
.viz_venv/bin/python viz/data/extract_real.py --n-sentences 600

# Step 2 — build all the static and interactive HTML
.viz_venv/bin/python viz/build_all.py

# Step 3 — open the showcase
open viz/output/index.html

# Step 4 — launch the live sentence inspector (separate terminal)
.viz_venv/bin/python viz/interactive/sentence_inspector.py
# Opens at http://localhost:7860
```

## What's where

```
viz/
├── thesis_data.py             # Single source of truth: every numerical result
│                                from the memoria, encoded as Python data
├── style.py                   # Plotly styling that matches plots/tfg_plot_style.py
├── data/
│   ├── extract_real.py        # Run BERT on N sentences, cache activations + probes
│   └── cache/                 # ~60MB — activations.npz, meta.json, probes.json
│                                (gitignored; regenerate by running extract_real)
├── static/
│   ├── pareto_3d.py           # 22 strategies + 3D phase-transition surface
│   ├── heads_matrix.py        # 144-head taxonomy with critical-head stars
│   ├── crystallization.py     # 23 emotions × 13 layers heatmap with cluster ribbon
│   ├── sunburst.py            # 6 emergent clusters + selectivity-norm bars
│   ├── fingerprints.py        # Emotion landscape + per-emotion radar
│   └── lesion_replay.py       # Activation patching: the "patient revives"
├── interactive/
│   ├── compression_sandbox.py # Pure HTML/JS — design your own SVD compression
│   ├── galaxy_formation.py    # 3D animated CLS evolution across 13 layers (REAL data)
│   ├── circuit_network.py     # Custom D3.js: emotion ↔ head ↔ cluster network
│   └── sentence_inspector.py  # Live Gradio app with the real fine-tuned model
├── build_all.py               # Renders every HTML + the scrollytelling index
└── output/                    # Generated HTML files (gitignored)
```

## The complete tour

| # | Where | What | Tech |
|---|-------|------|------|
| 1 | `01_pareto_landscape` | Frontera de Pareto + acantilado 3D | Plotly |
| 2 | `02_heads_matrix` | 144 cabezas + 11 críticas marcadas | Plotly |
| 3 | `03_crystallization` | Heatmap 23×13 con cluster ribbon | Plotly |
| 4 | `04_sunburst_clusters` | 6 clusters + barras de selectividad | Plotly |
| 5 | `05_compression_sandbox` | 18 sliders, F1 estimado en vivo | HTML+JS puro |
| 6 | `06_emotional_landscape` | Paisaje 2D + radar polar dinámico | Plotly |
| 7 | `07_galaxy_formation` | Galaxy 3D con **datos reales** | Plotly + sklearn |
| 8 | `08_lesion_replay` | Activation patching dramático | Plotly |
| 9 | `09_circuit_network` | Grafo emoción↔cabeza↔cluster (D3.js custom) | D3.js |
| ★ | `sentence_inspector` | **Gradio live · BERT real · SVD live** | torch + transformers + gradio |
| △ | `index.html` | Scrollytelling Distill-style con scroll triggers | vanilla JS |

## Real vs estimated data

The visualizations mix three data tiers:

**Real** (run on actual model):
- Galaxy formation 3D — 588 GoEmotions sentences, real CLS embeddings at all
  13 layers, real PCA projection.
- Sentence inspector — real model loaded in memory, real attention weights,
  real SVD compression applied to the actual weight matrices, recomputed
  predictions on every slider change.
- Crystallization heatmap shape — 13 trained linear probes (one per layer)
  with real test accuracies.

**From thesis tables** (verbatim):
- Pareto strategy points (22 strategies × F1 × ratio).
- Component sensitivities at r=128 (Tabla 9).
- Depth-band retention (Tabla 10).
- Activation patching restoration (Tablas 14, 15, 16).
- Critical heads per emotion (Tabla 19).
- Neuron counts and selectivity norms (Tablas 20, 21).
- Fine-tuning recovery per emotion (Tabla 29).

**Estimated/interpolated** (flagged with `ESTIMATED` in `thesis_data.py`):
- Crystallization layers for 12 of 23 emotions (the 11 named in Tabla 12 are
  exact).
- Critical attention heads for 13 emotions (10 from Tabla 19 are exact).
- Per-emotion neuron counts for 15 emotions (8 from Tabla 21 are exact).
- 144-head category grid: per-band totals from Tabla 18 are exact, exact
  category per (layer, head) is synthesized with seed 42.
- Sandbox F1 estimator: damage model fitted to real Tabla 9/10 retention
  values — accurate at the calibration points, interpolates between them.

To swap in real values for the estimated entries, regenerate the relevant
notebook (04 for crystallization, 06 for heads, 07 for neurons), update
`thesis_data.py`, and re-run `build_all.py`. The estimated entries are easy
to find:

```bash
grep -n ESTIMATED viz/thesis_data.py
```

## Model used for the live demo

`justin871030/bert-base-uncased-goemotions-original-finetuned` — a public
BERT-base-uncased fine-tuned on the **full 28-emotion** GoEmotions split (the
thesis filters to 23). Architecturally identical to the thesis model
(BERT-base-uncased, 12 layers, 12 heads, 768 hidden, 110M params); the
classifier head dimensions differ (28 vs 23 outputs) and the training data is
not filtered. To swap to *your* checkpoint, change `MODEL_NAME` in:

- `viz/data/extract_real.py`
- `viz/interactive/sentence_inspector.py`

…to a local path or HF Hub identifier, and regenerate the cache.

## Defense workflow

For the live defense:

1. Pre-launch the inspector in a terminal: `.viz_venv/bin/python viz/interactive/sentence_inspector.py`
2. Open the showcase: `open viz/output/index.html`
3. During the talk: scroll through the showcase as your slide deck. The
   sidebar nav doubles as a TOC.
4. When asked about a specific emotion: switch to the Emotion Landscape
   panel (or the radar) and select it.
5. When asked "is this real or simulated?": switch to the live Sentence
   Inspector, type the example, mash the compression sliders, and watch the
   probabilities re-rank in real time.

For LinkedIn / Twitter: record a 30-second screen capture of the galaxy
animation and a 60-second one of the sandbox sliders. Both designed to look
striking in motion.

## Performance notes

- Activation extraction: 588 sentences × forward + 13 probes ≈ **3.5s on Mac
  M-series (MPS)**, ~30s on CPU.
- Sentence inspector first-load: ~10s (model + tokenizer + probes).
- Inference per text input: ~50ms.
- SVD compression first apply: ~6s (decomposes 48-72 weight matrices). Cached
  per rank-config.
- All static HTMLs render in <1s each via `build_all.py`.

## Acknowledgements

Stylesheet ported from `plots/tfg_plot_style.py` for visual coherence with
the matplotlib figures already in the LaTeX memoria.
