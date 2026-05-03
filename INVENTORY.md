# Inventario de CSVs · qué hay, quién lo usa, qué borrar

Generado automáticamente desde el cross-reference de `results/csvs/` contra `latex_figures/*.ipynb` y `viz/`.

Convenciones:
- ✅ usado por al menos una figura
- 🔄 compartido entre PDF y web
- 🗑 huérfano (existe pero nadie lo usa)
- ⚠ referenciado pero no presente en `results/csvs/`

---

## `results/csvs/notebook2/`

- **`adaptive_compression_results.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`adaptive_rank_distribution.csv`** — ✅ sólo PDF
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
- **`mixed_compression_results.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`qkv_subspace_overlap.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`rank_matrix_k90.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`rank_matrix_k95.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`rank_matrix_k99.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`singular_values_by_component.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`spectral_analysis_results.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`uniform_compression_results.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`uniform_emotion_f1.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)

## `results/csvs/notebook3/`

- **`analysis_component_results.csv`** — 🗑 huérfano
- **`analysis_depth_results.csv`** — 🗑 huérfano
- **`baseline_info.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`component_f1_matrix.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`component_sensitivity.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`component_silhouette_matrix.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`depth_f1_matrix.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`depth_sensitivity.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap4_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`depth_silhouette_matrix.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`per_emotion_results.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`sample_points.csv`** — 🗑 huérfano
- **`sensitivity_master.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`top_emotions.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`tsne_bounds.csv`** — 🗑 huérfano
- **`tsne_component.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`tsne_depth.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)

## `results/csvs/notebook4/`

- **`crystallization_layers.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap5_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
  - viz/interactive/iterative_inference.py (web, Plotly)
  - viz/interactive/lens_vs_probe.py (web, Plotly)
- **`emotion_frequency.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap5_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`layer_information_gain.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap5_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
  - viz/interactive/lexical_to_semantic.py (web, Plotly)
- **`probe_results.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap5_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
  - viz/interactive/lens_vs_probe.py (web, Plotly)
- **`probe_results_long.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`probe_vs_lesion.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`probing_correlations.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap5_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`probing_summary.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)

## `results/csvs/notebook5/`

- **`activation_patching_per_component.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`activation_patching_per_layer.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`activation_patching_summary.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`baseline_vs_compressed.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`critical_layer_per_emotion.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`mean_restoration_per_layer.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`patching_master_summary.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`patching_vs_lesion.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`restoration_matrix.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)

## `results/csvs/notebook6/`

- **`head_ablation_long.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`head_ablation_results.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`head_analysis_summary.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`head_categories.csv`** — 🔄 compartido (PDF + web)
  - latex_figures/generate_cap5_figures.ipynb (PDF, matplotlib)
  - viz/data/load_results.py (web, Plotly)
- **`head_correlation_matrix.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`head_importance_matrix.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`head_importance_matrix.npy`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`head_redundancy_pairs.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`head_top_emotions.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`layer_attention_importance.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`layer_redundancy_matrix.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`top_heads_per_emotion.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)

## `results/csvs/notebook7/`

- **`neuron_catalog.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
  - viz/extractors/extract_neuron_activations.py (web, Plotly)
- **`neuron_emotion_clusters.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`neuron_selectivity.pkl`** — 🗑 huérfano
- **`neuron_significant_counts.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`svd_neuron_connection.pkl`** — 🗑 huérfano

## `results/csvs/notebook8/`

- **`lesion_per_component.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`lesion_per_layer.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`lesion_summary.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)

## `results/csvs/notebook9/`

- **`compression_comparison.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
  - viz/interactive/greedy_replay.py (web, Plotly)
- **`finetuning_recovery.csv`** — ✅ sólo web
  - viz/data/load_results.py (web, Plotly)
- **`greedy_50_ranks.csv`** — 🗑 huérfano
- **`greedy_60_ranks.csv`** — 🗑 huérfano
- **`greedy_70_ranks.csv`** — 🗑 huérfano
- **`greedy_75_ranks.csv`** — 🗑 huérfano
- **`greedy_80_ranks.csv`** — 🗑 huérfano
- **`greedy_85_ranks.csv`** — 🗑 huérfano
- **`greedy_90_ranks.csv`** — 🗑 huérfano
- **`greedy_95_ranks.csv`** — 🗑 huérfano
- **`informed_aggressive_ranks.csv`** — 🗑 huérfano
- **`informed_light_ranks.csv`** — 🗑 huérfano
- **`informed_moderate_ranks.csv`** — 🗑 huérfano

---

## Resumen

- **77 archivos** en `results/csvs/`
- **14 compartidos** (PDF + web)
- **1 sólo PDF**
- **45 sólo web**
- **17 huérfanos** (candidatos a borrar al cierre del TFG)

## Verificación en Drive

Los notebooks `notebooks/01-09.ipynb` se ejecutan en Colab y guardan los CSVs en Drive. Para que tu pipeline local funcione, todos estos archivos deben estar copiados desde Drive a `results/csvs/`. La auditoría de arriba confirma que **todos los archivos referenciados están presentes localmente**.

Si reejecutas un notebook en Colab y quieres sincronizar, copia la carpeta correspondiente de Drive a tu repo local:

```bash
# Ejemplo: notebook 4 reejecutado en Colab
rsync -av /Volumes/GoogleDrive/MyDrive/transformer-structural-compression-v2/results/csvs/notebook4/ results/csvs/notebook4/
```