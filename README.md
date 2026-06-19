# Anatomía Emocional de un Modelo Transformer

**Compresión selectiva e interpretabilidad mecánica de BERT-base sobre GoEmotions, mediante SVD.**

Trabajo de Fin de Grado · Guido Biosca Lasa · Facultat d'Informàtica de Barcelona (FIB-UPC) · 2026
Director: Lluís Padró Cirera.

| Recurso | Enlace |
| --- | --- |
| 📄 **Memoria final (PDF)** | [`TFG___Final.pdf`](TFG___Final.pdf) |
| 🌐 **Web interactiva** | <https://anatomy.guidobiosca.com> |
| 📝 **Fuente LaTeX de la memoria** | [`deliver/`](deliver/) |

---

## De qué va

El proyecto persigue dos objetivos encadenados sobre **BERT-base-uncased**
fine-tuneado en **GoEmotions** para clasificación multi-etiqueta. El baseline
inicial usa las 28 etiquetas del dataset (notebook 01); los análisis finales
de la memoria trabajan sobre un modelo filtrado a **23 emociones** (notebook
01b, que excluye *neutral* y cuatro emociones con F1 ≈ 0 en el baseline).

1. **Interpretabilidad mecánica** — entender *dónde* y *cómo* procesa el
   modelo cada emoción, contrastando **cinco** técnicas complementarias:
   probing lineal por capa, *logit lens*, *activation patching*, ablación de
   las 144 cabezas de atención y análisis de selectividad de las neuronas FFN.
2. **Compresión informada** — usar ese conocimiento para comprimir el modelo
   vía **SVD** protegiendo los componentes críticos para la emoción, en lugar
   de comprimir a ciegas.

La descomposición SVD reemplaza cada `nn.Linear` por dos capas de rango bajo
(`Vh_k` → `U_k·diag(S_k)`), con rangos uniformes, adaptativos por energía
espectral o seleccionados por una búsqueda greedy informada.

---

## Estructura del repositorio

```
transformer-structural-compression-v2/
│
├── TFG___Final.pdf                 ← la memoria final
├── deliver/                        ← fuente LaTeX de la memoria (capítulos, .bib)
│
├── src/                            código compartido (instalable como paquete)
│   ├── data/                         carga GoEmotions, tokenización, multi-hot
│   ├── models/                       BERT con cabeza multi-label (BCEWithLogits)
│   ├── compression/                  SVD, rangos adaptativos, greedy informado
│   └── utils/                        métricas (F1 macro/micro/por emoción)
│
├── notebooks/                      LA INVESTIGACIÓN (ejecutar en orden)
│   ├── 01_finetuning.ipynb           baseline 28 emociones
│   ├── 01b_finetuning_23emotions.ipynb  modelo final (23 emociones)
│   ├── 02_spectral_analysis.ipynb    SVD, k95, frontera de Pareto    ┐ cap. 5
│   ├── 03_compression_sensitivity.ipynb  sensibilidad por componente ┘
│   ├── 04_probing.ipynb              probes lineales por capa         ┐
│   ├── 05_activation_patching.ipynb  localización causal              │
│   ├── 06_head_analysis.ipynb        ablación de las 144 cabezas      │ cap. 6
│   ├── 07_neuron_analysis.ipynb      neuronas FFN especializadas      │
│   ├── 08_emotional_map.ipynb        lesiones + genealogía emocional  │
│   ├── 08b_logit_lens.ipynb          logit lens (5ª técnica)          ┘
│   ├── 09_informed_compression.ipynb informada vs ciega, recuperación ┐ cap. 7
│   ├── 09b_finetuning_control.ipynb  control de la hipótesis reguladora ┘
│   └── archive/                      versiones antiguas
│
├── results/                        SALIDAS DE LOS NOTEBOOKS
│   ├── checkpoints/                  el modelo, 418 MB (gitignored, en Drive)
│   ├── csvs/notebookN/               tablas de resultados
│   └── figures/notebookN/            PNGs preliminares
│
├── latex_figures/                  FIGURAS DEL PDF (matplotlib)
│   ├── tfg_plot_style.py             estilo matplotlib (tipografía Pagella)
│   ├── generate_cap4_figures.ipynb   figuras del cap. 4
│   ├── generate_cap5_figures.ipynb   figuras del cap. 5
│   ├── generate_extra_figures.py     figuras sueltas adicionales
│   └── figures/cap{N}_*_{es,en}.png  PNGs finales para LaTeX
│
├── viz/                            PIPELINE DE FIGURAS (Plotly + extractores)
│   ├── data/, extractors/            caches del modelo y cómo generarlos
│   ├── interactive/, plots/          figuras Plotly
│   └── style.py                      estilo común
│
└── web/                            WEB INTERACTIVA (el site editorial)
    ├── index.html, sobre.html        la web publicada
    ├── _site_mode.py                 estilo Plotly del site
    ├── build_figures.py              genera las figuras (importa de viz/)
    ├── build_index.py                ensambla index.html
    ├── build_pdf_figures.py          puente web → PNG estáticos del PDF
    └── build_all.py                  todo de golpe
```

---

## El modelo mental: dos productos, una fuente de verdad

```
                      notebooks/  →  results/csvs/  + checkpoint
                          │  (los ejecutas una sola vez)
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  PDF DE LA MEMORIA                    WEB INTERACTIVA
  ════════════════════                ════════════════════
  latex_figures/  (matplotlib)        viz/ + web/  (Plotly)
    + tfg_plot_style.py                 + _site_mode.py
    ↓                                   ↓
  latex_figures/figures/              web/figures/*.html
    cap{N}_*_{es,en}.png              web/index.html
```

**Puente opcional**: `web/build_pdf_figures.py` exporta figuras Plotly como
PNG estáticos en `latex_figures/figures/` con el naming `cap{N}_{name}_es.png`.
Útil cuando una figura existe en la web y no quieres redibujarla en matplotlib.

---

## Setup

```bash
pip install -r requirements.txt
```

Los notebooks corren tanto en local como en Google Colab Pro: detectan Colab
al inicio (`IN_COLAB = 'google.colab' in sys.modules`) y montan Drive
automáticamente. Los checkpoints se guardan en `results/checkpoints/`.

### API de Python

```python
from src.data import load_goemotions
from src.models import load_bert_classifier
from src.compression import apply_svd_compression, compute_singular_value_energy
from src.utils import compute_metrics
```

---

## Workflows

**Editar una figura del PDF** (matplotlib, vive en el notebook):

```bash
jupyter notebook latex_figures/generate_cap{4,5}_figures.ipynb
# → latex_figures/figures/cap{N}_*_{es,en}.png
```

**Editar una figura de la web** (Plotly, vive en `viz/interactive/` o `viz/plots/`):

```bash
python web/build_all.py          # regenera figuras + index
# → web/figures/*.html, web/index.html
```

**Llevar una figura de la web al PDF** sin redibujarla:

```bash
python web/build_pdf_figures.py  # → latex_figures/figures/cap{N}_{name}_es.png
```

**Ver la web en local:**

```bash
python -m http.server -d web 8080   # http://localhost:8080
```

---

## Convenciones

- **Tipografía**: Pagella en todo (PDF y web). Misma cadena de fallback en
  `latex_figures/tfg_plot_style.py` y `web/_site_mode.py`.
- **Paleta**: idéntica en ambos. `INK #1A1A1A · TERRA #C1553A · BLUE #3A6EA5
  · SAGE #5A8F7B · SAND #D4A843`.
- **Naming de figuras del PDF**: `cap{N}_{name}_{es,en}.png` siempre, lo
  genere matplotlib o Plotly.
- **Naming en la web**: `web/figures/{name}.html` (sin capítulo, sin idioma).
- **Etiquetas de capa**: `Emb · L0 · L1 · … · L11` para las 13 capas.

---

## Qué está gitignored y por qué

- `results/checkpoints/` — el modelo (418 MB). Vive en Drive.
- `viz/data/cache/*.npz` — caches del modelo, regenerables (el mayor,
  `activations.npz`, supera el límite de 100 MB de GitHub).
- `.viz_venv/` — virtualenv local (Plotly + kaleido).
- `web/_tmp/`, `viz/output/` — outputs de build intermedios, regenerables.

Cualquiera que clone el repo puede regenerar los caches con los scripts de
`viz/extractors/`.

---

## Licencia

[MIT](LICENSE) © 2026 Guido Biosca Lasa.
