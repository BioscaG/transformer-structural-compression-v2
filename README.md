# Anatomía Emocional de un Modelo Transformer

TFG · Compresión selectiva e interpretabilidad mecánica de BERT-base sobre GoEmotions.
Guido Biosca Lasa · FIB-UPC · 2026 · Director: Lluís Padró Cirera.

---

## Estructura del proyecto en una página

```
transformer-structural-compression-v2/
│
├── src/                            código compartido
│   ├── data/                         carga GoEmotions, tokenización
│   ├── models/                       BERT con cabeza multi-label
│   ├── compression/                  SVD, ranks adaptativos, greedy
│   └── utils/                        métricas (F1 macro/micro)
│
├── notebooks/                      LA INVESTIGACIÓN
│   ├── 01_finetuning.ipynb           entrena el modelo
│   ├── 02_spectral_analysis.ipynb    SVD, k95, frontera Pareto
│   ├── 03_compression_sensitivity.ipynb
│   ├── 04_probing.ipynb              probes lineales por capa
│   ├── 05_activation_patching.ipynb
│   ├── 06_head_analysis.ipynb        ablación de las 144 cabezas
│   ├── 07_neuron_analysis.ipynb      36 864 neuronas FFN
│   ├── 08_emotional_map.ipynb
│   └── 09_informed_compression.ipynb
│
├── results/                        SALIDAS DE LOS NOTEBOOKS
│   ├── checkpoints/23emo-final/      el modelo (gitignored, en Drive)
│   ├── csvs/notebookN/               61 tablas de resultados
│   └── figures/notebookN/            PNGs preliminares
│
├── latex_figures/                  PDF DE LA MEMORIA
│   ├── tfg_plot_style.py             estilo matplotlib (Pagella)
│   ├── generate_cap4_figures.ipynb   ← genera figuras del cap. 4
│   ├── generate_cap5_figures.ipynb   ← genera figuras del cap. 5
│   └── figures/cap{N}_*_{es,en}.png  ← PNGs finales para LaTeX
│
└── viz/                            WEB INTERACTIVA
    ├── data/cache/*.npz              caches del modelo (gitignored)
    ├── extractors/                   genera los caches
    ├── interactive/, plots/          27 figuras Plotly
    ├── site/                         pipeline del site editorial
    │   ├── index.html, sobre.html      la web final
    │   ├── figures/                    figuras Plotly site-mode
    │   ├── _site_mode.py               estilo Plotly del site
    │   ├── build_figures.py            genera figuras del site
    │   ├── build_index.py              genera index.html
    │   └── build_pdf_figures.py        web → PDF (figuras estáticas)
    └── output/                       outputs originales (no del site)
```

---

## El modelo mental: **dos productos, una fuente de verdad**

```
                      notebooks/  →  results/csvs/  + checkpoint
                          ↓ (los ejecutas, una sola vez)
                          ▼
                          ▼
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼

  PDF DE LA MEMORIA                    WEB INTERACTIVA
  ════════════════════                ════════════════════
  latex_figures/                      viz/
    + matplotlib                        + plotly
    + tfg_plot_style.py                 + _site_mode.py
    + generate_cap{N}.ipynb             + build_figures.py
    ↓                                   ↓
  latex_figures/figures/              viz/site/figures/
    cap{N}_*_es.png                     *.html
    cap{N}_*_en.png                   viz/site/index.html
```

**Y un puente opcional**: `viz/site/build_pdf_figures.py` exporta figuras
Plotly como PNG estáticos directamente en `latex_figures/figures/` con el
naming `cap{N}_{name}_es.png`. Lo usas si una figura existe en la web pero
no quieres redibujarla en matplotlib.

---

## Workflows (lo que tú haces día a día)

### 1. Cuando quieres editar / añadir una figura del PDF

Es matplotlib, vive en el notebook correspondiente:

```bash
jupyter notebook latex_figures/generate_cap{4,5}_figures.ipynb
```

Editas la celda. La ejecutas. Aparece en `latex_figures/figures/cap{N}_*_{es,en}.png`.

### 2. Cuando quieres editar / añadir una figura de la web

Es Plotly, vive en `viz/interactive/X.py` o `viz/plots/X.py`. Editas el
fichero. Luego:

```bash
python viz/site/build_figures.py     # regenera todas
python viz/site/build_index.py       # actualiza el index
```

O todo de golpe:

```bash
python viz/site/build_all.py
```

Aparece en `viz/site/figures/X.html`.

### 3. Cuando quieres una figura de la web también en el PDF

Sin redibujarla en matplotlib:

```bash
python viz/site/build_pdf_figures.py
```

Aparece en `latex_figures/figures/cap{N}_{name}_es.png` (mismo naming
convention que el notebook).

### 4. Cuando ejecutas en Colab y los datos están en Drive

Los notebooks de `latex_figures/` y `notebooks/` detectan Colab al inicio
(`IN_COLAB = 'google.colab' in sys.modules`) y montan Drive
automáticamente. La estructura de paths se calcula.

---

## Convenciones (lo que NO tienes que pensar)

- **Tipografía**: Pagella en TODO (PDF y web). Misma cadena de fallback en
  `tfg_plot_style.py` y `viz/site/_site_mode.py`.
- **Paleta**: misma en ambos. `INK #1A1A1A · TERRA #C1553A · BLUE #3A6EA5
  · SAGE #5A8F7B · SAND #D4A843` (definidos a la vez en los dos archivos
  de estilo, bit a bit idénticos).
- **Naming de figuras del PDF**: `cap{N}_{name}_{es,en}.png` siempre. Da
  igual si la generó matplotlib o Plotly.
- **Naming en la web**: `viz/site/figures/{name}.html` (sin capítulo,
  sin idioma).
- **Etiquetas**: `Emb · L0 · L1 · … · L11` para las 13 capas. Siempre.

---

## Lo que está gitignored y por qué

- `results/checkpoints/` — el modelo, 455 MB. Vive en Drive.
- `viz/data/cache/*.npz` — caches del modelo, regenerables. El más grande
  (`activations.npz`) pesa 212 MB.
- `.viz_venv/` — virtualenv local (Plotly + kaleido).

Cualquiera que clone el repo puede regenerar el cache:

```bash
python viz/extractors/extract_real.py
python viz/extractors/extract_compression_decay.py
python viz/extractors/extract_neuron_activations.py
python viz/extractors/extract_token_trajectories.py
```

---

## Independencia de los notebooks de `latex_figures/`

Las figuras del cap. 5 (lens vs probe, U-curve agregada) **no dependen
del cache de `viz/`**. Cargan GoEmotions desde HuggingFace, ejecutan el
checkpoint y computan todo desde cero. La separación entre la pipeline
del PDF y la de la web es estricta.

---

## Para abrir la web localmente

```bash
python -m http.server -d viz/site 8080
# luego abre http://localhost:8080
```

---

## Setup mínimo

```bash
pip install -r requirements.txt
# Para regenerar la web:
.viz_venv/bin/python viz/site/build_all.py
# Para regenerar las figuras del PDF:
jupyter notebook latex_figures/
# Para una figura específica de la web hacia el PDF:
.viz_venv/bin/python viz/site/build_pdf_figures.py NOMBRE
```
