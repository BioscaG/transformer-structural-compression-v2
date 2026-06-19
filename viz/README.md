# `viz/` — Pipeline de figuras Plotly

Genera las figuras interactivas (Plotly + un grafo D3.js) que consume la web
editorial de [`web/`](../web). Es la fuente de las visualizaciones de
*Anatomía Emocional de un Modelo Transformer*; **no** se ejecuta por sí sola
para publicar: el ensamblado del site vive en `web/build_all.py`, que importa
de aquí.

> Para ver o reconstruir la web, mira el [README principal](../README.md) y
> [`web/`](../web). Este documento describe solo el paquete `viz/`.

## Qué hay aquí

```
viz/
├── thesis_data.py        # Fuente única: resultados numéricos de la memoria como datos
├── style.py              # Estilo Plotly, coherente con latex_figures/tfg_plot_style.py
├── data/
│   ├── load_results.py   # Carga los CSV de results/csvs/ para las figuras
│   └── cache/            # activations.npz, meta.json, … (gitignored, regenerables)
├── extractors/           # Generan los caches ejecutando el modelo real
│   ├── extract_real.py
│   ├── extract_compression_decay.py
│   ├── extract_neuron_activations.py
│   └── extract_token_trajectories.py
├── plots/                # Figuras Plotly estáticas (Pareto, heads, sunburst, …)
├── interactive/          # Figuras Plotly animadas/interactivas (galaxy, atlas, …)
└── output/               # HTML intermedios (gitignored)
```

## Regenerar los caches

Los caches de `data/cache/*.npz` están gitignored (el mayor,
`activations.npz`, supera el límite de 100 MB de GitHub) pero son
reproducibles desde el checkpoint:

```bash
.viz_venv/bin/python viz/extractors/extract_real.py
.viz_venv/bin/python viz/extractors/extract_compression_decay.py
.viz_venv/bin/python viz/extractors/extract_neuron_activations.py
.viz_venv/bin/python viz/extractors/extract_token_trajectories.py
```

Luego reconstruye las figuras y la web desde `web/`:

```bash
.viz_venv/bin/python web/build_all.py
```

## Procedencia de los datos

`thesis_data.py` codifica los resultados de las tablas de la memoria de forma
literal. Las figuras que dependen del modelo (galaxy formation, atlas de
atención, trayectorias de tokens, heatmap de cristalización) se calculan
desde los caches reales generados por los extractores. Las entradas
estimadas/interpoladas se marcan con `ESTIMATED`:

```bash
grep -n ESTIMATED viz/thesis_data.py
```

## Entorno

```bash
python3 -m venv .viz_venv
.viz_venv/bin/pip install plotly numpy scipy scikit-learn pandas matplotlib \
                          kaleido torch transformers datasets
```
