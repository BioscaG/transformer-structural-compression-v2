# Transformer Structural Compression

TFG sobre análisis de compresión estructural de modelos transformer.

## Estructura del proyecto

- `src/data/` — Carga y preprocesado del dataset GoEmotions
- `src/models/` — Definición del clasificador BERT
- `src/utils/` — Funciones de métricas (F1 macro, micro, por emoción)
- `notebooks/` — Notebooks de experimentación
- `results/` — Checkpoints, métricas y visualizaciones

## Instalación

```bash
pip install -r requirements.txt
```

## Fase 1: Fine-tuning de BERT en GoEmotions

Clasificación multi-label de 28 emociones con BERT base (`bert-base-uncased`).

Ejecutar el notebook `notebooks/01_finetuning.ipynb` (compatible con Google Colab).

### Métricas esperadas

- F1 macro: ~0.45–0.55
- F1 micro: ~0.50–0.60
