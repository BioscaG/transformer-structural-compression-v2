"""Methodology pipeline — interactive flow diagram (Ch 3 of the thesis).

Replaces the static tikz diagram from the thesis with an interactive
flow graph: each box is clickable and jumps to the corresponding
section of the parent page. Designed as a "you are here" map for a
defense audience that needs to understand the experimental flow at a
glance.

Layout (vertical):
  Datos → Modelo → Baseline
                      ↓
              [Compresión SVD]                [Interpretabilidad mecánica]
              4 familias                       5 técnicas
                      ↘                            ↙
                       [Compresión informada]
                        Greedy (data-driven)
                              ↓
                       [Fine-tuning recovery]

Built as raw HTML+CSS so it inherits the editorial design system from
the host page when injected via inject_site_mode_into_html.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))


LANG = {
    "es": {
        "title": "Pipeline experimental",
        "data": {
            "title": "Datos",
            "head":  "GoEmotions",
            "stat":  "23 emociones · multi-label",
            "sub":   "43 410 train · 5 426 val · 5 427 test",
            "anchor": "#parte-1",
        },
        "model": {
            "title": "Modelo",
            "head":  "BERT-base",
            "stat":  "110 M parámetros · 12 capas",
            "sub":   "84,9 M en las 72 matrices del encoder",
            "anchor": "#arquitectura",
        },
        "tune": {
            "title": "Fine-tune",
            "head":  "AdamW · LR 2e-5",
            "stat":  "4 épocas · batch 32 · warmup 10 %",
            "sub":   "BCE multi-label · max_len 128",
            "anchor": "#parte-1",
        },
        "baseline": {
            "title": "Baseline",
            "head":  "F1 macro 0,577",
            "stat":  "F1 micro 0,627",
            "sub":   "Checkpoint: 23emo-final",
            "anchor": "#parte-1",
        },
        "compression": {
            "title": "Compresión SVD",
            "head":  "4 familias · 22 estrategias",
            "items": [
                ("Uniforme",  "Mismo rango a 72 matrices"),
                ("Adaptativa", "Por umbral de energía espectral"),
                ("Heurística", "Reglas de interpretabilidad"),
                ("Greedy",    "Data-driven · §07.2"),
            ],
            "anchor": "#parte-2",
        },
        "interpret": {
            "title": "Interpretabilidad mecánica",
            "head":  "5 técnicas complementarias",
            "items": [
                ("Probing lineal",      "276 probes · §03.1",          "#crystallization"),
                ("Logit lens",          "2 300 frases · §03.4",        "#iterative"),
                ("Activation patching", "Sobre rango 64 · §05.1",      "#lesion"),
                ("Ablación cabezas",    "144 cabezas · §04.1",         "#heads"),
                ("Análisis neuronal",   "36 864 neuronas · §06.1",     "#neurons"),
            ],
            "anchor": "#parte-3",
        },
        "informed": {
            "title": "Compresión informada",
            "head":  "Greedy data-driven",
            "stat":  "Domina 8 de 9 puntos Pareto-óptimos",
            "sub":   "Q y K primero · FFN tardía nunca tocada",
            "anchor": "#greedy",
        },
        "recovery": {
            "title": "Fine-tuning recovery",
            "head":  "+3 épocas tras compresión",
            "stat":  "F1 macro 0,591 con 86,4 % params",
            "sub":   "embarrassment +90 % relativo",
            "anchor": "#recovery",
        },
        "tag_setup":   "Setup",
        "tag_split":   "Dos brazos",
        "tag_synth":   "Síntesis",
        "hint":        "Click en cualquier bloque para saltar a la sección",
    },
    "en": {
        "title": "Experimental pipeline",
        "data": {
            "title": "Data",
            "head":  "GoEmotions",
            "stat":  "23 emotions · multi-label",
            "sub":   "43,410 train · 5,426 val · 5,427 test",
            "anchor": "#parte-1",
        },
        "model": {
            "title": "Model",
            "head":  "BERT-base",
            "stat":  "110 M parameters · 12 layers",
            "sub":   "84.9 M across the 72 encoder matrices",
            "anchor": "#arquitectura",
        },
        "tune": {
            "title": "Fine-tune",
            "head":  "AdamW · LR 2e-5",
            "stat":  "4 epochs · batch 32 · 10 % warmup",
            "sub":   "Multi-label BCE · max_len 128",
            "anchor": "#parte-1",
        },
        "baseline": {
            "title": "Baseline",
            "head":  "F1 macro 0.577",
            "stat":  "F1 micro 0.627",
            "sub":   "Checkpoint: 23emo-final",
            "anchor": "#parte-1",
        },
        "compression": {
            "title": "SVD compression",
            "head":  "4 families · 22 strategies",
            "items": [
                ("Uniform",   "Same rank to 72 matrices"),
                ("Adaptive",  "By spectral-energy threshold"),
                ("Heuristic", "Rules from interpretability"),
                ("Greedy",    "Data-driven · §07.2"),
            ],
            "anchor": "#parte-2",
        },
        "interpret": {
            "title": "Mechanistic interpretability",
            "head":  "5 complementary techniques",
            "items": [
                ("Linear probing",      "276 probes · §03.1",          "#crystallization"),
                ("Logit lens",          "2,300 sentences · §03.4",     "#iterative"),
                ("Activation patching", "Over rank 64 · §05.1",        "#lesion"),
                ("Head ablation",       "144 heads · §04.1",           "#heads"),
                ("Neural analysis",     "36,864 neurons · §06.1",      "#neurons"),
            ],
            "anchor": "#parte-3",
        },
        "informed": {
            "title": "Informed compression",
            "head":  "Greedy data-driven",
            "stat":  "Dominates 8 of 9 Pareto-optimal points",
            "sub":   "Q and K first · late FFN never touched",
            "anchor": "#greedy",
        },
        "recovery": {
            "title": "Fine-tuning recovery",
            "head":  "+3 epochs after compression",
            "stat":  "F1 macro 0.591 with 86.4 % params",
            "sub":   "embarrassment +90 % relative",
            "anchor": "#recovery",
        },
        "tag_setup":   "Setup",
        "tag_split":   "Two arms",
        "tag_synth":   "Synthesis",
        "hint":        "Click any block to jump to the section",
    },
}


def _box_setup(box: dict) -> str:
    return f"""
    <a class="pl-box pl-setup" href="{box['anchor']}" target="_top">
      <div class="pl-title">{box['title']}</div>
      <div class="pl-head">{box['head']}</div>
      <div class="pl-stat">{box['stat']}</div>
      <div class="pl-sub">{box['sub']}</div>
    </a>
    """


def _box_compression(c: dict) -> str:
    items_html = "".join(
        f'<div class="pl-item"><span class="pl-item-label">{lbl}</span>'
        f'<span class="pl-item-desc">{desc}</span></div>'
        for lbl, desc in c["items"]
    )
    return f"""
    <a class="pl-box pl-branch pl-compression" href="{c['anchor']}" target="_top">
      <div class="pl-title">{c['title']}</div>
      <div class="pl-head">{c['head']}</div>
      <div class="pl-items">{items_html}</div>
    </a>
    """


def _box_interpret(i: dict) -> str:
    items_html = "".join(
        f'<a class="pl-item pl-link" href="{anchor}" target="_top">'
        f'<span class="pl-item-label">{lbl}</span>'
        f'<span class="pl-item-desc">{desc}</span></a>'
        for lbl, desc, anchor in i["items"]
    )
    return f"""
    <a class="pl-box pl-branch pl-interpret" href="{i['anchor']}" target="_top">
      <div class="pl-title">{i['title']}</div>
      <div class="pl-head">{i['head']}</div>
      <div class="pl-items">{items_html}</div>
    </a>
    """


def build_html(out_path: pathlib.Path, lang: str = "es") -> pathlib.Path:
    L = LANG[lang]
    setup_row = "".join(_box_setup(L[k]) for k in ("data", "model", "tune", "baseline"))
    branch_row = _box_compression(L["compression"]) + _box_interpret(L["interpret"])
    synth_row = "".join(_box_setup(L[k]) for k in ("informed", "recovery"))

    html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="utf-8" />
<style>
  html, body {{
    margin: 0; padding: 0;
    background: transparent;
    font-family: "Geist", "Inter", -apple-system, sans-serif;
    color: #141413;
    height: 100%;
    overflow: hidden;
  }}
  .pipeline {{
    display: flex; flex-direction: column;
    gap: 18px;
    padding: 18px 22px 26px 22px;
    height: 100%; box-sizing: border-box;
  }}
  .pl-tag {{
    font-family: "Geist Mono", monospace;
    font-size: 9.5px;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #1F4E6C;
    margin-top: 4px;
    padding-bottom: 6px;
    border-bottom: 0.5px solid #D8D7D1;
  }}
  .pl-tag:first-child {{ margin-top: 0; }}
  .pl-row {{
    display: flex; gap: 12px;
    align-items: stretch;
  }}
  .pl-row.setup {{ }}
  .pl-row.branches {{
    flex: 1 1 auto; min-height: 0;
  }}
  .pl-row.synth {{ gap: 12px; }}

  .pl-box {{
    flex: 1 1 0;
    background: #FFFFFF;
    border: 0.5px solid #D8D7D1;
    border-radius: 3px;
    padding: 14px 16px;
    text-decoration: none; color: inherit;
    display: flex; flex-direction: column;
    gap: 4px;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
  }}
  .pl-box:hover {{
    border-color: #1F4E6C;
    background: #F7F6F2;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px -10px rgba(20,20,19,0.18);
  }}
  .pl-box::after {{
    content: "↗";
    position: absolute;
    top: 12px; right: 12px;
    font-size: 11px; color: #8A8A82;
    opacity: 0;
    transition: opacity 0.18s ease;
  }}
  .pl-box:hover::after {{ opacity: 1; }}

  .pl-title {{
    font-family: "Geist Mono", monospace;
    font-size: 10px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #1F4E6C;
    margin-bottom: 2px;
  }}
  .pl-head {{
    font-family: "Fraunces", serif;
    font-size: 16px;
    line-height: 1.2;
    color: #141413;
    font-weight: 400;
    font-variation-settings: "opsz" 36, "SOFT" 50;
  }}
  .pl-stat {{
    font-size: 12.5px;
    color: #4A4A47;
    line-height: 1.35;
  }}
  .pl-sub {{
    font-family: "Geist Mono", monospace;
    font-size: 10.5px;
    color: #8A8A82;
    line-height: 1.4;
    margin-top: auto;
  }}

  .pl-branch {{
    flex: 1 1 0;
  }}
  .pl-compression {{ background: #F7F6F2; }}
  .pl-interpret {{ background: #FFFFFF; }}

  .pl-items {{
    display: flex; flex-direction: column;
    gap: 5px;
    margin-top: 10px;
  }}
  .pl-item {{
    display: flex; flex-direction: column;
    gap: 1px;
    padding: 6px 8px;
    border-left: 2px solid #D8D7D1;
    background: rgba(255,255,255,0.6);
    border-radius: 0 2px 2px 0;
    text-decoration: none; color: inherit;
    transition: border-color 0.15s ease, background 0.15s ease;
  }}
  .pl-link:hover {{
    border-left-color: #1F4E6C;
    background: rgba(31,78,108,0.06);
  }}
  .pl-item-label {{
    font-size: 12.5px; font-weight: 500;
    color: #141413;
  }}
  .pl-item-desc {{
    font-family: "Geist Mono", monospace;
    font-size: 10px;
    color: #8A8A82;
  }}

  .pl-arrow {{
    text-align: center;
    color: #8A8A82;
    font-size: 16px;
    line-height: 1;
    height: 14px;
    margin: -4px 0;
  }}

  .pl-hint {{
    font-family: "Geist Mono", monospace;
    font-size: 10px;
    color: #8A8A82;
    letter-spacing: 0.06em;
    text-align: right;
    margin-top: -8px;
  }}

  @media (max-width: 720px) {{
    .pl-row {{ flex-direction: column; }}
  }}
</style>
</head>
<body>

<div class="pipeline">
  <div class="pl-tag">{L['tag_setup']}</div>
  <div class="pl-row setup">
    {setup_row}
  </div>

  <div class="pl-arrow">↓</div>
  <div class="pl-tag">{L['tag_split']}</div>
  <div class="pl-row branches">
    {branch_row}
  </div>

  <div class="pl-arrow">↓</div>
  <div class="pl-tag">{L['tag_synth']}</div>
  <div class="pl-row synth">
    {synth_row}
  </div>

  <div class="pl-hint">{L['hint']}</div>
</div>

</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    out = pathlib.Path(__file__).resolve().parents[2] / "viz" / "output" / "methodology.html"
    build_html(out, lang="es")
    print(f"✓ {out}")
