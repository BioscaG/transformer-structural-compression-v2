"""Generate web/index.html from sections.py.

This is the central assembly: takes structured section data and
produces the editorial site HTML using the design system established
in the prototype.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from web.sections import (
    SECTIONS, HERO_STATS, FIG_HEIGHTS, HERO, OUTRO, NAV, FOOTER,
    COMMENTS,
)


def bi(d: dict | str, tag: str = "span", classes: str = "") -> str:
    """Render a bilingual ``{"es": ..., "en": ...}`` dict as two HTML
    elements tagged with ``lang``. Use ``tag="p"`` for paragraphs.
    A plain string passes through (single-language fallback).
    """
    if isinstance(d, str):
        return d
    cls = f' class="{classes}"' if classes else ""
    return (
        f'<{tag} lang="es"{cls}>{d["es"]}</{tag}>'
        f'<{tag} lang="en"{cls}>{d["en"]}</{tag}>'
    )


def bi_text(d: dict | str) -> tuple[str, str]:
    """Return (es, en) tuple. For plain strings returns same value twice."""
    if isinstance(d, str):
        return d, d
    return d["es"], d["en"]


SITE_DIR = pathlib.Path(__file__).resolve().parent
OUT = SITE_DIR / "index.html"


def render_hero() -> str:
    stats_html = "\n".join(
        f'      <div class="stat">\n'
        f'        <span class="num">{num}</span>\n'
        f'        {bi(label, tag="span", classes="label")}\n'
        f'      </div>'
        for num, label in HERO_STATS
    )
    left_es, left_en = bi_text(HERO["left"])
    right_es, right_en = bi_text(HERO["right"])
    return f"""
<section class="hero" id="top">
  <canvas id="hero-canvas" aria-hidden="true"></canvas>

  <div class="hero-meta reveal">
    <div class="left">
      <span lang="es">{left_es}</span><span lang="en">{left_en}</span>
    </div>
    <div class="right">
      <span lang="es">{right_es}</span><span lang="en">{right_en}</span>
    </div>
  </div>

  <div class="hero-content reveal" style="--delay: 120ms">
    {bi(HERO["tag"], tag="div", classes="hero-tag")}
    <h1 class="hero-title">
      <span lang="es">{HERO["title_1"]["es"]}<br>{HERO["title_2"]["es"]}<br><em>{HERO["title_3"]["es"]}</em></span>
      <span lang="en">{HERO["title_1"]["en"]}<br>{HERO["title_2"]["en"]}<br><em>{HERO["title_3"]["en"]}</em></span>
      <span class="small">
        <span lang="es">{HERO["lead"]["es"]}</span>
        <span lang="en">{HERO["lead"]["en"]}</span>
      </span>
    </h1>

    <div class="hero-stats">
{stats_html}
    </div>
  </div>

  <div class="hero-foot reveal" style="--delay: 240ms">
    <div class="scroll-cue">
      <span class="arrow"></span>
      {bi(HERO["scroll"], tag="span")}
    </div>
    <div>
      <a href="sobre.html">Guido Biosca Lasa</a> · {bi(HERO["director"], tag="span")}
    </div>
  </div>
</section>
"""


def render_part(s: dict) -> str:
    intro_html = "\n".join(
        f"    {bi(p, tag='p')}" for p in s["intro"]
    )
    # Big numeral (asymmetric layout). Pull just the digits from the
    # bilingual ``num`` field (e.g. "Parte 04" → "04"). Same for both langs.
    num_es = s["num"]["es"].split()[-1] if isinstance(s["num"], dict) else s["num"].split()[-1]
    num_en = s["num"]["en"].split()[-1] if isinstance(s["num"], dict) else s["num"].split()[-1]
    return f"""
<section class="part" id="{s['id']}">
  <div class="part-grid">
    <div class="part-numeral reveal" aria-hidden="true">
      <span lang="es">{num_es}</span><span lang="en">{num_en}</span>
    </div>
    <div class="part-content">
      {bi(s['num'], tag='div', classes='part-meta reveal')}
      {bi(s['title'], tag='h2', classes='part-title reveal')}
      <div class="part-intro reveal" style="--delay: 200ms">
{intro_html}
      </div>
    </div>
  </div>
</section>
"""


HERO_FIGURES = {"galaxy_formation", "lesion_theater", "sentence_trajectory"}


def render_figure(s: dict) -> str:
    body_html = "\n".join(
        f'    {bi(p, tag="p", classes=("reveal lead" if i == 0 else "reveal"))}'
        for i, p in enumerate(s["body"])
    )
    pull_html = ""
    if s.get("pull"):
        pull_html = (
            f'\n  {bi(s["pull"], tag="blockquote", classes="pull reveal")}\n'
        )

    fig_h = FIG_HEIGHTS.get(s["figure"], 800)
    wrap_h = fig_h + 64  # 32px padding × 2 (top + bottom of card)

    is_hero = s["figure"] in HERO_FIGURES
    section_class = "chapter hero-figure" if is_hero else "chapter"

    chapter_html  = bi(s["chapter"],  tag="div", classes="ch-num reveal")
    title_html    = bi(s["title"],    tag="h2",  classes="ch-title reveal")
    subtitle_html = bi(s["subtitle"], tag="p",   classes="ch-sub reveal")
    caption_html  = bi(s["caption"],  tag="div", classes="caption")

    return f"""
<section class="{section_class}" id="{s['id']}">
  <div class="ch-head">
    {chapter_html}
    {title_html}
    {subtitle_html}
  </div>

  <div class="ch-body">
{body_html}
  </div>
{pull_html}
  <div class="figure reveal">
    <div class="figure-wrap" style="height: {wrap_h}px">
      <iframe
        data-src-es="figures/{s['figure']}.html"
        data-src-en="figures/{s['figure']}.en.html"
        loading="lazy" scrolling="no"></iframe>
    </div>
    <div class="figure-meta">
      {caption_html}
      <div class="id">Fig. <strong>{s['fig_id']}</strong></div>
    </div>
  </div>
</section>
"""


def _diagram_bisturi() -> str:
    """SVD as measurement instrument — three-stage horizontal flow."""
    return """
<div class="diagram diagram-bisturi">
  <div class="bist-stage">
    <div class="bist-stage-num"><span>01</span></div>
    <div class="bist-stage-glyph bist-mat-full" aria-hidden="true">
      <span class="lbl">W</span>
      <span class="sub">m × n</span>
    </div>
    <div class="bist-stage-cap">
      <strong lang="es">Matriz original</strong>
      <strong lang="en">Original matrix</strong>
      <span lang="es">Una de las 72 del encoder<br>de BERT.</span>
      <span lang="en">One of BERT's 72 encoder<br>matrices.</span>
    </div>
  </div>
  <div class="bist-arrow">
    <span class="bist-arrow-lbl">SVD<sub><em>k</em></sub></span>
    <span class="bist-arrow-line"></span>
  </div>
  <div class="bist-stage">
    <div class="bist-stage-num"><span>02</span></div>
    <div class="bist-stage-glyph bist-mat-svd" aria-hidden="true">
      <span class="bist-block u">U<sub>k</sub></span>
      <span class="bist-block s">Σ<sub>k</sub></span>
      <span class="bist-block v">V<sub>k</sub><sup>T</sup></span>
    </div>
    <div class="bist-stage-cap">
      <strong lang="es">Truncada al rango <em>k</em></strong>
      <strong lang="en">Truncated to rank <em>k</em></strong>
      <span lang="es">Pérdida controlada de información.<br>Eckart-Young, 1936.</span>
      <span lang="en">Controlled information loss.<br>Eckart-Young, 1936.</span>
    </div>
  </div>
  <div class="bist-arrow">
    <span class="bist-arrow-lbl"><span lang="es">aplicar al modelo</span><span lang="en">apply to model</span></span>
    <span class="bist-arrow-line"></span>
  </div>
  <div class="bist-stage">
    <div class="bist-stage-num"><span>03</span></div>
    <div class="bist-stage-glyph bist-delta" aria-hidden="true">
      <span class="bist-delta-symbol">Δ F1</span>
    </div>
    <div class="bist-stage-cap">
      <strong lang="es">Medida funcional</strong>
      <strong lang="en">Functional measurement</strong>
      <span lang="es">La caída de F1 cuantifica<br>la importancia de <em>W</em>.</span>
      <span lang="en">The F1 drop quantifies the<br>importance of <em>W</em>.</span>
    </div>
  </div>
  <div class="bist-thesis">
    <span lang="es">Lo que se rompe al quitarlo <strong>mide</strong> lo que importaba.</span>
    <span lang="en">What breaks when you remove it <strong>measures</strong> what mattered.</span>
  </div>
</div>
"""


def _diagram_tripartita() -> str:
    """Three columns: Representation, Alignment, Access."""
    cols = [
        {
            "num": "01",
            "title_es": "Representación",
            "title_en": "Representation",
            "def_es": "¿Qué información está <em>presente</em> en esta capa, linealmente accesible?",
            "def_en": "What information is <em>present</em> at this layer, linearly accessible?",
            "method_es": "Probing lineal",
            "method_en": "Linear probing",
            "method_sub_es": "clasificador entrenado<br>sobre cada capa",
            "method_sub_en": "fresh classifier trained<br>on each layer",
            "curve": "monotone",
            "result_es": "Sube monótonamente.<br>L0 → 61 %. L11 → 100 %.",
            "result_en": "Rises monotonically.<br>L0 → 61 %. L11 → 100 %.",
        },
        {
            "num": "02",
            "title_es": "Alineación",
            "title_en": "Alignment",
            "def_es": "¿Está esa información <em>proyectada</em> sobre la base del clasificador?",
            "def_en": "Is that information <em>projected</em> onto the classifier's basis?",
            "method_es": "Logit lens",
            "method_en": "Logit lens",
            "method_sub_es": "cabeza de L11<br>aplicada en cada capa",
            "method_sub_en": "L11 head applied<br>at every layer",
            "curve": "u-shape",
            "result_es": "Curva en U.<br>Capas medias no leíbles.",
            "result_en": "U-shaped curve.<br>Middle layers unreadable.",
        },
        {
            "num": "03",
            "title_es": "Acceso",
            "title_en": "Access",
            "def_es": "¿Es este componente <em>suficiente</em> para que el clasificador opere?",
            "def_en": "Is this component <em>sufficient</em> for the classifier to operate?",
            "method_es": "Activation patching",
            "method_en": "Activation patching",
            "method_sub_es": "restaurar pesos desde<br>un modelo colapsado",
            "method_sub_en": "restore weights from<br>a collapsed model",
            "curve": "spike",
            "result_es": "Sólo la FFN de L11:<br>100 % del F1.",
            "result_en": "L11 FFN alone:<br>100 % of F1.",
        },
    ]
    curves = {
        # SVG path snippets for each curve type, 120×40 viewBox
        "monotone": '<polyline points="6,34 24,28 42,22 60,16 78,12 96,8 114,6" '
                    'fill="none" stroke="currentColor" stroke-width="1.6" '
                    'stroke-linecap="round" />',
        "u-shape":  '<polyline points="6,12 24,20 42,30 60,34 78,30 96,18 114,6" '
                    'fill="none" stroke="currentColor" stroke-width="1.6" '
                    'stroke-linecap="round" />',
        "spike":    '<polyline points="6,34 36,34 60,34 84,34 100,34 108,6 114,6" '
                    'fill="none" stroke="currentColor" stroke-width="1.6" '
                    'stroke-linecap="round" />',
    }
    cells = []
    for c in cols:
        svg = (
            f'<svg viewBox="0 0 120 40" class="tri-curve" '
            f'aria-hidden="true">{curves[c["curve"]]}</svg>'
        )
        cells.append(f"""
      <div class="tri-col">
        <div class="tri-num"><span>{c['num']}</span></div>
        <h3 class="tri-title">
          <span lang="es">{c['title_es']}</span>
          <span lang="en">{c['title_en']}</span>
        </h3>
        <p class="tri-def">
          <span lang="es">{c['def_es']}</span>
          <span lang="en">{c['def_en']}</span>
        </p>
        <div class="tri-divider"></div>
        <div class="tri-method-lbl">
          <span lang="es">Medido por</span><span lang="en">Measured by</span>
        </div>
        <div class="tri-method">
          <span lang="es">{c['method_es']}</span>
          <span lang="en">{c['method_en']}</span>
        </div>
        <p class="tri-method-sub">
          <span lang="es">{c['method_sub_es']}</span>
          <span lang="en">{c['method_sub_en']}</span>
        </p>
        {svg}
        <p class="tri-result">
          <span lang="es">{c['result_es']}</span>
          <span lang="en">{c['result_en']}</span>
        </p>
      </div>
""")
    return f"""
<div class="diagram diagram-tripartita">
  <div class="tri-grid">
{''.join(cells)}
  </div>
  <div class="tri-coda">
    <span lang="es"><strong>Las tres mediciones son consistentes.</strong> Confundir representación con acceso, o alineación con presencia, produce las paradojas aparentes que el trabajo resuelve.</span>
    <span lang="en"><strong>The three measurements are consistent.</strong> Conflating representation with access, or alignment with presence, produces the apparent paradoxes the thesis resolves.</span>
  </div>
</div>
"""


def _diagram_convergencia() -> str:
    """5 techniques × 3 depth bands. Each cell shows the technique's reading."""
    # Density code: 3 = high activity, 2 = medium, 1 = low, 0 = none
    # rows: probing, logit-lens, patching, head-ablation, neurons
    techniques = [
        ("Probing", "Probing",
         "F1 por capa", "F1 per layer",
         [3, 1, 1],
         "L0 absorbe 61 %", "L0 absorbs 61 %",
         "+0,01 a +0,05", "+0.01 to +0.05",
         "casi nulo", "near zero"),
        ("Logit lens", "Logit lens",
         "Σ sigmoides", "Σ sigmoids",
         [2, 0, 3],
         "5,4 difuso", "5.4 diffuse",
         "0,2 colapso", "0.2 collapsed",
         "1,3 focalizado", "1.3 focused"),
        ("Activation patching", "Activation patching",
         "% F1 restaurado", "% F1 restored",
         [0, 0, 3],
         "0 %", "0 %",
         "0 %", "0 %",
         "100 % en L11", "100 % at L11"),
        ("Head ablation", "Head ablation",
         "% cabezas críticas", "% critical heads",
         [1, 2, 3],
         "27 %", "27 %",
         "53 %", "53 %",
         "77 % (L11 = 100 %)", "77 % (L11 = 100 %)"),
        ("Neuron selectivity", "Neuron selectivity",
         "Neuronas |d| > 2", "Neurons |d| > 2",
         [1, 2, 3],
         "11 (0,3 %)", "11 (0.3 %)",
         "570 (16 %)", "570 (16 %)",
         "3 061 (84 %)", "3,061 (84 %)"),
    ]
    band_headers = [
        ("Tempranas", "Early", "Emb · L0–L2", "Emb · L0–L2",
         "Señal léxica bruta", "Raw lexical signal"),
        ("Medias", "Middle", "L3–L7", "L3–L7",
         "Cómputo de transición", "Transition computation"),
        ("Tardías", "Late", "L8–L11", "L8–L11",
         "Alineación con el clasificador", "Classifier alignment"),
    ]
    header_html = ""
    for es, en, sub_es, sub_en, role_es, role_en in band_headers:
        header_html += f"""
      <div class="conv-band-head">
        <div class="conv-band-name">
          <span lang="es">{es}</span><span lang="en">{en}</span>
        </div>
        <div class="conv-band-sub">{sub_es}</div>
        <div class="conv-band-role">
          <span lang="es">{role_es}</span><span lang="en">{role_en}</span>
        </div>
      </div>"""

    def dots(d):
        if d == 3:
            return '<span class="dots d3">● ● ●</span>'
        if d == 2:
            return '<span class="dots d2">● ● <span class="dim">●</span></span>'
        if d == 1:
            return '<span class="dots d1">● <span class="dim">● ●</span></span>'
        return '<span class="dots d0"><span class="dim">● ● ●</span></span>'

    rows_html = ""
    for row in techniques:
        (name_es, name_en, metric_es, metric_en, densities,
         e_es, e_en, m_es, m_en, l_es, l_en) = row
        cells_es = [e_es, m_es, l_es]
        cells_en = [e_en, m_en, l_en]
        cells_html = ""
        for i, d in enumerate(densities):
            cells_html += f"""
        <div class="conv-cell d{d}">
          {dots(d)}
          <div class="conv-cell-val">
            <span lang="es">{cells_es[i]}</span>
            <span lang="en">{cells_en[i]}</span>
          </div>
        </div>"""
        rows_html += f"""
      <div class="conv-row">
        <div class="conv-tech">
          <div class="conv-tech-name">
            <span lang="es">{name_es}</span><span lang="en">{name_en}</span>
          </div>
          <div class="conv-tech-metric">
            <span lang="es">{metric_es}</span><span lang="en">{metric_en}</span>
          </div>
        </div>
        {cells_html}
      </div>"""

    return f"""
<div class="diagram diagram-convergencia">
  <div class="conv-grid">
    <div class="conv-corner">
      <span lang="es">técnica  /  banda funcional</span>
      <span lang="en">technique  /  functional band</span>
    </div>
{header_html}
{rows_html}
  </div>
  <div class="conv-coda">
    <span lang="es">Densidad de marca = magnitud del hallazgo. Cinco filas independientes, una columna dominante.</span>
    <span lang="en">Mark density = magnitude of the finding. Five independent rows, one dominant column.</span>
  </div>
</div>
"""


def _diagram_predicciones() -> str:
    """5 falsifiable predictions, as cards."""
    preds = [
        {
            "tag_es": "Arquitectónica", "tag_en": "Architectural",
            "title_es": "Asimetría espectral en BERT-large",
            "title_en": "Spectral asymmetry in BERT-large",
            "predict_es": "El cociente <em>k</em>₉₅(Q)/<em>k</em>₉₅(FFN) debe caer en [0,55 ; 0,75]. En BERT-base es 0,64.",
            "predict_en": "The <em>k</em>₉₅(Q)/<em>k</em>₉₅(FFN) ratio should land in [0.55 ; 0.75]. BERT-base is 0.64.",
            "falsify_es": "Un cociente cercano a 1 falsa la generalizabilidad.",
            "falsify_en": "A ratio near 1 falsifies the generalisability.",
        },
        {
            "tag_es": "Geométrica", "tag_en": "Geometric",
            "title_es": "Decoder-only ≠ encoder-only en patching",
            "title_en": "Decoder-only ≠ encoder-only in patching",
            "predict_es": "En GPT-2/LLaMA, la restauración por patching debe repartirse entre varias capas tardías, no concentrarse en una.",
            "predict_en": "In GPT-2/LLaMA, patching restoration should spread across several late layers, not concentrate in one.",
            "falsify_es": "Si una capa sola recupera el 100 %, la explicación geométrica del [CLS] se cae.",
            "falsify_en": "If a single layer recovers 100 %, the [CLS]-based geometric explanation falls.",
        },
        {
            "tag_es": "Semántica", "tag_en": "Semantic",
            "title_es": "Cristalización robusta a arquitectura",
            "title_en": "Crystallisation robust to architecture",
            "predict_es": "La distribución 9/8/6 emociones por banda y ρ = −0,67 con F1 máximo deben replicarse en RoBERTa-base.",
            "predict_en": "The 9/8/6 emotion-per-band split and ρ = −0.67 with max F1 should replicate on RoBERTa-base.",
            "falsify_es": "Si el orden cambia mucho, el corpus de pre-train decide, no la semántica.",
            "falsify_en": "If the order changes much, pre-training corpus drives it, not semantics.",
        },
        {
            "tag_es": "Estadística", "tag_en": "Statistical",
            "title_es": "Regularización en clases raras",
            "title_en": "Regularisation on rare classes",
            "predict_es": "Compresión + fine-tuning debe favorecer clases infrarrepresentadas también en NER o Reuters-21578.",
            "predict_en": "Compression + fine-tuning should help under-represented classes on NER or Reuters-21578 too.",
            "falsify_es": "Ausencia del efecto refuta la regularización implícita por SVD.",
            "falsify_en": "Absence of the effect refutes the implicit SVD regularisation.",
        },
        {
            "tag_es": "Metodológica", "tag_en": "Methodological",
            "title_es": "Heurística cualitativa = baseline ciego",
            "title_en": "Qualitative heuristic = blind baseline",
            "predict_es": "Cualquier regla cualitativa para asignar rangos convergerá sobre la frontera ciega, salvo que use datos cuantitativos.",
            "predict_en": "Any qualitative rule for assigning ranks will collapse onto the blind frontier, unless it uses quantitative data.",
            "falsify_es": "Una heurística que domine Pareto en otro dominio rompe la observación central.",
            "falsify_en": "A heuristic dominating Pareto elsewhere breaks the central observation.",
        },
    ]
    cards = ""
    for i, p in enumerate(preds, 1):
        cards += f"""
    <article class="pred-card">
      <div class="pred-num">{i:02d}</div>
      <div class="pred-tag">
        <span lang="es">{p['tag_es']}</span><span lang="en">{p['tag_en']}</span>
      </div>
      <h3 class="pred-title">
        <span lang="es">{p['title_es']}</span>
        <span lang="en">{p['title_en']}</span>
      </h3>
      <div class="pred-block pred-predict">
        <span class="pred-block-lbl">
          <span lang="es">Predicción</span><span lang="en">Prediction</span>
        </span>
        <span lang="es">{p['predict_es']}</span>
        <span lang="en">{p['predict_en']}</span>
      </div>
      <div class="pred-block pred-falsify">
        <span class="pred-block-lbl">
          <span lang="es">Se rompe si</span><span lang="en">Falsified if</span>
        </span>
        <span lang="es">{p['falsify_es']}</span>
        <span lang="en">{p['falsify_en']}</span>
      </div>
    </article>"""
    return f"""
<div class="diagram diagram-predicciones">
  <div class="pred-grid">{cards}
  </div>
  <div class="pred-coda">
    <span lang="es">Las cinco predicciones son ortogonales. La confirmación conjunta es un test estricto del framework; la refutación individual delimita qué parte del análisis pertenece a este caso.</span>
    <span lang="en">The five predictions are orthogonal. Joint confirmation is a strict test of the framework; individual refutation marks off which part of the analysis belongs to this case.</span>
  </div>
</div>
"""


_DIAGRAMS = {
    "bisturi":      _diagram_bisturi,
    "tripartita":   _diagram_tripartita,
    "convergencia": _diagram_convergencia,
    "predicciones": _diagram_predicciones,
}


def render_concept(s: dict) -> str:
    """Concept section: chapter-style frame with an inline diagram
    instead of an iframe figure. Reuses .chapter so presentation mode
    layout still works."""
    body_html = "\n".join(
        f'    {bi(p, tag="p", classes=("reveal lead" if i == 0 else "reveal"))}'
        for i, p in enumerate(s["body"])
    )
    pull_html = ""
    if s.get("pull"):
        pull_html = (
            f'\n  {bi(s["pull"], tag="blockquote", classes="pull reveal")}\n'
        )

    diagram_name = s["diagram"]
    diagram_html = _DIAGRAMS[diagram_name]()

    chapter_html  = bi(s["chapter"],  tag="div", classes="ch-num reveal")
    title_html    = bi(s["title"],    tag="h2",  classes="ch-title reveal")
    subtitle_html = bi(s["subtitle"], tag="p",   classes="ch-sub reveal")
    caption_html  = bi(s["caption"],  tag="div", classes="caption")

    return f"""
<section class="chapter concept concept-{diagram_name}" id="{s['id']}">
  <div class="ch-head">
    {chapter_html}
    {title_html}
    {subtitle_html}
  </div>

  <div class="ch-body">
{body_html}
  </div>
{pull_html}
  <div class="figure reveal">
    <div class="diagram-wrap">
      {diagram_html}
    </div>
    <div class="figure-meta">
      {caption_html}
      <div class="id">Fig. <strong>{s['fig_id']}</strong></div>
    </div>
  </div>
</section>
"""


def render_outro() -> str:
    label_html = bi(OUTRO["label"], tag="div", classes="ch-num reveal")
    title_html = bi(OUTRO["title"], tag="h2",  classes="ch-title reveal")
    sub_html   = bi(OUTRO["sub"],   tag="p",   classes="ch-sub reveal")

    blocks_html = ""
    for block in OUTRO.get("blocks", []):
        block_label = bi(block["label"], tag="div",
                         classes="outro-label reveal")
        block_title = bi(block["title"], tag="h3",
                         classes="outro-subtitle reveal")
        paragraphs_html = "\n    ".join(
            bi(p, tag="p", classes="reveal") for p in block["paragraphs"]
        )
        blocks_html += f"""
  <div class="outro-block">
    {block_label}
    {block_title}
    {paragraphs_html}
  </div>
"""
    coda_html = ""
    if "coda" in OUTRO:
        coda_html = bi(OUTRO["coda"], tag="p", classes="reveal outro-coda")

    return f"""
<section class="outro" id="cierre">
  {label_html}
  {title_html}
  {sub_html}

  <div class="ch-body">
    {blocks_html}
    {coda_html}
  </div>
</section>
"""


def render_comments() -> str:
    """Giscus-backed comments block. Lives between OUTRO and footer.

    Renders a single discussion thread per page (mapping=pathname).
    The Giscus iframe is themed `light` because it integrates better
    with the editorial palette than `dark`. Lang is initialised from
    document.documentElement.lang at load time and re-emits on the
    site-lang-change event.
    """
    g = COMMENTS["giscus"]
    label_html = bi(COMMENTS["label"], tag="div", classes="ch-num reveal")
    title_html = bi(COMMENTS["title"], tag="h2", classes="ch-title reveal")
    sub_html   = bi(COMMENTS["sub"],   tag="p",  classes="ch-sub reveal")

    return f"""
<section class="comments" id="comentarios">
  {label_html}
  {title_html}
  {sub_html}
  <div id="giscus-container" class="reveal"></div>

  <script>
    (function() {{
      const container = document.getElementById('giscus-container');
      function loadGiscus(lang) {{
        container.innerHTML = '';
        const s = document.createElement('script');
        s.src = 'https://giscus.app/client.js';
        s.setAttribute('data-repo', '{g["repo"]}');
        s.setAttribute('data-repo-id', '{g["repo_id"]}');
        s.setAttribute('data-category', '{g["category"]}');
        s.setAttribute('data-category-id', '{g["category_id"]}');
        s.setAttribute('data-mapping', '{g["mapping"]}');
        s.setAttribute('data-strict', '0');
        s.setAttribute('data-reactions-enabled', '{g["reactions"]}');
        s.setAttribute('data-emit-metadata', '0');
        s.setAttribute('data-input-position', '{g["input_pos"]}');
        s.setAttribute('data-theme', 'https://anatomy.guidobiosca.com/giscus.css');
        s.setAttribute('data-lang', lang === 'en' ? 'en' : 'es');
        s.setAttribute('data-loading', 'lazy');
        s.crossOrigin = 'anonymous';
        s.async = true;
        container.appendChild(s);
      }}
      const initial = document.documentElement.lang || 'es';
      loadGiscus(initial);
      window.addEventListener('site-lang-change', (ev) => {{
        loadGiscus(ev.detail.lang);
      }});
    }})();
  </script>
</section>
"""


def render_footer() -> str:
    return f"""
<footer class="foot">
  <div class="foot-grid">
    <div class="brand-cell">
      {bi(FOOTER["brand_title"], tag="h2")}
      {bi(FOOTER["brand_sub"], tag="p")}
      {bi(FOOTER["brand_school"], tag="p")}
    </div>
    <div>
      {bi(FOOTER["project"], tag="h3")}
      <a href="#">{bi(FOOTER["project_repo"], tag="span")}</a>
      <a href="#">{bi(FOOTER["project_pdf"], tag="span")}</a>
      <a href="sobre.html">{bi(FOOTER["project_about"], tag="span")}</a>
    </div>
    <div>
      {bi(FOOTER["author"], tag="h3")}
      <a href="mailto:guido.biosca0@gmail.com">guido.biosca0@gmail.com</a>
      <a href="#">LinkedIn</a>
      <a href="#">GitHub</a>
    </div>
    <div>
      {bi(FOOTER["tribunal"], tag="h3")}
      {bi(FOOTER["director"], tag="p")}
      <p>FIB · UPC · 2026</p>
    </div>
  </div>
  <div class="foot-bottom">
    <span>© 2026 Guido Biosca Lasa</span>
    {bi(FOOTER["stack"], tag="span")}
  </div>
</footer>
"""


def render_nav() -> str:
    # Group sections by part for the TOC
    parts = []
    current = None
    for s in SECTIONS:
        if s["kind"] == "part":
            current = {"part": s, "items": []}
            parts.append(current)
        elif s["kind"] in ("figure", "concept") and current is not None:
            current["items"].append(s)

    def _strip_em(s: str) -> str:
        return s.replace("<em>", "").replace("</em>", "")

    toc_html = ""
    for p in parts:
        part_title_es = _strip_em(p["part"]["title"]["es"])
        part_title_en = _strip_em(p["part"]["title"]["en"])
        part_num_es, part_num_en = bi_text(p["part"]["num"])
        toc_html += f'<div class="toc-part">'
        toc_html += (
            f'<a href="#{p["part"]["id"]}" class="toc-part-title">'
            f'<span class="toc-num"><span lang="es">{part_num_es}</span>'
            f'<span lang="en">{part_num_en}</span></span> '
            f'<span lang="es">{part_title_es}</span>'
            f'<span lang="en">{part_title_en}</span></a>'
        )
        toc_html += f'<ul>'
        for it in p["items"]:
            it_title_es = _strip_em(it["title"]["es"])
            it_title_en = _strip_em(it["title"]["en"])
            toc_html += (
                f'<li><a href="#{it["id"]}">'
                f'<span class="toc-fig">{it["fig_id"]}</span>'
                f' <span lang="es">{it_title_es}</span>'
                f'<span lang="en">{it_title_en}</span>'
                f'</a></li>'
            )
        toc_html += f'</ul></div>'

    brand_html    = bi(NAV["brand"],    tag="span")
    chapters_html = bi(NAV["chapters"], tag="span")
    data_html     = bi(NAV["data"],     tag="span")
    comments_html = bi(NAV["comments"], tag="span")
    about_html    = bi(NAV["about"],    tag="span")
    index_html    = bi(NAV["index"],    tag="span")

    # TOC: append a Comments entry at the bottom so the panel index
    # matches the actual page structure.
    toc_html += (
        '<div class="toc-part toc-comments">'
        '<a href="#comentarios" class="toc-part-title">'
        '<span class="toc-num">●</span> '
        f'{comments_html}'
        '</a></div>'
    )

    return f"""
<nav class="top" id="topnav">
  <a href="#top" class="brand"><span class="dot"></span>{brand_html}</a>
  <ul class="nav-links">
    <li><a class="link" href="#parte-1">{chapters_html}</a></li>
    <li><a class="link" href="#cierre">{data_html}</a></li>
    <li><a class="link link-comments" href="#comentarios">{comments_html}</a></li>
    <li><a class="link" href="sobre.html">{about_html}</a></li>
  </ul>
  <div class="nav-tools">
    <button class="lang-btn" id="lang-btn" aria-label="Toggle language">
      <span class="lang-current"></span>
    </button>
    <button class="present-btn" id="present-btn" aria-label="Presentation mode" title="Presentation mode (P)">▶</button>
    <button class="toc-btn" id="toc-btn">{index_html}</button>
  </div>
</nav>

<div id="present-overlay" aria-hidden="true">
  <div class="present-bar"><div class="present-bar-fill" id="present-bar-fill"></div></div>
  <div class="present-meta">
    <div class="present-section" id="present-section"></div>
    <div class="present-progress"><span id="present-counter">1 / 1</span></div>
  </div>
  <div class="present-hint" id="present-hint">
    <span lang="es">← → para navegar  ·  P o ESC para salir</span>
    <span lang="en">← → to navigate  ·  P or ESC to exit</span>
  </div>
</div>

<aside id="toc-panel" class="toc-panel">
  <div class="toc-head">
    {index_html}
    <button id="toc-close" aria-label="Cerrar">×</button>
  </div>
  <div class="toc-body">
    {toc_html}
  </div>
</aside>
<div id="toc-overlay"></div>
"""


def render_styles() -> str:
    """All the editorial CSS, kept inline for portability."""
    return r"""
<style>
  :root {
    --bg:        #F7F6F2;
    --bg-2:      #FFFFFF;
    --bg-3:      #EFEEE9;
    --ink:       #141413;
    --ink-2:     #4A4A47;
    --ink-3:     #8A8A82;
    --ink-4:     #B8B8B0;
    --line:      #D8D7D1;
    --line-2:    #E8E7E2;
    --accent:    #1F4E6C;
    --accent-d:  #163D55;
    --accent-l:  #3A7AA0;

    --serif:  "Fraunces", "Iowan Old Style", "Palatino", serif;
    --sans:   "Geist", "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
    --mono:   "Geist Mono", "JetBrains Mono", "SF Mono", Menlo, monospace;

    --max-narrow: 680px;
    --max-wide:   1320px;
    --pad-x:      max(28px, 5vw);
    --easing:     cubic-bezier(0.16, 1, 0.3, 1);
  }

  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; -webkit-font-smoothing: antialiased; }

  /* ─── BILINGUAL ──────────────────────────────────────────────────── */
  /* Hide the inactive language. Both versions are rendered; CSS picks. */
  html[lang="es"] [lang="en"] { display: none !important; }
  html[lang="en"] [lang="es"] { display: none !important; }

  /* ─── HERO CANVAS (ambient particle animation) ──────────────────── */
  #hero-canvas {
    position: absolute; inset: 0;
    width: 100%; height: 100%;
    z-index: 0; pointer-events: none;
    opacity: 0.55;
    mask-image: radial-gradient(ellipse 80% 70% at 70% 50%, #000 30%, transparent 80%);
    -webkit-mask-image: radial-gradient(ellipse 80% 70% at 70% 50%, #000 30%, transparent 80%);
  }

  /* ─── HERO-FIGURE (full-bleed key visualizations) ───────────────── */
  section.chapter.hero-figure .figure {
    max-width: none;
    width: 100%;
    margin: 56px 0 0 0;
  }
  section.chapter.hero-figure .figure-wrap {
    border-radius: 0;
    border-left: none; border-right: none;
    box-shadow:
      0 1px 0 rgba(20,20,19,0.02),
      0 24px 60px -32px rgba(20,20,19,0.18);
    padding-left: max(28px, 5vw);
    padding-right: max(28px, 5vw);
  }
  section.chapter.hero-figure .figure-meta {
    max-width: var(--max-wide);
    margin: 18px auto 0 auto;
    padding: 0 max(28px, 5vw);
  }

  /* ─── PRESENTATION MODE ──────────────────────────────────────────── */
  body.present-mode { overflow: hidden; }
  body.present-mode #progress,
  body.present-mode footer.foot,
  body.present-mode section.outro,
  body.present-mode .ch-body,
  body.present-mode .pull,
  body.present-mode .figure-meta,
  body.present-mode .part-intro,
  body.present-mode .part-meta,
  body.present-mode .hero-meta,
  body.present-mode .hero-stats,
  body.present-mode .hero-foot,
  body.present-mode .hero-tag,
  body.present-mode .hero-title .small,
  body.present-mode #toc-overlay,
  body.present-mode #toc-panel { display: none !important; }
  body.present-mode nav.top { background: transparent; border-bottom: none; }
  body.present-mode nav.top ul.nav-links { display: none; }
  body.present-mode nav.top .brand { opacity: 0.4; }
  body.present-mode main {
    height: 100vh;
    overflow-y: scroll;
    scroll-snap-type: y mandatory;
    scroll-behavior: smooth;
  }
  body.present-mode section.hero,
  body.present-mode section.part {
    min-height: 100vh; height: 100vh;
    padding: 12vh max(28px, 4vw) 8vh max(28px, 4vw);
    scroll-snap-align: start;
    scroll-snap-stop: always;
    display: flex; flex-direction: column;
    justify-content: center;
    margin: 0;
    position: relative;
    overflow: hidden;
  }
  body.present-mode section.hero .hero-content { align-self: center; }

  /* Asymmetric split layout for figure slides:
     left ~34% (chapter num + title + subtitle on top, caption + fig
     id at bottom), right ~66% (figure spanning full slide height). */
  body.present-mode section.chapter {
    min-height: 100vh; height: 100vh;
    padding: 5vh max(28px, 3vw);
    scroll-snap-align: start;
    scroll-snap-stop: always;
    display: grid;
    grid-template-columns: minmax(280px, 34%) 1fr;
    grid-template-rows: auto 1fr auto;
    grid-template-areas:
      "head fig"
      "head fig"
      "meta fig";
    column-gap: 4vw;
    row-gap: 0;
    align-items: stretch;
    margin: 0;
    position: relative;
    overflow: hidden;
  }
  body.present-mode section.chapter .ch-head {
    grid-area: head;
    max-width: none;
    margin: 0;
    width: 100%;
    text-align: left;
    align-self: center;
    display: flex; flex-direction: column;
    gap: 14px;
  }
  body.present-mode .ch-title {
    font-size: clamp(24px, 2.6vw, 40px);
    margin: 0;
    line-height: 1.08;
  }
  body.present-mode .ch-sub {
    font-size: clamp(13px, 1.05vw, 17px);
    margin: 0;
    color: var(--ink-2);
    line-height: 1.5;
  }
  /* Flatten .figure so its children (figure-wrap + figure-meta)
     become direct grid items of the section. */
  body.present-mode section.chapter .figure {
    display: contents;
  }
  body.present-mode section.chapter .figure-wrap {
    grid-area: fig;
    height: 100% !important;
    width: 100%;
    border-radius: 4px;
    padding: 14px;
    align-self: stretch;
  }
  body.present-mode section.chapter .figure-meta {
    grid-area: meta;
    display: flex !important;
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
    margin: 0;
    padding: 16px 0 0 0;
    max-width: none;
    border-top: 0.5px solid var(--line);
    align-self: end;
  }
  body.present-mode section.chapter .figure-meta .caption {
    font-family: var(--serif);
    font-size: 13.5px; font-style: italic;
    color: var(--ink-3); line-height: 1.45;
  }
  body.present-mode section.chapter .figure-meta .id {
    font-family: var(--mono); font-size: 10.5px;
    color: var(--ink-3); letter-spacing: 0.1em;
    text-transform: uppercase;
    white-space: nowrap;
  }
  body.present-mode section.part .part-grid {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    gap: 32px;
    max-width: var(--max-wide); margin: 0 auto;
    text-align: center;
    height: 100%;
  }
  body.present-mode section.part .part-numeral {
    font-size: clamp(96px, 14vw, 200px);
    text-align: center; opacity: 1;
    position: static;
    color: var(--accent);
    line-height: 1;
    padding: 0;
    font-variation-settings: "opsz" 144, "SOFT" 30;
  }
  body.present-mode section.part .part-content {
    text-align: center;
    max-width: var(--max-wide);
  }
  body.present-mode section.part .part-content .part-title {
    margin: 0;
  }
  body.present-mode section.part .part-content {
    text-align: center; position: relative; z-index: 2;
  }
  body.present-mode #present-overlay {
    display: block;
    position: fixed; inset: 0; pointer-events: none; z-index: 80;
  }
  #present-overlay { display: none; }
  .present-bar {
    position: absolute; top: 0; left: 0; right: 0;
    height: 2px; background: rgba(20,20,19,0.06);
    z-index: 81;
  }
  .present-bar-fill {
    height: 100%; width: 0%;
    background: var(--accent);
    transition: width 0.45s var(--easing);
  }
  .present-meta {
    position: absolute; bottom: 22px; right: 28px;
    display: flex; align-items: center; gap: 18px;
    background: rgba(247,246,242,0.92);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 8px 14px;
    border: 0.5px solid var(--line); border-radius: 2px;
    max-width: min(60vw, 640px);
  }
  .present-section {
    font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.06em; color: var(--ink-2);
    text-transform: uppercase;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    min-width: 0;
  }
  .present-section em {
    color: var(--ink); font-style: normal; font-weight: 500;
    letter-spacing: 0.04em;
  }
  .present-progress {
    font-family: var(--mono); font-size: 12px;
    letter-spacing: 0.1em; color: var(--ink);
    white-space: nowrap;
    border-left: 0.5px solid var(--line);
    padding-left: 16px;
  }
  .present-hint {
    position: absolute; bottom: 22px; left: 28px;
    font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.06em; color: var(--ink-3);
    background: rgba(247,246,242,0.88);
    padding: 6px 12px;
    border: 0.5px solid var(--line); border-radius: 2px;
    transition: opacity 0.6s var(--easing);
  }
  .present-hint.fade { opacity: 0; pointer-events: none; }
  nav.top button.present-btn {
    font-family: var(--mono); font-size: 11.5px;
    color: var(--ink); background: transparent;
    border: 0.5px solid var(--line);
    padding: 7px 12px; border-radius: 1px; cursor: pointer;
    transition: all 0.2s var(--easing);
    line-height: 1;
  }
  nav.top button.present-btn:hover {
    border-color: var(--accent); color: var(--accent);
    background: var(--bg-2);
  }

  /* ─── PART DIVIDER (asymmetric) ─────────────────────────────────── */
  .part-grid {
    display: grid;
    grid-template-columns: minmax(140px, 0.42fr) minmax(0, 0.58fr);
    gap: clamp(36px, 6vw, 96px);
    align-items: start;
    max-width: 1280px;
    margin: 0 auto;
  }
  .part-numeral {
    font-family: var(--serif);
    font-weight: 250;
    font-size: clamp(120px, 17vw, 240px);
    line-height: 0.84;
    letter-spacing: -0.04em;
    color: var(--ink);
    font-variation-settings: "opsz" 144, "SOFT" 30;
    align-self: start;
    text-align: right;
    padding-top: 18px;
    font-feature-settings: "tnum";
  }
  .part-content { padding-top: 0; }
  @media (max-width: 980px) {
    .part-grid {
      grid-template-columns: 1fr;
      gap: 18px;
    }
    .part-numeral { text-align: left; padding-top: 0; font-size: clamp(96px, 22vw, 160px); }
  }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: var(--sans);
    font-size: 16.5px;
    line-height: 1.6;
    font-feature-settings: "ss01", "ss02", "cv01";
    text-rendering: optimizeLegibility;
  }
  ::selection { background: var(--accent); color: var(--bg-2); }

  /* ─── NAV ───────────────────────────────────────────────────────── */
  nav.top {
    position: fixed; top: 0; left: 0; right: 0;
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px var(--pad-x);
    background: rgba(247,246,242,0.78);
    backdrop-filter: blur(20px) saturate(1.4);
    -webkit-backdrop-filter: blur(20px) saturate(1.4);
    border-bottom: 0.5px solid var(--line);
    z-index: 50;
    transition: transform 0.32s var(--easing);
  }
  nav.top.hidden { transform: translateY(-100%); }
  nav.top .brand {
    font-family: var(--mono); font-size: 12px;
    letter-spacing: 0.08em; color: var(--ink);
    text-decoration: none; text-transform: uppercase;
  }
  nav.top .brand .dot {
    display: inline-block; width: 5px; height: 5px;
    background: var(--accent); border-radius: 50%;
    margin-right: 10px; transform: translateY(-1px);
  }
  nav.top ul.nav-links {
    display: flex; gap: 22px; list-style: none; padding: 0; margin: 0;
  }
  nav.top a.link {
    font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.05em;
    color: var(--ink-2); text-decoration: none;
    transition: color 0.18s var(--easing);
    text-transform: uppercase;
  }
  nav.top a.link:hover { color: var(--accent); }
  nav.top .nav-tools {
    display: flex; align-items: center; gap: 10px;
  }
  nav.top button.toc-btn,
  nav.top button.lang-btn {
    font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.05em;
    color: var(--ink); background: transparent;
    border: 0.5px solid var(--line);
    padding: 7px 12px; border-radius: 1px; cursor: pointer;
    text-transform: uppercase;
    transition: all 0.2s var(--easing);
  }
  nav.top button.toc-btn:hover,
  nav.top button.lang-btn:hover {
    border-color: var(--accent); color: var(--accent);
    background: var(--bg-2);
  }
  nav.top button.lang-btn {
    min-width: 38px;
  }

  #progress {
    position: fixed; top: 0; left: 0; height: 1.5px;
    background: var(--accent);
    width: var(--scroll, 0%);
    z-index: 60;
    transition: width 0.05s linear;
  }

  /* ─── TOC PANEL (slide-out) ─────────────────────────────────────── */
  aside.toc-panel {
    position: fixed; top: 0; right: -480px;
    width: 460px; max-width: 90vw; height: 100vh;
    background: var(--bg-2);
    border-left: 0.5px solid var(--line);
    box-shadow: -20px 0 60px -30px rgba(20,20,19,0.18);
    z-index: 70;
    transition: right 0.4s var(--easing);
    overflow-y: auto;
    padding: 24px 32px 60px 32px;
  }
  aside.toc-panel.open { right: 0; }
  .toc-head {
    display: flex; justify-content: space-between; align-items: center;
    padding-bottom: 18px; border-bottom: 0.5px solid var(--line);
    margin-bottom: 22px;
    font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-3);
  }
  #toc-close {
    background: transparent; border: 0; cursor: pointer;
    font-size: 26px; line-height: 1; color: var(--ink-2);
    padding: 0 6px;
  }
  #toc-close:hover { color: var(--accent); }
  .toc-part { margin-bottom: 26px; }
  .toc-part-title {
    display: block; text-decoration: none;
    font-family: var(--serif); font-size: 19px;
    color: var(--ink); margin-bottom: 10px;
    font-variation-settings: "opsz" 36, "SOFT" 60;
    line-height: 1.2;
  }
  .toc-num {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.1em; color: var(--accent);
    margin-right: 8px; text-transform: uppercase;
    display: block; margin-bottom: 2px;
  }
  .toc-part ul {
    list-style: none; padding: 0; margin: 0 0 0 18px;
    border-left: 0.5px solid var(--line-2);
  }
  .toc-part ul li a {
    display: flex; gap: 14px;
    padding: 7px 0 7px 14px;
    font-family: var(--sans); font-size: 14px;
    color: var(--ink-2); text-decoration: none;
    transition: color 0.18s var(--easing);
    line-height: 1.3;
  }
  .toc-part ul li a:hover { color: var(--accent); }
  .toc-fig {
    font-family: var(--mono); font-size: 10.5px;
    color: var(--ink-3); flex-shrink: 0; padding-top: 1px;
  }
  #toc-overlay {
    position: fixed; inset: 0;
    background: rgba(20,20,19,0.18);
    backdrop-filter: blur(4px);
    z-index: 65;
    opacity: 0; pointer-events: none;
    transition: opacity 0.35s var(--easing);
  }
  #toc-overlay.show { opacity: 1; pointer-events: auto; }

  /* ─── HERO ─────────────────────────────────────────────────────── */
  section.hero {
    min-height: 100vh;
    padding: 14vh var(--pad-x) 6vh var(--pad-x);
    display: grid;
    grid-template-rows: auto 1fr auto;
    position: relative;
    overflow: hidden;
  }
  section.hero::before {
    content: "";
    position: absolute; inset: 0;
    background-image:
      linear-gradient(to right, var(--line-2) 1px, transparent 1px),
      linear-gradient(to bottom, var(--line-2) 1px, transparent 1px);
    background-size: 88px 88px;
    background-position: -1px -1px;
    mask-image: radial-gradient(ellipse 90% 70% at 50% 45%, #000 30%, transparent 78%);
    -webkit-mask-image: radial-gradient(ellipse 90% 70% at 50% 45%, #000 30%, transparent 78%);
    opacity: 0.85;
    pointer-events: none;
  }
  .hero-meta {
    display: flex; justify-content: space-between; align-items: flex-start;
    font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em;
    color: var(--ink-3); text-transform: uppercase;
    z-index: 2;
  }
  .hero-meta .left, .hero-meta .right { line-height: 1.7; }
  .hero-meta .right { text-align: right; }
  .hero-meta strong { color: var(--ink-2); font-weight: 500; }
  .hero-content { align-self: center; max-width: 1200px; z-index: 2; }
  .hero-tag {
    font-family: var(--mono); font-size: 11.5px;
    color: var(--accent); letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 32px;
    display: inline-flex; align-items: center; gap: 12px;
  }
  .hero-tag::before {
    content: ""; display: inline-block;
    width: 28px; height: 1px; background: var(--accent);
  }
  .hero-title {
    font-family: var(--serif);
    font-weight: 280;
    font-size: clamp(48px, 7.5vw, 108px);
    line-height: 0.96;
    letter-spacing: -0.025em;
    margin: 0;
    font-variation-settings: "opsz" 144, "SOFT" 30;
    color: var(--ink);
  }
  .hero-title em {
    font-style: italic; font-weight: 280; color: var(--accent);
    font-variation-settings: "opsz" 144, "SOFT" 100;
  }
  .hero-title .small {
    display: block;
    font-size: 0.42em; color: var(--ink-2);
    font-weight: 320; margin-top: 22px;
    letter-spacing: -0.01em; line-height: 1.32;
    max-width: 720px;
    font-variation-settings: "opsz" 72, "SOFT" 100;
  }
  .hero-stats {
    display: flex; gap: 56px;
    margin-top: 60px; flex-wrap: wrap;
  }
  .hero-stats .stat {
    border-left: 0.5px solid var(--line);
    padding-left: 16px;
  }
  .hero-stats .stat .num {
    font-family: var(--serif); font-size: 32px;
    font-weight: 350; color: var(--ink);
    letter-spacing: -0.02em; line-height: 1;
    display: block;
    font-feature-settings: "tnum";
    font-variation-settings: "opsz" 36, "SOFT" 80;
  }
  .hero-stats .stat .label {
    font-family: var(--mono); font-size: 10.5px;
    color: var(--ink-3); letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 9px; display: block;
  }
  .hero-foot {
    display: flex; justify-content: space-between;
    align-items: flex-end;
    font-family: var(--mono); font-size: 11px;
    color: var(--ink-3); letter-spacing: 0.06em;
    text-transform: uppercase; z-index: 2;
  }
  .hero-foot .scroll-cue {
    display: flex; align-items: center; gap: 12px;
    color: var(--ink-2);
  }
  .hero-foot .scroll-cue .arrow {
    width: 1px; height: 28px; background: var(--ink-2);
    position: relative;
  }
  .hero-foot .scroll-cue .arrow::after {
    content: ""; position: absolute;
    left: -3px; bottom: -1px;
    width: 7px; height: 7px;
    border-right: 1px solid var(--ink-2);
    border-bottom: 1px solid var(--ink-2);
    transform: rotate(45deg) translate(-1px,-1px);
  }
  .hero-foot a {
    color: var(--ink-2); text-decoration: none;
    transition: color 0.2s var(--easing);
  }
  .hero-foot a:hover { color: var(--accent); }

  /* ─── PART DIVIDER ─────────────────────────────────────────────── */
  section.part {
    padding: 22vh var(--pad-x) 14vh var(--pad-x);
    text-align: left;
    max-width: 1100px;
    margin: 0 auto;
  }
  .part-meta {
    font-family: var(--mono); font-size: 11.5px;
    color: var(--accent); letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-bottom: 32px;
    display: inline-flex; align-items: center; gap: 14px;
  }
  .part-meta::before {
    content: ""; display: inline-block;
    width: 36px; height: 1px; background: var(--accent);
  }
  .part-title {
    font-family: var(--serif);
    font-weight: 300;
    font-size: clamp(48px, 6.5vw, 88px);
    line-height: 1.0;
    letter-spacing: -0.025em;
    margin: 0 0 32px 0;
    color: var(--ink);
    font-variation-settings: "opsz" 144, "SOFT" 50;
  }
  .part-intro {
    max-width: 660px;
  }
  .part-intro p {
    font-family: var(--serif);
    font-size: clamp(19px, 1.8vw, 23px);
    line-height: 1.45;
    color: var(--ink-2);
    margin: 0 0 18px 0;
    font-style: italic;
    font-weight: 350;
    font-variation-settings: "opsz" 36, "SOFT" 80;
  }

  /* ─── CHAPTER (figure section) ─────────────────────────────────── */
  section.chapter {
    padding: 14vh var(--pad-x) 8vh var(--pad-x);
    position: relative;
  }
  .ch-head { max-width: var(--max-wide); margin: 0 auto 64px auto; }
  .ch-num {
    font-family: var(--mono); font-size: 11px;
    color: var(--accent); letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 24px;
    display: inline-flex; align-items: center; gap: 12px;
  }
  .ch-num::before {
    content: ""; display: inline-block;
    width: 36px; height: 0.5px; background: var(--accent);
  }
  .ch-title {
    font-family: var(--serif);
    font-weight: 320;
    font-size: clamp(36px, 4.5vw, 64px);
    line-height: 1.02;
    letter-spacing: -0.022em;
    margin: 0 0 18px 0;
    color: var(--ink);
    max-width: 980px;
    font-variation-settings: "opsz" 96, "SOFT" 50;
  }
  .ch-title em {
    font-style: italic; font-weight: 320; color: var(--ink);
    font-variation-settings: "opsz" 96, "SOFT" 100;
  }
  .ch-sub {
    font-family: var(--serif);
    font-size: clamp(18px, 1.8vw, 22px);
    font-style: italic;
    font-weight: 350;
    color: var(--ink-2);
    margin: 0; max-width: 780px;
    line-height: 1.4;
    font-variation-settings: "opsz" 36, "SOFT" 70;
  }
  .ch-body {
    max-width: var(--max-narrow);
    margin: 0 auto;
    color: var(--ink-2);
  }
  .ch-body p {
    margin: 0 0 22px 0; font-size: 17px; line-height: 1.65;
    hyphens: auto;
  }
  .ch-body > p.lead::first-letter {
    font-family: var(--serif);
    font-size: 4.6em; line-height: 0.9;
    float: left; color: var(--ink);
    padding: 6px 12px 0 0;
    font-weight: 320;
    font-variation-settings: "opsz" 144, "SOFT" 30;
  }
  .ch-body > p.lead { color: var(--ink); }
  .ch-body strong { color: var(--ink); font-weight: 500; }
  .ch-body em { font-style: italic; color: var(--ink-2); }
  .ch-body code {
    font-family: var(--mono); font-size: 0.86em;
    background: var(--bg-3); padding: 2px 6px;
    border-radius: 1px; color: var(--ink);
    border: 0.5px solid var(--line);
  }
  .ch-body a {
    color: var(--accent); text-decoration: none;
    border-bottom: 0.5px solid var(--accent-l);
    padding-bottom: 1px;
    transition: opacity 0.2s var(--easing);
  }
  .ch-body a:hover { opacity: 0.7; }

  /* Pull quote */
  .pull {
    max-width: 1080px; margin: 84px auto 56px auto;
    border-top: 0.5px solid var(--line);
    border-bottom: 0.5px solid var(--line);
    padding: 42px 0;
    font-family: var(--serif);
    font-size: clamp(24px, 2.4vw, 34px);
    font-style: italic;
    line-height: 1.25;
    color: var(--ink);
    font-weight: 320;
    letter-spacing: -0.012em;
    font-variation-settings: "opsz" 60, "SOFT" 100;
    text-align: left;
  }
  .pull::before {
    content: "\201C"; color: var(--accent);
    font-size: 1.4em; line-height: 0;
    vertical-align: -0.18em; margin-right: 4px;
  }
  .pull::after {
    content: "\201D"; color: var(--accent);
    font-size: 1.4em; line-height: 0;
    vertical-align: -0.18em; margin-left: 2px;
  }

  /* Figure */
  .figure { max-width: var(--max-wide); margin: 56px auto 0 auto; }
  .figure-wrap {
    position: relative;
    background: var(--bg-2);
    border-radius: 3px;
    padding: 32px;
    overflow: hidden;
    border: 0.5px solid var(--line-2);
    box-shadow:
      0 1px 0 rgba(20,20,19,0.02),
      0 12px 32px -16px rgba(20,20,19,0.10),
      0 36px 80px -40px rgba(20,20,19,0.10);
    transition: height 0.45s var(--easing),
                box-shadow 0.4s var(--easing),
                transform 0.4s var(--easing);
  }
  .figure-wrap:hover {
    box-shadow:
      0 1px 0 rgba(20,20,19,0.03),
      0 16px 40px -16px rgba(20,20,19,0.14),
      0 48px 100px -40px rgba(20,20,19,0.14);
    transform: translateY(-1px);
  }
  .figure-wrap iframe {
    width: 100%; height: 100%;
    border: none; display: block; background: transparent;
  }
  .figure-wrap::before {
    content: "Cargando figura\2026";
    position: absolute; inset: 0;
    display: flex; align-items: center; justify-content: center;
    background: var(--bg-2); color: var(--ink-3);
    font-family: var(--mono); font-size: 12px;
    letter-spacing: 0.08em; text-transform: uppercase;
    z-index: 0; transition: opacity 0.4s var(--easing);
  }
  .figure-wrap.loaded::before { opacity: 0; pointer-events: none; }

  .figure-meta {
    display: flex; justify-content: space-between;
    align-items: flex-start; gap: 32px;
    margin-top: 18px; padding: 0 4px;
  }
  .figure-meta .caption {
    flex: 1; max-width: 720px;
    font-family: var(--mono); font-size: 11.5px;
    line-height: 1.55; color: var(--ink-2);
    letter-spacing: 0.01em;
  }
  .figure-meta .id {
    font-family: var(--mono); font-size: 10.5px;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--ink-3); white-space: nowrap;
  }
  .figure-meta .id strong { color: var(--accent); font-weight: 500; }

  /* ─── CONCEPT DIAGRAMS ─────────────────────────────────────────── */
  section.chapter.concept .diagram-wrap {
    position: relative;
    background: var(--bg-2);
    border-radius: 3px;
    padding: 56px 48px;
    border: 0.5px solid var(--line-2);
    box-shadow:
      0 1px 0 rgba(20,20,19,0.02),
      0 12px 32px -16px rgba(20,20,19,0.10),
      0 36px 80px -40px rgba(20,20,19,0.10);
  }
  .diagram { color: var(--ink); }

  /* — Diagram 02.1 · Bisturí experimental — */
  .diagram-bisturi {
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1fr;
    gap: 18px;
    align-items: stretch;
    row-gap: 24px;
  }
  .bist-stage {
    display: flex; flex-direction: column;
    align-items: center;
    gap: 16px;
    text-align: center;
  }
  .bist-stage-num {
    font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.18em; color: var(--ink-3);
    text-transform: uppercase;
  }
  .bist-stage-num span {
    display: inline-block;
    padding: 4px 10px;
    border: 0.5px solid var(--line);
    border-radius: 999px;
  }
  .bist-stage-glyph {
    width: 140px; height: 120px;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #F0EFEA 0%, #E5E4DE 100%);
    border-radius: 4px;
    border: 0.5px solid var(--line);
    position: relative;
    box-shadow: inset 0 0 0 1px rgba(20,20,19,0.02);
  }
  .bist-mat-full .lbl {
    font-family: var(--serif); font-size: 38px;
    font-style: italic; color: var(--ink);
    line-height: 1;
  }
  .bist-mat-full .sub {
    position: absolute; bottom: 10px;
    font-family: var(--mono); font-size: 10px;
    color: var(--ink-3); letter-spacing: 0.08em;
  }
  .bist-mat-svd {
    display: flex; gap: 6px; align-items: center; padding: 0 14px;
  }
  .bist-block {
    flex: 1;
    height: 78px;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--serif); font-size: 18px; font-style: italic;
    color: var(--accent);
    background: rgba(31,78,108,0.05);
    border: 0.5px solid rgba(31,78,108,0.25);
    border-radius: 3px;
  }
  .bist-block.u { transform: scaleY(1.0); }
  .bist-block.s { transform: scaleX(0.45); flex: 0 0 22px;
                  background: rgba(31,78,108,0.12); }
  .bist-block.v { transform: scaleY(0.55); }
  .bist-delta {
    background: linear-gradient(135deg, #FBF4ED 0%, #F4E8DA 100%);
    border-color: rgba(155,90,40,0.30);
  }
  .bist-delta-symbol {
    font-family: var(--serif); font-size: 30px;
    color: #8B4F1F; font-style: italic;
  }
  .bist-stage-cap {
    display: flex; flex-direction: column; gap: 6px;
    max-width: 200px;
  }
  .bist-stage-cap strong {
    font-family: var(--serif); font-size: 16px;
    font-weight: 500; color: var(--ink); letter-spacing: -0.005em;
  }
  .bist-stage-cap em { font-style: italic; }
  .bist-stage-cap > span {
    font-family: var(--mono); font-size: 11px;
    color: var(--ink-3); line-height: 1.55;
  }
  .bist-arrow {
    display: flex; flex-direction: column;
    justify-content: center; align-items: center;
    gap: 6px; min-width: 80px;
    padding-top: 36px;
  }
  .bist-arrow-lbl {
    font-family: var(--mono); font-size: 10.5px;
    color: var(--accent); letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  .bist-arrow-lbl sub {
    font-size: 9px; vertical-align: -0.2em; margin-left: 1px;
  }
  .bist-arrow-line {
    width: 60px; height: 1px;
    background: var(--ink-4);
    position: relative;
  }
  .bist-arrow-line::after {
    content: ""; position: absolute; right: -1px; top: -3px;
    width: 0; height: 0;
    border-top: 4px solid transparent;
    border-bottom: 4px solid transparent;
    border-left: 7px solid var(--ink-4);
  }
  .bist-thesis {
    grid-column: 1 / -1;
    margin-top: 16px;
    padding-top: 22px;
    border-top: 0.5px solid var(--line);
    text-align: center;
    font-family: var(--serif); font-size: 18px;
    font-style: italic; color: var(--ink-2);
    line-height: 1.5;
  }
  .bist-thesis strong {
    font-weight: 500; color: var(--accent); font-style: normal;
  }

  /* — Diagram 05.1 · Tripartita — */
  .diagram-tripartita { color: var(--ink); }
  .tri-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0;
    align-items: stretch;
  }
  .tri-col {
    padding: 10px 28px;
    display: flex; flex-direction: column;
    gap: 12px;
    border-right: 0.5px solid var(--line);
  }
  .tri-col:last-child { border-right: none; }
  .tri-num {
    font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.18em; color: var(--ink-3);
    text-transform: uppercase;
  }
  .tri-num span {
    display: inline-block;
    padding: 4px 10px;
    border: 0.5px solid var(--line);
    border-radius: 999px;
  }
  .tri-title {
    margin: 4px 0 0 0;
    font-family: var(--serif); font-size: 26px;
    font-weight: 400; letter-spacing: -0.012em;
    color: var(--ink); line-height: 1.05;
  }
  .tri-def {
    margin: 0;
    font-family: var(--serif); font-size: 15px;
    line-height: 1.5; color: var(--ink-2);
  }
  .tri-def em {
    color: var(--accent); font-style: italic;
  }
  .tri-divider {
    height: 1px; background: var(--line-2);
    margin: 6px 0;
  }
  .tri-method-lbl {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--ink-3);
  }
  .tri-method {
    font-family: var(--serif); font-size: 17px;
    font-style: italic; color: var(--accent);
    margin-top: -4px;
  }
  .tri-method-sub {
    margin: 0;
    font-family: var(--mono); font-size: 11px;
    color: var(--ink-3); line-height: 1.55;
  }
  .tri-curve {
    width: 100%; height: 44px;
    color: var(--accent); margin-top: 4px;
  }
  .tri-result {
    margin: 0;
    font-family: var(--serif); font-size: 13.5px;
    color: var(--ink-2); line-height: 1.5;
    border-top: 0.5px solid var(--line-2);
    padding-top: 10px;
  }
  .tri-coda {
    margin-top: 28px;
    padding: 18px 22px;
    background: var(--bg-3);
    border-left: 2px solid var(--accent);
    font-family: var(--serif); font-size: 15px;
    line-height: 1.55; color: var(--ink-2);
  }
  .tri-coda strong { color: var(--ink); font-weight: 500; }

  /* — Diagram 06.4 · Convergencia — */
  .diagram-convergencia { color: var(--ink); }
  .conv-grid {
    display: grid;
    grid-template-columns: minmax(180px, 1.1fr) repeat(3, 2fr);
    gap: 0;
    border: 0.5px solid var(--line);
    border-radius: 3px;
    overflow: hidden;
  }
  .conv-corner, .conv-band-head, .conv-tech, .conv-cell {
    padding: 14px 16px;
    border-right: 0.5px solid var(--line-2);
    border-bottom: 0.5px solid var(--line-2);
  }
  .conv-grid > div:nth-child(4n) { border-right: none; }
  .conv-row { display: contents; }
  .conv-corner {
    background: var(--bg-3);
    display: flex; flex-direction: column; justify-content: center;
    font-family: var(--mono); font-size: 10.5px;
    letter-spacing: 0.08em; color: var(--ink-3);
    text-transform: uppercase;
  }
  .conv-band-head {
    background: var(--bg-3);
    display: flex; flex-direction: column; gap: 4px;
  }
  .conv-band-name {
    font-family: var(--serif); font-size: 17px;
    font-weight: 500; color: var(--ink);
  }
  .conv-band-sub {
    font-family: var(--mono); font-size: 10.5px;
    color: var(--ink-3); letter-spacing: 0.04em;
  }
  .conv-band-role {
    font-family: var(--serif); font-size: 12.5px;
    font-style: italic; color: var(--ink-2);
  }
  .conv-tech {
    background: var(--bg-2);
    display: flex; flex-direction: column;
    gap: 3px; justify-content: center;
  }
  .conv-tech-name {
    font-family: var(--serif); font-size: 15px;
    font-weight: 500; color: var(--ink); letter-spacing: -0.005em;
  }
  .conv-tech-metric {
    font-family: var(--mono); font-size: 10.5px;
    color: var(--ink-3); letter-spacing: 0.04em;
  }
  .conv-cell {
    background: var(--bg-2);
    display: flex; flex-direction: column; gap: 6px;
    justify-content: flex-start;
  }
  .conv-cell.d3 { background: rgba(31,78,108,0.06); }
  .conv-cell.d2 { background: rgba(31,78,108,0.03); }
  .conv-cell.d1 { background: var(--bg-2); }
  .conv-cell.d0 { background: var(--bg-3); }
  .dots {
    font-size: 14px; letter-spacing: 0.04em;
    color: var(--accent); line-height: 1;
  }
  .dots .dim { color: var(--ink-4); opacity: 0.4; }
  .conv-cell-val {
    font-family: var(--mono); font-size: 11.5px;
    color: var(--ink-2); line-height: 1.5;
  }
  .conv-coda {
    margin-top: 22px;
    text-align: center;
    font-family: var(--mono); font-size: 11px;
    color: var(--ink-3); letter-spacing: 0.03em;
  }

  /* — Diagram 08.1 · Cinco predicciones — */
  .diagram-predicciones { color: var(--ink); }
  .pred-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 22px;
  }
  .pred-card {
    position: relative;
    padding: 24px 22px 22px 22px;
    background: var(--bg-2);
    border: 0.5px solid var(--line);
    border-top: 2px solid var(--accent);
    border-radius: 2px;
    display: flex; flex-direction: column;
    gap: 12px;
  }
  .pred-num {
    position: absolute; top: 14px; right: 18px;
    font-family: var(--serif); font-size: 32px;
    font-weight: 300; color: var(--ink-4);
    line-height: 1; letter-spacing: -0.02em;
  }
  .pred-tag {
    font-family: var(--mono); font-size: 10px;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--accent);
  }
  .pred-title {
    margin: 0;
    font-family: var(--serif); font-size: 18px;
    font-weight: 500; letter-spacing: -0.012em;
    line-height: 1.18; color: var(--ink);
    max-width: 80%;
  }
  .pred-block {
    display: flex; flex-direction: column;
    gap: 4px;
    font-family: var(--serif); font-size: 14px;
    line-height: 1.5; color: var(--ink-2);
  }
  .pred-block-lbl {
    font-family: var(--mono); font-size: 9.5px;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--ink-3);
  }
  .pred-block em {
    color: var(--accent); font-style: italic;
  }
  .pred-falsify {
    padding-top: 10px;
    border-top: 0.5px dashed var(--line);
    font-style: italic; color: var(--ink-2);
  }
  .pred-falsify .pred-block-lbl { font-style: normal; }
  .pred-coda {
    margin-top: 26px;
    padding: 16px 20px;
    background: var(--bg-3);
    border-left: 2px solid var(--ink-3);
    font-family: var(--serif); font-size: 14px;
    line-height: 1.55; color: var(--ink-2);
  }

  /* Concept blocks in presentation mode: fill the figure column with
     the diagram instead of an iframe. Scroll if it overflows. */
  body.present-mode section.chapter.concept .diagram-wrap {
    grid-area: fig;
    height: 100%;
    width: 100%;
    padding: 24px 32px;
    overflow-y: auto;
    align-self: stretch;
    border-radius: 4px;
  }
  body.present-mode section.chapter.concept .diagram-bisturi {
    grid-template-columns: 1fr auto 1fr auto 1fr;
    gap: 14px;
  }
  body.present-mode section.chapter.concept .bist-stage-glyph {
    width: 110px; height: 96px;
  }
  body.present-mode section.chapter.concept .pred-grid {
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 14px;
  }

  /* Responsive collapse for the tripartita and convergencia diagrams */
  @media (max-width: 880px) {
    .diagram-bisturi {
      grid-template-columns: 1fr;
    }
    .bist-arrow {
      flex-direction: row; padding-top: 0;
    }
    .bist-arrow-line { width: 1px; height: 36px; }
    .bist-arrow-line::after {
      right: -3px; top: auto; bottom: -1px;
      border-left: 4px solid transparent;
      border-right: 4px solid transparent;
      border-top: 7px solid var(--ink-4);
      border-bottom: none;
    }
    .tri-grid { grid-template-columns: 1fr; }
    .tri-col {
      border-right: none;
      border-bottom: 0.5px solid var(--line);
      padding: 24px 0;
    }
    .tri-col:last-child { border-bottom: none; }
    .conv-grid { grid-template-columns: 1fr; }
    .conv-grid > div { border-right: none; }
  }

  /* ─── OUTRO ────────────────────────────────────────────────────── */
  section.outro {
    padding: 14vh var(--pad-x) 8vh var(--pad-x);
  }
  section.outro .ch-num,
  section.outro .ch-title,
  section.outro .ch-sub,
  section.outro .ch-body {
    max-width: var(--max-narrow);
    margin-left: auto; margin-right: auto;
  }
  section.outro .ch-num { display: inline-flex; }
  section.outro .outro-block {
    margin-top: 56px;
    padding-top: 28px;
    border-top: 0.5px solid var(--line);
  }
  section.outro .outro-block:first-child {
    margin-top: 64px;
    border-top: 0.5px solid var(--line);
  }
  section.outro .outro-label {
    font-family: var(--mono);
    font-size: 11px;
    color: var(--accent);
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 12px;
  }
  section.outro .outro-subtitle {
    font-family: var(--serif);
    font-weight: 350;
    font-size: 26px;
    color: var(--ink);
    line-height: 1.18;
    margin: 0 0 20px 0;
    font-variation-settings: "opsz" 60, "SOFT" 60;
  }
  section.outro .outro-subtitle em {
    font-style: italic; color: var(--accent);
  }
  section.outro .outro-coda {
    margin-top: 56px;
    padding-top: 28px;
    border-top: 0.5px solid var(--accent);
    font-family: var(--serif);
    font-style: italic;
    color: var(--ink-2);
    font-size: 18px;
    line-height: 1.55;
  }

  /* TOC: comments entry stands a bit apart from chapters */
  aside.toc-panel .toc-comments {
    margin-top: 28px;
    padding-top: 20px;
    border-top: 0.5px dashed var(--line);
  }

  /* ─── COMMENTS (Giscus) ────────────────────────────────────────── */
  section.comments {
    padding: 8vh var(--pad-x) 12vh var(--pad-x);
    border-top: 0.5px solid var(--line);
  }
  section.comments .ch-num,
  section.comments .ch-title,
  section.comments .ch-sub,
  section.comments #giscus-container {
    max-width: var(--max-narrow);
    margin-left: auto; margin-right: auto;
  }
  section.comments .ch-num { display: inline-flex; }
  section.comments #giscus-container {
    margin-top: 32px;
  }
  /* Hide presentation mode + section is irrelevant when projected */
  body.present-mode section.comments { display: none !important; }

  /* ─── REVEAL ────────────────────────────────────────────────────── */
  .reveal {
    opacity: 0; transform: translateY(14px);
    transition: opacity 0.85s var(--easing), transform 0.85s var(--easing);
    transition-delay: var(--delay, 0ms);
  }
  .reveal.in { opacity: 1; transform: translateY(0); }

  /* ─── FOOTER ────────────────────────────────────────────────────── */
  footer.foot {
    margin-top: 18vh;
    padding: 80px var(--pad-x) 52px var(--pad-x);
    border-top: 0.5px solid var(--line);
    background: var(--bg);
  }
  .foot-grid {
    max-width: var(--max-wide); margin: 0 auto;
    display: grid;
    grid-template-columns: 2fr 1fr 1fr 1fr;
    gap: 56px; align-items: start;
  }
  .foot-grid h3 {
    font-family: var(--mono); font-size: 10.5px;
    letter-spacing: 0.12em; color: var(--ink-3);
    text-transform: uppercase;
    margin: 0 0 18px 0; font-weight: 500;
  }
  .foot-grid p {
    font-family: var(--sans); font-size: 13.5px;
    color: var(--ink-2); margin: 0 0 8px 0;
    line-height: 1.55;
  }
  .foot-grid a {
    display: block;
    font-family: var(--mono); font-size: 12px;
    color: var(--ink); text-decoration: none;
    margin-bottom: 8px;
    border-bottom: 0.5px solid transparent;
    width: fit-content; padding-bottom: 1px;
    transition: all 0.2s var(--easing);
  }
  .foot-grid a:hover {
    color: var(--accent); border-bottom-color: var(--accent);
  }
  .foot-grid .brand-cell h2 {
    font-family: var(--serif); font-weight: 350;
    font-size: 24px; margin: 0 0 14px 0;
    color: var(--ink); letter-spacing: -0.012em;
    line-height: 1.15;
    font-variation-settings: "opsz" 36, "SOFT" 80;
  }
  .foot-grid .brand-cell em {
    font-style: italic; color: var(--accent);
  }
  .foot-bottom {
    max-width: var(--max-wide);
    margin: 56px auto 0 auto;
    padding-top: 24px;
    border-top: 0.5px solid var(--line-2);
    display: flex; justify-content: space-between;
    align-items: center; gap: 24px;
    font-family: var(--mono); font-size: 10.5px;
    letter-spacing: 0.06em; color: var(--ink-3);
    text-transform: uppercase;
  }

  /* ─── MOBILE ────────────────────────────────────────────────────── */
  @media (max-width: 980px) {
    nav.top ul.nav-links { display: none; }
    .hero-stats { gap: 32px; }
    .hero-foot { flex-direction: column; gap: 24px; align-items: flex-start; }
    .foot-grid { grid-template-columns: 1fr 1fr; gap: 40px 32px; }
    .ch-body > p:first-of-type::first-letter {
      font-size: 3.6em; padding: 4px 10px 0 0;
    }
    section.part { padding: 14vh var(--pad-x) 8vh var(--pad-x); }
  }
  @media (max-width: 620px) {
    .foot-grid { grid-template-columns: 1fr; }
    .hero-meta { font-size: 10px; }
    .pull { font-size: 22px; padding: 28px 0; margin: 56px auto; }
    .figure-meta { flex-direction: column; gap: 12px; }
  }

  @media (prefers-reduced-motion: reduce) {
    .reveal { opacity: 1; transform: none; transition: none; }
    nav.top { transition: none; }
  }
</style>
"""


def render_scripts() -> str:
    return r"""
<script>
  // ─── Hero canvas: ambient particle drift ────────────────────────
  // Six color groups, each gently orbiting its own anchor on the
  // right half of the hero. Anchors are spread apart enough that
  // groups stay visually distinct; particles never fully merge.
  // Pure aesthetics — no semantic claim.
  (function() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    const COLORS = [
      "#D4A843", "#C1553A", "#7B5E7B",
      "#3A6EA5", "#5A8F7B", "#A0A09A",
      "#B5555B", "#8E6E55",
    ];

    let W = 0, H = 0, particles = [], centers = [], t0 = performance.now();

    function layout() {
      const rect = canvas.getBoundingClientRect();
      W = rect.width; H = rect.height;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // 8 anchors spread across the right two thirds of the hero.
      // Pseudo-random offsets keep them feeling organic rather than
      // a tight grid.
      const xs = [0.52, 0.92, 0.62, 0.86, 0.70, 0.95, 0.58, 0.80];
      const ys = [0.16, 0.24, 0.46, 0.58, 0.78, 0.84, 0.68, 0.36];
      centers = xs.map((fx, i) => ({ x: W * fx, y: H * ys[i] }));
    }

    function spawn() {
      particles = [];
      const NC = centers.length;
      const N = Math.min(190, Math.max(96, Math.floor(W * H / 8800)));
      for (let i = 0; i < N; i++) {
        const cluster = i % NC;
        const c = centers[cluster] || { x: W * 0.7, y: H * 0.5 };
        const a = Math.random() * Math.PI * 2;
        const r = 30 + Math.random() * 60;
        particles.push({
          x: c.x + Math.cos(a) * r,
          y: c.y + Math.sin(a) * r,
          cluster,
          color: COLORS[cluster],
          phase: Math.random() * Math.PI * 2,
          orbitR: 22 + Math.random() * 34,   // local orbit radius
          orbitW: 0.14 + Math.random() * 0.18, // angular speed
          radius: 1.3 + Math.random() * 1.8,
        });
      }
    }

    function frame(now) {
      const t = (now - t0) / 1000;
      ctx.clearRect(0, 0, W, H);

      for (const p of particles) {
        const c = centers[p.cluster];
        // Each particle orbits its own anchor with a unique phase.
        // Anchors don't move; particles never merge into one blob.
        const tx = c.x + Math.cos(t * p.orbitW + p.phase) * p.orbitR;
        const ty = c.y + Math.sin(t * p.orbitW * 1.3 + p.phase) * p.orbitR * 0.7;
        p.x += (tx - p.x) * 0.05;
        p.y += (ty - p.y) * 0.05;

        ctx.fillStyle = p.color;
        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
        ctx.fill();
      }
      requestAnimationFrame(frame);
    }

    function resize() { layout(); spawn(); }
    window.addEventListener('resize', resize, { passive: true });
    resize();
    requestAnimationFrame(frame);
  })();

  // Language toggle (ES ↔ EN). Persists in localStorage, applied
  // before any reveal/scroll logic so the right text is visible from
  // the first paint.
  (function() {
    const html = document.documentElement;
    const stored = localStorage.getItem('site-lang');
    const initial = stored === 'es' || stored === 'en'
      ? stored
      : (navigator.language || 'es').toLowerCase().startsWith('en') ? 'en' : 'es';
    html.lang = initial;

    function updateBtn(lang) {
      const cur = document.querySelector('.lang-current');
      if (cur) cur.textContent = lang === 'es' ? 'EN' : 'ES';
    }
    updateBtn(initial);

    const btn = document.getElementById('lang-btn');
    if (btn) {
      btn.addEventListener('click', () => {
        const next = html.lang === 'es' ? 'en' : 'es';
        html.lang = next;
        localStorage.setItem('site-lang', next);
        updateBtn(next);
        window.dispatchEvent(new CustomEvent('site-lang-change',
          { detail: { lang: next } }));
      });
    }
  })();

  // Reading progress
  function progress() {
    const s = window.scrollY;
    const t = document.documentElement.scrollHeight - window.innerHeight;
    const pct = Math.min(100, (s / Math.max(t, 1)) * 100);
    document.documentElement.style.setProperty('--scroll', pct + '%');
  }
  window.addEventListener('scroll', progress, { passive: true });
  progress();

  // Hide nav on scroll-down
  let lastY = 0;
  const topnav = document.getElementById('topnav');
  window.addEventListener('scroll', () => {
    const y = window.scrollY;
    if (y > 100 && y > lastY) topnav.classList.add('hidden');
    else topnav.classList.remove('hidden');
    lastY = y;
  }, { passive: true });

  // Reveal on scroll
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('in');
        io.unobserve(e.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -8% 0px' });
  document.querySelectorAll('.reveal').forEach(el => io.observe(el));

  // Lazy-load iframes (language-aware). Each iframe carries
  // data-src-es and data-src-en; pick the one matching <html lang>.
  function srcForLang(iframe, lang) {
    return lang === 'en' ? iframe.dataset.srcEn : iframe.dataset.srcEs;
  }
  const figObserver = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const wrap = e.target;
      const iframe = wrap.querySelector('iframe');
      const lang = document.documentElement.lang || 'es';
      if (iframe && !iframe.src) {
        iframe.src = srcForLang(iframe, lang);
        iframe.addEventListener('load', () => wrap.classList.add('loaded'),
          { once: true });
      }
      figObserver.unobserve(wrap);
    });
  }, { rootMargin: '300px 0px' });
  document.querySelectorAll('.figure-wrap').forEach(w => figObserver.observe(w));

  // When language changes, swap every loaded iframe to its other-language
  // sibling. iframes that haven't been lazy-loaded yet pick up automatically
  // because srcForLang reads the current lang at observe time.
  window.addEventListener('site-lang-change', (ev) => {
    const lang = ev.detail.lang;
    document.querySelectorAll('.figure-wrap iframe').forEach(iframe => {
      const wantedSrc = srcForLang(iframe, lang);
      if (!wantedSrc) return;
      // Only swap if already loaded (has src) and target differs.
      if (iframe.src && !iframe.src.endsWith(wantedSrc)) {
        const wrap = iframe.closest('.figure-wrap');
        if (wrap) wrap.classList.remove('loaded');
        iframe.src = wantedSrc;
        iframe.addEventListener('load', () => {
          if (wrap) wrap.classList.add('loaded');
        }, { once: true });
      }
    });
  });

  // TOC toggle
  const tocBtn     = document.getElementById('toc-btn');
  const tocPanel   = document.getElementById('toc-panel');
  const tocClose   = document.getElementById('toc-close');
  const tocOverlay = document.getElementById('toc-overlay');
  function openToc() {
    tocPanel.classList.add('open');
    tocOverlay.classList.add('show');
  }
  function closeToc() {
    tocPanel.classList.remove('open');
    tocOverlay.classList.remove('show');
  }
  tocBtn.addEventListener('click', openToc);
  tocClose.addEventListener('click', closeToc);
  tocOverlay.addEventListener('click', closeToc);
  tocPanel.querySelectorAll('a').forEach(a => {
    a.addEventListener('click', () => setTimeout(closeToc, 180));
  });

  // ─── Presentation mode ──────────────────────────────────────────
  // Activated via ?present=1 URL param OR the ▶ button in the nav.
  // Hides body text, captions, footer; converts the page into
  // viewport-snapped slides; ← → navigate between them; ESC exits.
  (function() {
    const body = document.body;
    const main = document.querySelector('main');
    const presentBtn = document.getElementById('present-btn');
    const counter = document.getElementById('present-counter');
    const sectionLabel = document.getElementById('present-section');
    const barFill = document.getElementById('present-bar-fill');
    const hint = document.getElementById('present-hint');
    let hintTimer = null;

    function getSlides() {
      return Array.from(document.querySelectorAll(
        'main > section.hero, main > section.part, main > section.chapter'
      ));
    }
    let slides = getSlides();
    let idx = 0;

    // For each chapter/concept slide, find which preceding part section
    // it belongs to, so we can show "Parte 02 · La SVD como bisturí".
    function buildSectionContext() {
      const ctx = new Map();
      let currentPartEs = '';
      let currentPartEn = '';
      let currentNumEs = '';
      let currentNumEn = '';
      for (const slide of slides) {
        if (slide.classList.contains('hero')) {
          ctx.set(slide, {es: '', en: ''});
          continue;
        }
        if (slide.classList.contains('part')) {
          const t = slide.querySelector('.part-title');
          const n = slide.querySelector('.part-meta');
          if (t) {
            const es = t.querySelector('[lang="es"]');
            const en = t.querySelector('[lang="en"]');
            currentPartEs = es ? es.textContent : t.textContent;
            currentPartEn = en ? en.textContent : t.textContent;
          }
          if (n) {
            const es = n.querySelector('[lang="es"]');
            const en = n.querySelector('[lang="en"]');
            currentNumEs = es ? es.textContent : n.textContent;
            currentNumEn = en ? en.textContent : n.textContent;
          }
          ctx.set(slide, {es: currentPartEs, en: currentPartEn,
                          numEs: currentNumEs, numEn: currentNumEn});
          continue;
        }
        // chapter / concept slide
        ctx.set(slide, {es: currentPartEs, en: currentPartEn,
                        numEs: currentNumEs, numEn: currentNumEn});
      }
      return ctx;
    }
    let sectionContext = buildSectionContext();

    function snapToCurrent() {
      slides = getSlides();
      sectionContext = buildSectionContext();
      const target = slides[idx];
      if (target) target.scrollIntoView({ block: 'start' });
      updateCounter();
    }
    function updateCounter() {
      if (counter) counter.textContent = `${idx + 1} / ${slides.length}`;
      if (barFill) {
        const pct = slides.length > 1
          ? (idx / (slides.length - 1)) * 100
          : 100;
        barFill.style.width = pct.toFixed(2) + '%';
      }
      if (sectionLabel) {
        const target = slides[idx];
        const meta = target ? sectionContext.get(target) : null;
        const lang = document.documentElement.lang || 'es';
        if (meta && meta.es) {
          const num = lang === 'en' ? (meta.numEn || '') : (meta.numEs || '');
          const ttl = lang === 'en' ? meta.en : meta.es;
          sectionLabel.innerHTML = num
            ? `${num.replace(/\s+/g, ' ')} · <em>${ttl}</em>`
            : `<em>${ttl}</em>`;
        } else {
          sectionLabel.innerHTML = '';
        }
      }
    }
    function showHintBriefly() {
      if (!hint) return;
      hint.classList.remove('fade');
      if (hintTimer) clearTimeout(hintTimer);
      hintTimer = setTimeout(() => hint.classList.add('fade'), 4500);
    }
    function findCurrentIndex() {
      // Pick the slide whose top is closest to the viewport top
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < slides.length; i++) {
        const r = slides[i].getBoundingClientRect();
        const d = Math.abs(r.top);
        if (d < bestDist) { bestDist = d; best = i; }
      }
      return best;
    }
    function loadIframe(s) {
      const fr = s && s.querySelector('iframe[data-src-es]');
      if (!fr || fr.src) return null;
      const lang = document.documentElement.lang || 'es';
      fr.src = lang === 'en' ? fr.dataset.srcEn : fr.dataset.srcEs;
      const wrap = fr.closest('.figure-wrap');
      if (wrap) {
        fr.addEventListener('load', () => wrap.classList.add('loaded'),
          { once: true });
      }
      return fr;
    }

    // Sliding-window bg-loader. Keeps at most PRELOAD_AHEAD slides
    // loaded ahead of the current position. When user navigates, the
    // window slides with them and the next gap fills in. One load at
    // a time so Plotly never initializes multiple figures concurrently.
    const PRELOAD_AHEAD = 8;
    let queueRunning = false;
    function ensureQueue() {
      if (queueRunning) return;
      if (!body.classList.contains('present-mode')) return;
      const limit = Math.min(slides.length, idx + PRELOAD_AHEAD + 1);
      for (let i = idx; i < limit; i++) {
        const s = slides[i];
        const fr = s && s.querySelector('iframe[data-src-es]');
        if (!fr || fr.src) continue;
        queueRunning = true;
        loadIframe(s);
        let advanced = false;
        const advance = () => {
          if (advanced) return;
          advanced = true;
          queueRunning = false;
          setTimeout(ensureQueue, 500);
        };
        fr.addEventListener('load', advance, { once: true });
        setTimeout(advance, 3000);
        return;
      }
    }

    function enter() {
      body.classList.add('present-mode');
      slides = getSlides();
      sectionContext = buildSectionContext();
      idx = findCurrentIndex();
      // Preload immediate window inline so the first few slides feel instant
      slides.slice(Math.max(0, idx - 1), idx + 4).forEach(loadIframe);
      snapToCurrent();
      showHintBriefly();
      const url = new URL(window.location);
      if (url.searchParams.get('present') !== '1') {
        url.searchParams.set('present', '1');
        history.replaceState(null, '', url.toString());
      }
      // Then start the bounded sliding-window queue.
      setTimeout(ensureQueue, 1200);
    }
    function exit() {
      body.classList.remove('present-mode');
      const url = new URL(window.location);
      url.searchParams.delete('present');
      history.replaceState(null, '', url.toString());
    }
    function go(delta) {
      slides = getSlides();
      idx = findCurrentIndex();
      idx = Math.max(0, Math.min(slides.length - 1, idx + delta));
      snapToCurrent();
      // Snap-load destination + next in case queue hasn't reached
      // them yet (user racing through slides faster than queue).
      loadIframe(slides[idx]);
      if (slides[idx + 1]) loadIframe(slides[idx + 1]);
      // Slide the bg-load window forward.
      ensureQueue();
    }

    if (presentBtn) {
      presentBtn.addEventListener('click', () => {
        if (body.classList.contains('present-mode')) exit();
        else enter();
      });
    }

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        if (body.classList.contains('present-mode')) { exit(); return; }
        closeToc();
        return;
      }
      const inPresent = body.classList.contains('present-mode');
      if (e.key === 'p' || e.key === 'P') {
        if (inPresent) exit(); else enter();
        return;
      }
      if (!inPresent) return;
      if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') {
        e.preventDefault(); go(+1);
      } else if (e.key === 'ArrowLeft' || e.key === 'PageUp') {
        e.preventDefault(); go(-1);
      } else if (e.key === 'Home') {
        idx = 0; snapToCurrent();
      } else if (e.key === 'End') {
        idx = getSlides().length - 1; snapToCurrent();
      }
    });

    // Update counter as the user scrolls inside present mode
    if (main) {
      main.addEventListener('scroll', () => {
        if (!body.classList.contains('present-mode')) return;
        idx = findCurrentIndex();
        updateCounter();
      }, { passive: true });
    }

    // Auto-enter from URL param
    const initial = new URL(window.location).searchParams.get('present');
    if (initial === '1') enter();
    updateCounter();
  })();
</script>
"""


def build() -> pathlib.Path:
    def _render_section(s: dict) -> str:
        kind = s["kind"]
        if kind == "part":
            return render_part(s)
        if kind == "figure":
            return render_figure(s)
        if kind == "concept":
            return render_concept(s)
        return ""

    # SECTIONS doesn't include hero — we prepend it
    body = render_hero() + "\n".join(
        _render_section(s) for s in SECTIONS
    ) + render_outro() + render_comments() + render_footer()

    fonts = (
        '<link rel="preconnect" href="https://fonts.googleapis.com" />\n'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />\n'
        '<link href="https://fonts.googleapis.com/css2?'
        'family=Fraunces:opsz,wght,SOFT@9..144,300..900,30..100&'
        'family=Geist:wght@300..700&'
        'family=Geist+Mono:wght@300..600&display=swap" rel="stylesheet" />'
    )

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Anatomía Emocional · Guido Biosca</title>
<meta name="description" content="Compresión selectiva e interpretabilidad mecánica de BERT-base sobre GoEmotions. TFG de Guido Biosca, FIB-UPC 2026." />
<meta property="og:title" content="Anatomía emocional de un transformer" />
<meta property="og:description" content="TFG de Guido Biosca. Compresión selectiva e interpretabilidad mecánica de BERT-base." />
<meta property="og:type" content="website" />
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml;utf8,&lt;svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'&gt;&lt;rect width='32' height='32' rx='4' fill='%23F7F6F2'/&gt;&lt;circle cx='16' cy='16' r='5' fill='%231F4E6C'/&gt;&lt;/svg&gt;" />
{fonts}
{render_styles()}
</head>
<body>

<div id="progress"></div>

{render_nav()}

<main>
{body}
</main>

{render_scripts()}

</body>
</html>
"""

    OUT.write_text(html, encoding="utf-8")
    print(f"✓ wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    return OUT


if __name__ == "__main__":
    build()
