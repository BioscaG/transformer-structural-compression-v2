"""Generate viz/site/index.html from sections.py.

This is the central assembly: takes structured section data and
produces the editorial site HTML using the design system established
in the prototype.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.site.sections import SECTIONS, HERO_STATS, FIG_HEIGHTS


SITE_DIR = pathlib.Path(__file__).resolve().parent
OUT = SITE_DIR / "index.html"


def render_hero() -> str:
    stats_html = "\n".join(
        f'      <div class="stat">\n'
        f'        <span class="num">{num}</span>\n'
        f'        <span class="label">{label}</span>\n'
        f'      </div>'
        for num, label in HERO_STATS
    )
    return f"""
<section class="hero" id="top">
  <div class="hero-meta reveal">
    <div class="left">
      <strong>TFG · 2026</strong><br>
      Universitat Politècnica<br>
      de Catalunya · FIB
    </div>
    <div class="right">
      <strong>Vol. 01</strong><br>
      Compresión selectiva<br>
      Interpretabilidad mecánica
    </div>
  </div>

  <div class="hero-content reveal" style="--delay: 120ms">
    <div class="hero-tag">Trabajo de fin de grado · 2026</div>
    <h1 class="hero-title">
      Anatomía<br>
      emocional<br>
      <em>de un transformer</em>
      <span class="small">
        Compresión selectiva e interpretabilidad mecánica
        de BERT-base sobre GoEmotions. Un manual visual para
        desarmar el modelo capa por capa.
      </span>
    </h1>

    <div class="hero-stats">
{stats_html}
    </div>
  </div>

  <div class="hero-foot reveal" style="--delay: 240ms">
    <div class="scroll-cue">
      <span class="arrow"></span>
      <span>Scroll · Empezar el recorrido</span>
    </div>
    <div>
      <a href="sobre.html">Guido Biosca Lasa</a> · Director: Lluís Padró Cirera
    </div>
  </div>
</section>
"""


def render_part(s: dict) -> str:
    intro_html = "\n".join(f"    <p>{p}</p>" for p in s["intro"])
    return f"""
<section class="part" id="{s['id']}">
  <div class="part-meta reveal">{s['num']}</div>
  <h2 class="part-title reveal" style="--delay: 100ms">{s['title']}</h2>
  <div class="part-intro reveal" style="--delay: 200ms">
{intro_html}
  </div>
</section>
"""


def render_figure(s: dict) -> str:
    body_html = "\n".join(f'    <p class="reveal">{p}</p>'
                          for p in s["body"])
    pull_html = ""
    if s.get("pull"):
        pull_html = f'\n  <blockquote class="pull reveal">{s["pull"]}</blockquote>\n'

    fig_h = FIG_HEIGHTS.get(s["figure"], 800)
    wrap_h = fig_h + 64  # 32px padding × 2 (top + bottom of card)

    return f"""
<section class="chapter" id="{s['id']}">
  <div class="ch-head">
    <div class="ch-num reveal">{s['chapter']}</div>
    <h2 class="ch-title reveal" style="--delay: 80ms">{s['title']}</h2>
    <p class="ch-sub reveal" style="--delay: 160ms">{s['subtitle']}</p>
  </div>

  <div class="ch-body">
{body_html}
  </div>
{pull_html}
  <div class="figure reveal">
    <div class="figure-wrap" style="height: {wrap_h}px">
      <iframe data-src="figures/{s['figure']}.html" loading="lazy" scrolling="no"></iframe>
    </div>
    <div class="figure-meta">
      <div class="caption">{s['caption']}</div>
      <div class="id">Fig. <strong>{s['fig_id']}</strong></div>
    </div>
  </div>
</section>
"""


def render_outro() -> str:
    return """
<section class="outro" id="cierre">
  <div class="ch-num reveal">Cierre</div>
  <h2 class="ch-title reveal" style="--delay: 80ms">Cómo está hecho</h2>
  <p class="ch-sub reveal" style="--delay: 160ms">Stack, datos y créditos.</p>

  <div class="ch-body">
    <p class="reveal">
      Datos numéricos. Las visualizaciones se alimentan directamente
      de los CSVs de los notebooks 2 al 9. Resultados reales del
      fine-tune sobre BERT-base-uncased y 23 emociones de GoEmotions.
      61 tablas exportadas: probing por capa, ablación de 144 cabezas,
      especialización neuronal, activation patching, frontera de
      Pareto completa con 22 estrategias evaluadas.
    </p>
    <p class="reveal">
      Activaciones y geometría. Galaxy formation, sentence trajectory,
      compression decay y spectral flowers se computan ejecutando el
      checkpoint <code>23emo-final</code> (109.5M parámetros, 23
      emociones) sobre frases del test set. Pooler y classifier
      reales aplicados en cada paso. LDA fija ajustada en L12 para
      coordenadas consistentes capa a capa.
    </p>
    <p class="reveal">
      Visualización. 27 piezas: 19 estáticas en Plotly, 7
      interactivas con HTML+JS custom, 1 grafo en D3.js puro. Cada
      figura es un HTML autocontenido. La página que las une es
      static HTML+CSS+JS. Sin frameworks, sin servidor, sin
      backend.
    </p>
    <p class="reveal">
      Memoria. Guido Biosca Lasa. Director: Lluís Padró Cirera.
      FIB-UPC, 2026. <a href="sobre.html">Más sobre el proyecto</a>.
    </p>
  </div>
</section>
"""


def render_footer() -> str:
    return """
<footer class="foot">
  <div class="foot-grid">
    <div class="brand-cell">
      <h2>Anatomía emocional<br><em>de un transformer.</em></h2>
      <p>Trabajo de fin de grado.</p>
      <p>Facultat d'Informàtica de Barcelona · 2026.</p>
    </div>
    <div>
      <h3>Proyecto</h3>
      <a href="#">Repositorio</a>
      <a href="#">Memoria · PDF</a>
      <a href="sobre.html">Cómo está hecho</a>
    </div>
    <div>
      <h3>Autor</h3>
      <a href="mailto:guido.biosca0@gmail.com">guido.biosca0@gmail.com</a>
      <a href="#">LinkedIn</a>
      <a href="#">GitHub</a>
    </div>
    <div>
      <h3>Tribunal</h3>
      <p>Director: Lluís Padró Cirera</p>
      <p>FIB · UPC · 2026</p>
    </div>
  </div>
  <div class="foot-bottom">
    <span>© 2026 Guido Biosca Lasa</span>
    <span>Built with Python · Plotly · D3 · no frameworks</span>
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
        elif s["kind"] == "figure" and current is not None:
            current["items"].append(s)

    toc_html = ""
    for p in parts:
        toc_html += f'<div class="toc-part">'
        toc_html += (f'<a href="#{p["part"]["id"]}" class="toc-part-title">'
                     f'<span class="toc-num">{p["part"]["num"]}</span> '
                     f'{p["part"]["title"]}</a>')
        toc_html += f'<ul>'
        for it in p["items"]:
            toc_html += (f'<li><a href="#{it["id"]}">'
                         f'<span class="toc-fig">{it["fig_id"]}</span>'
                         f' {it["title"].replace("<em>", "").replace("</em>", "")}'
                         f'</a></li>')
        toc_html += f'</ul></div>'

    return f"""
<nav class="top" id="topnav">
  <a href="#top" class="brand"><span class="dot"></span>Anatomía Emocional</a>
  <ul class="nav-links">
    <li><a class="link" href="#parte-1">Capítulos</a></li>
    <li><a class="link" href="#cierre">Datos</a></li>
    <li><a class="link" href="sobre.html">Sobre</a></li>
  </ul>
  <button class="toc-btn" id="toc-btn">Índice</button>
</nav>

<aside id="toc-panel" class="toc-panel">
  <div class="toc-head">
    <span>Índice</span>
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
  nav.top button.toc-btn {
    font-family: var(--mono); font-size: 11.5px; letter-spacing: 0.05em;
    color: var(--ink); background: transparent;
    border: 0.5px solid var(--line);
    padding: 7px 12px; border-radius: 1px; cursor: pointer;
    text-transform: uppercase;
    transition: all 0.2s var(--easing);
  }
  nav.top button.toc-btn:hover {
    border-color: var(--accent); color: var(--accent);
    background: var(--bg-2);
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
  .ch-body > p:first-of-type::first-letter {
    font-family: var(--serif);
    font-size: 4.6em; line-height: 0.9;
    float: left; color: var(--ink);
    padding: 6px 12px 0 0;
    font-weight: 320;
    font-variation-settings: "opsz" 144, "SOFT" 30;
  }
  .ch-body > p:first-of-type { color: var(--ink); }
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

  // Lazy-load iframes
  const figObserver = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (!e.isIntersecting) return;
      const wrap = e.target;
      const iframe = wrap.querySelector('iframe');
      if (iframe && iframe.dataset.src && !iframe.src) {
        iframe.src = iframe.dataset.src;
        iframe.addEventListener('load', () => wrap.classList.add('loaded'),
          { once: true });
      }
      figObserver.unobserve(wrap);
    });
  }, { rootMargin: '300px 0px' });
  document.querySelectorAll('.figure-wrap').forEach(w => figObserver.observe(w));

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
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeToc();
  });
</script>
"""


def build() -> pathlib.Path:
    sections_html = []
    for s in SECTIONS:
        if s["kind"] == "hero":
            sections_html.append(render_hero())
        elif s["kind"] == "part":
            sections_html.append(render_part(s))
        elif s["kind"] == "figure":
            sections_html.append(render_figure(s))

    # SECTIONS doesn't include hero — we prepend it
    body = render_hero() + "\n".join(
        render_part(s) if s["kind"] == "part" else
        render_figure(s) if s["kind"] == "figure" else ""
        for s in SECTIONS
    ) + render_outro() + render_footer()

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
