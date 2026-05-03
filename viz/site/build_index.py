"""Generate viz/site/index.html from sections.py.

This is the central assembly: takes structured section data and
produces the editorial site HTML using the design system established
in the prototype.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.site.sections import (
    SECTIONS, HERO_STATS, FIG_HEIGHTS, HERO, OUTRO, NAV, FOOTER,
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


def render_outro() -> str:
    label_html = bi(OUTRO["label"], tag="div", classes="ch-num reveal")
    title_html = bi(OUTRO["title"], tag="h2",  classes="ch-title reveal")
    sub_html   = bi(OUTRO["sub"],   tag="p",   classes="ch-sub reveal")
    p1 = bi(OUTRO["p1"], tag="p", classes="reveal")
    p2 = bi(OUTRO["p2"], tag="p", classes="reveal")
    p3 = bi(OUTRO["p3"], tag="p", classes="reveal")
    p4 = bi(OUTRO["p4"], tag="p", classes="reveal")
    return f"""
<section class="outro" id="cierre">
  {label_html}
  {title_html}
  {sub_html}

  <div class="ch-body">
    {p1}
    {p2}
    {p3}
    {p4}
  </div>
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
        elif s["kind"] == "figure" and current is not None:
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
    about_html    = bi(NAV["about"],    tag="span")
    index_html    = bi(NAV["index"],    tag="span")

    return f"""
<nav class="top" id="topnav">
  <a href="#top" class="brand"><span class="dot"></span>{brand_html}</a>
  <ul class="nav-links">
    <li><a class="link" href="#parte-1">{chapters_html}</a></li>
    <li><a class="link" href="#cierre">{data_html}</a></li>
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
  <div class="present-progress"><span id="present-counter">1 / 1</span></div>
  <div class="present-hint">← →   ESC to exit</div>
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
  .present-progress {
    position: absolute; bottom: 24px; right: 32px;
    font-family: var(--mono); font-size: 12px;
    letter-spacing: 0.1em; color: var(--ink-3);
    background: rgba(247,246,242,0.85); padding: 6px 12px;
    border: 0.5px solid var(--line); border-radius: 2px;
  }
  .present-hint {
    position: absolute; bottom: 24px; left: 32px;
    font-family: var(--mono); font-size: 11px;
    letter-spacing: 0.08em; color: var(--ink-3);
    text-transform: uppercase;
  }
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
    ];

    let W = 0, H = 0, particles = [], centers = [], t0 = performance.now();

    function layout() {
      const rect = canvas.getBoundingClientRect();
      W = rect.width; H = rect.height;
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      // 6 anchors spread across the right two thirds of the hero.
      // Pseudo-random offsets keep them feeling organic rather than
      // a tight, geometric hexagon.
      const xs = [0.55, 0.92, 0.62, 0.86, 0.70, 0.95];
      const ys = [0.18, 0.28, 0.52, 0.62, 0.82, 0.78];
      centers = xs.map((fx, i) => ({ x: W * fx, y: H * ys[i] }));
    }

    function spawn() {
      particles = [];
      const N = Math.min(160, Math.max(72, Math.floor(W * H / 9500)));
      for (let i = 0; i < N; i++) {
        const cluster = i % 6;
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

    function getSlides() {
      return Array.from(document.querySelectorAll(
        'main > section.hero, main > section.part, main > section.chapter'
      ));
    }
    let slides = getSlides();
    let idx = 0;

    function snapToCurrent() {
      slides = getSlides();
      const target = slides[idx];
      if (target) target.scrollIntoView({ block: 'start' });
      updateCounter();
    }
    function updateCounter() {
      if (counter) counter.textContent = `${idx + 1} / ${slides.length}`;
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
      idx = findCurrentIndex();
      // Preload immediate window inline so the first few slides feel instant
      slides.slice(Math.max(0, idx - 1), idx + 4).forEach(loadIframe);
      snapToCurrent();
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
