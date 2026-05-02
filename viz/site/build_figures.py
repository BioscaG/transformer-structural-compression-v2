"""Build all 27 figures into viz/site/figures/ in site-mode.

Strategy:
  - Plotly figures (build_X_figure -> go.Figure): retheme via
    apply_site_mode, write with write_site_figure.
  - HTML+JS custom (build_html writes self-contained HTML): build the
    original first to viz/output/, then post-process via
    inject_site_mode_into_html.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# Plotly figures
from viz.plots import (crystallization, fingerprints, heads_matrix,
                       lesion_theater, pareto_3d, spectral_flowers, sunburst)
from viz.interactive import (bert_architecture, compression_decay,
                             confusion_evolution, confusion_volume,
                             finetuning_diff, galaxy_formation,
                             greedy_replay, internal_compression,
                             lens_vs_probe, lexical_to_semantic,
                             probe_constellations, spectral_landscape,
                             token_trajectories)
# HTML+JS custom
from viz.interactive import (attention_atlas, circuit_network,
                             compression_sandbox, decision_fingerprint,
                             emotion_cards, iterative_inference,
                             neuron_gallery, sentence_trajectory)

from viz.site._site_mode import write_site_figure, inject_site_mode_into_html


SITE_DIR = pathlib.Path(__file__).resolve().parent
SITE_FIGS = SITE_DIR / "figures"
LEGACY_OUT = SITE_DIR.parent / "output"


def _build_galaxy_combined() -> pathlib.Path:
    """Special: galaxy_formation gets both 2D and 3D in a single HTML
    with a toggle. Each figure is fully rendered and styled site-mode."""
    from viz.site._site_mode import (apply_site_mode, FONT_STACK,
                                     PAGE_INK_2, PAGE_INK_3, PAGE_ACCENT,
                                     PAGE_LINE, PAGE_LINE_2, PAGE_BG_PLOT,
                                     GOOGLE_FONTS_LINK)

    print("  Building galaxy_formation 3D + 2D combined…")
    fig_3d = galaxy_formation.build_galaxy_figure(dim=3)
    apply_site_mode(fig_3d)
    fig_2d = galaxy_formation.build_galaxy_figure(dim=2)
    apply_site_mode(fig_2d)

    hi_dpi = {
        "displayModeBar": "hover", "displaylogo": False,
        "responsive": True,
        "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
        "toImageButtonOptions": {
            "format": "png", "filename": "galaxy_formation",
            "height": 1200, "width": 1800, "scale": 3,
        },
    }
    html_3d = fig_3d.to_html(full_html=False, include_plotlyjs="cdn",
                             config=hi_dpi, div_id="galaxy-fig-3d")
    html_2d = fig_2d.to_html(full_html=False, include_plotlyjs=False,
                             config=hi_dpi, div_id="galaxy-fig-2d")

    template = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Galaxy formation</title>
{GOOGLE_FONTS_LINK}
<style>
  html, body {{
    margin: 0; padding: 0;
    height: 100%; width: 100%;
    background: transparent;
    font-family: {FONT_STACK};
    color: {PAGE_INK_2};
    overflow: hidden;
  }}
  .galaxy-toggle {{
    position: absolute;
    top: 6px; right: 6px;
    display: inline-flex;
    background: rgba(255,255,255,0.92);
    border: 0.5px solid {PAGE_LINE};
    border-radius: 2px;
    z-index: 10;
    padding: 2px;
  }}
  .galaxy-toggle button {{
    background: transparent; border: 0;
    font-family: "Geist Mono", monospace;
    font-size: 10.5px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {PAGE_INK_3};
    padding: 6px 14px;
    cursor: pointer;
    border-radius: 1px;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
  }}
  .galaxy-toggle button:hover {{
    color: {PAGE_INK_2};
  }}
  .galaxy-toggle button.active {{
    background: {PAGE_ACCENT};
    color: white;
  }}
  .galaxy-fig {{
    width: 100%; height: 100%;
  }}
  .galaxy-fig.hidden {{ display: none; }}
  .modebar {{ background: transparent !important; }}
  .modebar-btn path {{ fill: {PAGE_INK_3} !important; }}
  .modebar-btn:hover path {{ fill: {PAGE_ACCENT} !important; }}
  .gtitle {{ display: none !important; }}
</style>
</head>
<body>

<div class="galaxy-toggle" role="tablist">
  <button id="btn-3d" class="active" type="button">3D</button>
  <button id="btn-2d" type="button">2D</button>
</div>

<div class="galaxy-fig" id="wrap-3d">{html_3d}</div>
<div class="galaxy-fig hidden" id="wrap-2d">{html_2d}</div>

<script>
  const btn3 = document.getElementById('btn-3d');
  const btn2 = document.getElementById('btn-2d');
  const wrap3 = document.getElementById('wrap-3d');
  const wrap2 = document.getElementById('wrap-2d');

  function show3d() {{
    btn3.classList.add('active'); btn2.classList.remove('active');
    wrap3.classList.remove('hidden'); wrap2.classList.add('hidden');
    if (window.Plotly) {{
      Plotly.Plots.resize(document.querySelector('#galaxy-fig-3d'));
    }}
  }}
  function show2d() {{
    btn2.classList.add('active'); btn3.classList.remove('active');
    wrap2.classList.remove('hidden'); wrap3.classList.add('hidden');
    if (window.Plotly) {{
      Plotly.Plots.resize(document.querySelector('#galaxy-fig-2d'));
    }}
  }}
  btn3.addEventListener('click', show3d);
  btn2.addEventListener('click', show2d);
</script>

</body>
</html>
"""

    out = SITE_FIGS / "galaxy_formation.html"
    out.write_text(template, encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"  ✓ galaxy_formation.html ({kb:.0f} KB) [3D + 2D toggle]")
    return out


# ── Plotly figures: (name, module, build_function) ────────────────────────
PLOTLY_FIGS = [
    ("bert_architecture",     bert_architecture.build_architecture_figure),
    ("lexical_to_semantic",   lexical_to_semantic.build_figure),
    ("internal_compression",  internal_compression.build_figure),
    ("lens_vs_probe",         lens_vs_probe.build_figure),
    ("spectral_flowers",      spectral_flowers.build_flowers_figure),
    ("spectral_landscape",    spectral_landscape.build_landscape_figure),
    ("pareto_3d",             pareto_3d.build_pareto_figure),
    ("compression_decay",     compression_decay.build_decay_figure),
    ("crystallization",       crystallization.build_crystallization_figure),
    # galaxy_formation handled separately (2D + 3D toggle)
    ("heads_matrix",          heads_matrix.build_heads_figure),
    ("probe_constellations",  probe_constellations.build_constellation_figure),
    ("lesion_theater",        lesion_theater.build_lesion_theater),
    ("sunburst",              sunburst.build_sunburst_figure),
    ("emotional_landscape",   fingerprints.build_landscape_figure),
    ("confusion_evolution",   confusion_evolution.build_confusion_figure),
    ("confusion_volume",      confusion_volume.build_volume_figure),
    ("greedy_replay",         greedy_replay.build_replay_figure),
    ("finetuning_diff",       finetuning_diff.build_diff_figure),
]


# ── HTML+JS custom figures: (name, build_html_callable) ───────────────────
def _make_html_builder(module, name):
    """Wrap a module's build_html into a callable that writes to a temp
    path and returns it. Each module's `build_html` takes the output path."""
    def _build(out_path: pathlib.Path) -> pathlib.Path:
        return module.build_html(out_path)
    return _build


def _token_trajectories_builder(out_path: pathlib.Path) -> pathlib.Path:
    """token_trajectories.build_trajectories_figure() returns HTML string,
    not a Plotly figure — wrap it as an HTML builder."""
    html = token_trajectories.build_trajectories_figure()
    out_path.write_text(html, encoding="utf-8")
    return out_path


HTML_FIGS = [
    ("attention_atlas",       attention_atlas.build_html),
    ("circuit_network",       circuit_network.build_html),
    ("compression_sandbox",   compression_sandbox.build_html),
    ("decision_fingerprint",  decision_fingerprint.build_html),
    ("emotion_cards",         emotion_cards.build_html),
    ("iterative_inference",   iterative_inference.build_html),
    ("neuron_gallery",        neuron_gallery.build_html),
    ("sentence_trajectory",   sentence_trajectory.build_html),
    ("token_trajectories",    _token_trajectories_builder),
]


# ── Build everything ──────────────────────────────────────────────────────

def build_all() -> dict[str, pathlib.Path]:
    SITE_FIGS.mkdir(parents=True, exist_ok=True)
    results: dict[str, pathlib.Path] = {}

    # Special: galaxy with 2D + 3D toggle
    try:
        results["galaxy_formation"] = _build_galaxy_combined()
    except Exception as exc:
        print(f"  ✗ galaxy_formation: {type(exc).__name__}: {exc}")

    print(f"Building {len(PLOTLY_FIGS)} Plotly figures…")
    for name, builder in PLOTLY_FIGS:
        try:
            fig = builder()
            out = write_site_figure(fig, SITE_FIGS / f"{name}.html")
            kb = out.stat().st_size / 1024
            print(f"  ✓ {name}.html ({kb:.0f} KB)")
            results[name] = out
        except Exception as exc:
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")

    print(f"\nBuilding {len(HTML_FIGS)} HTML+JS custom figures…")
    LEGACY_OUT.mkdir(parents=True, exist_ok=True)
    tmp_dir = SITE_DIR / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in HTML_FIGS:
        try:
            tmp = tmp_dir / f"{name}.html"
            builder(tmp)
            out = inject_site_mode_into_html(tmp, SITE_FIGS / f"{name}.html")
            kb = out.stat().st_size / 1024
            print(f"  ✓ {name}.html ({kb:.0f} KB)")
            results[name] = out
        except Exception as exc:
            print(f"  ✗ {name}: {type(exc).__name__}: {exc}")

    return results


if __name__ == "__main__":
    print("─" * 60)
    print("Building site-mode figures")
    print("─" * 60)
    build_all()
    print("─" * 60)
    print(f"Output: {SITE_FIGS}")
