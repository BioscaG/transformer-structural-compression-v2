"""One-shot build of the entire site.

  .viz_venv/bin/python viz/site/build_all.py

Produces:
  viz/site/figures/*.html  (27 figures)
  viz/site/index.html
  viz/site/sobre.html      (already authored, not generated)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from viz.site import build_figures, build_index


if __name__ == "__main__":
    print("─" * 60)
    print("Anatomía Emocional · build")
    print("─" * 60)
    build_figures.build_all()
    print()
    build_index.build()
    print("─" * 60)
    print("Done. Open viz/site/index.html (or serve via local HTTP).")
    print("  python3 -m http.server -d viz/site 8080")
    print("─" * 60)
