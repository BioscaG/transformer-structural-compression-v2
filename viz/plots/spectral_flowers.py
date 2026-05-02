"""Hero plot 9 — Spectral flowers: 72 polar plots, one per weight matrix.

Each weight matrix (12 layers × 6 components) gets visualized as a polar
"flower" where each petal is a singular value. The petal length encodes its
magnitude relative to σ₁. The result:

  - Q, K matrices: flowers with 1-3 huge petals + tiny rest. The "spectrum
    is concentrated" — most of the action lives in a low-rank subspace.
  - V, Attn Output: intermediate flowers — gradual decay.
  - FFN matrices: round, full flowers — every singular value contributes
    meaningfully. The "spectrum is flat" — no compression-friendly structure.

Why this matters: the SHAPE of these flowers is the §4.1 finding made
visual. The k95 rank (Tabla 6) is mathematically what these shapes
encode. Concentrated flowers → compressible. Round flowers → fragile.

Computes real SVD on the actual fine-tuned model. Heavy-ish (~40s) but
runs once and caches the spectra to JSON.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import style as st


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"
SPECTRA_PATH = CACHE_DIR / "spectra.json"


def compute_or_load_spectra(top_k: int = 64, force: bool = False) -> dict:
    """Return per-(layer, component) spectrum vectors. Computes on the model
    if not cached.

    Output: { "L{layer}.{component}": [σ₁, σ₂, …, σ_top_k] (normalized to σ₁) }
    """
    if SPECTRA_PATH.exists() and not force:
        return json.loads(SPECTRA_PATH.read_text())

    print("Computing real SVD on the user's 23-emotion model (≈40s)...")
    import torch
    from transformers import AutoModelForSequenceClassification

    MODEL_NAME = str(pathlib.Path(__file__).resolve().parents[2] / "results" / "checkpoints" / "23emo-final")
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.eval()

    # Path templates
    components = {
        "query":            "bert.encoder.layer.{L}.attention.self.query",
        "key":              "bert.encoder.layer.{L}.attention.self.key",
        "value":            "bert.encoder.layer.{L}.attention.self.value",
        "attn_output":      "bert.encoder.layer.{L}.attention.output.dense",
        "ffn_intermediate": "bert.encoder.layer.{L}.intermediate.dense",
        "ffn_output":       "bert.encoder.layer.{L}.output.dense",
    }

    spectra = {}
    for L in range(12):
        for cname, tpl in components.items():
            mod = mdl.get_submodule(tpl.format(L=L))
            W = mod.weight.detach().float()
            S = torch.linalg.svdvals(W).cpu().numpy()
            S_norm = (S / S[0]).tolist()[:top_k]   # normalize to σ₁, keep top_k
            # Also full energy curve (for k95 calculation)
            energy = (S**2).cumsum() / (S**2).sum()
            k95 = int(np.searchsorted(energy, 0.95) + 1)
            k90 = int(np.searchsorted(energy, 0.90) + 1)
            spectra[f"L{L}.{cname}"] = {
                "spectrum": S_norm,
                "k95": k95,
                "k90": k90,
                "rank": int(min(W.shape)),
                "shape": list(W.shape),
            }
            print(f"  L{L:02d} {cname:18s}  k95={k95:4d}  k90={k90:4d}")
    SPECTRA_PATH.write_text(json.dumps(spectra, indent=2))
    print(f"✓ saved {SPECTRA_PATH}")
    return spectra


def build_flowers_figure(top_k: int = 32) -> go.Figure:
    spectra = compute_or_load_spectra(top_k=64)

    components = ["query", "key", "value", "attn_output", "ffn_intermediate", "ffn_output"]
    comp_label = {"query": "Q", "key": "K", "value": "V",
                  "attn_output": "Attn-O", "ffn_intermediate": "FFN-i", "ffn_output": "FFN-o"}
    comp_color = {
        "query": st.BLUE, "key": st.BLUE_L, "value": st.TERRA,
        "attn_output": st.ROSE, "ffn_intermediate": st.TEAL, "ffn_output": st.TEAL_L,
    }

    n_layers = 12
    n_comp = len(components)

    # Build a 12 (rows) × 6 (cols) grid of polar subplots
    fig = make_subplots(
        rows=n_layers, cols=n_comp,
        specs=[[{"type": "polar"}] * n_comp for _ in range(n_layers)],
        horizontal_spacing=0.012, vertical_spacing=0.014,
        column_titles=[comp_label[c] for c in components],
        row_titles=[f"L{i}" for i in range(n_layers)],
    )

    for L in range(n_layers):
        for ci, cname in enumerate(components):
            key = f"L{L}.{cname}"
            sp = spectra[key]
            spectrum = sp["spectrum"][:top_k]
            n_pts = len(spectrum)

            # Each petal at angle 360 * i / n_pts, length = σ_i (already normalized)
            theta = np.linspace(0, 360, n_pts, endpoint=False)
            r = np.array(spectrum)

            # Lay petals as bars
            fig.add_trace(go.Barpolar(
                r=r, theta=theta,
                width=360 / n_pts * 0.92,
                marker=dict(color=comp_color[cname], opacity=0.85,
                            line=dict(color="white", width=0.3)),
                hovertemplate=(f"<b>L{L} · {comp_label[cname]}</b><br>"
                               "σᵢ / σ₁ = %{r:.3f}<br>"
                               f"k95 = {sp['k95']} (de {sp['rank']})<br>"
                               f"shape = {sp['shape'][0]}×{sp['shape'][1]}<extra></extra>"),
                showlegend=False,
            ), row=L+1, col=ci+1)

    # Strip all polar axes
    for L in range(n_layers):
        for ci in range(n_comp):
            polar_id = "polar" if (L == 0 and ci == 0) else f"polar{L * n_comp + ci + 1}"
            fig.update_layout(**{polar_id: dict(
                radialaxis=dict(showticklabels=False, showgrid=False, ticks="",
                                visible=False, range=[0, 1.05]),
                angularaxis=dict(showticklabels=False, showgrid=False, ticks="",
                                 visible=False),
                bgcolor="white",
            )})

    # Style row/column titles
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=10.5, color=st.INK_2, family="serif")

    fig.update_layout(
        **st.thesis_layout(
            title=("Spectral flowers · 72 matrices, 72 huellas espectrales"
                   "<br><sub>Cada flor es una matriz de pesos. Pétalo i = σᵢ/σ₁. "
                   "Q/K son flores espigadas (espectro concentrado, compresibles); "
                   "FFN son flores rellenas (espectro plano, frágiles).</sub>"),
            height=1200, width=1100,
        ),
    )
    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_flowers_figure()
    out = out_dir / "12_spectral_flowers.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
