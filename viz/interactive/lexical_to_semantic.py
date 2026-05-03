"""De diccionario a clasificador · cómo BERT olvida palabras y aprende emociones.

Three layer-wise curves computed on the user's fine-tuned 23emo model,
all from the existing token_trajectories cache:

  1. Retención léxica = mean cosine similarity between hidden[L, token]
     and the input embedding hidden[0, token]. Decays from 1.0 to ~0 — the
     model gradually "forgets" which word each position originally held.

  2. Anisotropía = mean pairwise cosine similarity between DIFFERENT
     tokens within the same sentence. Stays low until L8, then explodes —
     all tokens converge toward a single "context" vector. The classic
     anisotropy phenomenon (Ethayarajh, EMNLP 2019).

  3. Emergencia semántica = mean probe F1 per layer (already computed in
     notebook 4). Grows monotonically — emotion information becomes
     linearly extractable from CLS.

The three curves cross around L8-L9 — that is the lexical→semantic hinge
point in this fine-tune. This replicates the well-known finding of
Tenney et al. ("BERT Rediscovers the NLP Pipeline", ACL 2019) on the
user's model and quantifies WHERE the transition happens for emotion.

Companion right-side view: a (layer × token) heatmap of cosine
similarity to L0 for 4 example sentences. Visually shows token identity
decay — early rows have varied colors (each token preserves identity),
late rows are uniform (all tokens have lost their original embedding).
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viz import style as st


CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "cache"
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PROBE_CSV = PROJECT_ROOT / "results" / "csvs" / "notebook4" / "layer_information_gain.csv"


def _compute_curves():
    """Return dict with the three curves and the per-token cosine-to-L0 grid."""
    d = np.load(CACHE_DIR / "token_trajectories.npz")
    H = d["hidden"].astype(np.float32)  # (N, 13, T, 768)
    meta = json.loads((CACHE_DIR / "token_trajectories_meta.json").read_text())
    tokens = meta["tokens"]
    sentences = meta["sentences"]
    labels = meta["labels"]

    n_sent, n_layers, T, _ = H.shape

    # Mask of "real" content tokens (drop PAD/CLS/SEP).
    # The cache stores special tokens as ⟨CLS⟩/⟨SEP⟩ and pad as "".
    SPECIAL = {"⟨CLS⟩", "⟨SEP⟩", "[CLS]", "[SEP]", "[PAD]", ""}
    mask = np.zeros((n_sent, T), dtype=bool)
    for si, toks in enumerate(tokens):
        for ti, tok in enumerate(toks):
            if ti < T and tok not in SPECIAL:
                mask[si, ti] = True

    # Per-token cosine to L0: (N, n_layers, T)
    Hn = H / (np.linalg.norm(H, axis=-1, keepdims=True) + 1e-8)
    cos_to_L0 = np.einsum("slth,sth->slt", Hn, Hn[:, 0])  # (N, L, T)

    # 1) Lexical retention (mean over masked tokens & sentences)
    lex = np.zeros(n_layers)
    for L in range(n_layers):
        vals = cos_to_L0[:, L][mask]
        lex[L] = float(np.mean(vals))

    # 2) Anisotropy: mean pairwise cos between distinct tokens within sentence
    ani = np.zeros(n_layers)
    for L in range(n_layers):
        sims = []
        for si in range(n_sent):
            valid = np.where(mask[si])[0]
            if len(valid) < 2:
                continue
            v = Hn[si, L, valid]                 # (k, 768)
            cos_mat = v @ v.T
            iu = np.triu_indices(len(valid), k=1)
            sims.extend(cos_mat[iu].tolist())
        ani[L] = float(np.mean(sims))

    # 3) Probe F1 per layer (precomputed)
    probe = pd.read_csv(PROBE_CSV)
    f1 = probe["mean_probe_f1"].to_numpy(dtype=np.float32)
    if len(f1) < n_layers:
        f1 = np.concatenate([f1, np.full(n_layers - len(f1), np.nan)])

    return {
        "lex": lex,
        "ani": ani,
        "f1":  f1[:n_layers],
        "n_layers": n_layers,
        "cos_to_L0": cos_to_L0,
        "tokens": tokens,
        "sentences": sentences,
        "labels": labels,
        "mask": mask,
    }


def _pick_examples(data, n: int = 4) -> list[int]:
    """Pick n short clean sentences with diverse emotion labels."""
    picks: list[int] = []
    seen_labels: set[str] = set()
    sents = data["sentences"]
    labs = data["labels"]
    toks = data["tokens"]
    # Score by token count (prefer 3-7 content tokens)
    SPECIAL = {"⟨CLS⟩", "⟨SEP⟩", "[CLS]", "[SEP]", "[PAD]", ""}
    candidates = []
    for i, ss in enumerate(sents):
        n_real = sum(1 for t in toks[i] if t not in SPECIAL)
        if 3 <= n_real <= 8 and "[NAME]" not in ss and "[" not in ss[1:5]:
            candidates.append((n_real, i))
    candidates.sort()
    for _, i in candidates:
        if labs[i] not in seen_labels:
            picks.append(i)
            seen_labels.add(labs[i])
            if len(picks) == n:
                break
    while len(picks) < n and len(picks) < len(sents):
        for i in range(len(sents)):
            if i not in picks:
                picks.append(i)
                if len(picks) == n:
                    break
    return picks


LANG = {
    "es": {
        "subplot_curves":  "tres curvas, una historia · capa por capa",
        "lex":             "retención léxica",
        "ani":             "anisotropía",
        "f1":              "emergencia semántica (probe F1)",
        "lex_hover":       "cos(hidden, embedding)",
        "ani_hover":       "cos entre tokens",
        "f1_hover":        "F1 probe lineal",
        "phase_lex_b":     "fase léxica",
        "phase_lex_s":     "tokens preservan identidad",
        "phase_mix_b":     "mezcla contextual",
        "phase_mix_s":     "identidad léxica se diluye",
        "phase_sem_b":     "fase semántica",
        "phase_sem_s":     "emoción cristalizada, tokens convergen",
        "hinge":           "punto de bisagra",
        "axis_layer":      "capa",
        "axis_value":      "valor",
        "tok_hover":       "token <b>%{x}</b><br>capa <b>%{y}</b><br>cos a L0: %{z:.3f}<extra></extra>",
        "cbar_title":      "cos<br>↔ L0",
    },
    "en": {
        "subplot_curves":  "three curves, one story · layer by layer",
        "lex":             "lexical retention",
        "ani":             "anisotropy",
        "f1":              "semantic emergence (probe F1)",
        "lex_hover":       "cos(hidden, embedding)",
        "ani_hover":       "cos between tokens",
        "f1_hover":        "linear-probe F1",
        "phase_lex_b":     "lexical phase",
        "phase_lex_s":     "tokens keep their identity",
        "phase_mix_b":     "contextual mixing",
        "phase_mix_s":     "lexical identity dissolves",
        "phase_sem_b":     "semantic phase",
        "phase_sem_s":     "emotion crystallises, tokens converge",
        "hinge":           "hinge point",
        "axis_layer":      "layer",
        "axis_value":      "value",
        "tok_hover":       "token <b>%{x}</b><br>layer <b>%{y}</b><br>cos to L0: %{z:.3f}<extra></extra>",
        "cbar_title":      "cos<br>↔ L0",
    },
}


def build_figure(lang: str = "es") -> go.Figure:
    L = LANG[lang]
    data = _compute_curves()
    n_layers = data["n_layers"]
    layer_x = list(range(n_layers))
    layer_labels = ["Emb"] + [f"L{i}" for i in range(n_layers - 1)]

    examples = _pick_examples(data, n=4)

    fig = make_subplots(
        rows=2, cols=4,
        specs=[[{"colspan": 4}, None, None, None],
               [{}, {}, {}, {}]],
        row_heights=[0.58, 0.42],
        subplot_titles=(
            L["subplot_curves"],
            *[f"<b>[{data['labels'][i]}]</b> "
              + (data['sentences'][i][:38] + "…"
                 if len(data['sentences'][i]) > 40
                 else data['sentences'][i])
              for i in examples]
        ),
        horizontal_spacing=0.025,
        vertical_spacing=0.16,
    )

    # ─── Curves ───────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=layer_x, y=data["lex"],
        mode="lines+markers", name=L["lex"],
        line=dict(color=st.BLUE, width=3, shape="spline", smoothing=0.6),
        marker=dict(size=8, color=st.BLUE, line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: %{y:.3f}<extra>" + L["lex_hover"] + "</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=layer_x, y=data["ani"],
        mode="lines+markers", name=L["ani"],
        line=dict(color=st.SAND, width=3, shape="spline", smoothing=0.6),
        marker=dict(size=8, color=st.SAND, line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: %{y:.3f}<extra>" + L["ani_hover"] + "</extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=layer_x, y=data["f1"],
        mode="lines+markers", name=L["f1"],
        line=dict(color=st.TERRA, width=3, shape="spline", smoothing=0.6),
        marker=dict(size=8, color=st.TERRA,
                    line=dict(color="white", width=1.5)),
        hovertemplate="%{x}: %{y:.3f}<extra>" + L["f1_hover"] + "</extra>",
    ), row=1, col=1)

    # Phase shading
    shapes = [
        dict(type="rect", xref="x", yref="paper",
             x0=-0.5, x1=4.5, y0=0, y1=1,
             fillcolor="rgba(58,110,165,0.07)", line=dict(width=0),
             layer="below"),
        dict(type="rect", xref="x", yref="paper",
             x0=4.5, x1=8.5, y0=0, y1=1,
             fillcolor="rgba(212,168,67,0.06)", line=dict(width=0),
             layer="below"),
        dict(type="rect", xref="x", yref="paper",
             x0=8.5, x1=12.5, y0=0, y1=1,
             fillcolor="rgba(193,85,58,0.08)", line=dict(width=0),
             layer="below"),
        # Hinge line at L8.5
        dict(type="line", xref="x", yref="paper",
             x0=8.5, x1=8.5, y0=0, y1=1,
             line=dict(color=st.INK_3, width=1, dash="dot")),
    ]

    annotations = [
        dict(x=2, y=0.97, xref="x", yref="paper", showarrow=False,
             text=f"<b>{L['phase_lex_b']}</b><br><span style='font-size:10px'>"
                  f"{L['phase_lex_s']}</span>",
             font=dict(size=11, color=st.BLUE), align="center"),
        dict(x=6.5, y=0.97, xref="x", yref="paper", showarrow=False,
             text=f"<b>{L['phase_mix_b']}</b><br><span style='font-size:10px'>"
                  f"{L['phase_mix_s']}</span>",
             font=dict(size=11, color=st.SAND), align="center"),
        dict(x=10.5, y=0.97, xref="x", yref="paper", showarrow=False,
             text=f"<b>{L['phase_sem_b']}</b><br><span style='font-size:10px'>"
                  f"{L['phase_sem_s']}</span>",
             font=dict(size=11, color=st.TERRA), align="center"),
        dict(x=8.5, y=0.04, xref="x", yref="paper", showarrow=True,
             ax=0, ay=-22, arrowhead=0, arrowwidth=1, arrowcolor=st.INK_3,
             text=L["hinge"], font=dict(size=10, color=st.INK_3)),
    ]

    # ─── Per-sentence heatmaps (layer × token, cos to L0) ────────────────
    SPECIAL_HEATMAP = {"⟨SEP⟩", "[SEP]", "[PAD]", ""}
    for col_i, sent_idx in enumerate(examples, start=1):
        toks = data["tokens"][sent_idx]
        # Keep CLS + content tokens (drop SEP and PAD), in order
        valid_pos = [t for t, tk in enumerate(toks)
                     if tk not in SPECIAL_HEATMAP][:9]
        token_labels = [toks[t] if t < len(toks) else "" for t in valid_pos]
        token_labels = [tl.replace("##", "·").replace("⟨CLS⟩", "CLS")
                        for tl in token_labels]

        z = data["cos_to_L0"][sent_idx][:, valid_pos]  # (n_layers, T_valid)

        fig.add_trace(go.Heatmap(
            z=z,
            x=token_labels,
            y=layer_labels,
            colorscale=[
                [0.0, "#FFFFFF"],
                [0.30, st.SAND_L],
                [0.65, st.SAND],
                [1.0, st.BLUE],
            ],
            zmin=0, zmax=1,
            showscale=(col_i == 4),
            colorbar=dict(
                thickness=10, len=0.34, x=1.01, y=0.21,
                title=dict(text=L["cbar_title"],
                           font=dict(size=10, color=st.INK_3)),
                tickfont=dict(size=9, color=st.INK_3),
            ) if col_i == 4 else None,
            hovertemplate=L["tok_hover"],
            xgap=1, ygap=1,
        ), row=2, col=col_i)

    # ─── Layout ───────────────────────────────────────────────────────────
    fig.update_layout(
        **st.thesis_layout(
            title=("De diccionario a clasificador · cómo BERT olvida "
                   "palabras y aprende emociones"
                   "<br><sub>Replicación en el modelo fine-tuneado de "
                   "Tenney et al. (ACL 2019) y Ethayarajh (EMNLP 2019). "
                   "El cruce entre fase léxica y semántica ocurre en "
                   "<b>L8-L9</b>.</sub>"),
            height=900, width=1400,
        ),
        legend=dict(
            orientation="h",
            x=0.5, y=0.46, xanchor="center", yanchor="top",
            bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
            borderwidth=0.5, font=dict(size=11),
        ),
        shapes=shapes,
        annotations=list(fig.layout.annotations) + annotations,
    )

    fig.update_xaxes(
        title=dict(text=L["axis_layer"], font=dict(size=12, color=st.INK_2)),
        tickmode="array", tickvals=layer_x, ticktext=layer_labels,
        gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.INK_3),
        row=1, col=1,
    )
    fig.update_yaxes(
        title=dict(text=L["axis_value"], font=dict(size=12, color=st.INK_2)),
        range=[-0.05, 1.05],
        gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.INK_3),
        row=1, col=1,
    )

    # Style heatmap subplot axes
    for col_i in range(1, 5):
        fig.update_xaxes(
            tickfont=dict(size=9, color=st.INK_2, family="monospace"),
            tickangle=-45, side="bottom",
            showgrid=False, linecolor=st.SPINE, showline=True,
            row=2, col=col_i,
        )
        fig.update_yaxes(
            tickfont=dict(size=8, color=st.INK_3),
            autorange="reversed",
            showgrid=False, linecolor=st.SPINE, showline=True,
            row=2, col=col_i,
        )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_figure()
    out = out_dir / "28_lexical_to_semantic.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
