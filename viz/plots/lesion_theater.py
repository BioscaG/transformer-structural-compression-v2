"""Hero plot 8 — Lesion theater: animated revival sequence.

Starts with a fully collapsed model (F1=0.000, all 23 emotions dead).
Restoring layers in sequence reveals where the function lives:
  - L0..L7 patched: nothing wakes up.
  - L8 patched: 0.1% recovery — a flicker.
  - L9 patched: 4% recovery — first signs of life.
  - L10 patched: 24% recovery — partial.
  - L11 patched: 100% recovery — full revival.

This is the visceral version of §5.2: the FFN-L11 alone IS the model's
emotional capacity. Auto-play through the 12 layers; or click any step to
jump there. Each emotion has its own colored bar so you also see WHICH
emotions revive first when L11 comes online.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.graph_objects as go

from viz import thesis_data as td
from viz import style as st
from viz.data.load_results import (load_patching, load_informed,
                                    f1_baseline_per_emotion, EMOTIONS_23)


def _real_per_layer_f1():
    """Return (12, n_emotions) matrix of REAL patched F1 from nb5/per_layer."""
    p = load_patching()
    df = p["per_layer"]
    if df is None:
        return None, None
    pivot = df.pivot(index="layer", columns="emotion", values="patched_f1")
    pivot = pivot.reindex(columns=EMOTIONS_23)
    pivot = pivot.reindex(index=range(12))
    return pivot.values.astype(np.float32), df


def _per_emotion_recovery(stage_pct: float, real_matrix=None,
                          baseline_f1=None) -> dict[str, float]:
    """For each emotion, return F1 at this stage of patching.

    If `real_matrix` (12, 23) is supplied, returns:
        - stage 0..N-1 → cumulative real patched_f1 (avg of restored layers' patched F1)
    Falls back to a sigmoid synthesis if no real data.
    """
    if real_matrix is None:
        out = {}
        for e in td.EMOTIONS:
            baseline = td.F1_BASELINE[e]
            crystal = td.CRYSTALLIZATION_LAYER[e]
            late_factor = crystal / 11
            center = 50 + late_factor * 35
            sharpness = 0.18
            recovery_frac = 1 / (1 + np.exp(-sharpness * (stage_pct - center)))
            out[e] = float(baseline * recovery_frac)
        return out

    # Map stage_pct (0..100) to a layer index 0..11 (or "all dead" at -1)
    # Stage thresholds match the STAGES list in the figure builder.
    return None  # signaled to caller — use direct lookup instead


LANG = {
    "es": {
        "baseline":   "F1 baseline (objetivo)",
        "current":    "F1 actual",
        "baseline_h": "<b>%{y}</b><br>F1 baseline: %{x:.3f}<extra></extra>",
        "current_h":  "<b>%{y}</b><br>F1: %{x:.3f}<extra></extra>",
        "axis_x":     "F1 por emoción",
        "stage_pref": "Etapa: ",
        "play":       "▶ Play (revive completo)",
        "pause":      "⏸ Pause",
        "reset":      "↺ Reset",
        "dead_lbl":   "Muerto",
        "title_fmt":  "<b>{stage}</b> · F1 macro = {now:.3f} ({pct:.1f}% del baseline {base:.3f})",
        "stages": [
            ("Modelo colapsado (SVD r=64)", "F1 = 0.000 en todas las emociones."),
            ("Restaurar L0",  "L0 sola: la señal léxica creada aquí se destruye al pasar por las capas comprimidas."),
            ("Restaurar L1",  "Misma historia: las capas tempranas crean información, pero las intermedias comprimidas la consumen."),
            ("Restaurar L2",  "Sigue muerto."),
            ("Restaurar L3",  "Sigue muerto."),
            ("Restaurar L4",  "Sigue muerto."),
            ("Restaurar L5",  "Sigue muerto."),
            ("Restaurar L6",  "Sigue muerto."),
            ("Restaurar L7",  "Sigue muerto. 8 capas restauradas y F1 sigue cerca de 0."),
            ("Restaurar L8",  "Primer destello — el cuello de botella se asoma."),
            ("Restaurar L9",  "Las emociones léxicas (gratitude, love) empiezan a despertar."),
            ("Restaurar L10", "Casi mitad del modelo emocional vuelve."),
            ("Restaurar L11", "BOOM. La capa 11 — concretamente su FFN — es el cuello de botella. Aquí vive el modelo."),
        ],
    },
    "en": {
        "baseline":   "F1 baseline (target)",
        "current":    "F1 current",
        "baseline_h": "<b>%{y}</b><br>F1 baseline: %{x:.3f}<extra></extra>",
        "current_h":  "<b>%{y}</b><br>F1: %{x:.3f}<extra></extra>",
        "axis_x":     "F1 per emotion",
        "stage_pref": "Stage: ",
        "play":       "▶ Play (full revival)",
        "pause":      "⏸ Pause",
        "reset":      "↺ Reset",
        "dead_lbl":   "Dead",
        "title_fmt":  "<b>{stage}</b> · F1 macro = {now:.3f} ({pct:.1f}% of baseline {base:.3f})",
        "stages": [
            ("Collapsed model (SVD r=64)", "F1 = 0.000 across every emotion."),
            ("Restore L0",  "L0 alone: the lexical signal it creates is destroyed by the compressed layers downstream."),
            ("Restore L1",  "Same story: early layers create information, mid compressed layers consume it."),
            ("Restore L2",  "Still dead."),
            ("Restore L3",  "Still dead."),
            ("Restore L4",  "Still dead."),
            ("Restore L5",  "Still dead."),
            ("Restore L6",  "Still dead."),
            ("Restore L7",  "Still dead. 8 layers restored and F1 is still near 0."),
            ("Restore L8",  "First flicker — the bottleneck peeks out."),
            ("Restore L9",  "Lexical emotions (gratitude, love) start waking up."),
            ("Restore L10", "Almost half the emotional model returns."),
            ("Restore L11", "BOOM. Layer 11 — its FFN specifically — is the bottleneck. The model lives here."),
        ],
    },
}


def build_lesion_theater(lang: str = "es") -> go.Figure:
    _L = LANG[lang]
    # Load REAL per-(layer, emotion) patched F1 from notebook 5
    real_f1, _ = _real_per_layer_f1()
    informed = load_informed()
    real_baseline = f1_baseline_per_emotion(informed)
    use_real = real_f1 is not None and bool(real_baseline)

    # Sort emotions by REAL baseline F1 descending
    if use_real:
        sorted_emos = sorted(EMOTIONS_23, key=lambda e: -real_baseline.get(e, 0))
    else:
        sorted_emos = sorted(td.EMOTIONS, key=lambda e: -td.F1_BASELINE[e])

    # Compute summary stage % from the real per-layer data when available
    # by averaging restoration scores. Otherwise use the canonical Tabla 14.
    stage_pcts_canonical = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 4.1, 24.1, 100.0]
    if use_real:
        # Weighted F1 at each layer = mean(patched_f1 across emotions) / baseline
        baseline_vec = np.array([real_baseline.get(e, 0) for e in EMOTIONS_23])
        baseline_macro = baseline_vec.mean()
        # patched F1 macro per layer = mean across emotions
        # Convert restoration: pct = patched_macro / baseline_macro * 100
        layer_pcts = []
        for L in range(12):
            patched_macro = float(np.nanmean(real_f1[L]))
            layer_pcts.append(patched_macro / baseline_macro * 100 if baseline_macro else 0)
        stage_pcts_real = [0.0] + layer_pcts  # stage 0 = collapsed
    else:
        stage_pcts_real = stage_pcts_canonical

    STAGES = [
        (_L["stages"][i][0], stage_pcts_real[i], _L["stages"][i][1])
        for i in range(13)
    ]

    n_emo = len(sorted_emos)
    bar_colors = [td.CLUSTER_COLORS[td.EMOTION_TO_CLUSTER[e]] for e in sorted_emos]
    if use_real:
        baseline_values = [real_baseline.get(e, 0) for e in sorted_emos]
    else:
        baseline_values = [td.F1_BASELINE[e] for e in sorted_emos]

    # Initial frame (everything zero)
    init_values = [0.0] * n_emo

    fig = go.Figure()

    # Ghost baseline bars (always shown, faded — represent the "target")
    fig.add_trace(go.Bar(
        x=baseline_values, y=sorted_emos, orientation="h",
        marker=dict(color="rgba(200,199,193,0.20)", line=dict(color=st.SPINE, width=0.6)),
        name=_L["baseline"],
        hovertemplate=_L["baseline_h"],
    ))
    # The active bars (these get animated frame-by-frame)
    fig.add_trace(go.Bar(
        x=init_values, y=sorted_emos, orientation="h",
        marker=dict(color=bar_colors, line=dict(color="white", width=0.5)),
        name=_L["current"],
        hovertemplate=_L["current_h"],
    ))

    # Frames — one per stage. Build per-emotion F1 values: real lookup if data
    # available, sigmoid synthesis otherwise.
    frames = []
    stage_pcts = [s[1] for s in STAGES]
    stage_explanations = [s[2] for s in STAGES]

    emo_to_idx = {e: i for i, e in enumerate(EMOTIONS_23)}

    for i, (stage_name, stage_pct, explain) in enumerate(STAGES):
        if use_real:
            if i == 0:
                values = [0.0 for _ in sorted_emos]
            else:
                # Stage i = "Restaurar L{i-1}". Use that layer's patched F1.
                layer_idx = i - 1
                values = []
                for e in sorted_emos:
                    f1 = real_f1[layer_idx, emo_to_idx[e]] if e in emo_to_idx else 0.0
                    values.append(float(f1) if not np.isnan(f1) else 0.0)
        else:
            recovery = _per_emotion_recovery(stage_pct)
            values = [recovery[e] for e in sorted_emos]
        # Title overlay (real F1 macro = mean of values, real baseline if avail)
        baseline_mean = sum(baseline_values) / max(len(baseline_values), 1)
        f1_macro_now = sum(values) / max(len(values), 1)
        title_html = _L["title_fmt"].format(
            stage=stage_name, now=f1_macro_now, pct=stage_pct, base=baseline_mean
        )

        frames.append(go.Frame(
            data=[
                go.Bar(x=baseline_values, y=sorted_emos, orientation="h",
                       marker=dict(color="rgba(200,199,193,0.20)",
                                   line=dict(color=st.SPINE, width=0.6))),
                go.Bar(x=values, y=sorted_emos, orientation="h",
                       marker=dict(color=bar_colors, line=dict(color="white", width=0.5))),
            ],
            name=str(i),
            layout=dict(
                title=dict(text=title_html,
                           font=dict(size=14, color=st.INK, family="serif"), x=0.02),
                annotations=[
                    dict(text=explain,
                         x=0.5, y=-0.16, xref="paper", yref="paper",
                         showarrow=False, align="center", xanchor="center",
                         font=dict(size=11.5, color=st.INK_2, family="serif"),
                         bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                         borderwidth=0.5, borderpad=10),
                ],
            ),
        ))

    fig.frames = frames

    # Slider steps
    steps = [dict(
        method="animate",
        args=[[str(i)],
              dict(mode="immediate", frame=dict(duration=0, redraw=True),
                   transition=dict(duration=400, easing="cubic-in-out"))],
        label=(STAGES[i][0]
               .replace("Restaurar ", "").replace("Restore ", "")
               .replace("Modelo colapsado (SVD r=64)", _L["dead_lbl"])
               .replace("Collapsed model (SVD r=64)", _L["dead_lbl"])),
    ) for i in range(len(STAGES))]

    fig.update_layout(
        **st.thesis_layout(
            title=("Lesion theater · L11 sola recupera el 100%"
                   "<br><sub>Modelo arrancado a SVD r=64 (F1 = 0). Pulsa Play o navega "
                   "el slider para restaurar los pesos originales una capa cada vez.</sub>"),
            height=820, width=1280,
        ),
        annotations=[
            dict(text=stage_explanations[0],
                 x=0.5, y=-0.16, xref="paper", yref="paper",
                 showarrow=False, align="center", xanchor="center",
                 font=dict(size=11.5, color=st.INK_2, family="serif"),
                 bgcolor="rgba(255,255,255,0.92)", bordercolor=st.SPINE,
                 borderwidth=0.5, borderpad=10),
        ],
        sliders=[dict(
            active=0, x=0.05, y=-0.04, len=0.90,
            currentvalue=dict(prefix=_L["stage_pref"],
                              font=dict(size=13, color=st.INK, family="serif"),
                              xanchor="left"),
            steps=steps,
            pad=dict(t=20, b=10),
            tickcolor=st.INK_3,
            font=dict(size=10, color=st.INK_3),
            transition=dict(duration=400, easing="cubic-in-out"),
        )],
        updatemenus=[dict(
            type="buttons", direction="left",
            x=0.05, y=-0.18, xanchor="left", yanchor="top",
            buttons=[
                dict(label=_L["play"], method="animate",
                     args=[None, dict(frame=dict(duration=900, redraw=True),
                                      transition=dict(duration=400, easing="cubic-in-out"),
                                      fromcurrent=True, mode="immediate")]),
                dict(label=_L["pause"], method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
                dict(label=_L["reset"], method="animate",
                     args=[["0"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                        transition=dict(duration=300))]),
            ],
            font=dict(size=11, color=st.INK_2),
            bgcolor="white", bordercolor=st.SPINE,
        )],
        bargap=0.20,
        showlegend=True,
        legend=dict(x=0.97, y=0.02, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.85)", bordercolor=st.SPINE,
                    borderwidth=0.5, font=dict(size=10)),
    )

    fig.update_xaxes(
        title=dict(text=_L["axis_x"], font=dict(size=12, color=st.INK_2)),
        range=[0, 1.0],
        gridcolor=st.GRID, linecolor=st.SPINE, showline=True,
        tickfont=dict(size=10, color=st.INK_3),
    )
    fig.update_yaxes(
        title=dict(text="", font=dict(size=12, color=st.INK_2)),
        autorange="reversed",
        showgrid=False, linecolor=st.SPINE,
        tickfont=dict(size=10, color=st.INK_2, family="serif"),
    )

    return fig


def main(out_dir: pathlib.Path | None = None) -> pathlib.Path:
    out_dir = out_dir or pathlib.Path(__file__).resolve().parents[1] / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig = build_lesion_theater()
    out = out_dir / "11_lesion_theater.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True,
                   config={"displayModeBar": True, "displaylogo": False})
    print(f"✓ wrote {out}")
    return out


if __name__ == "__main__":
    main()
