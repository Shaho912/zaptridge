#!/usr/bin/env python3
"""CAS semantic diversity — single vertical grouped bar chart.

Drop metric: (Exp1 oracle − joint) / total × 100 for both bars,
so both Exp1 and Exp2 are measured against the same baseline.

Usage:
    python scripts/plot_cas_semantics.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C1    = "#2a78d6"
C2    = "#eb6834"
GRAY  = "#6b7280"
LGRAY = "#e5e7eb"

# ── data ─────────────────────────────────────────────────────────────────────
# (label, exp1_oracle, exp1_joint, exp2_joint or None, total, anomaly)
# drop = (exp1_oracle - joint) / total * 100  (same baseline for both bars)
CONDITIONS = [
    ("4 ML papers\n(same domain)",         42, 36, 35, 45, False),
    ("4-domain\n(ML/FPGA/Bio/Astro)",      20, 17, 18, 21, False),
    ("Patient records\n4 patients (LH)",    8,  7, 10, 16, False),
    ("Patient records\n5 patients † (LH)",  6,  9, 10, 20, True),
    ("Distinct\nquery types (Setup 2)",    18, 18, None, 18, False),
]

def drop_pct(oracle, joint, total):
    return (oracle - joint) / total * 100

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

n = len(CONDITIONS)
x = np.arange(n)
bw  = 0.30
gap = 0.06

ax.axhspan(0, 22, color=C2, alpha=0.05, zorder=0)
ax.axhspan(-22, 0, color=C1, alpha=0.05, zorder=0)

for y in [-20, -15, -10, -5, 5, 10, 15, 20]:
    ax.axhline(y, color=LGRAY, linewidth=0.7, zorder=1)
ax.axhline(0, color="#9ca3af", linewidth=1.4, zorder=2)

ax.text(-0.4, 21,  "Interference",      ha="left", va="bottom", fontsize=8.5, color=C2, alpha=0.8)
ax.text(-0.4, -21, "Positive transfer", ha="left", va="top",    fontsize=8.5, color=C1, alpha=0.8)

for i, (lbl, e1_ora, e1_jnt, e2_jnt, total, anom) in enumerate(CONDITIONS):
    a1 = 0.4 if anom else 1.0
    d1 = drop_pct(e1_ora, e1_jnt, total)

    if e2_jnt is not None:
        d2 = drop_pct(e1_ora, e2_jnt, total)
        ax.bar(x[i] - bw/2 - gap/2, d1, width=bw, color=C1, alpha=a1, zorder=3)
        ax.bar(x[i] + bw/2 + gap/2, d2, width=bw, color=C2, zorder=3)
        for val, xoff, alpha in [(d1, -bw/2 - gap/2, a1), (d2, bw/2 + gap/2, 1.0)]:
            sign = "+" if val >= 0 else ""
            yoff = 0.7 if val >= 0 else -0.7
            va   = "bottom" if val >= 0 else "top"
            ax.text(x[i] + xoff, val + yoff, f"{sign}{val:.1f}%",
                    ha="center", va=va, fontsize=7.5, color=GRAY, alpha=alpha)
    else:
        ax.bar(x[i], d1, width=bw, color=C1, alpha=a1, zorder=3)
        ax.text(x[i], 1.2, "0%", ha="center", va="bottom", fontsize=7.5, color=GRAY)
        ax.text(x[i], -1.5, "no Exp 2\n(zero interference\nwithout CAS)",
                ha="center", va="top", fontsize=7, color=GRAY, style="italic", linespacing=1.4)

# separator between research papers and patient records
ax.axvline(1.5, color=LGRAY, linewidth=0.8, linestyle="--")

ax.set_xticks(x)
ax.set_xticklabels([c[0] for c in CONDITIONS], fontsize=9, linespacing=1.4)
ax.set_ylabel("% drop  =  (Exp1 oracle − joint) / n × 100", fontsize=9.5, color=GRAY)
ax.set_ylim(-22, 24)
ax.yaxis.set_tick_params(labelsize=9, colors=GRAY)
ax.tick_params(bottom=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(LGRAY)
ax.spines["bottom"].set_color(LGRAY)

p1 = mpatches.Patch(color=C1, label="Exp 1 — independent training")
p2 = mpatches.Patch(color=C2, label="Exp 2 — CAS (p=0.75 distractor)")
ax.legend(handles=[p1, p2], fontsize=9, loc="upper right",
          framealpha=0.9, edgecolor=LGRAY)

ax.set_title(
    "CAS Effect on Joint Accuracy — by Corpus Type  (Qwen3-8B student)",
    fontsize=11, fontweight="bold", pad=12, loc="left",
)

fig.text(0.01, -0.02,
         "† 5-patient Exp 1 bar dimmed: lh_p05 anomaly inflates joint accuracy.  "
         "4-domain Exp 2 shows Run A only.  4 ML papers Exp 2: 100% distractor.",
         fontsize=7.5, color=GRAY)

plt.tight_layout()
out = os.path.join(OUT_DIR, "cas_drop_by_corpus.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
