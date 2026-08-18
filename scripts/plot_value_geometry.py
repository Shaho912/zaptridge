#!/usr/bin/env python3
"""Compare two value cosim curves per layer:
  1. Cross-patient  (p01 vs p02, same training — different corpora)
  2. Cross-condition (p01 independent vs p01 CAS — same corpus, different training)

Both show early-layer divergence converging by layer ~20, suggesting CAS
redirects early-layer values into a similar geometric space as natural
patient-to-patient variation.

Usage:
    python scripts/plot_value_geometry.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C_CROSS_PATIENT   = "#eb6834"   # orange
C_CROSS_CONDITION = "#2a78d6"   # blue
GRAY  = "#6b7280"
LGRAY = "#e5e7eb"

# ── cross-condition: p01 independent vs p01 CAS (all 36 layers) ──────────────
layers_cc = list(range(36))
val_cc = [
    0.416111, 0.685036, 0.780358, 0.868371, 0.903447, 0.930741, 0.953252, 0.976271,
    0.985522, 0.987006, 0.994450, 0.987991, 0.991631, 0.990908, 0.994927, 0.995506,
    0.997366, 0.996981, 0.998057, 0.998813, 0.998817, 0.999137, 0.999429, 0.999527,
    0.999774, 0.999727, 0.999826, 0.999869, 0.999888, 0.999930, 0.999922, 0.999937,
    0.999948, 0.999938, 0.999939, 0.999940,
]

# ── cross-patient: p01 vs p02 (same training — sparse layer samples) ─────────
layers_cp = [0, 1, 2, 5, 10, 22, 27, 33]
val_cp    = [0.286, 0.494, 0.597, 0.804, 0.983, 0.997, 0.9994, 0.9999]

# clip both to layers 0–20
MAX_LAYER = 20

layers_cc_clip = [l for l in layers_cc if l <= MAX_LAYER]
val_cc_clip    = [val_cc[l] for l in layers_cc_clip]

layers_cp_clip = [l for l in layers_cp if l <= MAX_LAYER]
val_cp_clip    = [val_cp[layers_cp.index(l)] for l in layers_cp_clip]

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for y in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ax.axhline(y, color=LGRAY, linewidth=0.7, zorder=1)

# cross-condition — solid line
ax.plot(layers_cc_clip, val_cc_clip,
        color=C_CROSS_CONDITION, linewidth=2, linestyle="-", zorder=3)

# cross-patient — dashed line with markers (sparse)
ax.plot(layers_cp_clip, val_cp_clip,
        color=C_CROSS_PATIENT, linewidth=2, linestyle="--",
        marker="o", markersize=7, zorder=4)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.5, MAX_LAYER + 0.5)
ax.set_ylim(0.22, 1.04)
ax.set_xticks([0, 5, 10, 15, 20])
ax.set_xticklabels(["0", "5", "10", "15", "20+"])
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Layer", fontsize=10, color=GRAY)
ax.set_ylabel("Value cosine similarity", fontsize=10, color=GRAY)
ax.tick_params(colors=GRAY, labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(LGRAY)
ax.spines["bottom"].set_color(LGRAY)

# ── legend ────────────────────────────────────────────────────────────────────
l1 = mlines.Line2D([], [], color=C_CROSS_CONDITION, linewidth=2, linestyle="-",
                   label="Same patient, Exp 1 vs Exp 2  (independent vs CAS)")
l2 = mlines.Line2D([], [], color=C_CROSS_PATIENT, linewidth=2, linestyle="--",
                   marker="o", markersize=6,
                   label="Different patients, same training  (p01 vs p02)")
ax.legend(handles=[l1, l2], fontsize=9, loc="lower right",
          framealpha=0.9, edgecolor=LGRAY)

ax.set_title(
    "Value Geometry Per Layer: Cross-Patient vs Cross-Condition\n"
    "Qwen3-8B · 1024 slots · LongHealth",
    fontsize=10.5, fontweight="bold", pad=10, loc="left",
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "value_geometry_cosim.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
