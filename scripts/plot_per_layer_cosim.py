#!/usr/bin/env python3
"""Per-layer cosine similarity — 4 lines:
  - Cross-patient keys   (p01 vs p02, same training)
  - Cross-condition keys (p01 independent vs p01 CAS)
  - Cross-patient values (p01 vs p02, same training)
  - Cross-condition values (p01 independent vs p01 CAS)

Usage:
    python scripts/plot_per_layer_cosim.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C_KEY = "#2a78d6"
C_VAL = "#eb6834"
GRAY  = "#6b7280"
LGRAY = "#e5e7eb"

MAX_LAYER = 10

# ── cross-condition: p01 independent vs p01 CAS (all 36 layers) ──────────────
layers_cc = list(range(36))
key_cc = [
    1.000269, 1.000004, 0.999846, 0.999858, 1.000033, 1.000032, 0.999907, 0.999858,
    1.000031, 0.999836, 1.000062, 0.999969, 0.999916, 0.999846, 0.999920, 0.999864,
    0.999849, 0.999865, 0.999882, 0.999882, 0.999901, 0.999907, 0.999903, 0.999938,
    0.999912, 0.999917, 0.999891, 0.999884, 0.999893, 0.999909, 0.999928, 0.999900,
    0.999869, 0.999865, 0.999897, 0.999996,
]
val_cc = [
    0.416111, 0.685036, 0.780358, 0.868371, 0.903447, 0.930741, 0.953252, 0.976271,
    0.985522, 0.987006, 0.994450, 0.987991, 0.991631, 0.990908, 0.994927, 0.995506,
    0.997366, 0.996981, 0.998057, 0.998813, 0.998817, 0.999137, 0.999429, 0.999527,
    0.999774, 0.999727, 0.999826, 0.999869, 0.999888, 0.999930, 0.999922, 0.999937,
    0.999948, 0.999938, 0.999939, 0.999940,
]

# ── cross-patient: p01 vs p02, independent training (all 36 layers) ──────────
layers_cp_all = list(range(36))
key_cp_all = [
    1.000263, 0.999946, 0.999618, 0.999627, 0.999918, 0.999999, 0.999710, 0.999593,
    0.999923, 0.999545, 0.999965, 0.999846, 0.999713, 0.999541, 0.999737, 0.999584,
    0.999527, 0.999526, 0.999555, 0.999586, 0.999529, 0.999567, 0.999548, 0.999699,
    0.999483, 0.999641, 0.999375, 0.999355, 0.999475, 0.999319, 0.999604, 0.999406,
    0.999404, 0.999185, 0.999487, 0.999778,
]
val_cp_all = [
    0.285636, 0.493989, 0.596713, 0.701744, 0.759315, 0.803595, 0.858473, 0.928919,
    0.954332, 0.959583, 0.982608, 0.957374, 0.967774, 0.966561, 0.981438, 0.982219,
    0.989396, 0.987781, 0.991148, 0.994555, 0.994037, 0.995643, 0.997151, 0.997592,
    0.998860, 0.998597, 0.999137, 0.999436, 0.999560, 0.999825, 0.999848, 0.999883,
    0.999914, 0.999931, 0.999930, 0.999894,
]

# clip to MAX_LAYER
cc_idx = [i for i, l in enumerate(layers_cc) if l <= MAX_LAYER]
layers_cc_c = [layers_cc[i] for i in cc_idx]
key_cc_c    = [key_cc[i]    for i in cc_idx]
val_cc_c    = [val_cc[i]    for i in cc_idx]

cp_idx = [i for i, l in enumerate(layers_cp_all) if l <= MAX_LAYER]
layers_cp_c = [layers_cp_all[i] for i in cp_idx]
key_cp_c    = [key_cp_all[i]    for i in cp_idx]
val_cp_c    = [val_cp_all[i]    for i in cp_idx]

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for y in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ax.axhline(y, color=LGRAY, linewidth=0.7, zorder=1)

# cross-condition — solid lines (dense, no markers)
ax.plot(layers_cc_c, key_cc_c, color=C_KEY, linewidth=2, linestyle="-",  zorder=3)
ax.plot(layers_cc_c, val_cc_c, color=C_VAL, linewidth=2, linestyle="-",  zorder=3)

# cross-patient — dashed lines with markers (sparse)
ax.plot(layers_cp_c, key_cp_c, color=C_KEY, linewidth=2, linestyle="--",
        marker="o", markersize=6, zorder=4)
ax.plot(layers_cp_c, val_cp_c, color=C_VAL, linewidth=2, linestyle="--",
        marker="o", markersize=6, zorder=4)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.5, MAX_LAYER + 0.5)
ax.set_ylim(0.22, 1.04)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(["0", "2", "4", "6", "8", "10+"])
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.set_xlabel("Layer", fontsize=10, color=GRAY)
ax.set_ylabel("Cosine similarity", fontsize=10, color=GRAY)
ax.tick_params(colors=GRAY, labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(LGRAY)
ax.spines["bottom"].set_color(LGRAY)

# ── legend ────────────────────────────────────────────────────────────────────
l1 = mlines.Line2D([], [], color=C_KEY, linewidth=2, linestyle="-",
                   label="Keys — same patient, Ind vs CAS")
l2 = mlines.Line2D([], [], color=C_KEY, linewidth=2, linestyle="--",
                   marker="o", markersize=5, label="Keys — p01 vs p02 (cross-patient)")
l3 = mlines.Line2D([], [], color=C_VAL, linewidth=2, linestyle="-",
                   label="Values — same patient, Ind vs CAS")
l4 = mlines.Line2D([], [], color=C_VAL, linewidth=2, linestyle="--",
                   marker="o", markersize=5, label="Values — p01 vs p02 (cross-patient)")
ax.legend(handles=[l1, l2, l3, l4], fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor=LGRAY)

ax.set_title(
    "Per-Layer Cosine Similarity: Cross-Patient vs Cross-Condition\n"
    "Qwen3-8B · 1024 slots · LongHealth",
    fontsize=10.5, fontweight="bold", pad=10, loc="left",
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "per_layer_cosim.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
