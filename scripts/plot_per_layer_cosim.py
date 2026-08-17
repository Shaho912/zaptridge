#!/usr/bin/env python3
"""Per-layer cosine similarity: same patient (p01), independent vs CAS training.

Shows that keys are corpus-agnostic (flat at ~1.000 across all layers) while
values reorganize significantly in early layers (0-10) under CAS training.

Data from: compare_cartridge_keys.py --per-layer
  cart-a: outputs/exp_rotation_1/lh_p01_cartridge.pt  (independent)
  cart-b: outputs/exp_rotation_2/lh_p01_cartridge.pt  (CAS p=0.75)

Usage:
    python scripts/plot_per_layer_cosim.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ── per-layer data (exp_rotation_1 p01 vs exp_rotation_2 p01) ────────────────
layers = list(range(36))

key_cosim = [
    1.000269, 1.000004, 0.999846, 0.999858, 1.000033, 1.000032, 0.999907, 0.999858,
    1.000031, 0.999836, 1.000062, 0.999969, 0.999916, 0.999846, 0.999920, 0.999864,
    0.999849, 0.999865, 0.999882, 0.999882, 0.999901, 0.999907, 0.999903, 0.999938,
    0.999912, 0.999917, 0.999891, 0.999884, 0.999893, 0.999909, 0.999928, 0.999900,
    0.999869, 0.999865, 0.999897, 0.999996,
]

val_cosim = [
    0.416111, 0.685036, 0.780358, 0.868371, 0.903447, 0.930741, 0.953252, 0.976271,
    0.985522, 0.987006, 0.994450, 0.987991, 0.991631, 0.990908, 0.994927, 0.995506,
    0.997366, 0.996981, 0.998057, 0.998813, 0.998817, 0.999137, 0.999429, 0.999527,
    0.999774, 0.999727, 0.999826, 0.999869, 0.999888, 0.999930, 0.999922, 0.999937,
    0.999948, 0.999938, 0.999939, 0.999940,
]

C_KEYS = "#2a78d6"
C_VALS = "#eb6834"
GRAY   = "#6b7280"
LGRAY  = "#e5e7eb"

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# ── early-layer shaded zone ───────────────────────────────────────────────────
ax.axvspan(-0.5, 10.5, color=C_VALS, alpha=0.06, zorder=0)
ax.text(5, 0.375, "early-layer\nreorganization\n(layers 0–10)",
        ha="center", va="bottom", fontsize=8, color=C_VALS, alpha=0.75,
        linespacing=1.5)

# ── convergence zone ──────────────────────────────────────────────────────────
ax.axvspan(26.5, 35.5, color=LGRAY, alpha=0.45, zorder=0)
ax.text(31, 0.96, "converged\n(layers 27+)",
        ha="center", va="top", fontsize=8, color=GRAY, alpha=0.85,
        linespacing=1.5)

# ── gridlines ─────────────────────────────────────────────────────────────────
for y in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ax.axhline(y, color=LGRAY, linewidth=0.7, zorder=1)

# ── key cosim line ────────────────────────────────────────────────────────────
ax.plot(layers, key_cosim, color=C_KEYS, linewidth=2, label="Keys", zorder=3)
# annotate key line
ax.text(35.3, key_cosim[-1], "Keys  ~1.000", color=C_KEYS,
        va="center", fontsize=8.5, fontweight="500")

# ── value cosim line ──────────────────────────────────────────────────────────
ax.plot(layers, val_cosim, color=C_VALS, linewidth=2, label="Values", zorder=3)
# dots at layer 0 and convergence
ax.scatter([0], [val_cosim[0]], color=C_VALS, s=48, zorder=4)
ax.scatter([27], [val_cosim[27]], color=C_VALS, s=28, zorder=4, alpha=0.6)

# ── layer 0 annotation ───────────────────────────────────────────────────────
ax.annotate(
    f"Layer 0: {val_cosim[0]:.3f}",
    xy=(0, val_cosim[0]),
    xytext=(4.5, 0.44),
    fontsize=8.5,
    color=C_VALS,
    arrowprops=dict(arrowstyle="-", color=C_VALS, lw=0.9, alpha=0.6),
)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.8, 37)
ax.set_ylim(0.35, 1.025)
ax.set_xlabel("Layer", fontsize=10, color=GRAY)
ax.set_ylabel("Cosine similarity", fontsize=10, color=GRAY)
ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35])
ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.tick_params(colors=GRAY, labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(LGRAY)
ax.spines["bottom"].set_color(LGRAY)

# ── legend ────────────────────────────────────────────────────────────────────
patch_k = mpatches.Patch(color=C_KEYS, label="Keys  (independent vs CAS)")
patch_v = mpatches.Patch(color=C_VALS, label="Values  (independent vs CAS)")
ax.legend(handles=[patch_k, patch_v], fontsize=8.5,
          loc="lower right", framealpha=0.9, edgecolor=LGRAY)

# ── title ─────────────────────────────────────────────────────────────────────
ax.set_title(
    "Per-Layer Cosine Similarity: Same Patient (p01), Independent vs CAS Training\n"
    "Qwen3-8B · 1024 slots · 240 steps",
    fontsize=10.5, fontweight="bold", pad=10, loc="left",
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "per_layer_cosim.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
