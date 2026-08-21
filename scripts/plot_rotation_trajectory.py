#!/usr/bin/env python3
"""Key/Value rotation trajectory: independent vs CAS training.

Static PNG from hardcoded rotation table output.

Usage:
    python scripts/plot_rotation_trajectory.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C_KEY = "#2a78d6"
C_VAL = "#eb6834"
GRAY  = "#6b7280"
LGRAY = "#e5e7eb"

steps     = [50, 100, 150, 200]
indep_key = [0.857, 0.815, 0.815, 0.728]
cas_key   = [0.857, 0.815, 0.815, 0.728]
indep_val = [9.923, 7.377, 5.373, 4.439]
cas_val   = [9.695, 7.964, 5.455, 4.407]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for y in [2, 4, 6, 8, 10]:
    ax.axhline(y, color=LGRAY, linewidth=0.7, zorder=1)

ms = 7  # marker size

ax.plot(steps, indep_val, color=C_VAL, linewidth=2, linestyle="--",
        marker="o", markersize=ms, zorder=3, label="Values — independent")
ax.plot(steps, cas_val,   color=C_VAL, linewidth=2, linestyle="-",
        marker="s", markersize=ms, zorder=3, label="Values — CAS")
ax.plot(steps, indep_key, color=C_KEY, linewidth=2, linestyle="--",
        marker="o", markersize=ms, zorder=3, label="Keys — independent")
ax.plot(steps, cas_key,   color=C_KEY, linewidth=2, linestyle="-",
        marker="s", markersize=ms, zorder=3, label="Keys — CAS")

# value labels on val lines
for i, step in enumerate(steps):
    ax.text(step, indep_val[i] + 0.25, f"{indep_val[i]:.2f}°",
            ha="center", va="bottom", fontsize=7.5, color=GRAY)
    ax.text(step, cas_val[i] - 0.45, f"{cas_val[i]:.2f}°",
            ha="center", va="top", fontsize=7.5, color=GRAY)

# key annotation (flat, both overlap)
ax.annotate("Keys ≈ 0.8° (both conditions overlap)",
            xy=(175, 0.815), xytext=(130, 2.0),
            fontsize=8, color=C_KEY,
            arrowprops=dict(arrowstyle="-", color=C_KEY, lw=0.8),
            ha="center")

ax.set_xticks(steps)
ax.set_xticklabels([str(s) for s in steps], fontsize=9)
ax.set_xlabel("Training step", fontsize=10, color=GRAY)
ax.set_ylabel("Rotation (degrees per 50-step window)", fontsize=10, color=GRAY)
ax.set_xlim(30, 220)
ax.set_ylim(0, 11.5)
ax.tick_params(colors=GRAY, labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(LGRAY)
ax.spines["bottom"].set_color(LGRAY)

ax.legend(fontsize=9, loc="upper right", framealpha=0.9, edgecolor=LGRAY)

ax.set_title(
    "Key vs Value Rotation: Independent vs CAS Training\n"
    "Qwen3-8B · 1024 slots · avg. 4 patients",
    fontsize=10.5, fontweight="bold", pad=10, loc="left",
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "rotation_trajectory.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
