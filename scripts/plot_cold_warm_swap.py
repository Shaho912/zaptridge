#!/usr/bin/env python3
"""Cold vs warm swap scaling: drop % as cartridge count grows.

Usage:
    python scripts/plot_cold_warm_swap.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C_COLD = "#eb6834"   # orange — cold swap
C_WARM = "#2a78d6"   # blue   — warm swap
GRAY   = "#6b7280"
LGRAY  = "#e5e7eb"

# drop % = (oracle - joint) / total * 100
# 4 CAS: 11/32 oracle, 12/32 joint  (8Q, shared by both lines)
# 5-cart cold: 13/40, 14/40        (8Q, shared — warm not tested separately)
# 6-cart cold: 14/24, 10/24        (4Q)
# 6-cart warm: 11/24, 14/24        (4Q)

n_carts  = [4,    5,    6]
cold_drop = [-3.1, -2.5, +16.7]
warm_drop = [-3.1, -2.5, -12.5]

fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# background zones
ax.axhspan(0,  22, color=C_COLD, alpha=0.05, zorder=0)
ax.axhspan(-22, 0, color=C_WARM, alpha=0.05, zorder=0)

for y in [-15, -10, -5, 5, 10, 15, 20]:
    ax.axhline(y, color=LGRAY, linewidth=0.7, zorder=1)
ax.axhline(0, color="#9ca3af", linewidth=1.4, zorder=2)

# lines
ax.plot(n_carts, cold_drop, color=C_COLD, linewidth=2.5, linestyle="-",
        marker="o", markersize=8, zorder=4, label="Cold swap (independent)")
ax.plot(n_carts, warm_drop, color=C_WARM, linewidth=2.5, linestyle="-",
        marker="s", markersize=8, zorder=4, label="Warm swap (distractor-trained)")

# value labels
offsets = {
    (4, "cold"): (-0.08,  1.2, "right",  "bottom"),
    (5, "cold"): (-0.08,  1.2, "right",  "bottom"),
    (6, "cold"): ( 0.08,  1.2, "left",   "bottom"),
    (4, "warm"): (-0.08, -1.2, "right",  "top"),
    (5, "warm"): (-0.08, -1.2, "right",  "top"),
    (6, "warm"): ( 0.08, -1.2, "left",   "top"),
}
for i, x in enumerate(n_carts):
    for tag, vals in [("cold", cold_drop), ("warm", warm_drop)]:
        v = vals[i]
        xo, yo, ha, va = offsets[(x, tag)]
        sign = "+" if v >= 0 else ""
        ax.text(x + xo, v + yo, f"{sign}{v:.1f}%",
                ha=ha, va=va, fontsize=8.5, color=GRAY)

# zone labels
ax.text(4.05, 21,  "Interference",     ha="left", va="bottom", fontsize=8.5, color=C_COLD, alpha=0.8)
ax.text(4.05, -21, "Positive transfer", ha="left", va="top",    fontsize=8.5, color=C_WARM, alpha=0.8)

# shared point annotation
ax.annotate("Same result\n(1 newcomer)", xy=(5, -2.5), xytext=(5.2, -9),
            fontsize=7.5, color=GRAY, ha="left",
            arrowprops=dict(arrowstyle="-", color=LGRAY, lw=1.0))

ax.set_xticks(n_carts)
ax.set_xticklabels(["4 cartridges\n(CAS baseline)", "5 cartridges\n(+1 cold swap)",
                    "6 cartridges\n(+2 newcomers)"], fontsize=9)
ax.set_ylabel("% drop  =  (oracle − joint) / n × 100", fontsize=9.5, color=GRAY)
ax.set_xlim(3.6, 6.6)
ax.set_ylim(-22, 24)
ax.tick_params(bottom=False, colors=GRAY, labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color(LGRAY)
ax.spines["bottom"].set_color(LGRAY)

ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor=LGRAY)
ax.set_title(
    "Cold vs Warm Swap: Scaling to 6 Cartridges\n"
    "4 CAS-trained base  ·  Qwen3-8B  ·  LongHealth",
    fontsize=10.5, fontweight="bold", pad=10, loc="left",
)


plt.tight_layout()
out = os.path.join(OUT_DIR, "cold_warm_swap.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
