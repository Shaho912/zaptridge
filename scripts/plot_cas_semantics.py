#!/usr/bin/env python3
"""Generate CAS semantic diversity charts as PNG files.

Produces two figures:
  figures/cas_drop_by_corpus.png  - diverging bar chart: % drop (oracle-joint) by corpus type
  figures/cas_domain_breakdown.png - per-domain interference for 3-domain 8B experiment

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

C1 = "#2a78d6"   # Exp1 independent — blue
C2 = "#eb6834"   # Exp2 CAS       — orange
C1L = "#d0e4f7"  # light blue zone
C2L = "#fde8dd"  # light orange zone
GRAY = "#6b7280"

# ── experiment data ──────────────────────────────────────────────────────────
# Each entry: (label_line1, label_line2, exp1_oracle, exp1_joint, exp1_n, exp2_oracle, exp2_joint, exp2_n, anomaly)
EXPERIMENTS = [
    ("Research papers",  "3-domain · 4B student\n(ML, FPGA, Bio)",    7,    5,  15,  7,    6,  15, False),
    ("Research papers",  "3-domain · 8B student\n(ML, FPGA, Bio)",   16,   13,  16, 16,   13,  16, False),
    ("Research papers",  "4-domain · 8B student\n(ML, FPGA, Bio, Astro)", 20, 17, 21, 20, 18,  21, False),
    ("Patient records",  "4 patients\n(LongHealth, avg 2 runs)",       8,    7,  16,  7.5, 10,  16, False),
    ("Patient records",  "5 patients †\n(LongHealth)",                 6,    9,  20,  8,   10,  20, True),
]

labels      = [f"{e[0]}\n{e[1]}" for e in EXPERIMENTS]
exp1_drop   = [(e[2] - e[3]) / e[4] * 100 for e in EXPERIMENTS]
exp2_drop   = [(e[5] - e[6]) / e[7] * 100 for e in EXPERIMENTS]
anomaly     = [e[8] for e in EXPERIMENTS]

# ── Figure 1: diverging horizontal bar chart ─────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

N = len(EXPERIMENTS)
y_pos = np.arange(N)
bar_h = 0.32
gap   = 0.06
offsets = [-bar_h/2 - gap/2, bar_h/2 + gap/2]  # Exp1 above, Exp2 below

# background zones
ax.axvspan(0, 25, alpha=0.07, color=C2, zorder=0)
ax.axvspan(-25, 0, alpha=0.07, color=C1, zorder=0)

# zone labels at the top
ax.text(-12.5, N - 0.05, "← Positive transfer  (joint > solo)", ha="center", va="bottom",
        fontsize=8.5, color=C1, alpha=0.85, transform=ax.get_xaxis_transform())
ax.text( 12.5, N - 0.05, "Interference  (joint < solo) →", ha="center", va="bottom",
        fontsize=8.5, color=C2, alpha=0.85, transform=ax.get_xaxis_transform())

for i, (d1, d2, anom) in enumerate(zip(exp1_drop, exp2_drop, anomaly)):
    yi = y_pos[i]

    # Exp1 bar
    alpha1 = 0.45 if anom else 1.0
    ax.barh(yi + offsets[0], d1, height=bar_h, color=C1, alpha=alpha1,
            left=0, zorder=3, clip_on=True)
    # Exp2 bar
    ax.barh(yi + offsets[1], d2, height=bar_h, color=C2, alpha=1.0,
            left=0, zorder=3, clip_on=True)

    # end labels
    for val, off, alpha in [(d1, offsets[0], alpha1), (d2, offsets[1], 1.0)]:
        sign = "+" if val >= 0 else ""
        lbl  = f"{sign}{val:.1f}%"
        xpos = val + (0.5 if val >= 0 else -0.5)
        ha   = "left" if val >= 0 else "right"
        ax.text(xpos, yi + off, lbl, va="center", ha=ha,
                fontsize=8, color=GRAY, alpha=alpha)

# separator between papers and patients
sep_y = 2.5  # between index 2 and 3
ax.axhline(sep_y, color="#d1d5db", linewidth=0.8, linestyle="--", zorder=1)
ax.text(0, sep_y + 0.08, "— more semantically distinct corpora below —",
        ha="center", va="bottom", fontsize=7.5, color=GRAY, style="italic")

# zero line
ax.axvline(0, color="#9ca3af", linewidth=1.4, zorder=2)

# gridlines
for x in [-15, -10, -5, 5, 10, 15, 20]:
    ax.axvline(x, color="#e5e7eb", linewidth=0.7, zorder=1)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels, fontsize=9.5)
ax.set_xlabel("% drop  =  (oracle − joint) / n × 100", fontsize=9, color=GRAY)
ax.set_xlim(-22, 25)
ax.set_ylim(-0.6, N - 0.3)
ax.xaxis.set_tick_params(labelsize=9, colors=GRAY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False)

# legend
patch1 = mpatches.Patch(color=C1, label="Exp 1 — independent training")
patch2 = mpatches.Patch(color=C2, label="Exp 2 — CAS (p=0.75 distractor)")
ax.legend(handles=[patch1, patch2], fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor="#e5e7eb")

ax.set_title("CAS Effect on Joint Accuracy — by Corpus Type\n"
             "Positive drop = interference. Negative drop = positive transfer (joint > solo).",
             fontsize=10.5, fontweight="bold", pad=10, loc="left")

fig.text(0.01, -0.02,
         "† Patient records 5-patient: lh_p05 anomaly inflates joint accuracy — Exp 1 bar dimmed.",
         fontsize=7.5, color=GRAY)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "cas_drop_by_corpus.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out1}")

# ── Figure 2: per-domain breakdown (3-domain 8B) ─────────────────────────────
DOMAINS = [
    ("ML",   6, 2, 2),   # (domain, n, exp1_drop, exp2_drop)
    ("FPGA", 5, 1, 1),
    ("Bio",  5, 0, 0),
]

fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

x = np.arange(len(DOMAINS))
bw = 0.32
gap = 0.06

for i, (domain, n, d1, d2) in enumerate(DOMAINS):
    ax.bar(i - bw/2 - gap/2, d1, width=bw, color=C1, zorder=3, label="Exp 1" if i == 0 else "")
    ax.bar(i + bw/2 + gap/2, d2, width=bw, color=C2, zorder=3, label="Exp 2 (CAS)" if i == 0 else "")

    # value labels
    for val, xoff in [(d1, -bw/2 - gap/2), (d2, bw/2 + gap/2)]:
        ax.text(i + xoff, val + 0.04, str(val), ha="center", va="bottom", fontsize=9.5, color=GRAY)

for y in [1, 2]:
    ax.axhline(y, color="#e5e7eb", linewidth=0.7, zorder=1)

ax.set_xticks(x)
ax.set_xticklabels([d[0] for d in DOMAINS], fontsize=11)
ax.set_ylabel("Questions dropped  (oracle − joint)", fontsize=9, color=GRAY)
ax.set_ylim(0, 2.6)
ax.set_yticks([0, 1, 2])
ax.yaxis.set_tick_params(labelsize=9, colors=GRAY)
ax.xaxis.set_tick_params(bottom=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#e5e7eb")

ax.set_title("Per-Domain Interference — Research Papers, 3-domain (Qwen3-8B)\n"
             "Exp 1 and Exp 2 drop identically — CAS has zero effect.",
             fontsize=10, fontweight="bold", pad=10, loc="left")

fig.text(0.05, -0.04,
         "ML papers share vocabulary (KV cache, attention, compression) preventing distractor-based routing.",
         fontsize=7.5, color=GRAY)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "cas_domain_breakdown.png")
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out2}")
