#!/usr/bin/env python3
"""CAS semantic diversity charts — updated with full per-domain data.

Outputs:
  figures/cas_drop_by_corpus.png    — main diverging bar chart
  figures/cas_domain_breakdown.png  — per-domain oracle vs joint (3-domain and 4-domain 8B)

Usage:
    python scripts/plot_cas_semantics.py
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUT_DIR, exist_ok=True)

C1   = "#2a78d6"   # Exp1 independent — blue
C2   = "#eb6834"   # Exp2 CAS        — orange
GRAY = "#6b7280"
LGRAY = "#e5e7eb"

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# FIGURE 1: main diverging bar chart
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

# (label, sublabel, exp1_oracle, exp1_joint, exp1_n, exp2_oracle, exp2_joint, exp2_n, exp2_note, anomaly)
# exp2_n=None means no Exp2 run
EXPERIMENTS = [
    (
        "Research papers — 3 domains",
        "ML / FPGA / Bio  ·  8B student",
        16, 13, 16,   16, 13, 16,   None, False,
    ),
    (
        "Research papers — 4 domains",
        "ML / FPGA / Bio / Astro  ·  8B student",
        20, 17, 21,   20, 18, 21,   "Run A", False,
    ),
    (
        "Patient records — 4 patients",
        "LongHealth medical  ·  avg. 2 runs",
        8, 7, 16,   7.5, 10, 16,   None, False,
    ),
    (
        "Patient records — 5 patients †",
        "LongHealth medical",
        6, 9, 20,   8, 10, 20,   None, True,
    ),
]

# Distinct query types (no Exp2 — zero interference without CAS)
SETUP2 = ("Distinct query types", "Skills / Methods / Preferences / Research", 18, 18, 18)

fig, ax = plt.subplots(figsize=(11, 5.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

N = len(EXPERIMENTS) + 1  # +1 for Setup 2
bar_h = 0.30
gap   = 0.08
y_pos = np.arange(N)

# background zones
ax.axvspan(0, 25, alpha=0.06, color=C2, zorder=0)
ax.axvspan(-25, 0, alpha=0.06, color=C1, zorder=0)

ax.text(-12, N - 0.08, "← Positive transfer", ha="center", va="bottom",
        fontsize=8.5, color=C1, alpha=0.8, transform=ax.get_xaxis_transform())
ax.text( 12, N - 0.08, "Interference →", ha="center", va="bottom",
        fontsize=8.5, color=C2, alpha=0.8, transform=ax.get_xaxis_transform())

# gridlines
for x in [-15, -10, -5, 5, 10, 15, 20]:
    ax.axvline(x, color=LGRAY, linewidth=0.7, zorder=1)
ax.axvline(0, color="#9ca3af", linewidth=1.4, zorder=2)

def pct(drop, n):
    return drop / n * 100

def draw_bar(ax, y, val, color, alpha=1.0):
    ax.barh(y, val, height=bar_h, color=color, alpha=alpha, left=0, zorder=3)
    if abs(val) > 0.5:
        sign = "+" if val >= 0 else ""
        xpos = val + (0.4 if val >= 0 else -0.4)
        ha = "left" if val >= 0 else "right"
        ax.text(xpos, y, f"{sign}{val:.1f}%", va="center", ha=ha,
                fontsize=8, color=GRAY, alpha=alpha)

# draw Setup 2 (top row, Exp1 only — zero drop)
yi = y_pos[-1]
draw_bar(ax, yi, 0.0, C1)
# zero-drop tick mark
ax.plot([0], [yi], marker="|", color=C1, markersize=10, markeredgewidth=2, zorder=4)
ax.text(0.5, yi, "0%", va="center", ha="left", fontsize=8, color=GRAY)
ax.text(-0.5, yi - 0.45, "no Exp 2 — zero interference without CAS",
        va="top", ha="right", fontsize=7.5, color=GRAY, style="italic")

# draw experiments
for i, (lbl, sub, o1, j1, n1, o2, j2, n2, note2, anom) in enumerate(EXPERIMENTS):
    yi = y_pos[i]
    d1 = pct(o1 - j1, n1)
    d2 = pct(o2 - j2, n2)
    a1 = 0.40 if anom else 1.0

    draw_bar(ax, yi + gap/2 + bar_h/2, d1, C1, alpha=a1)
    exp2_lbl = f"Exp 2{f' ({note2})' if note2 else ''}"
    draw_bar(ax, yi - gap/2 - bar_h/2, d2, C2)

# y-axis labels
all_labels = [(e[0], e[1]) for e in EXPERIMENTS] + [(SETUP2[0], SETUP2[1])]
ax.set_yticks(y_pos)
ax.set_yticklabels(
    [f"{l}\n{s}" for l, s in all_labels],
    fontsize=9.5,
)

ax.set_xlabel("% drop  =  (oracle − joint) / n × 100", fontsize=9, color=GRAY)
ax.set_xlim(-23, 25)
ax.set_ylim(-0.7, N - 0.2)
ax.xaxis.set_tick_params(labelsize=9, colors=GRAY)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False)

# separator between research papers and patient records
ax.axhline(1.5, color=LGRAY, linewidth=0.8, linestyle="--")

# legend
p1 = mpatches.Patch(color=C1, label="Exp 1 — independent training")
p2 = mpatches.Patch(color=C2, label="Exp 2 — CAS (p=0.75 distractor)")
ax.legend(handles=[p1, p2], fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor=LGRAY)

ax.set_title(
    "CAS Effect on Joint Accuracy — by Corpus Type  (Qwen3-8B student)\n"
    "Positive drop = interference. Negative = positive transfer (joint > oracle).",
    fontsize=10.5, fontweight="bold", pad=10, loc="left",
)

fig.text(0.01, -0.03,
         "† 5-patient: lh_p05 anomaly inflates joint accuracy — Exp 1 bar dimmed.\n"
         "4-domain Exp 2 Run A shown (Run B: 17/21 joint, drop +4 — mixed result).",
         fontsize=7.5, color=GRAY)

plt.tight_layout()
out1 = os.path.join(OUT_DIR, "cas_drop_by_corpus.png")
fig.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out1}")

# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──
# FIGURE 2: per-domain breakdown  (3-domain 8B  and  4-domain 8B)
# ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ── ──

# 3-domain 8B (Exp 1 = Exp 2, CAS zero effect)
D3 = [
    ("ML",   6, 6, 4, 6),    # (domain, oracle_n, oracle_k, joint_k, joint_n)
    ("FPGA", 5, 5, 4, 5),
    ("Bio",  5, 5, 5, 5),
]

# 4-domain 8B (Exp 1 only — Exp 2 ambiguous)
D4 = [
    ("ML",   6, 6, 5, 6),
    ("FPGA", 5, 4, 3, 5),
    ("Bio",  5, 5, 5, 5),
    ("Astro",5, 5, 4, 5),
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
fig.patch.set_facecolor("white")

def draw_oracle_joint(ax, data, title, has_cas=False):
    ax.set_facecolor("white")
    n_dom = len(data)
    x = np.arange(n_dom)
    bw = 0.25
    gap = 0.05

    for i, (domain, total, oracle, joint, jtotal) in enumerate(data):
        oracle_pct = oracle / total * 100
        joint_pct  = joint  / jtotal * 100
        drop       = oracle - joint

        # oracle bar
        ax.bar(x[i] - bw/2 - gap/2, oracle_pct, width=bw, color=C1,
               alpha=0.85, zorder=3)
        # joint bar
        ax.bar(x[i] + bw/2 + gap/2, joint_pct, width=bw, color=C2,
               alpha=0.85, zorder=3)

        # drop label between bars
        mid_y = min(oracle_pct, joint_pct) - 5
        sign = "+" if drop >= 0 else ""
        ax.text(x[i], max(oracle_pct, joint_pct) + 2,
                f"drop {sign}{drop}", ha="center", va="bottom",
                fontsize=8, color=GRAY)

    for yg in [20, 40, 60, 80, 100]:
        ax.axhline(yg, color=LGRAY, linewidth=0.7, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([d[0] for d in data], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=9, color=GRAY)
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.yaxis.set_tick_params(labelsize=9, colors=GRAY)
    ax.xaxis.set_tick_params(bottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8, loc="left")

    # legend
    p_oracle = mpatches.Patch(color=C1, alpha=0.85, label="Oracle (solo)")
    p_joint  = mpatches.Patch(color=C2, alpha=0.85, label="Joint (all loaded)")
    ax.legend(handles=[p_oracle, p_joint], fontsize=8, loc="lower right",
              framealpha=0.9, edgecolor=LGRAY)

draw_oracle_joint(
    axes[0], D3,
    "3-domain 8B  (ML / FPGA / Bio)\nExp 1 = Exp 2 — CAS has zero effect",
)
draw_oracle_joint(
    axes[1], D4,
    "4-domain 8B  (ML / FPGA / Bio / Astro)\nExp 1 only  (Exp 2 ambiguous)",
)

# shared note
fig.text(0.5, -0.04,
         "Bio is immune in every run. ML 'institution' question bleeds into FPGA answers — "
         "persistent failure mode across all research-paper runs.",
         ha="center", fontsize=8, color=GRAY)

plt.tight_layout()
out2 = os.path.join(OUT_DIR, "cas_domain_breakdown.png")
fig.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out2}")
