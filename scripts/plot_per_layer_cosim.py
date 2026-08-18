#!/usr/bin/env python3
"""Per-layer cosine similarity — 6 lines:
  - Cross-condition keys/values  (p01 ind vs p01 CAS, same patient)
  - Cross-patient ind keys/values (p01 ind vs p02 ind)
  - Cross-patient CAS keys/values (p01 CAS vs p02 CAS)

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

# ── cross-patient ind: p01 ind vs p02 ind (all 36 layers) ────────────────────
layers_cp_all = list(range(36))
key_cp_ind = [
    0.964690, 0.905371, 0.716290, 0.700161, 0.869636, 0.843152, 0.804066, 0.635698,
    0.855917, 0.512079, 0.880403, 0.789063, 0.700809, 0.466059, 0.738464, 0.564530,
    0.591623, 0.497704, 0.523410, 0.582552, 0.461780, 0.554560, 0.621922, 0.622715,
    0.473887, 0.670040, 0.516345, 0.599120, 0.604082, 0.476958, 0.701558, 0.549988,
    0.652607, 0.424804, 0.627398, 0.829965,
]
val_cp_ind = [
    0.083274, 0.111995, 0.101465, 0.142588, 0.112223, 0.152963, 0.108540, 0.195934,
    0.167838, 0.176399, 0.194605, 0.188496, 0.213127, 0.253637, 0.189015, 0.254924,
    0.207132, 0.244023, 0.256156, 0.193247, 0.186894, 0.184981, 0.182076, 0.176013,
    0.163268, 0.178051, 0.152111, 0.178865, 0.181704, 0.099354, 0.182463, 0.123204,
    0.148579, 0.102571, 0.210696, 0.170526,
]

# ── cross-patient CAS: p01 CAS vs p02 CAS (all 36 layers) ────────────────────
key_cp_cas = [
    0.964690, 0.905370, 0.716269, 0.700166, 0.869631, 0.843158, 0.804081, 0.635688,
    0.855920, 0.512105, 0.880419, 0.789076, 0.700836, 0.466023, 0.738511, 0.564541,
    0.591636, 0.497694, 0.523395, 0.582573, 0.461766, 0.554611, 0.621932, 0.622731,
    0.473895, 0.670043, 0.516338, 0.599179, 0.604094, 0.476969, 0.701596, 0.550007,
    0.652638, 0.424848, 0.627379, 0.829984,
]
val_cp_cas = [
    0.081102, 0.116384, 0.105046, 0.146259, 0.107595, 0.153201, 0.107658, 0.193555,
    0.168286, 0.175770, 0.193784, 0.188336, 0.214386, 0.253906, 0.189241, 0.255600,
    0.207432, 0.243706, 0.256093, 0.193106, 0.187080, 0.184784, 0.182033, 0.176012,
    0.163259, 0.177983, 0.152176, 0.178861, 0.181761, 0.099361, 0.182468, 0.123214,
    0.148576, 0.102568, 0.210694, 0.170522,
]

# clip to MAX_LAYER
cc_idx = [i for i, l in enumerate(layers_cc) if l <= MAX_LAYER]
layers_cc_c = [layers_cc[i] for i in cc_idx]
key_cc_c    = [key_cc[i]    for i in cc_idx]
val_cc_c    = [val_cc[i]    for i in cc_idx]

cp_idx = [i for i, l in enumerate(layers_cp_all) if l <= MAX_LAYER]
layers_cp_c   = [layers_cp_all[i] for i in cp_idx]
key_cp_ind_c  = [key_cp_ind[i]    for i in cp_idx]
val_cp_ind_c  = [val_cp_ind[i]    for i in cp_idx]
key_cp_cas_c  = [key_cp_cas[i]    for i in cp_idx]
val_cp_cas_c  = [val_cp_cas[i]    for i in cp_idx]

# ── figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for y in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ax.axhline(y, color=LGRAY, linewidth=0.7, zorder=1)

# cross-condition — solid lines
ax.plot(layers_cc_c, key_cc_c, color=C_KEY, linewidth=2, linestyle="-",  zorder=3)
ax.plot(layers_cc_c, val_cc_c, color=C_VAL, linewidth=2, linestyle="-",  zorder=3)

# cross-patient independent — dashed lines
ax.plot(layers_cp_c, key_cp_ind_c, color=C_KEY, linewidth=2, linestyle="--", zorder=4)
ax.plot(layers_cp_c, val_cp_ind_c, color=C_VAL, linewidth=2, linestyle="--", zorder=4)

# cross-patient CAS — dotted lines
ax.plot(layers_cp_c, key_cp_cas_c, color=C_KEY, linewidth=2, linestyle=":",  zorder=5)
ax.plot(layers_cp_c, val_cp_cas_c, color=C_VAL, linewidth=2, linestyle=":",  zorder=5)

# ── axes ──────────────────────────────────────────────────────────────────────
ax.set_xlim(-0.5, MAX_LAYER + 0.5)
ax.set_ylim(0.04, 1.04)
ax.set_xticks([0, 2, 4, 6, 8, 10])
ax.set_xticklabels(["0", "2", "4", "6", "8", "10+"])
ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
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
                   label="Keys — p01 ind vs p02 ind")
l3 = mlines.Line2D([], [], color=C_KEY, linewidth=2, linestyle=":",
                   label="Keys — p01 CAS vs p02 CAS")
l4 = mlines.Line2D([], [], color=C_VAL, linewidth=2, linestyle="-",
                   label="Values — same patient, Ind vs CAS")
l5 = mlines.Line2D([], [], color=C_VAL, linewidth=2, linestyle="--",
                   label="Values — p01 ind vs p02 ind")
l6 = mlines.Line2D([], [], color=C_VAL, linewidth=2, linestyle=":",
                   label="Values — p01 CAS vs p02 CAS")
ax.legend(handles=[l1, l2, l3, l4, l5, l6], fontsize=8.5, loc="lower right",
          framealpha=0.9, edgecolor=LGRAY)

# ── overlap annotations ───────────────────────────────────────────────────────
# Keys: ind and CAS cross-patient are nearly identical (~0.965 at layer 0)
ax.annotate("Ind & CAS\noverlap",
            xy=(3, key_cp_ind_c[3]), xytext=(5.5, 0.77),
            fontsize=7.5, color=C_KEY, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=C_KEY, lw=0.9))

# Values: ind and CAS cross-patient are nearly identical (~0.08–0.09)
ax.annotate("Ind & CAS\noverlap",
            xy=(3, val_cp_ind_c[3]), xytext=(5.5, 0.32),
            fontsize=7.5, color=C_VAL, ha="center",
            arrowprops=dict(arrowstyle="-|>", color=C_VAL, lw=0.9))

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
