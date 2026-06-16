import matplotlib.pyplot as plt
import numpy as np

phases = ["build\ntraining\ndataset", "baseline\neval", "train\ncartridge", "cartridge\neval"]

memory = {
    "8B":   [22.2, 20.9, 17.7, 16.6],
    "4B":   [14.4, 13.1,  9.8,  8.7],
    "1.7B": [ 9.5,  7.9,  4.5,  3.9],
    "0.6B": [ 7.2,  5.6,  2.2,  1.7],
}

# FLOPs in TFLOPs (1 PFLOPs = 1000 TFLOPs)
flops = {
    "8B":   [15047, 2505, 85.2, 11.9],
    "4B":   [ 4000, 1330, 181,   6.3],
    "1.7B": [ 1710,  570,  77,   2.7],
    "0.6B": [  593,  197, 26.8,  0.93],
}

x = np.arange(len(phases))
width = 0.2
colors = {"8B": "#8172B3", "4B": "#4C72B0", "1.7B": "#DD8452", "0.6B": "#55A868"}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

# --- Memory ---
for i, (model, vals) in enumerate(memory.items()):
    ax1.bar(x + (i - 1.5) * width, vals, width, label=model, color=colors[model])

ax1.set_xticks(x)
ax1.set_xticklabels(phases)
ax1.set_ylabel("Peak memory (GB)")
ax1.set_title("Peak GPU memory per phase")
ax1.legend()
ax1.grid(True, axis="y", alpha=0.3)

# --- FLOPs (log scale) ---
for i, (model, vals) in enumerate(flops.items()):
    ax2.bar(x + (i - 1.5) * width, vals, width, label=model, color=colors[model])

ax2.set_yscale("log")
ax2.set_xticks(x)
ax2.set_xticklabels(phases)
ax2.set_ylabel("FLOPs (TFLOPs, log scale)")
ax2.set_title("FLOPs per phase")
ax2.legend()
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("phase_metrics.png", dpi=150)
print("Saved phase_metrics.png")
