import matplotlib.pyplot as plt
import numpy as np

phases = ["build\ntraining\ndataset", "baseline\neval", "train\ncartridge", "cartridge\neval"]

memory_240 = [7.3, 5.6, 2.2, 1.7]
memory_60  = [7.2, 5.6, 2.2, 1.7]
time_240   = [27.9, 8.5, 29.6, 6.4]
time_60    = [22.8, 8.6, 12.0, 6.3]

x = np.arange(len(phases))
width = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Qwen3-0.6B — 240 steps vs 60 steps", fontsize=13)

# --- Memory ---
ax1.bar(x - width/2, memory_240, width, label="240 steps", color="#4C72B0")
ax1.bar(x + width/2, memory_60,  width, label="60 steps",  color="#DD8452")
ax1.set_xticks(x)
ax1.set_xticklabels(phases)
ax1.set_ylabel("Peak memory (GB)")
ax1.set_title("Peak GPU memory")
ax1.set_ylim(0, 10)
ax1.legend()
ax1.grid(True, axis="y", alpha=0.3)

# --- Timing ---
ax2.bar(x - width/2, time_240, width, label="240 steps", color="#4C72B0")
ax2.bar(x + width/2, time_60,  width, label="60 steps",  color="#DD8452")
ax2.set_xticks(x)
ax2.set_xticklabels(phases)
ax2.set_ylabel("Time (s)")
ax2.set_title("Phase timing")
ax2.legend()
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("steps_ablation.png", dpi=150)
print("Saved steps_ablation.png")
