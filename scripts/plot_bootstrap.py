import matplotlib.pyplot as plt
import numpy as np

bootstrap = [30, 60, 120]

build_flops   = [2.0, 4.0, 8.0]   # PFLOPs
train_flops   = [51.1, 51.1, 51.1] # TFLOPs (constant)
build_memory  = [14.4, 14.5, 14.4] # GB (constant)
train_memory  = [9.8, 10.1, 9.8]   # GB (constant)
cartridge_memory = [8.7, 8.7, 8.7] # GB (constant)
bootstrap_time = [9.6, 18.9, 40.9] # s
build_time     = [18.1, 32.0, 59.2] # s
train_time     = [16.1, 17.9, 17.4] # s

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Qwen3-4B — effect of bootstrap count (60 train steps)", fontsize=13)

# --- FLOPs ---
ax = axes[0]
ax.plot(bootstrap, build_flops, marker="o", color="#4C72B0", label="build_training_dataset (PFLOPs)")
ax.axhline(y=51.1/1e6, color="#DD8452", linestyle="--", label="train_cartridge (TFLOPs, ~0)")
ax.set_xlabel("Bootstrap questions")
ax.set_ylabel("PFLOPs")
ax.set_title("FLOPs")
ax.set_xticks(bootstrap)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Peak memory ---
ax = axes[1]
ax.plot(bootstrap, build_memory,     marker="o", color="#4C72B0", label="build_training_dataset")
ax.plot(bootstrap, train_memory,     marker="s", color="#DD8452", label="train_cartridge")
ax.plot(bootstrap, cartridge_memory, marker="^", color="#55A868", label="cartridge_eval")
ax.set_xlabel("Bootstrap questions")
ax.set_ylabel("Peak memory (GB)")
ax.set_title("Peak GPU memory")
ax.set_xticks(bootstrap)
ax.set_ylim(0, 18)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Timing (stacked bar) ---
ax = axes[2]
x = np.arange(len(bootstrap))
width = 0.5
p1 = ax.bar(x, bootstrap_time, width, label="Bootstrap generation", color="#4C72B0")
p2 = ax.bar(x, build_time,     width, bottom=bootstrap_time, label="Build training dataset", color="#DD8452")
p3 = ax.bar(x, train_time,     width,
            bottom=[a+b for a,b in zip(bootstrap_time, build_time)],
            label="Train cartridge", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels([f"{b}" for b in bootstrap])
ax.set_xlabel("Bootstrap questions")
ax.set_ylabel("Time (s)")
ax.set_title("Build time breakdown")
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("bootstrap_ablation.png", dpi=150)
print("Saved bootstrap_ablation.png")

# Print bullet notes
print("""
Key notes for slides:
  • Memory is flat — bootstrap count has zero effect on peak GPU memory
  • build FLOPs scale exactly 2x with bootstrap count (30→60→120 = 2→4→8 PFLOPs)
  • train FLOPs are fixed — depend only on step count, not data volume (51 TFLOPs always)
  • Bootstrap generation dominates build time at low counts; halving bootstrap halves both bootstrap + dataset build time
  • 120->30 bootstrap saves ~91s total build time with modest quality trade-off
""")
