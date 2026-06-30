import matplotlib.pyplot as plt

slots        = [64, 128, 256, 512, 1024]
compression  = [129, 65, 32, 16, 8]
exact        = [0.60, 0.65, 0.70, 0.85, 0.85]
semantic     = [0.80, 0.80, 0.85, 0.90, 1.00]
baseline_exact    = 0.80
baseline_semantic = 1.00

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(slots, exact,    marker="o", label="Cartridge exact match",    color="#DD8452", linestyle="--")
ax.plot(slots, semantic, marker="s", label="Cartridge semantic match", color="#DD8452", linestyle="-")
ax.axhline(baseline_exact,    color="#4C72B0", linestyle="--", alpha=0.7, label="Baseline exact (0.80)")
ax.axhline(baseline_semantic, color="#4C72B0", linestyle="-",  alpha=0.7, label="Baseline semantic (1.00)")

# Annotate compression ratios above each x tick
for s, c, e in zip(slots, compression, exact):
    ax.annotate(f"{c}×", xy=(s, e), xytext=(0, -18),
                textcoords="offset points", ha="center", fontsize=8, color="#555555")

ax.set_xlabel("Cartridge slots")
ax.set_ylabel("Match rate")
ax.set_ylim(0, 1.10)
ax.set_xscale("log", base=2)
ax.set_xticks(slots)
ax.set_xticklabels(slots)
ax.set_title("Accuracy vs cartridge size (Qwen3-8B, 60 steps, 8192-token document)\ncompression ratio shown below each point")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("slot_accuracy.png", dpi=150)
print("Saved slot_accuracy.png")
