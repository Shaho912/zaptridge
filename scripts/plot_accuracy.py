import matplotlib.pyplot as plt

models = ["0.6B", "1.7B", "4B", "8B"]
baseline_exact    = [0.25, 0.70, 0.90, 0.75]
baseline_semantic = [0.50, 0.85, 1.00, 1.00]
cartridge_exact   = [0.40, 0.70, 0.75, 0.85]
cartridge_semantic= [0.60, 0.80, 1.00, 1.00]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(models, baseline_exact,    marker="o", label="Baseline exact",    color="#4C72B0", linestyle="--")
ax.plot(models, baseline_semantic, marker="o", label="Baseline semantic",  color="#4C72B0", linestyle="-")
ax.plot(models, cartridge_exact,   marker="s", label="Cartridge exact",   color="#DD8452", linestyle="--")
ax.plot(models, cartridge_semantic,marker="s", label="Cartridge semantic", color="#DD8452", linestyle="-")

ax.set_xlabel("Model size")
ax.set_ylabel("Match rate")
ax.set_ylim(0, 1.05)
ax.set_title("Accuracy vs model size (240 steps, 8× compression)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("accuracy_by_model.png", dpi=150)
print("Saved accuracy_by_model.png")
