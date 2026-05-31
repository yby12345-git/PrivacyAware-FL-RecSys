import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

df = pd.read_csv("outputs/tables/ablation_results.csv")

variants = ["Full\nModel", "w/o\nDP", "w/o\nIID"]
auc = df["AUC"].values

colors = ["#fdd0a2", "#fdae6b", "#74c476"]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)

# Left panel
ax = axes[0]
bars = ax.bar(
    variants,
    auc,
    color=colors,
    edgecolor="black",
    linewidth=0.35,
    width=0.58
)

ax.set_ylabel("AUC")
ax.set_ylim(0.50, 0.76)
ax.set_title("(a) Ablation on AUC", fontsize=10, pad=8)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
ax.tick_params(axis="x", pad=14)

for b, v in zip(bars, auc):
    ax.text(
        b.get_x() + b.get_width() / 2,
        v + 0.006,
        f"{v:.4f}",
        ha="center",
        va="bottom",
        fontsize=7
    )

# Right panel
ax = axes[1]

full_auc = auc[0]
drop = (full_auc - auc) / full_auc * 100

bars = ax.bar(
    variants,
    drop,
    color=colors,
    edgecolor="black",
    linewidth=0.35,
    width=0.58
)

ax.set_ylabel("Relative Change (%)")
ax.set_ylim(-5, 30)
ax.set_title("(b) Relative performance change", fontsize=10, pad=8)
ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
ax.axhline(0, color="black", linewidth=0.6)
ax.tick_params(axis="x", pad=16)

for b, v in zip(bars, drop):
    if v >= 0:
        y = v + 1.1
        va = "bottom"
    else:
        y = v - 1.5
        va = "top"

    ax.text(
        b.get_x() + b.get_width() / 2,
        y,
        f"{v:.1f}%",
        ha="center",
        va=va,
        fontsize=7
    )

legend_labels = [
    "Full Model",
    "w/o Differential Privacy",
    "w/o IID Distribution"
]

fig.legend(
    legend_labels,
    loc="lower center",
    ncol=3,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, 0.02)
)

plt.subplots_adjust(
    left=0.08,
    right=0.985,
    top=0.83,
    bottom=0.42,
    wspace=0.34
)

plt.savefig("outputs/figures/Fig5_ablation_study.png", dpi=600)
plt.savefig("outputs/figures/Fig5_ablation_study.pdf")
plt.savefig("outputs/figures/Fig5_ablation_study.svg")

print("Saved:")
print("outputs/figures/Fig5_ablation_study.png")
print("outputs/figures/Fig5_ablation_study.pdf")
print("outputs/figures/Fig5_ablation_study.svg")
