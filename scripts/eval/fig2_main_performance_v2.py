import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

methods = ["Centralized", "FedAvg-IID", "FedAvg-nonIID", "DP-FedAvg"]
colors = ["#f3c999", "#f2a65e", "#74c476", "#2b9eb3"]

gauc_ml = [0.782, 0.720, 0.547, 0.710]
gauc_az = [0.557, 0.516]

ndcg_ml = [0.742, 0.692, 0.503, 0.681]
ndcg_az = [0.521, 0.487]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.55), dpi=300)

def draw_panel(ax, ml_vals, az_vals, ylabel, title):
    bar_width = 0.16

    # MovieLens 四柱
    x_ml = np.array([-0.24, -0.08, 0.08, 0.24])

    # Amazon 两柱，单独居中
    x_az_center = 1.05
    x_az = np.array([x_az_center - 0.08, x_az_center + 0.08])

    for i, v in enumerate(ml_vals):
        ax.bar(
            x_ml[i], v,
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.35,
            label=methods[i]
        )
        ax.text(
            x_ml[i], v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5
        )

    for i, v in enumerate(az_vals):
        ax.bar(
            x_az[i], v,
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.35
        )
        ax.text(
            x_az[i], v + 0.012,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.5
        )

    ax.set_xticks([0, x_az_center])
    ax.set_xticklabels(["MovieLens-1M", "Amazon"], fontsize=9)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, pad=5)

    ax.set_ylim(0.45, 0.84)
    ax.set_xlim(-0.45, 1.25)

    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="both", labelsize=8, length=3)

draw_panel(axes[0], gauc_ml, gauc_az, "GAUC", "HR@10")
draw_panel(axes[1], ndcg_ml, ndcg_az, "NDCG@10", "NDCG@10")

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.01)
)

plt.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.82,
    bottom=0.30,
    wspace=0.24
)

plt.savefig("outputs/figures/Fig2_main_performance_v2.png", dpi=600, bbox_inches="tight")
plt.savefig("outputs/figures/Fig2_main_performance_v2.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/Fig2_main_performance_v2.svg", bbox_inches="tight")

print("Saved:")
print("outputs/figures/Fig2_main_performance_v2.png")
print("outputs/figures/Fig2_main_performance_v2.pdf")
print("outputs/figures/Fig2_main_performance_v2.svg")
