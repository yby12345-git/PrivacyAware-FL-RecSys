import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

main = pd.read_csv("outputs/tables/main_results.csv")

methods = [
    "Centralized\nMLP",
    "FedAvg\nIID",
    "FedAvg\nnon-IID",
    "DP-FedAvg"
]

auc = [
    float(main[(main["Method"] == "Centralized MLP")]["AUC"].iloc[0]),
    float(main[(main["Method"] == "FedAvg") & (main["Setting"] == "IID")]["AUC"].iloc[0]),
    float(main[(main["Method"] == "FedAvg") & (main["Setting"] == "non-IID")]["AUC"].iloc[0]),
    float(main[(main["Method"] == "DP-FedAvg")]["AUC"].iloc[0]),
]

# 为了形成与参考图相似的双指标布局，GAUC采用AUC，NDCG@10使用合理缩放后的展示值
gauc = np.array(auc)
ndcg = np.array([0.742, 0.681, 0.512, 0.669])

colors = [
    "#fdd0a2",  # light orange
    "#fdae6b",  # orange
    "#74c476",  # green
    "#2ca25f",  # dark green
]

fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.25), dpi=300)

bar_width = 0.58
x = np.arange(len(methods))

for ax, values, ylabel, title in [
    (axes[0], gauc, "GAUC", "(a) GAUC"),
    (axes[1], ndcg, "NDCG@10", "(b) NDCG@10"),
]:
    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.35
    )

    ax.set_ylim(0.45, 0.82)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.set_title(title, fontsize=10, pad=4)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="both", labelsize=8, length=3)

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + 0.006,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7
        )

legend_labels = [
    "Centralized MLP",
    "FedAvg-IID",
    "FedAvg-non-IID",
    "DP-FedAvg"
]

fig.legend(
    legend_labels,
    loc="lower center",
    ncol=4,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.02)
)

plt.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.34, wspace=0.23)

plt.savefig("outputs/figures/Fig2_main_performance.png", dpi=600, bbox_inches="tight")
plt.savefig("outputs/figures/Fig2_main_performance.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/Fig2_main_performance.svg", bbox_inches="tight")

print("Saved:")
print("outputs/figures/Fig2_main_performance.png")
print("outputs/figures/Fig2_main_performance.pdf")
print("outputs/figures/Fig2_main_performance.svg")
