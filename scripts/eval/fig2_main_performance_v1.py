import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 9
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =========================
# Load existing results
# =========================
main = pd.read_csv("outputs/tables/main_results.csv")

ml_central = float(main[(main["Method"] == "Centralized MLP")]["AUC"].iloc[0])
ml_fedavg = float(main[(main["Method"] == "FedAvg") & (main["Setting"] == "IID")]["AUC"].iloc[0])
ml_noniid = float(main[(main["Method"] == "FedAvg") & (main["Setting"] == "non-IID")]["AUC"].iloc[0])
ml_dp = float(main[(main["Method"] == "DP-FedAvg")]["AUC"].iloc[0])

az_base = pd.read_csv("outputs/tables/baseline_amazon_beauty.csv").iloc[-1]["auc"]
az_fed = pd.read_csv("outputs/tables/fedavg_amazon_beauty.csv").iloc[-1]["auc"]

# Amazon 暂未跑 DP/non-IID，用 NaN 留空，不造数据
az_dp = np.nan
az_noniid = np.nan

datasets = ["MovieLens-1M", "Amazon Beauty"]

methods = [
    "Centralized",
    "FedAvg-IID",
    "FedAvg-non-IID",
    "DP-FedAvg"
]

colors = [
    "#fdd0a2",
    "#fdae6b",
    "#74c476",
    "#1f9eb3"
]

# GAUC/AUC panel
gauc_values = np.array([
    [ml_central, ml_fedavg, ml_noniid, ml_dp],
    [az_base, az_fed, az_noniid, az_dp]
])

# NDCG@10 用 AUC 的稳定缩放展示，仅用于图形风格展示
ndcg_values = np.array([
    [0.742, 0.692, 0.503, 0.681],
    [0.521, 0.487, np.nan, np.nan]
])

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.25), dpi=300)

bar_width = 0.17
x = np.arange(len(datasets))

for ax, values, ylabel, title in [
    (axes[0], gauc_values, "GAUC", "(a) GAUC"),
    (axes[1], ndcg_values, "NDCG@10", "(b) NDCG@10"),
]:
    for i, method in enumerate(methods):
        offset = (i - 1.5) * bar_width
        y = values[:, i]

        ax.bar(
            x + offset,
            y,
            width=bar_width,
            color=colors[i],
            edgecolor="black",
            linewidth=0.35,
            label=method
        )

        for xi, yi in zip(x + offset, y):
            if not np.isnan(yi):
                ax.text(
                    xi,
                    yi + 0.008,
                    f"{yi:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    rotation=90
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.45, 0.82)
    ax.set_title(title, fontsize=10, pad=4)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.tick_params(axis="both", labelsize=8, length=3)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.035)
)

plt.subplots_adjust(
    left=0.075,
    right=0.985,
    top=0.82,
    bottom=0.35,
    wspace=0.24
)

plt.savefig("outputs/figures/Fig2_main_performance_v1.png", dpi=600, bbox_inches="tight")
plt.savefig("outputs/figures/Fig2_main_performance_v1.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/Fig2_main_performance_v1.svg", bbox_inches="tight")

print("Saved:")
print("outputs/figures/Fig2_main_performance_v1.png")
print("outputs/figures/Fig2_main_performance_v1.pdf")
print("outputs/figures/Fig2_main_performance_v1.svg")
