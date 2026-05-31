import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

iid = pd.read_csv("outputs/tables/fedavg_iid_movielens.csv")
dp = pd.read_csv("outputs/tables/dp_fedavg_movielens_eps1.0.csv")
noniid = pd.read_csv("outputs/tables/fedavg_noniid_movielens.csv")

fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.45), dpi=300)

styles = {
    "FedAvg-IID": {
        "color": "#1f9eb3",
        "linestyle": "-",
        "linewidth": 1.9
    },
    "DP-FedAvg": {
        "color": "#e95c47",
        "linestyle": "-.",
        "linewidth": 1.6
    },
    "FedAvg-non-IID": {
        "color": "#2ca58d",
        "linestyle": "--",
        "linewidth": 1.6
    }
}

# Left: AUC convergence
ax = axes[0]
ax.plot(iid["round"], iid["auc"], label="FedAvg-IID", **styles["FedAvg-IID"])
ax.plot(dp["round"], dp["auc"], label="DP-FedAvg", **styles["DP-FedAvg"])
ax.plot(noniid["round"], noniid["auc"], label="FedAvg-non-IID", **styles["FedAvg-non-IID"])

ax.set_xlabel("Communication Round")
ax.set_ylabel("AUC")
ax.set_xlim(1, 12)
ax.set_ylim(0.50, 0.75)
ax.set_xticks([1, 3, 5, 7, 9, 11])
ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.55)
ax.set_title("(a) AUC convergence", fontsize=10, pad=5)

# Right: loss convergence
ax = axes[1]
ax.plot(iid["round"], iid["loss"], label="FedAvg-IID", **styles["FedAvg-IID"])
ax.plot(dp["round"], dp["loss"], label="DP-FedAvg", **styles["DP-FedAvg"])
ax.plot(noniid["round"], noniid["loss"], label="FedAvg-non-IID", **styles["FedAvg-non-IID"])

ax.set_xlabel("Communication Round")
ax.set_ylabel("Training Loss")
ax.set_xlim(1, 12)
ax.set_ylim(0.52, 0.69)
ax.set_xticks([1, 3, 5, 7, 9, 11])
ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.55)
ax.set_title("(b) Loss convergence", fontsize=10, pad=5)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.04)
)

plt.subplots_adjust(left=0.09, right=0.985, top=0.82, bottom=0.34, wspace=0.33)

plt.savefig("outputs/figures/Fig4_convergence_curve.png", dpi=600, bbox_inches="tight")
plt.savefig("outputs/figures/Fig4_convergence_curve.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/Fig4_convergence_curve.svg", bbox_inches="tight")

print("Saved:")
print("outputs/figures/Fig4_convergence_curve.png")
print("outputs/figures/Fig4_convergence_curve.pdf")
print("outputs/figures/Fig4_convergence_curve.svg")
