import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

df = pd.read_csv("outputs/tables/privacy_tradeoff_movielens_all.csv")
df = df.sort_values("epsilon")

eps = df["epsilon"].values
auc = df["final_auc"].values
noise = df["noise_std"].values

fig, ax1 = plt.subplots(figsize=(4.4, 2.55), dpi=300)

# AUC curve
ax1.plot(
    eps,
    auc,
    color="#e95c47",
    linestyle="-.",
    marker="o",
    linewidth=1.8,
    markersize=5,
    label="DP-FedAvg AUC"
)

ax1.set_xlabel(r"Privacy budget $\epsilon$")
ax1.set_ylabel("AUC")
ax1.set_ylim(0.704, 0.7115)
ax1.set_xticks(eps)
ax1.grid(True, linestyle="--", linewidth=0.55, alpha=0.55)

for x, y in zip(eps, auc):
    ax1.text(
        x,
        y + 0.00025,
        f"{y:.4f}",
        ha="center",
        va="bottom",
        fontsize=7
    )

# Noise curve, right axis
ax2 = ax1.twinx()
ax2.plot(
    eps,
    noise,
    color="#2ca58d",
    linestyle="--",
    marker="s",
    linewidth=1.6,
    markersize=4.5,
    label="Noise std"
)
ax2.set_ylabel(r"Noise multiplier $\sigma$")
ax2.set_ylim(0, 0.045)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

fig.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="lower center",
    ncol=2,
    frameon=True,
    fontsize=8,
    bbox_to_anchor=(0.5, -0.03)
)

ax1.set_title("Privacy–Utility Tradeoff", fontsize=10, pad=5)

plt.subplots_adjust(left=0.13, right=0.86, top=0.83, bottom=0.32)

plt.savefig("outputs/figures/Fig3_privacy_utility_tradeoff.png", dpi=600, bbox_inches="tight")
plt.savefig("outputs/figures/Fig3_privacy_utility_tradeoff.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/Fig3_privacy_utility_tradeoff.svg", bbox_inches="tight")

print("Saved:")
print("outputs/figures/Fig3_privacy_utility_tradeoff.png")
print("outputs/figures/Fig3_privacy_utility_tradeoff.pdf")
print("outputs/figures/Fig3_privacy_utility_tradeoff.svg")
