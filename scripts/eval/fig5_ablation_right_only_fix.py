import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"

df = pd.read_csv("outputs/tables/ablation_results.csv")

variants = [
    "Full\nModel",
    "w/o\nDP",
    "w/o\nIID"
]

auc = df["AUC"].values

full_auc = auc[0]
drop = (full_auc - auc) / full_auc * 100

colors = [
    "#fdd0a2",
    "#fdae6b",
    "#74c476"
]

fig, ax = plt.subplots(figsize=(3.8, 2.8), dpi=300)

bars = ax.bar(
    variants,
    drop,
    color=colors,
    edgecolor="black",
    linewidth=0.35,
    width=0.58
)

ax.set_ylabel("Relative Change (%)")

ax.set_ylim(-4, 30)

ax.set_title(
    "(b) Relative performance change",
    fontsize=10,
    pad=8
)

ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)

ax.axhline(0, color="black", linewidth=0.6)

# 核心修复：增大label与坐标轴距离
ax.tick_params(axis='x', pad=10)

# 核心修复：增大底部margin
plt.subplots_adjust(bottom=0.32)

for b, v in zip(bars, drop):

    if v >= 0:
        ypos = v + 1.0
        valign = "bottom"
    else:
        ypos = v - 1.2
        valign = "top"

    ax.text(
        b.get_x() + b.get_width()/2,
        ypos,
        f"{v:.1f}%",
        ha="center",
        va=valign,
        fontsize=7
    )

plt.savefig(
    "outputs/figures/Fig5_ablation_right_fix.png",
    dpi=600,
    bbox_inches="tight"
)

print("Saved:")
print("outputs/figures/Fig5_ablation_right_fix.png")
