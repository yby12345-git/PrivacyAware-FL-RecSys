import os
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.linewidth"] = 1.0

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

# 关键：增大画布高度
fig, ax = plt.subplots(figsize=(4.2, 3.6), dpi=300)

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

ax.set_title(
    "(b) Relative performance change",
    fontsize=11,
    pad=10
)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.5,
    alpha=0.45
)

ax.axhline(
    0,
    color="black",
    linewidth=0.6
)

# 关键修复
ax.tick_params(axis='x', pad=18)

# 关键修复
plt.subplots_adjust(bottom=0.35)

for b, v in zip(bars, drop):

    if v >= 0:
        ypos = v + 1.0
        valign = "bottom"
    else:
        ypos = v - 1.5
        valign = "top"

    ax.text(
        b.get_x() + b.get_width()/2,
        ypos,
        f"{v:.1f}%",
        ha="center",
        va=valign,
        fontsize=8
    )

# 输出三种格式
plt.savefig(
    "outputs/figures/Fig5_right_panel_fix.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    "outputs/figures/Fig5_right_panel_fix.pdf",
    bbox_inches="tight"
)

plt.savefig(
    "outputs/figures/Fig5_right_panel_fix.svg",
    bbox_inches="tight"
)

print("Saved:")
print("outputs/figures/Fig5_right_panel_fix.png")
print("outputs/figures/Fig5_right_panel_fix.pdf")
print("outputs/figures/Fig5_right_panel_fix.svg")
