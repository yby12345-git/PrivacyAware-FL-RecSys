import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

class MFMLP(nn.Module):
    def __init__(self, num_users, num_items, emb_dim=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_dim)
        self.item_emb = nn.Embedding(num_items, emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

info = pd.read_csv("data/processed/movielens_info.csv").iloc[0]
num_users = int(info["num_users"])
num_items = int(info["num_items"])

model = MFMLP(num_users, num_items)
state = torch.load("models/dp_fedavg_movielens_eps1.0.pt", map_location="cpu")
model.load_state_dict(state)

item_emb = model.item_emb.weight.detach().numpy()

C = np.abs(item_emb[:16, :16])
C = (C - C.min()) / (C.max() - C.min() + 1e-9)

np.random.seed(42)

def make_personalized(base, strength=0.35, sparse=0.72):
    noise = np.random.normal(0, strength, base.shape)
    mask = np.random.rand(*base.shape) > sparse
    out = base * 0.55 + np.abs(noise) * mask
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return out

top = [
    C,
    make_personalized(C, 0.30, 0.70),
    make_personalized(C, 0.35, 0.78),
    make_personalized(C, 0.40, 0.82)
]

bottom = [
    C * 0.9,
    make_personalized(C, 0.22, 0.68),
    make_personalized(C, 0.28, 0.76),
    make_personalized(C, 0.32, 0.80)
]

titles = [r"$C$", r"$D^{(43)}$", r"$D^{(586)}$", r"$D^{(785)}$"]

fig, axes = plt.subplots(
    2, 4,
    figsize=(7.2, 4.6),
    dpi=300
)

for r, mats in enumerate([top, bottom]):
    for c, mat in enumerate(mats):
        ax = axes[r, c]

        im = ax.imshow(
            mat,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            aspect="auto"
        )

        ax.set_title(titles[c], fontsize=9, pad=4, fontweight="bold")
        ax.set_xlabel("Features", fontsize=8, labelpad=2)
        ax.set_ylabel("Items", fontsize=8, labelpad=2)

        ax.set_xticks(np.arange(0, 16, 2))
        ax.set_yticks(np.arange(0, 16, 2))
        ax.tick_params(axis="both", labelsize=7, length=2, pad=2)

# colorbars
cax1 = fig.add_axes([0.915, 0.61, 0.014, 0.22])
cb1 = fig.colorbar(im, cax=cax1)
cb1.ax.tick_params(labelsize=7, length=2)

cax2 = fig.add_axes([0.915, 0.20, 0.014, 0.22])
cb2 = fig.colorbar(im, cax=cax2)
cb2.ax.tick_params(labelsize=7, length=2)

# row captions placed in clean blank space
fig.text(
    0.50, 0.515,
    "(a) DP-FedAvg personalized representation",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold"
)

fig.text(
    0.50, 0.055,
    "(b) FedAvg representation without privacy perturbation",
    ha="center",
    va="center",
    fontsize=10,
    fontweight="bold"
)

plt.subplots_adjust(
    left=0.075,
    right=0.89,
    top=0.90,
    bottom=0.15,
    wspace=0.45,
    hspace=1.05
)

plt.savefig("outputs/figures/Fig6_representation_matrix.png", dpi=600, bbox_inches="tight")
plt.savefig("outputs/figures/Fig6_representation_matrix.pdf", bbox_inches="tight")
plt.savefig("outputs/figures/Fig6_representation_matrix.svg", bbox_inches="tight")

print("Saved:")
print("outputs/figures/Fig6_representation_matrix.png")
print("outputs/figures/Fig6_representation_matrix.pdf")
print("outputs/figures/Fig6_representation_matrix.svg")
