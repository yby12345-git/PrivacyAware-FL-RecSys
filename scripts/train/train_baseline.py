import os
import argparse
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("models", exist_ok=True)

def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class RecDataset(Dataset):
    def __init__(self, df):
        self.u = df["user"].values.astype(np.int64)
        self.i = df["item"].values.astype(np.int64)
        self.y = df["label"].values.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.u[idx], self.i[idx], self.y[idx]

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

    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=1)
        return self.mlp(x).squeeze(1)

def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for u, i, y in loader:
            u, i = u.to(device), i.to(device)
            p = torch.sigmoid(model(u, i)).cpu().numpy()
            ys.extend(y.numpy())
            ps.extend(p)
    try:
        auc = roc_auc_score(ys, ps)
    except:
        auc = 0.5
    return auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="movielens")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_all(args.seed)

    train = pd.read_csv(f"data/processed/{args.dataset}_train.csv")
    test = pd.read_csv(f"data/processed/{args.dataset}_test.csv")
    info = pd.read_csv(f"data/processed/{args.dataset}_info.csv").iloc[0]

    num_users = int(info["num_users"])
    num_items = int(info["num_items"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", args.dataset)
    print("Users:", num_users, "Items:", num_items)

    train_loader = DataLoader(
        RecDataset(train),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = DataLoader(
        RecDataset(test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )

    model = MFMLP(num_users, num_items).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    records = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []
        for u, i, y in tqdm(train_loader, desc=f"Epoch {ep}"):
            u, i, y = u.to(device), i.to(device), y.to(device)
            opt.zero_grad()
            logit = model(u, i)
            loss = loss_fn(logit, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        auc = evaluate(model, test_loader, device)
        mean_loss = float(np.mean(losses))

        print(f"Epoch {ep}: loss={mean_loss:.4f}, AUC={auc:.4f}")

        records.append({
            "dataset": args.dataset,
            "epoch": ep,
            "loss": mean_loss,
            "auc": auc
        })

    out = pd.DataFrame(records)
    out.to_csv(f"outputs/tables/baseline_{args.dataset}.csv", index=False)
    torch.save(model.state_dict(), f"models/baseline_{args.dataset}.pt")

    print("Saved:", f"outputs/tables/baseline_{args.dataset}.csv")

if __name__ == "__main__":
    main()
