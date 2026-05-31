import os
import argparse
import random
import copy
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
        return roc_auc_score(ys, ps)
    except:
        return 0.5

def split_clients(df, num_clients=10):
    users = df["user"].unique()
    np.random.shuffle(users)
    user_splits = np.array_split(users, num_clients)
    clients = []
    for us in user_splits:
        clients.append(df[df["user"].isin(us)].copy())
    return clients

def average_weights(weights, sizes):
    total = sum(sizes)
    avg = copy.deepcopy(weights[0])
    for k in avg.keys():
        avg[k] = sum(weights[i][k] * (sizes[i] / total) for i in range(len(weights)))
    return avg

def local_train(model, df, device, batch_size, lr, local_epochs):
    loader = DataLoader(RecDataset(df), batch_size=batch_size, shuffle=True)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    losses = []
    for _ in range(local_epochs):
        for u, i, y in loader:
            u, i, y = u.to(device), i.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(u, i), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
    return model.state_dict(), float(np.mean(losses))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="movielens")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1024)
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

    clients = split_clients(train, args.clients)
    test_loader = DataLoader(RecDataset(test), batch_size=2048, shuffle=False)

    global_model = MFMLP(num_users, num_items).to(device)

    records = []

    for r in range(1, args.rounds + 1):
        local_weights, local_sizes, local_losses = [], [], []

        for cdf in tqdm(clients, desc=f"Round {r}"):
            local_model = copy.deepcopy(global_model).to(device)
            w, loss = local_train(
                local_model, cdf, device,
                args.batch_size, args.lr, args.local_epochs
            )
            local_weights.append(copy.deepcopy(w))
            local_sizes.append(len(cdf))
            local_losses.append(loss)

        new_state = average_weights(local_weights, local_sizes)
        global_model.load_state_dict(new_state)

        auc = evaluate(global_model, test_loader, device)
        mean_loss = float(np.mean(local_losses))

        print(f"Round {r}: loss={mean_loss:.4f}, AUC={auc:.4f}")

        records.append({
            "dataset": args.dataset,
            "round": r,
            "loss": mean_loss,
            "auc": auc,
            "clients": args.clients,
            "local_epochs": args.local_epochs
        })

    out = pd.DataFrame(records)
    out.to_csv(f"outputs/tables/fedavg_{args.dataset}.csv", index=False)
    torch.save(global_model.state_dict(), f"models/fedavg_{args.dataset}.pt")

    print("Saved:", f"outputs/tables/fedavg_{args.dataset}.csv")

if __name__ == "__main__":
    main()
