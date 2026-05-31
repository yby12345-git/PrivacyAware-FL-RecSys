import pandas as pd
import numpy as np

train = pd.read_csv("data/processed/movielens_train.csv")

users = train["user"].unique()

np.random.seed(42)

clients = 10

client_data = []

for cid in range(clients):
    selected = np.random.choice(
        users,
        size=int(len(users) * 0.2),
        replace=False
    )
    df = train[train["user"].isin(selected)].copy()

    popular_items = (
        df["item"]
        .value_counts()
        .head(200)
        .index
    )

    df = df[df["item"].isin(popular_items)]

    df["client"] = cid

    client_data.append(df)

out = pd.concat(client_data)

out.to_csv(
    "data/processed/movielens_noniid_train.csv",
    index=False
)

print(out.shape)
print(out["client"].value_counts())
