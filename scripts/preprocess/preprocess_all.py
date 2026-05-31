import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def process_movielens():
    path = os.path.join(RAW_DIR, "ml-1m", "ratings.dat")
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    df["label"] = (df["rating"] >= 4).astype(int)
    df = df[["user_id", "item_id", "timestamp", "label"]]
    df["dataset"] = "movielens"

    df["user_id"] = "ml_u_" + df["user_id"].astype(str)
    df["item_id"] = "ml_i_" + df["item_id"].astype(str)

    return df


def process_amazon():
    path = os.path.join(RAW_DIR, "amazon_beauty", "reviews_Beauty_5.json")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            rows.append([
                x.get("reviewerID"),
                x.get("asin"),
                x.get("unixReviewTime"),
                1 if float(x.get("overall", 0)) >= 4 else 0
            ])
    df = pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp", "label"])
    df = df.dropna()
    df["dataset"] = "amazon_beauty"

    df["user_id"] = "az_u_" + df["user_id"].astype(str)
    df["item_id"] = "az_i_" + df["item_id"].astype(str)

    return df


def remap_ids(df):
    user_map = {u: i for i, u in enumerate(df["user_id"].unique())}
    item_map = {v: i for i, v in enumerate(df["item_id"].unique())}

    df["user"] = df["user_id"].map(user_map)
    df["item"] = df["item_id"].map(item_map)

    return df, user_map, item_map


def split_data(df, name):
    df = df.sort_values(["user", "timestamp"])

    train_list, test_list = [], []

    for _, g in df.groupby("user"):
        if len(g) < 3:
            train_list.append(g)
        else:
            test_list.append(g.tail(1))
            train_list.append(g.iloc[:-1])

    train = pd.concat(train_list)
    test = pd.concat(test_list) if len(test_list) > 0 else df.sample(frac=0.2, random_state=42)

    train.to_csv(os.path.join(OUT_DIR, f"{name}_train.csv"), index=False)
    test.to_csv(os.path.join(OUT_DIR, f"{name}_test.csv"), index=False)

    info = {
        "dataset": name,
        "num_users": int(df["user"].nunique()),
        "num_items": int(df["item"].nunique()),
        "num_interactions": int(len(df)),
        "train_size": int(len(train)),
        "test_size": int(len(test)),
        "positive_rate": float(df["label"].mean())
    }

    pd.DataFrame([info]).to_csv(os.path.join(OUT_DIR, f"{name}_info.csv"), index=False)

    print(f"\n===== {name} =====")
    print(info)


def main():
    ml = process_movielens()
    ml, _, _ = remap_ids(ml)
    split_data(ml, "movielens")

    az = process_amazon()
    az, _, _ = remap_ids(az)
    split_data(az, "amazon_beauty")

    print("\nDONE: preprocessing finished.")


if __name__ == "__main__":
    main()
