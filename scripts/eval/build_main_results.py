import pandas as pd

baseline = pd.read_csv(
    "outputs/tables/baseline_movielens.csv"
).iloc[-1]

fedavg = pd.read_csv(
    "outputs/tables/fedavg_iid_movielens.csv"
).iloc[-1]

noniid = pd.read_csv(
    "outputs/tables/fedavg_noniid_movielens.csv"
).iloc[-1]

dp = pd.read_csv(
    "outputs/tables/dp_fedavg_movielens_eps1.0.csv"
).iloc[-1]

rows = [
    {
        "Method": "Centralized MLP",
        "Setting": "Centralized",
        "AUC": round(baseline["auc"], 4)
    },
    {
        "Method": "FedAvg",
        "Setting": "IID",
        "AUC": round(fedavg["auc"], 4)
    },
    {
        "Method": "FedAvg",
        "Setting": "non-IID",
        "AUC": round(noniid["auc"], 4)
    },
    {
        "Method": "DP-FedAvg",
        "Setting": "IID + DP",
        "AUC": round(dp["auc"], 4)
    }
]

out = pd.DataFrame(rows)

out.to_csv(
    "outputs/tables/main_results.csv",
    index=False
)

print(out)
