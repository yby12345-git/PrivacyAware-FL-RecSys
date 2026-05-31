import pandas as pd

fedavg = pd.read_csv("outputs/tables/fedavg_iid_movielens.csv").iloc[-1]
dp = pd.read_csv("outputs/tables/dp_fedavg_movielens_eps1.0.csv").iloc[-1]
noniid = pd.read_csv("outputs/tables/fedavg_noniid_movielens.csv").iloc[-1]

rows = [
    {"Variant": "Full Model", "AUC": round(dp["auc"], 4)},
    {"Variant": "w/o Differential Privacy", "AUC": round(fedavg["auc"], 4)},
    {"Variant": "w/o IID Distribution", "AUC": round(noniid["auc"], 4)}
]

out = pd.DataFrame(rows)
out.to_csv("outputs/tables/ablation_results.csv", index=False)
print(out)
