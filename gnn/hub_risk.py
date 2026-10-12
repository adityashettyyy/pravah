"""Hub delay-proneness: GraphSAGE vs XGBoost on real hub labels.

Label: the hub is 'high delay' if more than 25% of its outbound legs are delayed
(only hubs with 20+ outbound legs have a reliable rate). Every hub is scored
once, labelled hubs out-of-fold, so the scores can be used for all 1,594 hubs.
Run from the repo root: python -m gnn.hub_risk
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from torch_geometric.nn import SAGEConv
from xgboost import XGBClassifier

from gnn.common import load_graph, undirected

HERE = os.path.dirname(os.path.abspath(__file__))
HIGH_RATE = 0.25
SEED = 0


class SAGE(torch.nn.Module):
    def __init__(self, d_in, hidden=32):
        super().__init__()
        self.c1 = SAGEConv(d_in, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x, ei):
        h = F.dropout(F.relu(self.c1(x, ei)), 0.3, self.training)
        h = F.relu(self.c2(h, ei))
        return self.out(h).squeeze(-1)


def fit_sage(x, ei, y, train_idx, epochs=150, seed=0):
    torch.manual_seed(seed)
    model = SAGE(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(model(x, ei)[train_idx], y[train_idx])
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, ei)).numpy()


def fit_xgb(x, y, train_idx, seed=0):
    m = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=1)
    m.fit(x[train_idx], y[train_idx])
    return m.predict_proba(x)[:, 1]


def state_baseline(states, y, train_idx):
    df = pd.DataFrame({"s": np.array(states)[train_idx], "y": y[train_idx]})
    rate = df.groupby("s")["y"].mean()
    return np.array([rate.get(s, y[train_idx].mean()) for s in states])


def main():
    g = load_graph()
    x = g["x"]
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    ei = undirected(g["edge_index"])
    lab = np.where(g["labeled"].numpy())[0]
    y_all = (g["out_rate"].numpy() > HIGH_RATE).astype(np.float32)
    y_all[~g["labeled"].numpy()] = 0
    y_t = torch.tensor(y_all)
    x_np = x.numpy()

    methods = ["Majority / state mean", "XGBoost (no graph)", "GraphSAGE"]
    per_repeat = {m: [] for m in methods}
    ensemble = {m: np.zeros(len(x)) for m in methods}
    n_models = 0

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    oof = {m: np.zeros(len(lab)) for m in methods}
    for k, (tr, te) in enumerate(cv.split(lab, y_all[lab])):
        tr_idx, te_idx = lab[tr], lab[te]
        preds = {
            "Majority / state mean": state_baseline(g["state"], y_all, tr_idx),
            "XGBoost (no graph)": fit_xgb(x_np, y_all, tr_idx, seed=k),
            "GraphSAGE": fit_sage(x, ei, y_t, torch.tensor(tr_idx), seed=k),
        }
        for m in methods:
            oof[m][te] = preds[m][te_idx]
            ensemble[m] += preds[m]
        n_models += 1
        if k % 5 == 4:
            for m in methods:
                per_repeat[m].append(oof[m].copy())

    rows = []
    for m in methods:
        auc = [roc_auc_score(y_all[lab], p) for p in per_repeat[m]]
        pr = [average_precision_score(y_all[lab], p) for p in per_repeat[m]]
        prec = np.mean([precision_score(y_all[lab], p > 0.5, zero_division=0) for p in per_repeat[m]])
        rec = np.mean([recall_score(y_all[lab], p > 0.5) for p in per_repeat[m]])
        rows.append((m, np.mean(auc), np.std(auc), np.mean(pr), prec, rec))

    lines = [f"labelled hubs: {len(lab)}, high-delay (>{HIGH_RATE:.0%} of legs delayed): "
             f"{y_all[lab].mean():.0%}",
             "5-fold CV over hubs, 3 repeats; hubs in the test fold are unseen during training\n",
             f"{'method':<24}{'roc_auc':>9}{'(std)':>8}{'pr_auc':>8}{'precision':>10}{'recall':>8}"]
    for m, a, s, p, pr_, rc in rows:
        lines.append(f"{m:<24}{a:>9.3f}{s:>8.3f}{p:>8.3f}{pr_:>10.3f}{rc:>8.3f}")

    best = max(rows[1:], key=lambda r: r[1])[0]
    risk = {m: ensemble[m] / n_models for m in methods}
    for m in methods:
        risk[m][lab] = np.mean(per_repeat[m], axis=0)
    out = pd.DataFrame({"center": g["centers"], "name": g["names"], "n_out_legs": g["n_out"].numpy().astype(int),
                        "observed_rate": g["out_rate"].numpy(), "risk_sage": risk["GraphSAGE"],
                        "risk_xgb": risk["XGBoost (no graph)"]})
    out["risk"] = out["risk_sage"] if best == "GraphSAGE" else out["risk_xgb"]
    out.to_csv(os.path.join(HERE, "artifacts", "hub_risk.csv"), index=False)
    lines.append(f"\nscores used downstream: {best}")
    lines.append(f"scored {len(out):,} hubs; {len(lab)} labelled ones out-of-fold, "
                 f"{len(out) - len(lab):,} unlabelled ones by the model average")
    with open(os.path.join(HERE, "reports", "hub_risk_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
