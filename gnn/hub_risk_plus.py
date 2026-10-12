"""Does the graph help predict a hub's delay-proneness? A fairer test than gnn/hub_risk.py.

gnn/hub_risk.py compared GraphSAGE with XGBoost on 20 hub-level features and the graph did not
help. Message passing over weak features is not the only way to use a graph though. The evidence
(analysis/reports/propagation_evidence.txt) says delay-proneness is regional: a hub's rate correlates
0.39 with its neighbours' and 0.43 with its state's. A model that is allowed to look at the
OBSERVED delay rates of nearby training hubs can exploit that, and message passing over node
features cannot. So this adds

  * GAT (the other architecture named in the case study),
  * a hybrid: XGBoost + label-aware neighbourhood features, computed only from the training hubs of
    each fold (graph neighbours' mean rate, state mean rate, inverse-distance mean of the k nearest
    training hubs). Training rows exclude themselves, so nothing leaks from a row's own label.

Same protocol as hub_risk.py: 5-fold CV over labelled hubs, 3 repeats, test-fold hubs unseen.
Run from the repo root:  python -m gnn.hub_risk_plus
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.nn import GATConv
from xgboost import XGBClassifier

from gnn.common import load_graph, undirected
from gnn.hub_risk import fit_sage, fit_xgb, HIGH_RATE, SEED

HERE = os.path.dirname(os.path.abspath(__file__))


class GAT(torch.nn.Module):
    def __init__(self, d_in, hidden=16, heads=4):
        super().__init__()
        self.c1 = GATConv(d_in, hidden, heads=heads, dropout=0.3)
        self.c2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.3)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x, ei):
        h = F.elu(self.c1(x, ei))
        h = F.elu(self.c2(h, ei))
        return self.out(h).squeeze(-1)


def fit_gat(x, ei, y, train_idx, epochs=120, seed=0):
    torch.manual_seed(seed)
    model = GAT(x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        F.binary_cross_entropy_with_logits(model(x, ei)[train_idx], y[train_idx]).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, ei)).numpy()


def haversine_matrix(lat, lon):
    p = np.pi / 180
    la, lo = lat[:, None] * p, lon[:, None] * p
    a = np.sin((la.T - la) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo.T - lo) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def neighbour_features(rate, known, adj, state_ids, dist, k=8):
    """Label-aware features for every hub from hubs flagged in `known` (the training hubs).
    A hub never uses its own rate."""
    n = len(rate)
    kn = known.astype(float)
    r = np.where(known, rate, 0.0)
    cnt = adj @ kn
    nb_mean = np.where(cnt > 0, (adj @ r) / np.maximum(cnt, 1), np.nan)
    st_sum = np.bincount(state_ids, weights=r, minlength=state_ids.max() + 1)
    st_cnt = np.bincount(state_ids, weights=kn, minlength=state_ids.max() + 1)
    own_r, own_k = r, kn
    st_mean = np.where(st_cnt[state_ids] - own_k > 0,
                       (st_sum[state_ids] - own_r) / np.maximum(st_cnt[state_ids] - own_k, 1), np.nan)
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    d[:, ~known] = np.inf
    nn = np.argsort(d, axis=1)[:, :k]
    dd = np.take_along_axis(d, nn, 1)
    w = 1.0 / (dd + 25.0)
    knn = (w * r[nn]).sum(1) / w.sum(1)
    return np.column_stack([nb_mean, cnt, st_mean, knn, dd.mean(1)])


def main():
    g = load_graph()
    x = g["x"]
    xs = (x - x.mean(0)) / (x.std(0) + 1e-6)
    ei = undirected(g["edge_index"])
    n = x.shape[0]
    lab = np.where(g["labeled"].numpy())[0]
    rate = g["out_rate"].numpy().astype(float)
    y_all = (rate > HIGH_RATE).astype(np.float32)
    y_all[~g["labeled"].numpy()] = 0
    y_t = torch.tensor(y_all)
    x_np = xs.numpy()
    adj = np.zeros((n, n))
    adj[ei[0].numpy(), ei[1].numpy()] = 1.0
    states = pd.factorize(np.array(g["state"]))[0]
    dist = haversine_matrix(g["lat"].numpy().astype(float), g["lon"].numpy().astype(float))

    methods = ["State mean", "XGBoost (hub features)", "GraphSAGE", "GAT",
               "XGBoost + neighbourhood labels (hybrid)", "Neighbourhood labels only"]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    oof = {m: np.zeros((3, len(lab))) for m in methods}
    for k, (tr, te) in enumerate(cv.split(lab, y_all[lab])):
        rep = k // 5
        tr_idx, te_idx = lab[tr], lab[te]
        known = np.zeros(n, bool)
        known[tr_idx] = True
        nbf = neighbour_features(rate, known, adj, states, dist)
        x_aug = np.column_stack([x_np, nbf])
        st = pd.DataFrame({"s": states[tr_idx], "y": y_all[tr_idx]}).groupby("s")["y"].mean()
        preds = {
            "State mean": np.array([st.get(s, y_all[tr_idx].mean()) for s in states]),
            "XGBoost (hub features)": fit_xgb(x_np, y_all, tr_idx, seed=k),
            "GraphSAGE": fit_sage(xs, ei, y_t, torch.tensor(tr_idx), seed=k),
            "GAT": fit_gat(xs, ei, y_t, torch.tensor(tr_idx), seed=k),
            "XGBoost + neighbourhood labels (hybrid)": fit_xgb(np.nan_to_num(x_aug, nan=-1), y_all, tr_idx, seed=k),
            "Neighbourhood labels only": fit_xgb(np.nan_to_num(nbf, nan=-1), y_all, tr_idx, seed=k),
        }
        for m in methods:
            oof[m][rep, te] = preds[m][te_idx]

    y_lab = y_all[lab]
    lines = [f"{len(lab)} labelled hubs, high-delay = more than {HIGH_RATE:.0%} of outbound legs delayed "
             f"({y_lab.mean():.0%} of hubs)", "5-fold CV over hubs x 3 repeats; test-fold hubs unseen in training; "
             "neighbourhood features use training-fold labels only\n",
             f"{'method':<42}{'ROC-AUC':>9}{'(sd)':>7}{'PR-AUC':>8}"]
    res = {}
    for m in methods:
        auc = [roc_auc_score(y_lab, oof[m][r]) for r in range(3)]
        pr = [average_precision_score(y_lab, oof[m][r]) for r in range(3)]
        res[m] = auc
        lines.append(f"{m:<42}{np.mean(auc):>9.3f}{np.std(auc):>7.3f}{np.mean(pr):>8.3f}")
    gain = np.mean(res["XGBoost + neighbourhood labels (hybrid)"]) - np.mean(res["XGBoost (hub features)"])
    lines += ["", f"hybrid minus plain XGBoost: {gain:+.3f} ROC-AUC",
              "message passing (GraphSAGE / GAT) over the 20 hub features does not use neighbours' observed",
              "delay rates; the hybrid does, which is why the graph/geography helps there and not in the GNNs."]
    out = "\n".join(lines)
    open(os.path.join(HERE, "reports", "hub_risk_plus_report.txt"), "w").write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
