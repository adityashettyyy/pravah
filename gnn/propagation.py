"""Propagation GNN: given a disruption at one hub, predict every hub's risk of being hit.

Trained on scenarios from graph/cascade.py (so it learns the assumed dynamics, not
real cascades). It replaces 100 Monte Carlo runs by one forward pass and takes
edge flows as input, so the RL environment can change flows and re-query it.
Test sources are hubs never used as a disruption origin in training.
Run from the repo root: python -m gnn.propagation
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GraphConv
from xgboost import XGBRegressor

from graph.cascade import Network, shares, simulate, RUNS

torch.set_num_threads(1)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "artifacts", "propagation.pt")
AT_RISK = 0.2
N_TRAIN, N_VAL, N_TEST = 2500, 300, 500


class PropGNN(torch.nn.Module):
    def __init__(self, d_in, hidden=32, layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [GraphConv(d_in if i == 0 else hidden, hidden, aggr="add") for i in range(layers)])
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_weight):
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_weight))
        return self.out(x).squeeze(-1)


def node_inputs(net, ego, severity):
    n = len(ego["nodes"])
    is_src = np.zeros(n, np.float32)
    is_src[0] = 1
    return np.column_stack([net.x[ego["nodes"]], net.vuln[ego["nodes"]], is_src, is_src * severity]).astype(np.float32)


def perturb(ego, rng):
    """Random flow changes (scaling, dropped and added edges), like the RL agent's reroutes."""
    n = len(ego["nodes"])
    w = ego["w"] * rng.lognormal(0, 0.3, len(ego["w"]))
    keep = rng.random(len(w)) > 0.1
    src, dst, w = ego["src"][keep], ego["dst"][keep], w[keep]
    k = rng.integers(0, 4)
    if k and n > 2:
        a, b = rng.integers(0, n, k), rng.integers(1, n, k)
        ok = a != b
        src = np.r_[src, a[ok]]
        dst = np.r_[dst, b[ok]]
        w = np.r_[w, np.full(ok.sum(), np.median(ego["w"]) if len(ego["w"]) else 1.0)]
    return src, dst, w


def make_scenarios(net, sources, count, rng, weights=None):
    out = []
    for _ in range(count):
        s = rng.choice(sources, p=weights)
        ego = net.ego(s)
        sev = rng.uniform(0.3, 1.0)
        src, dst, w = perturb(ego, rng)
        p = shares(ego, src, dst, w)
        y = simulate(net.vuln[ego["nodes"]], src, dst, p, sev, rng=rng)
        out.append({"source": s, "ego": ego, "sev": sev, "src": src, "dst": dst, "p": p, "y": y})
    return out


def to_data(net, sc):
    return Data(x=torch.tensor(node_inputs(net, sc["ego"], sc["sev"])),
                edge_index=torch.tensor(np.vstack([sc["src"], sc["dst"]]), dtype=torch.long),
                edge_weight=torch.tensor(sc["p"], dtype=torch.float32),
                y=torch.tensor(sc["y"], dtype=torch.float32))


def predict(model, x, edge_index, edge_weight):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(x, edge_index, edge_weight)).numpy()


def train(net, tr, va, epochs=25):
    torch.manual_seed(0)
    model = PropGNN(tr[0].x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=0.005)
    loader = DataLoader(tr, batch_size=32, shuffle=True)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        for b in loader:
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(b.x, b.edge_index, b.edge_weight), b.y).backward()
            opt.step()
        val = np.mean([np.abs(predict(model, d.x, d.edge_index, d.edge_weight) - d.y.numpy()).mean() for d in va])
        if val < best:
            best, best_state = val, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def hop_baseline(sc):
    """Hop-decay heuristic: 0.5^hops from the source along edge directions, no learning."""
    n = len(sc["ego"]["nodes"])
    hops = np.full(n, np.inf)
    hops[0] = 0
    for h in range(1, 4):
        reach = np.isin(np.arange(n), sc["dst"][np.isfinite(hops[sc["src"]]) & (hops[sc["src"]] == h - 1)])
        hops[reach & ~np.isfinite(hops)] = h
    return np.where(np.isfinite(hops), 0.5 ** hops * sc["sev"], 0.0)


def xgb_features(net, sc):
    ego = sc["ego"]
    n = len(ego["nodes"])
    s = ego["nodes"][0]
    km = 111.0 * np.sqrt((net.lat[ego["nodes"]] - net.lat[s]) ** 2 +
                         (np.cos(np.radians(net.lat[s])) * (net.lon[ego["nodes"]] - net.lon[s])) ** 2)
    same = np.array([net.state[i] == net.state[s] for i in ego["nodes"]], float)
    return np.column_stack([node_inputs(net, ego, sc["sev"]), km, same,
                            np.full(n, len(net.out_edges[s]))])


def score(name, pred, sc_list, rows):
    mae = np.mean([np.abs(p[1:] - s["y"][1:]).mean() for p, s in zip(pred, sc_list)])
    y = np.concatenate([s["y"][1:] for s in sc_list]) > AT_RISK
    p = np.concatenate([q[1:] for q in pred])
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    p5 = []
    for q, s in zip(pred, sc_list):
        true = s["y"][1:]
        if (true > AT_RISK).sum() >= 1 and len(true) >= 5:
            k = min(5, (true > AT_RISK).sum())
            p5.append(len(set(np.argsort(-q[1:])[:k]) & set(np.argsort(-true)[:k])) / k)
    rho = np.mean([spearmanr(q[1:], s["y"][1:])[0] for q, s in zip(pred, sc_list)
                   if s["y"][1:].std() > 0 and q[1:].std() > 0])
    rows.append(f"{name:<26}{mae:>8.3f}{rho:>10.3f}{auc:>11.3f}{np.mean(p5):>13.3f}")


def evaluate(net, model, xgb, sc_list, rng, title, lines):
    data = [to_data(net, s) for s in sc_list]
    gnn_pred = [predict(model, d.x, d.edge_index, d.edge_weight) for d in data]
    xgb_pred = [np.clip(xgb.predict(xgb_features(net, s)), 0, 1) for s in sc_list]
    rerun = [simulate(net.vuln[s["ego"]["nodes"]], s["src"], s["dst"], s["p"], s["sev"], rng=rng)
             for s in sc_list]
    sizes = [len(s["ego"]["nodes"]) for s in sc_list]
    rows = [f"{title} ({len(sc_list)} scenarios, local graph median {int(np.median(sizes))} hubs, "
            f"max {max(sizes)})",
            f"{'method':<26}{'MAE':>8}{'Spearman':>10}{'AUC(>0.2)':>11}{'precision@5':>13}"]
    score("GNN surrogate", gnn_pred, sc_list, rows)
    score("XGBoost (no edges/flows)", xgb_pred, sc_list, rows)
    score("Hop-decay heuristic", [hop_baseline(s) for s in sc_list], sc_list, rows)
    score("Simulator, second run", rerun, sc_list, rows)
    lines += rows + [""]


def latency(net, model, sc, rng):
    d = to_data(net, sc)
    t = time.time()
    for _ in range(50):
        predict(model, d.x, d.edge_index, d.edge_weight)
    gnn_ms = (time.time() - t) / 50 * 1000
    t = time.time()
    for _ in range(20):
        simulate(net.vuln[sc["ego"]["nodes"]], sc["src"], sc["dst"], sc["p"], sc["sev"], rng=rng)
    return gnn_ms, (time.time() - t) / 20 * 1000


def main():
    net = Network()
    rng = np.random.default_rng(1)
    sources = np.array([i for i in range(net.n) if net.out_edges[i]])
    rng.shuffle(sources)
    k1, k2 = int(0.7 * len(sources)), int(0.85 * len(sources))
    tr_s, va_s, te_s = sources[:k1], sources[k1:k2], sources[k2:]
    # major hubs are rare among sources but are the disruptions that matter, so oversample them
    w = np.array([1.0 + len(net.out_edges[i]) for i in tr_s])
    w /= w.sum()

    tr = make_scenarios(net, tr_s, N_TRAIN, rng, weights=w)
    va = make_scenarios(net, va_s, N_VAL, rng)
    te_all = make_scenarios(net, te_s, N_TEST, rng)
    major = np.array([i for i in te_s if len(net.ego(i)["nodes"]) >= 30])
    te_major = make_scenarios(net, major, 200, rng)

    model, val_mae = train(net, [to_data(net, s) for s in tr], [to_data(net, s) for s in va])
    torch.save({"state": model.state_dict(), "d_in": node_inputs(net, tr[0]["ego"], 0.5).shape[1]}, MODEL_PATH)

    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, n_jobs=1, random_state=0)
    xgb.fit(np.vstack([xgb_features(net, s) for s in tr]), np.concatenate([s["y"] for s in tr]))

    lines = [f"source hubs: train {len(tr_s)} / val {len(va_s)} / test {len(te_s)} (test hubs never seen as an origin)",
             f"scenarios: train {N_TRAIN} (oversampling major hubs), val {N_VAL}; "
             f"labels = share of {RUNS} simulation runs",
             f"test hubs with a local graph of 30+ hubs: {len(major)}",
             f"validation MAE of chosen checkpoint: {val_mae:.3f}\n"]
    evaluate(net, model, xgb, te_all, rng, "all source hubs", lines)
    evaluate(net, model, xgb, te_major, rng, "major source hubs only", lines)
    big = max(te_major, key=lambda s: len(s["ego"]["nodes"]))
    g_ms, s_ms = latency(net, model, big, rng)
    lines += ["'Simulator, second run' is the noise floor: two independent runs of the same scenario.",
              "The GNN learns the assumed dynamics. XGBoost cannot see edges or flows, which the",
              "simulator depends on, so that gap only shows the graph structure is needed.",
              f"\nlatency on the largest local graph ({len(big['ego']['nodes'])} hubs): "
              f"GNN {g_ms:.1f} ms, simulator ({RUNS} runs) {s_ms:.1f} ms"]
    with open(os.path.join(HERE, "reports", "propagation_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
