"""Pravah Criticality Index (PCI): which hubs would hurt the network most if they failed?

Classical centralities (degree, PageRank, betweenness) rank hubs by position in the graph. They do
not know that a disruption spreads in proportion to traffic share and downstream vulnerability.
PCI is the expected share of network traffic exposed when a hub is disrupted at severity 0.8,
estimated with the cascade simulator (and, as a fast approximation, with the propagation GNN).

Outputs
  novelty/artifacts/criticality.csv   one row per hub (used by the dashboard)
  novelty/reports/criticality_report.txt, fig_criticality.png
Run from the repo root:  python -m novelty.criticality
"""
import os
import json
import numpy as np
import pandas as pd
import networkx as nx
import torch
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graph.cascade import Network, shares, simulate
from gnn.propagation import MODEL_PATH, node_inputs, predict
from rl.env import load_gnn

HERE = os.path.dirname(os.path.abspath(__file__))
ART, REP = os.path.join(HERE, "artifacts"), os.path.join(HERE, "reports")
SEVERITY, RUNS = 0.8, 150


def build_nx(net):
    g = nx.DiGraph()
    g.add_nodes_from(range(net.n))
    for a, b, w in zip(net.src, net.dst, net.w):
        g.add_edge(int(a), int(b), weight=float(w), dist=1.0 / max(float(w), 1e-9))
    return g


def pci_all(net, gnn=None, severity=SEVERITY, runs=RUNS, seed=0):
    """Per hub: expected traffic share exposed (simulator and, optionally, GNN) and expected #hubs."""
    rng = np.random.default_rng(seed)
    traffic = net.inbound + np.bincount(net.src, weights=net.w, minlength=net.n)
    total = traffic.sum()
    rows = []
    for h in range(net.n):
        if not net.out_edges[h]:
            rows.append((h, traffic[h] / total * 100, 1.0, traffic[h] / total * 100))
            continue
        ego = net.ego(h)
        p = shares(ego, ego["src"], ego["dst"], ego["w"])
        prob = simulate(net.vuln[ego["nodes"]], ego["src"], ego["dst"], p, severity, runs=runs, rng=rng)
        nodes = ego["nodes"]
        exposed = (prob * traffic[nodes]).sum() / total * 100
        gnn_exposed = np.nan
        if gnn is not None:
            r = predict(gnn, torch.tensor(node_inputs(net, ego, severity)),
                        torch.tensor(np.vstack([ego["src"], ego["dst"]]), dtype=torch.long),
                        torch.tensor(p, dtype=torch.float32))
            gnn_exposed = (r * traffic[nodes]).sum() / total * 100
        rows.append((h, exposed, prob.sum(), gnn_exposed))
    return pd.DataFrame(rows, columns=["idx", "pci", "exp_hubs", "pci_gnn"]).set_index("idx")


def attack_curve(net, order, ks):
    """Share of shipments (legs) still served when the first k hubs of `order` are removed,
    and share of hubs in the largest weakly connected component."""
    src, dst, w = net.src, net.dst, net.w
    total = w.sum()
    served, lcc = [], []
    for k in ks:
        dead = np.zeros(net.n, bool)
        dead[order[:k]] = True
        ok = ~dead[src] & ~dead[dst]
        served.append(w[ok].sum() / total)
        g = nx.Graph()
        g.add_nodes_from(np.flatnonzero(~dead).tolist())
        g.add_edges_from(zip(src[ok].tolist(), dst[ok].tolist()))
        lcc.append(max(len(c) for c in nx.connected_components(g)) / (net.n - k))
    return np.array(served), np.array(lcc)


def main():
    os.makedirs(ART, exist_ok=True)
    os.makedirs(REP, exist_ok=True)
    net, gnn = Network(), load_gnn(MODEL_PATH)
    pci = pci_all(net, gnn)
    g = build_nx(net)
    deg = np.array([g.out_degree(i) + g.in_degree(i) for i in range(net.n)], float)
    strength = net.inbound + np.bincount(net.src, weights=net.w, minlength=net.n)
    pr = nx.pagerank(g, weight="weight")
    bet = nx.betweenness_centrality(g, k=300, weight="dist", seed=0)
    df = pd.DataFrame({"center": net.centers, "hub": net.names, "state": net.state,
                       "lat": net.lat, "lon": net.lon, "pci": pci["pci"].values,
                       "pci_gnn": pci["pci_gnn"].values, "exp_hubs": pci["exp_hubs"].values,
                       "traffic": strength, "degree": deg,
                       "pagerank": [pr[i] for i in range(net.n)],
                       "betweenness": [bet[i] for i in range(net.n)],
                       "delay_risk": net.risk})
    df["rank_pci"] = df["pci"].rank(ascending=False)
    df["rank_degree"] = df["degree"].rank(ascending=False)
    df["rank_betw"] = df["betweenness"].rank(ascending=False)
    df.to_csv(os.path.join(ART, "criticality.csv"), index=False)

    lines = [f"Pravah Criticality Index: expected share of network traffic exposed when a hub is disrupted "
             f"at severity {SEVERITY} ({RUNS} Monte Carlo runs per hub, {net.n:,} hubs)",
             "cascade rules are ASSUMED (see graph/cascade.py); traffic, graph and hub risk are real\n"]
    cmp = {"degree": "degree", "traffic": "traffic (legs through hub)", "pagerank": "PageRank",
           "betweenness": "betweenness"}
    lines.append("Spearman rank correlation of PCI with classical measures (1 = identical ranking):")
    corrs = {}
    for k, nm in cmp.items():
        rho = spearmanr(df["pci"], df[k])[0]
        corrs[k] = float(rho)
        lines.append(f"  PCI vs {nm:<28}{rho:>6.2f}")
    ok = df["pci_gnn"].notna()
    rho_gnn = spearmanr(df.loc[ok, "pci"], df.loc[ok, "pci_gnn"])[0]
    lines.append(f"  PCI (simulator) vs PCI (GNN surrogate): {rho_gnn:.2f}  (the GNN gives the same ranking)")

    top = df.sort_values("pci", ascending=False).head(15)
    lines += ["", "Top 15 hubs by PCI", f"{'hub':<40}{'PCI %':>7}{'exp.hubs':>9}{'rank deg':>9}{'rank betw':>10}"]
    for _, r in top.iterrows():
        lines.append(f"{r['hub'][:38]:<40}{r['pci']:>7.2f}{r['exp_hubs']:>9.1f}{r['rank_degree']:>9.0f}{r['rank_betw']:>10.0f}")

    hidden = df[(df["rank_pci"] <= 50) & (df["rank_degree"] > 150)]
    lines += ["", f"'Hidden' critical hubs: top-50 by PCI but outside the top-150 by degree: {len(hidden)}"]
    for _, r in hidden.sort_values("pci", ascending=False).head(8).iterrows():
        lines.append(f"  {r['hub'][:40]:<42}PCI {r['pci']:.2f}%  degree rank {r['rank_degree']:.0f}")

    # robustness under targeted removal
    ks = [0, 5, 10, 20, 40, 80]
    orders = {"PCI": np.argsort(-df["pci"].values), "traffic": np.argsort(-df["traffic"].values),
              "betweenness": np.argsort(-df["betweenness"].values)}
    curves = {k: attack_curve(net, o, ks) for k, o in orders.items()}
    rng = np.random.default_rng(0)
    rand = [attack_curve(net, rng.permutation(net.n), ks) for _ in range(8)]
    curves["random"] = (np.mean([r[0] for r in rand], 0), np.mean([r[1] for r in rand], 0))
    lines += ["", "Targeted removal: share of shipments still served after removing the top-k hubs",
              f"{'k':<6}" + "".join(f"{k:>12}" for k in curves)]
    for i, k in enumerate(ks):
        lines.append(f"{k:<6}" + "".join(f"{curves[c][0][i]:>12.1%}" for c in curves))
    lines.append("(lower = the ordering finds hubs the network depends on more)")
    open(os.path.join(REP, "criticality_report.txt"), "w").write("\n".join(lines) + "\n")
    json.dump({"spearman": corrs, "spearman_gnn": float(rho_gnn),
               "attack": {c: {"k": ks, "served": v[0].tolist(), "lcc": v[1].tolist()} for c, v in curves.items()}},
              open(os.path.join(ART, "criticality_summary.json"), "w"), indent=1)
    print("\n".join(lines))
    figure(df, curves, ks)


def figure(df, curves, ks):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    sc = ax[0].scatter(df["lon"], df["lat"], c=df["pci"], s=6 + 90 * df["pci"] / df["pci"].max(),
                       cmap="YlOrRd", alpha=0.85, edgecolors="none")
    ax[0].set_title("Hubs coloured by Pravah Criticality Index (%)", fontsize=10)
    ax[0].set_xlim(67, 98)
    ax[0].set_ylim(6, 36)
    ax[0].set_aspect("equal")
    plt.colorbar(sc, ax=ax[0], shrink=0.7)
    ax[1].scatter(df["rank_degree"], df["rank_pci"], s=6, alpha=0.5, color="#8FA9A3")
    ax[1].set_xlim(0, 400)
    ax[1].set_ylim(0, 400)
    ax[1].invert_xaxis()
    ax[1].set_xlabel("rank by degree")
    ax[1].set_ylabel("rank by PCI")
    ax[1].set_title("PCI finds hubs that degree misses (off-diagonal)", fontsize=10)
    ax[1].plot([0, 400], [0, 400], "--", color="#1A1815", lw=0.7)
    colors = {"PCI": "#C4622A", "traffic": "#2F6B4F", "betweenness": "#7A5C99", "random": "#B8A88A"}
    for k, (served, _) in curves.items():
        ax[2].plot(ks, np.array(served) * 100, "o-", label=k, color=colors[k])
    ax[2].set_xlabel("hubs removed (k)")
    ax[2].set_ylabel("shipments still served (%)")
    ax[2].set_title("Robustness under targeted hub failure", fontsize=10)
    ax[2].legend(frameon=False)
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(REP, "fig_criticality.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
