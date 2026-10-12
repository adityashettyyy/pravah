"""Monte Carlo cascade simulator on the hub graph (what-if tool).

The data shows no measurable hub-to-hub delay cascade (analysis/propagation_check.py),
so the propagation rules here are ASSUMED, not fitted. What comes from real data:
the graph, the traffic on each edge and each hub's delay-proneness.

A disruption starts at one hub with severity s in (0, 1]. For 3 hops, every newly
disrupted hub a can disrupt a downstream hub b with probability
    min(1, BETA * s_a * share(a->b) * vulnerability(b))
share = a's part of b's inbound traffic, vulnerability = 0.3 + 0.7 * hub risk score.
A hub disrupted by a gets severity s_a * DECAY. The output is, per hub, the share
of simulation runs in which it got disrupted.
"""
import os
import numpy as np
import pandas as pd
import torch

from paths import GRAPH_PATH

RISK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "gnn", "artifacts", "hub_risk.csv")
BETA, DECAY, HOPS, RUNS, MAX_NODES = 1.5, 0.7, 3, 100, 120
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


class Network:
    def __init__(self):
        g = torch.load(GRAPH_PATH, weights_only=False)
        self.g = g
        self.centers, self.names, self.state = g["centers"], g["names"], g["state"]
        self.n = len(self.centers)
        self.src, self.dst = g["edge_index"].numpy()
        self.w = g["edge_legs"].numpy().astype(float)
        self.inbound = np.bincount(self.dst, weights=self.w, minlength=self.n)
        self.out_edges = [[] for _ in range(self.n)]
        for e, a in enumerate(self.src):
            self.out_edges[a].append(e)
        risk = pd.read_csv(RISK_PATH).set_index("center")["risk"].reindex(self.centers).values
        self.risk = risk
        self.vuln = 0.3 + 0.7 * risk
        x = g["x"]
        self.x = ((x - x.mean(0)) / (x.std(0) + 1e-6)).numpy()
        self.lat, self.lon = g["lat"].numpy(), g["lon"].numpy()
        self._ego = {}

    def find(self, text):
        hits = [i for i, nm in enumerate(self.names) if text.lower() in str(nm).lower()]
        if not hits:
            raise KeyError(f"no hub matching '{text}'")
        return max(hits, key=lambda i: self.inbound[i] + sum(self.w[e] for e in self.out_edges[i]))

    def ego(self, source):
        """Hubs within HOPS downstream hops of source (capped at MAX_NODES), source first."""
        if source in self._ego:
            return self._ego[source]
        nodes, seen, layer = [source], {source}, [source]
        for _ in range(HOPS):
            cand = {}
            for a in layer:
                for e in self.out_edges[a]:
                    b = self.dst[e]
                    if b not in seen:
                        cand[b] = max(cand.get(b, 0), self.w[e])
            layer = [b for b, _ in sorted(cand.items(), key=lambda kv: -kv[1])][:MAX_NODES - len(nodes)]
            nodes += layer
            seen.update(layer)
            if len(nodes) >= MAX_NODES:
                break
        ego = self.subgraph(nodes)
        self._ego[source] = ego
        return ego

    def subgraph(self, nodes):
        """Local graph on the given hubs (first one is the disruption source)."""
        nodes = np.array(nodes)
        local = -np.ones(self.n, dtype=int)
        local[nodes] = np.arange(len(nodes))
        keep = (local[self.src] >= 0) & (local[self.dst] >= 0)
        return {"nodes": nodes, "src": local[self.src[keep]], "dst": local[self.dst[keep]],
                "w": self.w[keep], "inbound": self.inbound[nodes]}


def shares(ego, src, dst, w):
    """Each edge's share of its destination's inbound traffic, given (possibly changed) flows."""
    n = len(ego["nodes"])
    total = ego["inbound"] - np.bincount(ego["dst"], weights=ego["w"], minlength=n) \
        + np.bincount(dst, weights=w, minlength=n)
    return np.clip(w / np.maximum(total[dst], 1e-6), 0, 1)


def simulate(vuln, src, dst, p, severity, runs=RUNS, beta=BETA, rng=None):
    """Probability that each local hub (index 0 is the source) gets disrupted."""
    rng = rng or np.random.default_rng()
    n, E = len(vuln), len(src)
    if E == 0:
        out = np.zeros(n)
        out[0] = 1.0
        return out
    order = np.argsort(dst, kind="stable")
    s, d, pp = src[order], dst[order], p[order]
    starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    targets = d[starts]
    active = np.zeros((runs, n), bool)
    sev = np.zeros((runs, n))
    active[:, 0], sev[:, 0] = True, severity
    frontier = active.copy()
    for _ in range(HOPS):
        q = np.clip(beta * sev[:, s] * pp * vuln[d], 0, 1)
        hit = (rng.random((runs, E)) < q) & frontier[:, s]
        best = np.maximum.reduceat(np.where(hit, sev[:, s] * DECAY, 0.0), starts, axis=1)
        new = np.zeros((runs, n))
        new[:, targets] = best
        newly = (new > 0) & ~active
        sev[newly] = new[newly]
        active |= newly
        frontier = newly
    return active.mean(0)


def sensitivity(net, n_sources=40, severity=0.8):
    top = np.argsort(-net.inbound)[:n_sources]
    betas = [0.5, 1.5, 3.0]
    counts = {b: [] for b in betas}
    ranks = {b: [] for b in betas}
    rng = np.random.default_rng(0)
    for s in top:
        ego = net.ego(s)
        p = shares(ego, ego["src"], ego["dst"], ego["w"])
        for b in betas:
            prob = simulate(net.vuln[ego["nodes"]], ego["src"], ego["dst"], p, severity, beta=b, rng=rng)
            counts[b].append((prob[1:] > 0.2).sum())
            ranks[b].append(set(np.argsort(-prob[1:])[:5]))
    lines = [f"disruption of severity {severity} at each of the {n_sources} busiest hubs",
             f"assumed: BETA={BETA}, DECAY={DECAY}, {HOPS} hops, {RUNS} runs\n",
             f"{'BETA':<6}{'hubs at risk (P>0.2), mean':>28}{'median':>8}{'max':>6}"]
    for b in betas:
        c = np.array(counts[b])
        lines.append(f"{b:<6}{c.mean():>28.1f}{np.median(c):>8.0f}{c.max():>6}")
    ov = np.mean([len(a & b) / 5 for a, b in zip(ranks[0.5], ranks[3.0]) if a and b])
    lines.append(f"\ntop-5 most affected hubs, BETA 0.5 vs 3.0: {ov:.0%} overlap")
    lines.append("BETA sets how far a disruption spreads. Which hubs are hit first depends mostly on")
    lines.append("the traffic structure, so the ranking is more robust than the spread size.")
    return lines


if __name__ == "__main__":
    net = Network()
    lines = sensitivity(net)
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "cascade_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
