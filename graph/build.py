"""Builds the hub graph: data/processed/graph.pt

Nodes are hubs with coordinates. A directed edge a->b exists if at least one leg
ran from a to b. Node features do not use delay information, so they can be used
to predict delay-proneness. The delay labels are kept separate.
Run from the repo root: python -m graph.build
"""
import os
import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import LEGS_PATH, HUBS_PATH, GRAPH_PATH

DELAY_THRESHOLD = 2.6      # same definition as ml/problem.py
MIN_LABEL_LEGS = 20        # hubs with fewer outbound legs have too noisy a delay rate
HUB_TYPES = ["H", "HB", "D", "DC", "PC", "I", "IP", "L", "C"]
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def hub_type(name):
    m = re.search(r"_([A-Za-z]+)(?:_\d+)?\s*\(", str(name))
    t = m.group(1) if m else "none"
    return t if t in HUB_TYPES else "other"


def main():
    legs = pd.read_csv(LEGS_PATH, parse_dates=["od_start_time"])
    hubs = pd.read_csv(HUBS_PATH).sort_values("center").reset_index(drop=True)
    idx = {c: i for i, c in enumerate(hubs["center"])}
    legs = legs[legs["source_center"].isin(idx) & legs["destination_center"].isin(idx)].copy()
    legs["src"] = legs["source_center"].map(idx)
    legs["dst"] = legs["destination_center"].map(idx)
    legs["delayed"] = (legs["actual_time_min"] > DELAY_THRESHOLD * legs["osrm_time_min"]).astype(int)
    legs["ftl"] = (legs["route_type"] == "FTL").astype(int)
    n = len(hubs)

    edges = legs.groupby(["src", "dst"]).agg(n_legs=("delayed", "size")).reset_index()
    edges["km"] = haversine_km(hubs["lat"].values[edges["src"]], hubs["lon"].values[edges["src"]],
                               hubs["lat"].values[edges["dst"]], hubs["lon"].values[edges["dst"]])

    out = legs.groupby("src").agg(n_out=("delayed", "size"), out_rate=("delayed", "mean"),
                                  ftl_share=("ftl", "mean"))
    n_in = legs.groupby("dst").size()
    out_deg = edges.groupby("src").size()
    in_deg = edges.groupby("dst").size()
    mean_km = edges.groupby("src").apply(lambda g: np.average(g["km"], weights=g["n_legs"]),
                                         include_groups=False)

    nodes = pd.DataFrame(index=range(n))
    nodes["lat"] = (hubs["lat"] - hubs["lat"].mean()) / hubs["lat"].std()
    nodes["lon"] = (hubs["lon"] - hubs["lon"].mean()) / hubs["lon"].std()
    nodes["log_out_legs"] = np.log1p(out["n_out"].reindex(nodes.index).fillna(0))
    nodes["log_in_legs"] = np.log1p(n_in.reindex(nodes.index).fillna(0))
    nodes["log_out_deg"] = np.log1p(out_deg.reindex(nodes.index).fillna(0))
    nodes["log_in_deg"] = np.log1p(in_deg.reindex(nodes.index).fillna(0))
    nodes["log_mean_km"] = np.log1p(mean_km.reindex(nodes.index).fillna(0))
    nodes["ftl_share"] = out["ftl_share"].reindex(nodes.index).fillna(0)
    nodes["geo_exact"] = (hubs["geo_source"] == "exact").astype(float)
    types = hubs["name"].map(hub_type)
    for t in HUB_TYPES + ["other", "none"]:
        nodes[f"type_{t}"] = (types == t).astype(float)
    feats = nodes.astype(np.float32)

    labeled = (out["n_out"].reindex(nodes.index).fillna(0) >= MIN_LABEL_LEGS).values
    torch.save({
        "centers": hubs["center"].tolist(), "names": hubs["name"].tolist(),
        "state": hubs["state"].fillna("").tolist(),
        "lat": torch.tensor(hubs["lat"].values, dtype=torch.float32),
        "lon": torch.tensor(hubs["lon"].values, dtype=torch.float32),
        "x": torch.tensor(feats.values), "feature_names": list(feats.columns),
        "edge_index": torch.tensor(edges[["src", "dst"]].values.T, dtype=torch.long),
        "edge_legs": torch.tensor(edges["n_legs"].values, dtype=torch.float32),
        "edge_km": torch.tensor(edges["km"].values, dtype=torch.float32),
        "out_rate": torch.tensor(out["out_rate"].reindex(nodes.index).fillna(np.nan).values, dtype=torch.float32),
        "n_out": torch.tensor(out["n_out"].reindex(nodes.index).fillna(0).values, dtype=torch.float32),
        "labeled": torch.tensor(labeled),
    }, GRAPH_PATH)

    und = pd.concat([edges[["src", "dst"]], edges[["dst", "src"]].set_axis(["src", "dst"], axis=1)]).drop_duplicates()
    comp = connected_components(n, und.values)
    sizes = np.bincount(comp)
    deg = out_deg.reindex(range(n)).fillna(0) + in_deg.reindex(range(n)).fillna(0)
    top = hubs.assign(deg=deg.values).sort_values("deg", ascending=False).head(8)
    lines = [
        f"hubs (nodes): {n:,}   directed edges: {len(edges):,}   legs used: {len(legs):,}",
        f"features per node: {feats.shape[1]}",
        f"connected components (undirected): {len(sizes)}, largest {sizes.max():,} hubs ({sizes.max() / n:.1%})",
        f"edges with 5+ legs: {(edges['n_legs'] >= 5).sum():,}   median edge length {edges['km'].median():.0f} km",
        f"hubs with {MIN_LABEL_LEGS}+ outbound legs (labelled): {labeled.sum()}",
        f"  delayed rate of labelled hubs: median {np.nanmedian(out['out_rate'][labeled[out.index]]):.2f}, "
        f"{(out['out_rate'][labeled[out.index]] > 0.25).mean():.0%} above the 0.25 network average",
        "\nmost connected hubs (in + out degree):",
    ] + [f"  {r.name:<38} {int(r.deg):>3}" for r in top.itertuples()]
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "graph_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot_network(hubs, edges)


def connected_components(n, pairs):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def plot_network(hubs, edges):
    fig, ax = plt.subplots(figsize=(7, 8))
    busy = edges[edges["n_legs"] >= 5]
    for s, d in zip(busy["src"], busy["dst"]):
        ax.plot([hubs["lon"][s], hubs["lon"][d]], [hubs["lat"][s], hubs["lat"][d]],
                color="#1A1815", alpha=0.12, lw=0.6)
    ax.scatter(hubs["lon"], hubs["lat"], s=np.clip(hubs["n_legs"] / 8, 3, 90), color="#C4622A", alpha=0.7, lw=0)
    ax.set_title("Hub network (edges with 5+ legs, size = traffic)")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig_network.png"), dpi=120)
    plt.close()


if __name__ == "__main__":
    main()
