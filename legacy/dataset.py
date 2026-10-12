"""Turn hub_bucket_labels.csv + graph/{nodes,edges}.csv into PyG snapshots.

One snapshot per bucket. Node feature vector at bucket t:
  [delay_rate_t, log1p(n_legs_t), latitude, longitude]
Target: delay_rate_{t+1} for the same hub.
Edges: static hub graph (same edge_index every snapshot - we do not have
enough buckets to learn a time-varying topology, so topology is fixed and
only node state varies, which is the standard setup for this kind of
"diffusion over a fixed graph" GNN).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_PATH = os.path.join(ROOT, "gnn", "hub_bucket_labels.csv")
NODES_PATH = os.path.join(ROOT, "graph", "nodes.csv")
EDGES_PATH = os.path.join(ROOT, "graph", "edges.csv")

MIN_EDGE_LEGS = 2  # drop one-off pairs, keep edges seen at least twice


def build_snapshots():
    labels = pd.read_csv(LABELS_PATH)
    nodes = pd.read_csv(NODES_PATH).set_index("hub")
    edges = pd.read_csv(EDGES_PATH)
    edges = edges[edges.n_legs >= MIN_EDGE_LEGS]

    hubs = sorted(set(labels.hub) & set(nodes.index))
    hub_idx = {h: i for i, h in enumerate(hubs)}

    edges = edges[edges.src.isin(hub_idx) & edges.dst.isin(hub_idx)]
    src = edges.src.map(hub_idx).to_numpy()
    dst = edges.dst.map(hub_idx).to_numpy()
    # undirected: add both directions
    edge_index = np.concatenate([np.stack([src, dst]), np.stack([dst, src])], axis=1)
    edge_index = torch.tensor(edge_index, dtype=torch.long)

    lat = nodes.loc[hubs, "latitude"].to_numpy()
    lon = nodes.loc[hubs, "longitude"].to_numpy()
    # normalize coords to roughly [-1, 1] over India's bounding box
    lat_n = (lat - 22.0) / 15.0
    lon_n = (lon - 80.0) / 15.0

    hub_overall_mean = labels.groupby("hub").delay_rate.mean()

    buckets = sorted(labels.bucket.unique())
    snapshots = {}
    for b in buckets:
        row = labels[labels.bucket == b].set_index("hub")
        delay_rate = np.array([
            row.loc[h, "delay_rate"] if h in row.index else hub_overall_mean.get(h, 0.25)
            for h in hubs
        ])
        n_legs = np.array([
            row.loc[h, "n_legs"] if h in row.index else 0.0 for h in hubs
        ])
        x = torch.tensor(
            np.stack([delay_rate, np.log1p(n_legs), lat_n, lon_n], axis=1),
            dtype=torch.float,
        )
        snapshots[b] = Data(x=x, edge_index=edge_index)
        snapshots[b].delay_rate = torch.tensor(delay_rate, dtype=torch.float)

    return hubs, snapshots


if __name__ == "__main__":
    hubs, snaps = build_snapshots()
    print(f"{len(hubs)} hubs, {len(snaps)} bucket snapshots")
    b0 = snaps[0]
    print(f"edge_index shape {tuple(b0.edge_index.shape)}, x shape {tuple(b0.x.shape)}")
