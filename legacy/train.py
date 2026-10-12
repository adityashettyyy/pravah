"""Train the GNN and report it against two baselines that don't use the
graph at all: persistence (next bucket = this bucket) and global mean.

Data reality check: there are only 4 bucket snapshots -> 3 transitions.
Train on transitions (0->1) and (1->2), evaluate on the held-out, later
transition (2->3). This is a real temporal split (test bucket is strictly
after train buckets) but it means the test set is a SINGLE transition -
one snapshot's worth of hubs, not an average over many independent time
steps. Treat the numbers below as a directional signal, not a tight
estimate: with 3 transitions total there isn't enough independent data to
be confident the GNN generalizes across time, only that it does (or
doesn't) beat the baselines on this one held-out window.
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import build_snapshots  # noqa: E402
from model import DelayPropagationGNN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_PATH = os.path.join(ROOT, "gnn", "gnn_report.txt")
MODEL_PATH = os.path.join(ROOT, "gnn", "gnn_model.pt")

TRAIN_PAIRS = [(0, 1), (1, 2)]
TEST_PAIR = (2, 3)
EPOCHS = 300
LR = 0.01
SEED = 0


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def evaluate(model, snaps, pair):
    t_in, t_out = pair
    model.eval()
    with torch.no_grad():
        pred = model(snaps[t_in].x, snaps[t_in].edge_index).numpy()
    true = snaps[t_out].delay_rate.numpy()
    persistence = snaps[t_in].delay_rate.numpy()
    global_mean = np.full_like(true, snaps[t_in].delay_rate.numpy().mean())
    return {
        "gnn_mae": mae(pred, true),
        "persistence_mae": mae(persistence, true),
        "global_mean_mae": mae(global_mean, true),
        "n_hubs": len(true),
    }


def train():
    torch.manual_seed(SEED)
    hubs, snaps = build_snapshots()
    model = DelayPropagationGNN(in_dim=snaps[0].x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = 0.0
        for t_in, t_out in TRAIN_PAIRS:
            pred = model(snaps[t_in].x, snaps[t_in].edge_index)
            loss = loss + F.mse_loss(pred, snaps[t_out].delay_rate)
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"epoch {epoch:3d}  train_mse {loss.item():.5f}")

    torch.save(model.state_dict(), MODEL_PATH)

    train_metrics = [evaluate(model, snaps, p) for p in TRAIN_PAIRS]
    test_metrics = evaluate(model, snaps, TEST_PAIR)

    lines = []
    lines.append(f"Hubs: {len(hubs)}. Buckets: {len(snaps)} (3 transitions total).")
    lines.append(f"Train transitions {TRAIN_PAIRS}, held-out test transition {TEST_PAIR}")
    lines.append("")
    lines.append("Train-transition fit (not a generalization estimate, just a sanity check):")
    for p, m in zip(TRAIN_PAIRS, train_metrics):
        lines.append(f"  {p}: GNN MAE {m['gnn_mae']:.4f} | "
                      f"persistence MAE {m['persistence_mae']:.4f} | "
                      f"global-mean MAE {m['global_mean_mae']:.4f}  (n={m['n_hubs']})")
    lines.append("")
    lines.append("Held-out test transition (the number that matters):")
    m = test_metrics
    lines.append(f"  {TEST_PAIR}: GNN MAE {m['gnn_mae']:.4f} | "
                  f"persistence MAE {m['persistence_mae']:.4f} | "
                  f"global-mean MAE {m['global_mean_mae']:.4f}  (n={m['n_hubs']})")
    lines.append("")
    beat_persistence = m["gnn_mae"] < m["persistence_mae"]
    lines.append(f"GNN beats persistence baseline on held-out transition: {beat_persistence}")
    lines.append("")
    lines.append("Honest read: with only one held-out transition this is a single data "
                  "point, not a statistically reliable comparison. The result should be "
                  "read as 'does the graph-based prediction look reasonable', not as "
                  "proof the GNN generalizes. More weeks of data would be needed for a "
                  "real train/val/test split over independent time windows.")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    with open(os.path.join(ROOT, "gnn", "gnn_metrics.json"), "w") as f:
        json.dump({"train": train_metrics, "test": test_metrics}, f, indent=2)

    return model, hubs, snaps


if __name__ == "__main__":
    train()
