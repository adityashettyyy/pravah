"""Ground truth for the GNN: hub x time-bucket delay rate.

The raw data has no "delay propagated from hub A to hub B" label - Delhivery
does not record causal disruption chains. So we define a proxy, as flagged
in the case study review: bucket time into windows, compute each hub's
delay rate (share of legs originating there that were delayed) per window,
and train the GNN to predict a hub's next-window delay rate from this
window's network-wide state. That turns "does congestion at a hub affect
its neighbours" into a concrete, checkable prediction task.

Bucket size: the usable data is 12 Sep - 4 Oct 2018 (23 days; the last two
days have almost no legs and are dropped, see graph/graph_report.txt-style
check in build()). A hub gets a median of ~11 legs over the whole window, so
daily hub-level delay rates would mostly be computed on 0-1 legs - too noisy
to mean anything. We use 4 buckets (~5-6 days each) instead, which gives a
usable (if still small) number of legs per hub per bucket, at the cost of
only having 3 time transitions (bucket0->1, 1->2, 2->3) to learn from. That
is a real limitation, not a design choice made for convenience - see
gnn/train.py's docstring for how it affects evaluation.
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
from problem import load_clean  # noqa: E402

N_BUCKETS = 4
MIN_LEGS_PER_HUB_TOTAL = 8  # hub must have this many legs overall to be scored
OUT_PATH = os.path.join(ROOT, "gnn", "hub_bucket_labels.csv")


def build():
    df = load_clean()
    df = df[df.od_start_time < "2018-10-05"].copy()  # drop the near-empty tail

    t0, t1 = df.od_start_time.min(), df.od_start_time.max()
    edges = pd.date_range(t0, t1 + pd.Timedelta(seconds=1), periods=N_BUCKETS + 1)
    df["bucket"] = pd.cut(df.od_start_time, bins=edges, labels=False, include_lowest=True)

    # score a hub only if it has enough total volume to make its per-bucket
    # rate meaningful; this keeps the node set honest rather than including
    # hundreds of hubs with 1-2 legs each
    hub_totals = df.groupby("source_center").size()
    keep_hubs = hub_totals[hub_totals >= MIN_LEGS_PER_HUB_TOTAL].index
    scored = df[df.source_center.isin(keep_hubs)]

    labels = (
        scored.groupby(["source_center", "bucket"])
        .agg(delay_rate=("is_delayed", "mean"), n_legs=("is_delayed", "size"))
        .reset_index()
        .rename(columns={"source_center": "hub"})
    )
    labels.to_csv(OUT_PATH, index=False)

    print(f"Bucket edges: {list(edges)}")
    print(f"Hubs scored (>= {MIN_LEGS_PER_HUB_TOTAL} legs total): {len(keep_hubs)} "
          f"of {df.source_center.nunique()}")
    print(f"hub-bucket rows: {len(labels)} "
          f"(max possible {len(keep_hubs) * N_BUCKETS}, "
          f"so {len(labels) / (len(keep_hubs) * N_BUCKETS):.1%} of hub-bucket "
          f"cells actually have >=1 leg)")
    print(labels.groupby("bucket").agg(hubs=("hub", "nunique"),
                                        mean_delay_rate=("delay_rate", "mean"),
                                        mean_legs=("n_legs", "mean")))
    return labels


if __name__ == "__main__":
    build()
