"""Multi-scenario evaluation sweep (case study section 10 / week 11).

Runs the full closed loop (GNN propagation -> every rerouting policy) over many disrupted hubs,
severities and shipment manifests, then writes one results table plus latency statistics.

    python -m evaluation.sweep                    # held-out (test) hubs x 3 severities x 3 manifests
    python -m evaluation.sweep --split all        # every disruptable hub, split-tagged
    python -m evaluation.sweep --severities 0.5 1.0 --seeds 5 --max-hubs 10

Every hub is tagged train / val / test using the same split as rl/train.py. The RL agent saw the
train hubs during training, so the headline table uses held-out hubs only. All delays are simulated
(delay constants in rl/env.py are assumptions), these are policy comparisons, not measured savings.

Outputs (evaluation/reports/): sweep_raw.csv, sweep_summary.json, sweep_report.md
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from app.engine import Engine, POLICY_ORDER, AGENT
from paths import ROOT
from rl.train import split_hubs

REPORT_DIR = os.path.join(ROOT, "evaluation", "reports")


def hub_splits(net):
    tr, va, te = split_hubs(net)
    return {int(h): s for s, hubs in (("train", tr), ("val", va), ("test", te)) for h in hubs}


def select_hubs(eng, split, max_hubs):
    tags = hub_splits(eng.net)
    hubs = [(i, tags[i]) for i, _ in eng.hub_options() if i in tags]  # busiest first
    if split != "all":
        hubs = [(i, s) for i, s in hubs if s == split]
    return hubs[:max_hubs] if max_hubs else hubs


def run_sweep(eng, hubs, severities, seeds):
    rows, lat_gnn, lat_loop = [], [], []
    for hub, tag in hubs:
        for sev in severities:
            prop = eng.propagation(hub, sev)
            lat_gnn.append(prop["latency_ms"])
            for seed in seeds:
                t0 = time.perf_counter()
                cmp = eng.compare(hub, sev, seed)
                lat_loop.append((time.perf_counter() - t0) * 1000)
                for p in POLICY_ORDER:
                    r = cmp[p]
                    rows.append({"hub": eng.label(hub), "state": str(eng.net.state[hub]), "split": tag,
                                 "severity": sev, "seed": seed, "policy": p, "excess_h": r["excess_h"],
                                 "cost": r["cost"], "overloads": r["overloads"], "saving_pct": r["saving_pct"],
                                 "contained": r["contained"], "hubs_at_risk": prop["n_at_risk"],
                                 "traffic_exposed_pct": prop["traffic_exposed_pct"]})
    return pd.DataFrame(rows), np.array(lat_gnn), np.array(lat_loop)


def policy_table(df):
    g = df.groupby("policy")
    t = pd.DataFrame({"excess_h": g["excess_h"].mean(), "excess_sd": g["excess_h"].std(),
                      "cost": g["cost"].mean(), "overloads_per_ep": g["overloads"].mean(),
                      "contained_pct": 100 * g["contained"].mean(), "saving_pct": g["saving_pct"].mean()})
    return t.loc[[p for p in POLICY_ORDER if p in t.index]]


def paired_gap(df, a, b, n=2000, seed=0):
    """Mean of (excess of policy a - excess of policy b) over identical scenarios, with 95% bootstrap CI."""
    key = ["hub", "severity", "seed"]
    pa = df[df.policy == a].set_index(key)["excess_h"]
    pb = df[df.policy == b].set_index(key)["excess_h"]
    d = (pa - pb.loc[pa.index]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def latency_stats(x):
    return {"p50_ms": float(np.percentile(x, 50)), "p95_ms": float(np.percentile(x, 95)),
            "max_ms": float(x.max()), "n": int(len(x))}


def md_table(df, floatfmt="{:.2f}"):
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in row]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_outputs(df, lat_gnn, lat_loop, args, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "sweep_raw.csv"), index=False)
    overall = policy_table(df)
    gaps = {f"{AGENT} vs {o}": paired_gap(df, AGENT, o) for o in ["Keep planned route", "Risk-aware rule",
                                                                 "Nearest hub (greedy rule)"]}
    by_split = {s: policy_table(d) for s, d in df.groupby("split")}
    by_sev = {s: policy_table(d) for s, d in df.groupby("severity")}
    ag = df[df.policy == AGENT]
    by_state = (ag.groupby("state").agg(hubs=("hub", "nunique"), excess_h=("excess_h", "mean"),
                                        saving_pct=("saving_pct", "mean"), contained_pct=("contained", "mean"))
                .assign(contained_pct=lambda d: 100 * d["contained_pct"]).sort_values("hubs", ascending=False))
    summary = {
        "scenarios": int(df.groupby(["hub", "severity", "seed"]).ngroups), "hubs": int(df["hub"].nunique()),
        "split": args.split, "severities": args.severities, "seeds": args.seeds,
        "policies": overall.round(4).to_dict("index"),
        "paired_gap_hours_95ci": {k: [round(x, 3) for x in v] for k, v in gaps.items()},
        "latency": {"gnn_propagation": latency_stats(lat_gnn), "full_loop_all_policies": latency_stats(lat_loop)},
    }
    json.dump(summary, open(os.path.join(out_dir, "sweep_summary.json"), "w"), indent=2)

    n = summary["scenarios"]
    L = [f"# Evaluation sweep\n",
         f"{n} scenarios = {summary['hubs']} hubs x {len(args.severities)} severities {args.severities} x "
         f"{len(args.seeds)} shipment manifests; split filter: `{args.split}`. All delays are simulated "
         "(constants in `rl/env.py` are assumptions), so this compares policies, it does not measure savings "
         "on real shipments. The RL agent trained on `train` hubs only; use `test` rows as the unbiased result.\n",
         "## All policies\n", md_table(overall.rename_axis("policy")), "\n",
         "## Paired difference in excess delay (hours, negative = agent better, 95% bootstrap CI)\n",
         "| comparison | mean | low | high |", "|---|---|---|---|"]
    L += [f"| {k} | {m:+.2f} | {lo:+.2f} | {hi:+.2f} |" for k, (m, lo, hi) in gaps.items()]
    L += ["", "## By hub split (train = seen in RL training)\n"]
    for s in ["train", "val", "test"]:
        if s in by_split:
            L += [f"**{s}** ({df[df.split == s]['hub'].nunique()} hubs)\n", md_table(by_split[s].rename_axis("policy")), ""]
    L += ["## By severity\n"]
    for s, t in by_sev.items():
        L += [f"**severity {s}**\n", md_table(t.rename_axis("policy")), ""]
    L += ["## RL agent by state\n", md_table(by_state.rename_axis("state")), "",
          "## Latency (case study: real-time decision loop)\n",
          "| stage | p50 ms | p95 ms | max ms | n |", "|---|---|---|---|---|"]
    for k, v in summary["latency"].items():
        L.append(f"| {k} | {v['p50_ms']:.1f} | {v['p95_ms']:.1f} | {v['max_ms']:.1f} | {v['n']} |")
    open(os.path.join(out_dir, "sweep_report.md"), "w").write("\n".join(L) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["test", "val", "train", "all"], default="test")
    ap.add_argument("--severities", type=float, nargs="+", default=[0.5, 0.8, 1.0])
    ap.add_argument("--seeds", type=int, default=3, help="number of shipment manifests per scenario")
    ap.add_argument("--max-hubs", type=int, default=0, help="0 = no cap (busiest hubs first)")
    ap.add_argument("--out", default=REPORT_DIR)
    a = ap.parse_args()
    a.seeds = list(range(a.seeds))
    eng = Engine()
    hubs = select_hubs(eng, a.split, a.max_hubs)
    print(f"{len(hubs)} hubs x {len(a.severities)} severities x {len(a.seeds)} manifests", flush=True)
    df, lg, ll = run_sweep(eng, hubs, a.severities, a.seeds)
    s = write_outputs(df, lg, ll, a, a.out)
    print(md_table(pd.DataFrame(s["policies"]).T.rename_axis("policy")))
    print(f"\nwrote {os.path.relpath(a.out, ROOT)}/sweep_report.md  (GNN p95 {s['latency']['gnn_propagation']['p95_ms']:.1f} ms)")


if __name__ == "__main__":
    main()
