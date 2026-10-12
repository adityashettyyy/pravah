"""End to end: disrupt a hub -> GNN propagation risk -> RL rerouting plan -> before/after (JSON).

    python pipeline.py --hub "Bhiwandi" --severity 0.8 --out result.json

Shipment batches are synthetic (drawn from the hub's real outbound corridor shares); propagation
dynamics and delay constants are assumptions, see README.md.
"""
import argparse
import json
import time

import numpy as np

from app.engine import Engine, AGENT, POLICY_ORDER
from graph.cascade import BETA, DECAY, HOPS


def plain(o):
    """numpy scalars/arrays -> plain Python so the result is JSON-serialisable."""
    if isinstance(o, dict):
        return {k: plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def run(engine, hub, severity, seed=0):
    t0 = time.perf_counter()
    prop = engine.propagation(hub, severity)
    cmp = engine.compare(hub, severity, seed)
    total_ms = (time.perf_counter() - t0) * 1000
    top = prop["hubs"].iloc[1:].sort_values("risk", ascending=False).head(15)
    return plain({
        "disrupted_hub": engine.label(hub), "severity": severity, "seed": seed,
        "propagation": [{"hub": r.hub, "state": r.state, "risk": round(float(r.risk), 3), "hop": int(r.hop),
                         "lat": float(r.lat), "lon": float(r.lon)} for r in top.itertuples() if r.risk > 0.01],
        "hubs_at_risk": prop["n_at_risk"], "traffic_exposed_pct": round(prop["traffic_exposed_pct"], 2),
        "plan": cmp[AGENT]["steps"],
        "policies": {p: {"excess_delay_h": round(cmp[p]["excess_h"], 2), "cost": round(cmp[p]["cost"], 2),
                         "overloads": cmp[p]["overloads"], "saving_pct": round(cmp[p]["saving_pct"], 1),
                         "contained": cmp[p]["contained"]} for p in POLICY_ORDER},
        "latency_ms": {"gnn_propagation": round(prop["latency_ms"], 1), "full_loop_all_policies": round(total_ms, 1),
                       "agent_decisions_10_batches": round(cmp[AGENT]["decision_ms"], 1)},
        "agent": "MaskablePPO" if engine.agent_is_rl else "risk-aware rule (RL model file not found)",
        "assumptions": {"cascade": f"BETA={BETA}, DECAY={DECAY}, {HOPS} hops (assumed, no cascade in the data)",
                        "batches": "10 synthetic batches", "delay_constants": "rl/env.py"},
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", required=True, help="part of the hub name, e.g. Bhiwandi")
    ap.add_argument("--severity", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    eng = Engine()
    hub = eng.find(a.hub)
    if hub not in set(int(h) for h in eng.pool):
        raise SystemExit(f"{eng.label(hub)} has too few downstream hubs, pick a larger hub")
    res = run(eng, hub, a.severity, a.seed)
    print(f"{res['disrupted_hub']}  severity {a.severity}   agent: {res['agent']}")
    print(f"\nhubs at risk (>20%): {res['hubs_at_risk']}   traffic exposed: {res['traffic_exposed_pct']}%")
    print("\nmost exposed hubs:")
    for h in res["propagation"][:8]:
        print(f"  {h['risk']:.2f}  hop {h['hop']}  {h['hub']}")
    print("\nplan:")
    for s in res["plan"]:
        flag = " (urgent)" if s["urgent"] else ""
        print(f"  batch {s['batch']:>2} -> {s['to'][:28]:<28}{flag:<9} {s['action']}")
    print(f"\n{'policy':<30}{'excess h':>9}{'saving':>8}{'overloads':>10}")
    for p, v in res["policies"].items():
        print(f"{p:<30}{v['excess_delay_h']:>9.1f}{v['saving_pct']:>7.0f}%{v['overloads']:>10}")
    print(f"\nlatency: {res['latency_ms']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
