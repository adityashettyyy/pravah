"""Run the whole Pravah pipeline in order, stopping at the first failure.

    python run_all.py                 # everything (about 60-90 min on one CPU, mostly RL training)
    python run_all.py --fast          # fewer RL steps / Optuna trials, for a quick check (~15 min)
    python run_all.py --from gnn      # resume at the first stage whose name contains "gnn"
    python run_all.py --only tests    # a single stage
    python run_all.py --list
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))


def stages(fast):
    steps, trials = (60_000, 10) if fast else (250_000, 30)
    return [
        ("data: legs", [sys.executable, "data/prepare_data.py"], ROOT, {}),
        ("data: geocoding", [sys.executable, "data/prepare_geo.py"], ROOT, {}),
        ("baseline: eda", [sys.executable, "explore.py"], os.path.join(ROOT, "ml"), {}),
        ("baseline: xgboost", [sys.executable, "train.py"], os.path.join(ROOT, "ml"), {}),
        ("analysis: propagation evidence", [sys.executable, "-m", "analysis.propagation_check"], ROOT, {}),
        ("graph: build", [sys.executable, "-m", "graph.build"], ROOT, {}),
        ("gnn: hub risk", [sys.executable, "-m", "gnn.hub_risk"], ROOT, {}),
        ("gnn: hub risk plus (GAT, hybrid)", [sys.executable, "-m", "gnn.hub_risk_plus"], ROOT, {}),
        ("graph: cascade sensitivity", [sys.executable, "-m", "graph.cascade"], ROOT, {}),
        ("gnn: propagation surrogate", [sys.executable, "-m", "gnn.propagation"], ROOT, {}),
        ("novelty: criticality index", [sys.executable, "-m", "novelty.criticality"], ROOT, {}),
        ("advanced: leg-delay model", [sys.executable, "-m", "advanced.train"], ROOT, {"PRAVAH_TRIALS": str(trials)}),
        ("rl: masked ppo", [sys.executable, "-m", "rl.train", "--steps", str(steps), "--seeds", "3"], ROOT, {}),
        ("evaluation: scenario sweep", [sys.executable, "-m", "evaluation.sweep"], ROOT, {}),
        ("tests", [sys.executable, "-m", "pytest", "-q", "tests"], ROOT, {}),
        ("demo: pipeline", [sys.executable, "pipeline.py", "--hub", "Bhiwandi", "--severity", "0.8",
                            "--out", "result.json"], ROOT, {}),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--from", dest="start")
    ap.add_argument("--only")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    st = stages(a.fast)
    if a.list:
        for i, (n, *_ ) in enumerate(st):
            print(f"{i:>2}  {n}")
        return
    if a.only:
        st = [s for s in st if a.only in s[0]]
    elif a.start:
        i = next(i for i, s in enumerate(st) if a.start in s[0])
        st = st[i:]
    for name, cmd, cwd, env in st:
        print(f"\n=== {name} ===", flush=True)
        t = time.time()
        r = subprocess.run(cmd, cwd=cwd, env={**os.environ, "PYTHONPATH": ROOT, **env})
        print(f"--- {name}: {'ok' if r.returncode == 0 else 'FAILED'} in {time.time() - t:.0f}s", flush=True)
        if r.returncode:
            sys.exit(r.returncode)


if __name__ == "__main__":
    main()
