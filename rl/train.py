"""Main RL experiment: action-masked PPO with a proper train / validation / test split of hubs.

What is new compared with rl/train.py
  * disrupted hubs are split 60 / 15 / 25 into train / validation / test. Model selection
    (which seed) uses validation hubs only, the test hubs are touched once at the end;
  * invalid reroutes (alternate hub would be over capacity) are masked out (MaskablePPO), which
    is the standard way to enforce a hard constraint instead of hoping a penalty teaches it;
  * several seeds, mean +/- std, so one lucky run cannot be reported as "the result";
  * ablation: the same PPO without masking, trained on the same split;
  * stronger rule-based baseline (risk-aware) next to the case study's "nearest hub" rule;
  * paired bootstrap 95% CI of the excess-delay difference against each baseline.

Run from the repo root:  python -m rl.train --steps 250000 --seeds 3
"""
import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from graph.cascade import Network
from gnn.propagation import MODEL_PATH
from rl.env import RerouteEnv, load_gnn, scenario_pool, M, HOLD_COST
from rl.policies import (keep, hold, random_policy, nearest_hub, risk_aware, lookahead,
                         agent_policy)
from rl.evaluate import run_episode

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
REP = os.path.join(HERE, "reports")
BEST_PATH = os.path.join(ART, "maskppo_best")
N_VAL, N_TEST = int(os.environ.get("PRAVAH_NVAL", 100)), int(os.environ.get("PRAVAH_NTEST", 300))
VAL_SEEDS = range(20_000, 20_000 + N_VAL)
TEST_SEEDS = range(10_000, 10_000 + N_TEST)


def split_hubs(net, seed=0):
    pool = scenario_pool(net)
    rng = np.random.default_rng(seed)
    rng.shuffle(pool)
    a, b = int(0.6 * len(pool)), int(0.75 * len(pool))
    return pool[:a], pool[a:b], pool[b:]


def make_model(kind, env, seed):
    venv = VecNormalize(DummyVecEnv([lambda: env]), norm_obs=False, norm_reward=True, gamma=0.98)
    cls = MaskablePPO if kind == "masked" else PPO
    return cls("MlpPolicy", venv, n_steps=1024, batch_size=128, gamma=0.98, learning_rate=3e-4,
               ent_coef=0.01, policy_kwargs={"net_arch": [64, 64]}, seed=seed, verbose=0)


def episodes(env, policy, seeds):
    return [run_episode(env, policy, s) for s in seeds]


def summarize(eps, base):
    ex = np.array([e["excess"] for e in eps])
    ov = np.array([e["overload"] for e in eps])
    return {"excess_h": ex.mean(), "excess_sd": ex.std(), "cost": np.mean([e["cost"] for e in eps]),
            "overloads": ov.mean(), "reward": np.mean([e["reward"] for e in eps]),
            "contained": float(((ex <= 0.5 * base) & (ov == 0)).mean()),
            "reroutes": np.mean([e["reroutes"] for e in eps]),
            "holds": np.mean([e["holds"] for e in eps])}


def boot_diff(a, b, n=2000, seed=0):
    """Paired bootstrap 95% CI of mean(a - b) over episodes."""
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250_000)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(ART, exist_ok=True)
    os.makedirs(REP, exist_ok=True)

    net, gnn = Network(), load_gnn(MODEL_PATH)
    tr, va, te = split_hubs(net)
    print(f"hubs: train {len(tr)} / val {len(va)} / test {len(te)}", flush=True)
    train_env = lambda: RerouteEnv(net, gnn, tr)
    val_env, test_env = RerouteEnv(net, gnn, va), RerouteEnv(net, gnn, te)

    # every model is saved as soon as it is trained, and re-used if the script is restarted
    models = {}
    for seed in range(args.seeds):
        path = os.path.join(ART, f"maskppo_seed{seed}")
        if os.path.exists(path + ".zip"):
            models[f"MaskPPO seed {seed}"] = MaskablePPO.load(path)
            print(f"loaded masked seed {seed}", flush=True)
            continue
        m = make_model("masked", train_env(), seed)
        m.learn(total_timesteps=args.steps)
        m.save(path)
        models[f"MaskPPO seed {seed}"] = m
        print(f"trained masked seed {seed}", flush=True)
    path = os.path.join(ART, "ppo_plain_seed0")
    if os.path.exists(path + ".zip"):
        plain = PPO.load(path)
    else:
        plain = make_model("plain", train_env(), 0)
        plain.learn(total_timesteps=args.steps)
        plain.save(path)
    print("plain PPO ready", flush=True)

    # model selection on validation hubs only
    val_keep = np.array([e["excess"] for e in episodes(val_env, keep, VAL_SEEDS)])
    val_score = {}
    for name, m in models.items():
        eps = episodes(val_env, agent_policy(m), VAL_SEEDS)
        val_score[name] = np.mean([e["reward"] for e in eps])
    best_name = max(val_score, key=val_score.get)
    models[best_name].save(BEST_PATH)
    print("validation reward per seed:", {k: round(v, 2) for k, v in val_score.items()},
          "-> selected", best_name, flush=True)

    # final test, once
    policies = {"Keep planned route": keep, "Always hold": hold,
                "Random": random_policy(np.random.default_rng(1)),
                "Nearest hub (greedy rule)": nearest_hub, "Risk-aware rule": risk_aware,
                "PPO, no masking (ablation)": agent_policy(plain, masked=False)}
    for name, m in models.items():
        policies[name] = agent_policy(m)
    policies["One-step lookahead (oracle)"] = lookahead
    raw = {n: episodes(test_env, p, TEST_SEEDS) for n, p in policies.items()}
    base = np.array([e["excess"] for e in raw["Keep planned route"]])
    table = {n: summarize(e, base) for n, e in raw.items()}

    seed_names = [n for n in models]
    seed_ex = np.array([table[n]["excess_h"] for n in seed_names])
    seed_ct = np.array([table[n]["contained"] for n in seed_names])
    seed_ov = np.array([table[n]["overloads"] for n in seed_names])

    chosen = np.array([e["excess"] for e in raw[best_name]])
    comps = {}
    for other in ["Keep planned route", "Nearest hub (greedy rule)", "Risk-aware rule",
                  "PPO, no masking (ablation)", "One-step lookahead (oracle)"]:
        o = np.array([e["excess"] for e in raw[other]])
        comps[other] = boot_diff(chosen, o)

    lines = [f"MaskablePPO, {args.steps:,} steps per run, {args.seeds} seeds; hubs split "
             f"train {len(tr)} / val {len(va)} / test {len(te)} (test hubs never used for training "
             f"or selection)",
             f"selected by validation reward: {best_name}; test = {N_TEST} episodes x 10 batches",
             "excess delay = total delay minus normal transit; contained = excess cut 50%+ and no "
             "overload\n",
             f"{'policy':<30}{'excess h':>9}{'sd':>6}{'cost':>6}{'overl.':>7}{'contain':>8}{'reward':>8}"
             f"{'reroute':>8}{'hold':>6}"]
    for n, t in table.items():
        tag = "  <- selected" if n == best_name else ""
        lines.append(f"{n:<30}{t['excess_h']:>9.1f}{t['excess_sd']:>6.1f}{t['cost']:>6.1f}"
                     f"{t['overloads']:>7.2f}{t['contained']:>8.0%}{t['reward']:>8.1f}"
                     f"{t['reroutes']:>8.1f}{t['holds']:>6.1f}{tag}")
    lines += ["", f"across {args.seeds} seeds (masked): excess {seed_ex.mean():.1f} +/- {seed_ex.std():.1f} h, "
              f"contained {seed_ct.mean():.0%} +/- {seed_ct.std():.0%}, overloads {seed_ov.mean():.2f}",
              "", f"paired bootstrap, selected agent minus other, excess delay in hours "
              f"(negative = agent better), mean [95% CI]:"]
    for o, (m, lo, hi) in comps.items():
        lines.append(f"  vs {o:<30}{m:>7.2f}  [{lo:.2f}, {hi:.2f}]")
    lines += ["", "All delay constants in rl/env.py are assumptions: this compares policies inside a",
              "simulation, it is not a measured saving on real shipments."]
    with open(os.path.join(REP, "rl_report.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(REP, "rl_metrics.json"), "w") as f:
        json.dump({"selected": best_name, "val_reward": val_score, "table": table,
                   "paired_vs": {k: list(map(float, v)) for k, v in comps.items()},
                   "hubs": {"train": len(tr), "val": len(va), "test": len(te)}}, f, indent=1,
                  default=float)
    print("\n".join(lines), flush=True)
    plot(table, best_name)


def plot(table, best):
    names = list(table)
    cols = ["#C4622A" if n == best else "#8FA9A3" if "PPO" in n else "#B8A88A" for n in names]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for a, key, ttl, sc in zip(ax, ["excess_h", "contained", "overloads"],
                               ["Excess delay per episode (h), lower is better",
                                "Episodes contained (%), higher is better",
                                "Capacity overloads per episode, lower is better"], [1, 100, 1]):
        a.barh(names, [table[n][key] * sc for n in names], color=cols)
        a.set_title(ttl, fontsize=10)
        a.invert_yaxis()
        a.spines[["top", "right"]].set_visible(False)
        if a is not ax[0]:
            a.set_yticklabels([])
    plt.tight_layout()
    plt.savefig(os.path.join(REP, "fig_policies.png"), dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
