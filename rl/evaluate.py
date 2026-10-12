"""Episode runner shared by training, tests, pipeline and dashboard."""
import numpy as np

from rl.env import M


def run_episode(env, policy, seed):
    obs, _ = env.reset(seed=seed)
    total = {"delay": 0.0, "cost": 0.0, "overload": 0, "reward": 0.0, "reroutes": 0, "holds": 0}
    done = False
    while not done:
        a = policy(env, obs)
        obs, r, done, _, info = env.step(a)
        total["delay"] += info["delay_min"] / 60
        total["cost"] += info["cost"]
        total["overload"] += info["overload"]
        total["reward"] += r
        total["reroutes"] += int(1 <= a <= M)
        total["holds"] += int(a == M + 1)
    total["excess"] = total["delay"] - env.base_minutes() / 60
    return total


def evaluate(env, policies, seeds):
    results = {name: [run_episode(env, p, s) for s in seeds] for name, p in policies.items()}
    base = np.array([r["excess"] for r in results["Keep planned route"]])
    table = {}
    for name, rs in results.items():
        ex = np.array([r["excess"] for r in rs])
        ov = np.array([r["overload"] for r in rs])
        contained = ((ex <= 0.5 * base) & (ov == 0)).mean()
        table[name] = {"excess_h": ex.mean(), "delay_h": np.mean([r["delay"] for r in rs]),
                       "cost": np.mean([r["cost"] for r in rs]), "overloads": ov.mean(),
                       "reward": np.mean([r["reward"] for r in rs]), "contained": contained,
                       "reroutes": np.mean([r["reroutes"] for r in rs]),
                       "holds": np.mean([r["holds"] for r in rs])}
    return table
