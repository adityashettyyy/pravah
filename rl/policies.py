"""Rerouting policies used for evaluation, the pipeline and the dashboard.

Every policy has the signature policy(env, obs) -> action and only looks at things an
operator would know at decision time, except `lookahead` which queries the simulator and is
reported as a reference upper bound, not as a deployable policy.
"""
import numpy as np

from rl.env import M, RISK_MIN, DISRUPTION_MIN


def keep(env, obs):
    return 0


def hold(env, obs):
    return M + 1


def random_policy(rng):
    return lambda env, obs: int(rng.integers(M + 2))


def nearest_hub(env, obs):
    """Greedy rule from the case study: nearest alternate hub that still has room, else keep."""
    t = env.t
    for m in np.argsort(env.t_DA):
        if env.util_A[m] + env.size[t] / env.cap_A[m] <= 0.9:
            return int(m) + 1
    return 0


def risk_aware(env, obs):
    """Stronger hand-written rule: compares the estimated delay of keeping the route with each
    alternate (extra transit + GNN risk at the hubs touched + queueing), picks the cheapest."""
    t, c = env.t, env.corr[env.t]
    sev = env.sev(t)
    keep_cost = (env.scale * DISRUPTION_MIN * sev + env.r[env.jl[c]] * RISK_MIN)
    best, best_a = keep_cost, 0
    for m in range(M):
        if env.util_A[m] + env.size[t] / env.cap_A[m] > 0.95:
            continue
        cost = (env.t_DA[m] + env.t_AJ[m, c] - env.t_DJ[c] + env.net.vuln[env.A[m]] * 40
                + env.r[env.al[m]] * RISK_MIN + env.r[env.jl[c]] * RISK_MIN)
        if cost < best:
            best, best_a = cost, m + 1
    return best_a


def lookahead(env, obs):
    """One-step planner that tries every action in the simulator (reference only)."""
    snap = env.snapshot()
    best, best_a = -np.inf, 0
    for a in range(M + 2):
        _, r, _, _, _ = env.step(a)
        env.restore(snap)
        if r > best:
            best, best_a = r, a
    return best_a


def agent_policy(model, masked=True):
    """Wraps a trained (Maskable)PPO model as a policy."""
    if masked:
        return lambda env, obs: int(model.predict(obs, action_masks=env.action_masks(),
                                                  deterministic=True)[0])
    return lambda env, obs: int(model.predict(obs, deterministic=True)[0])
