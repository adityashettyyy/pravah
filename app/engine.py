"""Pravah engine: one object that holds every trained component and runs the closed loop
    predict (GNN) -> decide (RL) -> act (simulate) -> re-predict (GNN)
for the dashboard and for the command-line pipeline. No Streamlit code in here, so it is testable.
"""
import os
import time
import json
import numpy as np
import pandas as pd
import torch

from graph.cascade import Network, shares, BETA, DECAY, HOPS
from gnn.propagation import MODEL_PATH, node_inputs, predict
from rl.env import RerouteEnv, load_gnn, scenario_pool, M, N_BATCHES
from rl.policies import (keep, nearest_hub, risk_aware, lookahead, agent_policy)
from rl.train import BEST_PATH
from novelty import scenarios as scen

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRIT_PATH = os.path.join(ROOT, "novelty", "artifacts", "criticality.csv")
CRIT_SUMMARY = os.path.join(ROOT, "novelty", "artifacts", "criticality_summary.json")
RL_METRICS = os.path.join(ROOT, "rl", "reports", "rl_metrics.json")
ADV_METRICS = os.path.join(ROOT, "advanced", "artifacts", "advanced_metrics.json")
AT_RISK = 0.2

AGENT = "RL agent (MaskablePPO)"
POLICY_ORDER = ["Keep planned route", "Nearest hub (greedy rule)", "Risk-aware rule", AGENT,
                "Oracle one-step lookahead"]


class Engine:
    def __init__(self):
        torch.set_num_threads(1)
        self.net = Network()
        self.gnn = load_gnn(MODEL_PATH)
        self.pool = scenario_pool(self.net)
        self.env = RerouteEnv(self.net, self.gnn, self.pool)
        self.traffic = self.net.inbound + np.bincount(self.net.src, weights=self.net.w, minlength=self.net.n)
        self.agent, self.agent_error = None, None
        try:
            from sb3_contrib import MaskablePPO
            self.agent = MaskablePPO.load(BEST_PATH)
        except Exception as e:  # model file missing or sb3_contrib not installed
            self.agent_error = str(e)
        self.crit = pd.read_csv(CRIT_PATH) if os.path.exists(CRIT_PATH) else None
        self.policies = {"Keep planned route": keep, "Nearest hub (greedy rule)": nearest_hub,
                         "Risk-aware rule": risk_aware, "Oracle one-step lookahead": lookahead}
        self.policies[AGENT] = (agent_policy(self.agent) if self.agent is not None else risk_aware)
        self.agent_is_rl = self.agent is not None

    # ---------------------------------------------------------------- hubs
    def label(self, i):
        return str(self.net.names[i])

    def hub_options(self):
        """Hubs that can be disrupted in the simulator, busiest first: [(index, label)]."""
        order = sorted(self.pool, key=lambda i: -self.traffic[i])
        return [(int(i), self.label(i)) for i in order]

    def find(self, text):
        return self.net.find(text)

    # ---------------------------------------------------------------- propagation (GNN)
    def propagation(self, hub, severity):
        """GNN risk for every hub within 3 downstream hops of `hub`, plus hop distance and edges."""
        net = self.net
        t0 = time.perf_counter()
        ego = net.ego(hub)
        p = shares(ego, ego["src"], ego["dst"], ego["w"])
        risk = predict(self.gnn, torch.tensor(node_inputs(net, ego, severity)),
                       torch.tensor(np.vstack([ego["src"], ego["dst"]]), dtype=torch.long),
                       torch.tensor(p, dtype=torch.float32))
        risk[0] = max(risk[0], severity)
        n = len(ego["nodes"])
        hop = np.full(n, -1)
        hop[0] = 0
        for h in range(1, HOPS + 1):
            reach = np.unique(ego["dst"][(hop[ego["src"]] == h - 1)])
            reach = reach[hop[reach] < 0]
            hop[reach] = h
        nodes = ego["nodes"]
        df = pd.DataFrame({"idx": nodes, "hub": [self.label(i) for i in nodes],
                           "state": [net.state[i] for i in nodes], "lat": net.lat[nodes], "lon": net.lon[nodes],
                           "risk": risk, "hop": hop, "traffic": self.traffic[nodes]})
        keep_e = (p > 0.03) & (hop[ego["src"]] >= 0) & (hop[ego["dst"]] == hop[ego["src"]] + 1)
        edges = pd.DataFrame({"src": nodes[ego["src"][keep_e]], "dst": nodes[ego["dst"][keep_e]],
                              "share": p[keep_e]})
        exposed = float((risk * self.traffic[nodes]).sum() / self.traffic.sum() * 100)
        return {"hubs": df, "edges": edges, "latency_ms": (time.perf_counter() - t0) * 1000,
                "n_at_risk": int((risk[1:] > AT_RISK).sum()), "traffic_exposed_pct": exposed}

    # ---------------------------------------------------------------- rerouting (RL)
    def run_policy(self, hub, severity, seed, policy_name):
        env = self.env
        obs, _ = env.reset(seed=seed, options={"hub": hub, "severity": severity})
        policy = self.policies[policy_name]
        net, D = self.net, int(env.D)
        steps, done = [], False
        total_delay = total_cost = 0.0
        overloads = 0
        t0 = time.perf_counter()
        while not done:
            t, c = env.t, int(env.corr[env.t])
            urgent, size = bool(env.urgent[t]), float(env.size[t])
            risk_before = float(env.r[env.jl[c]])
            a = policy(env, obs)
            obs, r, done, _, info = env.step(a)
            risk_after = float(env.r[env.jl[c]]) if not done else risk_before
            via = None
            if 1 <= a <= M:
                via = int(env.A[a - 1])
                text = f"reroute via {self.label(via)}"
            elif a == M + 1:
                text = "hold 2 steps"
            else:
                text = "keep planned route"
            steps.append({"batch": t + 1, "to": self.label(env.J[c]), "to_idx": int(env.J[c]), "via_idx": via,
                          "urgent": urgent, "size": round(size, 2), "action": text, "action_id": int(a),
                          "delay_h": round(info["delay_min"] / 60, 2), "cost": round(info["cost"], 3),
                          "overload": bool(info["overload"]),
                          "corridor_risk_before": round(risk_before, 3), "corridor_risk_after": round(risk_after, 3)})
            total_delay += info["delay_min"] / 60
            total_cost += info["cost"]
            overloads += info["overload"]
        base = env.base_minutes() / 60
        return {"policy": policy_name, "steps": steps, "excess_h": total_delay - base, "delay_h": total_delay,
                "cost": total_cost, "overloads": int(overloads), "decision_ms": (time.perf_counter() - t0) * 1000,
                "disrupted_idx": D}

    def compare(self, hub, severity, seed):
        """All policies on the same disruption and the same shipment batches."""
        res = {p: self.run_policy(hub, severity, seed, p) for p in POLICY_ORDER}
        base = res["Keep planned route"]["excess_h"]
        for r in res.values():
            r["saving_pct"] = 100 * (base - r["excess_h"]) / base if base > 1e-9 else 0.0
            r["contained"] = bool(r["excess_h"] <= 0.5 * base and r["overloads"] == 0)
        return res

    def robustness(self, hub, severity, n=12, seed0=100):
        """Mean excess delay over several random shipment manifests at the same disrupted hub."""
        rows = {p: [] for p in POLICY_ORDER}
        for s in range(seed0, seed0 + n):
            for p in POLICY_ORDER:
                rows[p].append(self.run_policy(hub, severity, s, p)["excess_h"])
        return pd.DataFrame({p: [np.mean(v), np.std(v)] for p, v in rows.items()}, index=["mean", "sd"]).T

    # ---------------------------------------------------------------- scenarios / criticality
    def regional(self, kind, runs=150, **kw):
        r = scen.run_scenario(self.net, kind, runs=runs, **kw)
        net = self.net
        df = pd.DataFrame({"hub": [self.label(i) for i in range(net.n)], "state": net.state, "lat": net.lat,
                           "lon": net.lon, "p_all": r["p_all"], "p_cascade": r["p_cascade"],
                           "direct": r["sev0"], "traffic": self.traffic})
        return r, df

    def metrics_json(self, path):
        return json.load(open(path)) if os.path.exists(path) else None


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))
