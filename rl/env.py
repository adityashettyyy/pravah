"""Rerouting environment: a hub is disrupted, shipment batches must be routed around it.

Each episode: one disrupted hub D (severity decays over time), N batches planned
to leave D toward one of K corridor hubs. For each batch the agent picks
  0 keep the planned route through D
  1..M reroute via one of the M nearest alternate hubs
  M+1 hold the batch for 2 steps
After every decision the flows change (volume moves off D->J onto D->A->J) and the
propagation GNN is queried again, so the risk of the corridor and alternate hubs
reacts to the agent's own choices.

All constants below are ASSUMPTIONS for a simulation, not measured values.
Real inputs: hub locations, traffic, delay-proneness and the graph itself.
"""
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from graph.cascade import shares
from gnn.propagation import PropGNN, node_inputs, predict

K, M, N_BATCHES = 3, 4, 10
CIRCUITY = 1.3          # road km per straight km (typical for Indian highways)
KM_PER_MIN = 0.94       # median km per OSRM minute in the data
OPS_FACTOR = 2.0        # median actual/OSRM time ratio in the data
DISRUPTION_MIN = 240    # extra minutes on a batch through a fully disrupted hub
RISK_MIN = 90           # extra minutes at a hub that is certain to be disrupted
TAU = 6.0               # steps for the disruption to fade by 1/e
HOLD_STEPS, HOLD_COST = 2, 0.3
KM_COST = 0.01          # cost units per extra km (1 unit ~ 1 hour of delay)
OVERLOAD_PENALTY = 3.0
MIN_ALT_TRAFFIC = 30    # alternates must be real, busy hubs
OBS_DIM = 4 + K + K + M + M + 1 + M + M + 3 + M + 4  # last 4 = manifest look-ahead


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_gnn(path):
    ckpt = torch.load(path, weights_only=False)
    model = PropGNN(ckpt["d_in"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model


def scenario_pool(net):
    """Hubs that can be disrupted: at least K distinct downstream hubs and real traffic."""
    pool = []
    for d in range(net.n):
        out = net.out_edges[d]
        if len(out) >= K and net.w[out].sum() >= 30:
            pool.append(d)
    return np.array(pool)


class RerouteEnv(gym.Env):
    def __init__(self, net, gnn, hubs, disruption_scale=1.0, hold_cost=HOLD_COST):
        self.net, self.gnn, self.hubs = net, gnn, np.asarray(hubs)
        self.scale, self.hold_cost = disruption_scale, hold_cost
        self.action_space = spaces.Discrete(M + 2)
        self.observation_space = spaces.Box(-5, 5, shape=(OBS_DIM,), dtype=np.float32)
        traffic = net.inbound + np.bincount(net.src, weights=net.w, minlength=net.n)
        self.traffic = traffic
        self.traffic_pct = traffic.argsort().argsort() / (net.n - 1)
        self.busy = np.flatnonzero(traffic >= MIN_ALT_TRAFFIC)

    def km(self, a, b):
        n = self.net
        return haversine(n.lat[a], n.lon[a], n.lat[b], n.lon[b]) * CIRCUITY

    def minutes(self, a, b):
        return self.km(a, b) / KM_PER_MIN * OPS_FACTOR

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        rng = self.rng = np.random.default_rng(seed)
        net = self.net
        self.D = int(rng.choice(self.hubs))
        if "hub" in options:
            self.D = int(options["hub"])
        out = net.out_edges[self.D]
        top = sorted(out, key=lambda e: -net.w[e])[:K]
        self.J = [int(net.dst[e]) for e in top]
        self.J_share = net.w[top] / net.w[top].sum()

        # alternates: busy hubs with the smallest detour D -> A -> corridor hubs
        cand = np.array([h for h in self.busy if h != self.D and h not in self.J])
        to_a = self.km(np.full(len(cand), self.D), cand)
        a_to_j = sum(s * self.km(cand, np.full(len(cand), j)) for s, j in zip(self.J_share, self.J))
        self.A = [int(cand[i]) for i in np.argsort(to_a + a_to_j)[:M]]

        ego = net.ego(self.D)
        in_ego = set(ego["nodes"].tolist())
        extra = [h for h in self.J + self.A if h not in in_ego]
        self.ego = net.subgraph(np.r_[ego["nodes"], extra].astype(int))
        loc = {int(h): i for i, h in enumerate(self.ego["nodes"])}
        self.jl, self.al = [loc[j] for j in self.J], [loc[a] for a in self.A]

        # room for rerouted flow: D->A and A->J edges start at zero
        e = self.ego
        new_s = [0] * M + [a for a in self.al for _ in range(K)]
        new_d = self.al + [j for _ in range(M) for j in self.jl]
        self.src = np.r_[e["src"], new_s].astype(int)
        self.dst = np.r_[e["dst"], new_d].astype(int)
        self.w = np.r_[e["w"], np.zeros(len(new_s))].astype(float)
        self.e_prior = len(e["src"])
        self.dj_edge = [int(np.flatnonzero((e["src"] == 0) & (e["dst"] == j))[0]) for j in self.jl]
        self.d_out_total = float(e["w"][e["src"] == 0].sum())

        self.sev0 = options.get("severity", rng.uniform(0.5, 1.0))
        self.util_D = rng.uniform(0.5, 0.9)
        self.util_A = np.clip(0.3 + 0.4 * self.traffic_pct[self.A] + rng.normal(0, 0.08, M), 0.15, 0.9)
        self.cap_A = 3 + 9 * self.traffic_pct[self.A]
        self.t_DJ = np.array([self.minutes(self.D, j) for j in self.J])
        self.t_DA = np.array([self.minutes(self.D, a) for a in self.A])
        self.t_AJ = np.array([[self.minutes(a, j) for j in self.J] for a in self.A])
        self.km_DJ = np.array([self.km(self.D, j) for j in self.J])
        self.km_DA = np.array([self.km(self.D, a) for a in self.A])
        self.km_AJ = np.array([[self.km(a, j) for j in self.J] for a in self.A])

        self.corr = rng.choice(K, size=N_BATCHES, p=self.J_share)
        self.size = rng.uniform(0.5, 1.5, N_BATCHES)
        self.urgent = rng.random(N_BATCHES) < 0.3
        self.t = 0
        self._risk()
        return self._obs(), {}

    def snapshot(self):
        return self.t, self.w.copy(), self.util_A.copy(), self.r.copy()

    def restore(self, snap):
        self.t, w, u, r = snap
        self.w, self.util_A, self.r = w.copy(), u.copy(), r.copy()

    def base_minutes(self):
        """Transit time with no disruption, to separate disruption delay from normal transit."""
        return float(sum(self.t_DJ[c] for c in self.corr))

    def sev(self, t):
        return self.sev0 * np.exp(-t / TAU)

    def _risk(self):
        p = shares(self.ego, self.src, self.dst, self.w)
        x = torch.tensor(node_inputs(self.net, self.ego, self.sev(self.t)))
        ei = torch.tensor(np.vstack([self.src, self.dst]), dtype=torch.long)
        self.r = predict(self.gnn, x, ei, torch.tensor(p, dtype=torch.float32))

    def extra_ratio(self, c):
        return (self.t_DA + self.t_AJ[:, c] - self.t_DJ[c]) / 300.0

    def _manifest(self, t):
        """What a dispatcher knows from the day's manifest: how many urgent batches and how much
        volume are still to come, and what the next batch looks like."""
        rest = slice(t + 1, N_BATCHES)
        nxt = t + 1 < N_BATCHES
        return (self.urgent[rest].sum() / N_BATCHES, self.size[rest].sum() / N_BATCHES,
                float(self.urgent[t + 1]) if nxt else 0.0, self.size[t + 1] if nxt else 0.0)

    def _obs(self):
        t = min(self.t, N_BATCHES - 1)
        c = self.corr[t]
        onehot = np.eye(K)[c]
        vuln = self.net.vuln[self.A]
        return np.r_[self.sev(self.t), self.t / N_BATCHES, self.size[t], float(self.urgent[t]), onehot,
                     self.r[self.jl], self.r[self.al], self.util_A, self.util_D,
                     np.clip(self.extra_ratio(c), -5, 5), vuln,
                     self.J_share[c], self.t_DJ[c] / 300.0, self.sev(self.t + HOLD_STEPS),
                     self.util_A + self.size[t] / self.cap_A, *self._manifest(t)].astype(np.float32)

    def action_masks(self):
        """Valid actions: keep and hold are always allowed, a reroute only if the alternate
        hub would stay within capacity (utilisation <= 1) after taking the batch."""
        t = min(self.t, N_BATCHES - 1)
        mask = np.ones(M + 2, dtype=bool)
        mask[1:M + 1] = (self.util_A + self.size[t] / self.cap_A) <= 1.0
        return mask

    def step(self, action):
        t, c, size, urgent = self.t, self.corr[self.t], self.size[self.t], self.urgent[self.t]
        sev, dj = self.sev(t), self.t_DJ[c]
        disrupt = lambda s: self.scale * DISRUPTION_MIN * s * (1 + 0.5 * max(0.0, self.util_D - 0.5))
        moved = 0.1 * self.d_out_total * size
        cost, overload = 0.0, 0

        if action == 0:
            delay = dj + disrupt(sev) + self.r[self.jl[c]] * RISK_MIN
        elif action == M + 1:
            s2 = self.sev(t + HOLD_STEPS)
            delay = 60 * HOLD_STEPS + dj + disrupt(s2) + self.r[self.jl[c]] * RISK_MIN * s2 / max(sev, 1e-6)
            cost = self.hold_cost
        else:
            m = action - 1
            self.util_A[m] += size / self.cap_A[m]
            over = self.util_A[m]
            queue = 240 * max(0.0, over - 0.8)
            overload = int(over > 1.0)
            k = self.dj_edge[c]
            shift = min(moved, self.w[k])
            self.w[k] -= shift
            self.w[self.e_prior + m] += shift
            self.w[self.e_prior + M + m * K + c] += shift
            self._risk()
            delay = (self.t_DA[m] + self.t_AJ[m, c] + queue + self.net.vuln[self.A[m]] * 40
                     + self.r[self.al[m]] * RISK_MIN + self.r[self.jl[c]] * RISK_MIN)
            cost = KM_COST * (self.km_DA[m] + self.km_AJ[m, c] - self.km_DJ[c])

        self.util_A = np.maximum(self.util_A - 0.03, 0.1)
        reward = -((2.0 if urgent else 1.0) * delay / 60 + cost + OVERLOAD_PENALTY * overload)
        self.t += 1
        done = self.t >= N_BATCHES
        if not done:
            self._risk()
        obs = self._obs() if not done else np.zeros(OBS_DIM, np.float32)
        return obs, float(reward), done, False, {"delay_min": delay, "cost": cost, "overload": overload}
