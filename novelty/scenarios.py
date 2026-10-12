"""Regional and monsoon disruption scenarios (multi-hub cascades on the full network).

The case study asks for weather-grounded disruption scenarios. A flood or cyclone does not hit one
hub, it hits a whole region at once, so this module extends the single-hub cascade of
graph/cascade.py to MANY simultaneous sources on the full graph (no local-graph cap).

What is real and what is assumed
  * real: the graph, the traffic on each edge, each hub's delay-proneness, hub coordinates;
  * assumed: the cascade rule (same as graph/cascade.py), and the rainfall intensity per state.
    The IMD API is not reachable from the build environment, so STATE_RAIN_PRIOR below is a coarse,
    hand-written index of relative June-September monsoon rainfall intensity by state, taken from
    general climatology. It is NOT IMD data. If you place a CSV with columns `state,rainfall_mm`
    at data/raw/imd_state_rainfall.csv (e.g. exported from IMD / data.gov.in sub-division tables)
    `load_rain_index()` uses it instead, and the rest of the code does not change.
"""
import os
import numpy as np
import pandas as pd

from graph.cascade import BETA, DECAY, HOPS
from paths import ROOT

IMD_CSV = os.path.join(ROOT, "data", "raw", "imd_state_rainfall.csv")

# relative monsoon-rainfall intensity, 0 (dry) .. 1 (extreme). ASSUMED prior, see module docstring.
STATE_RAIN_PRIOR = {
    "Kerala": 1.0, "Goa": 1.0, "Meghalaya": 1.0, "Arunachal Pradesh": 0.9, "Assam": 0.9,
    "Mizoram": 0.8, "Tripura": 0.8, "Nagaland": 0.8, "Maharashtra": 0.7, "Karnataka": 0.65,
    "Orissa": 0.7, "West Bengal": 0.7, "Chhattisgarh": 0.6, "Jharkhand": 0.6, "Bihar": 0.6,
    "Uttarakhand": 0.65, "Himachal Pradesh": 0.6, "Daman & Diu": 0.7,
    "Dadra and Nagar Haveli": 0.7, "Uttar Pradesh": 0.5, "Madhya Pradesh": 0.5, "Gujarat": 0.45,
    "Andhra Pradesh": 0.4, "Telangana": 0.4, "Haryana": 0.35, "Delhi": 0.35, "Punjab": 0.35,
    "Chandigarh": 0.4, "Jammu & Kashmir": 0.3, "Tamil Nadu": 0.25, "Pondicherry": 0.25,
    "Rajasthan": 0.2,
}


def load_rain_index():
    """(index dict, source label). Uses real IMD-style data when the CSV is present."""
    if os.path.exists(IMD_CSV):
        d = pd.read_csv(IMD_CSV)
        mm = d.set_index("state")["rainfall_mm"].astype(float)
        idx = (mm / mm.max()).clip(0, 1).to_dict()
        return idx, f"IMD-style data from {os.path.basename(IMD_CSV)}"
    return dict(STATE_RAIN_PRIOR), "assumed climatological prior (not IMD data)"


def multi_source_cascade(net, sev0, runs=200, beta=BETA, hops=HOPS, rng=None):
    """Monte Carlo cascade from many simultaneously disrupted hubs.

    sev0: array (n_hubs,) with the probability/severity that a hub is directly disrupted.
    Each run: hub i is disrupted at start with probability sev0[i] (severity sev0[i]); then, for
    `hops` rounds, every newly disrupted hub a disrupts a downstream hub b with probability
    min(1, beta * s_a * share(a->b) * vulnerability(b)), the victim inheriting s_a * DECAY.
    Returns the share of runs in which each hub is disrupted (direct or by cascade), and the same
    for cascade only.
    """
    rng = rng or np.random.default_rng(0)
    n = net.n
    src, dst = net.src, net.dst
    share = np.clip(net.w / np.maximum(net.inbound[dst], 1e-6), 0, 1)
    order = np.argsort(dst, kind="stable")
    s, d, pp = src[order], dst[order], share[order]
    starts = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
    targets = d[starts]

    direct = rng.random((runs, n)) < sev0[None, :]
    sev = np.where(direct, sev0[None, :], 0.0)
    active = direct.copy()
    frontier = direct.copy()
    for _ in range(hops):
        q = np.clip(beta * sev[:, s] * pp * net.vuln[d], 0, 1)
        hit = (rng.random((runs, len(s))) < q) & frontier[:, s]
        best = np.maximum.reduceat(np.where(hit, sev[:, s] * DECAY, 0.0), starts, axis=1)
        new = np.zeros((runs, n))
        new[:, targets] = best
        newly = (new > 0) & ~active
        sev[newly] = new[newly]
        active |= newly
        frontier = newly
    return active.mean(0), (active & ~direct).mean(0)


def region_severity(net, kind, **kw):
    """Initial severity vector for a named scenario.

    kind = "state":   all hubs of kw["state"] get severity kw["severity"]
    kind = "radius":  hubs within kw["radius_km"] of (lat, lon), severity fades linearly to 0
    kind = "monsoon": every state gets severity rain_index * kw["intensity"] (national monsoon week)
    """
    sev = np.zeros(net.n)
    states = np.array(net.state)
    if kind == "state":
        sev[states == kw["state"]] = kw["severity"]
    elif kind == "radius":
        p = np.pi / 180
        lat, lon = np.asarray(net.lat), np.asarray(net.lon)
        a = (np.sin((lat - kw["lat"]) * p / 2) ** 2
             + np.cos(kw["lat"] * p) * np.cos(lat * p) * np.sin((lon - kw["lon"]) * p / 2) ** 2)
        km = 2 * 6371.0 * np.arcsin(np.sqrt(a))
        sev = np.clip(kw["severity"] * (1 - km / kw["radius_km"]), 0, 1)
    elif kind == "monsoon":
        idx, _ = load_rain_index()
        sev = np.array([idx.get(s, 0.3) for s in states]) * kw["intensity"]
    else:
        raise ValueError(kind)
    return np.clip(sev, 0, 1)


def run_scenario(net, kind, runs=200, seed=0, **kw):
    sev0 = region_severity(net, kind, **kw)
    p_all, p_casc = multi_source_cascade(net, sev0, runs=runs, rng=np.random.default_rng(seed))
    traffic = net.inbound + np.bincount(net.src, weights=net.w, minlength=net.n)
    states = np.array(net.state)
    hit_states = sev0 > 0
    region_states = set(states[hit_states])
    outside = ~np.isin(states, list(region_states))
    exposed = (p_all * traffic).sum() / traffic.sum()
    spill = (p_casc * traffic)[outside].sum() / max((p_all * traffic).sum(), 1e-9)
    return {"sev0": sev0, "p_all": p_all, "p_cascade": p_casc,
            "directly_hit_hubs": float((sev0).sum()),
            "expected_hubs_disrupted": float(p_all.sum()),
            "traffic_exposed_pct": float(100 * exposed),
            "spillover_share_pct": float(100 * spill),
            "regions": sorted(region_states)}


def worst_states(net, severity=0.8, runs=100):
    """Rank states by traffic exposed when the whole state is disrupted (state-level criticality)."""
    out = []
    for st in sorted(set(net.state)):
        n_hubs = sum(1 for s in net.state if s == st)
        if n_hubs < 5:
            continue
        r = run_scenario(net, "state", runs=runs, state=st, severity=severity)
        out.append({"state": st, "hubs": n_hubs, "traffic_exposed_pct": r["traffic_exposed_pct"],
                    "spillover_share_pct": r["spillover_share_pct"],
                    "expected_hubs_disrupted": r["expected_hubs_disrupted"]})
    return pd.DataFrame(out).sort_values("traffic_exposed_pct", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    from graph.cascade import Network
    net = Network()
    idx, src = load_rain_index()
    print("rain index source:", src)
    df = worst_states(net)
    print(df.head(12).to_string(index=False))
