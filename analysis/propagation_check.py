"""Is there delay propagation in the Delhivery data that a model could learn?

Four checks, written to analysis/reports/propagation_evidence.txt:
  1. trip chains: does a delayed leg make the next leg of the same trip late?
  2. hub to hub: does a hub's delay today predict its downstream neighbour's tomorrow?
  3. congestion: are legs slower when their hub is busier?
  4. neighbour similarity: do connected hubs have similar delay rates?
Run from the repo root: python -m analysis.propagation_check
"""
import os
import numpy as np
import pandas as pd

from paths import LEGS_PATH, HUBS_PATH
DELAY_THRESHOLD = 2.6  # same as ml/problem.py

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "propagation_evidence.txt")


def load():
    legs = pd.read_csv(LEGS_PATH, parse_dates=["od_start_time", "od_end_time"])
    legs["delayed"] = (legs["actual_time_min"] > DELAY_THRESHOLD * legs["osrm_time_min"]).astype(int)
    return legs


def trip_chains(legs, w):
    legs = legs.sort_values(["trip_uuid", "od_start_time"]).reset_index(drop=True)
    g = legs.groupby("trip_uuid")
    legs["prev_delayed"] = g["delayed"].shift(1)
    linked = g["destination_center"].shift(1) == legs["source_center"]
    ch = legs[legs["prev_delayed"].notna() & linked].copy()
    p1 = ch.loc[ch["prev_delayed"] == 1, "delayed"].mean()
    p0 = ch.loc[ch["prev_delayed"] == 0, "delayed"].mean()
    w(f"1. trip chains ({len(ch):,} consecutive leg pairs)")
    w(f"   next leg delayed after a delayed leg: {p1:.1%}, after an on-time leg: {p0:.1%}")
    res = ch["delayed"] - ch.groupby("source_center")["delayed"].transform("mean")
    w(f"   after removing each hub's own delay rate: {res[ch['prev_delayed'] == 1].mean():+.3f} "
      f"vs {res[ch['prev_delayed'] == 0].mean():+.3f}")
    w("   -> the raw gap is explained by which hub the next leg leaves from, not by a cascade")


def hub_to_hub(legs, w, hours=24, min_legs=4, seed=0):
    rng = np.random.default_rng(seed)
    t0 = legs["od_start_time"].min()
    legs = legs.assign(b=((legs["od_start_time"] - t0).dt.total_seconds() // (hours * 3600)).astype(int))
    hb = legs.groupby(["source_center", "b"])["delayed"].agg(["mean", "size"]).reset_index()
    hb = hb[hb["size"] >= min_legs]
    hb["res"] = hb["mean"] - hb.groupby("source_center")["mean"].transform("mean")
    res = hb.set_index(["source_center", "b"])["res"].to_dict()
    hubs = sorted({h for h, _ in res})

    edges = legs.groupby(["source_center", "destination_center"]).size().reset_index(name="n")
    edges = edges[(edges["n"] >= 10) & edges["source_center"].isin(hubs) & edges["destination_center"].isin(hubs)]
    days = sorted({d for _, d in res})
    pairs, control = [], []
    for a, b in zip(edges["source_center"], edges["destination_center"]):
        for d in days:
            if (a, d) in res and (b, d + 1) in res:
                pairs.append((res[(a, d)], res[(b, d + 1)]))
                r = hubs[rng.integers(len(hubs))]
                if r not in (a, b) and (r, d) in res:
                    control.append((res[(r, d)], res[(b, d + 1)]))
    p, c = np.array(pairs), np.array(control)
    own = np.array([(v, res[(h, d + 1)]) for (h, d), v in res.items() if (h, d + 1) in res])
    w(f"2. hub to hub, {hours}h buckets ({len(hubs)} hubs with {min_legs}+ legs/bucket, {len(edges)} edges)")
    w(f"   corr(upstream delay at t, downstream delay at t+1): {np.corrcoef(p.T)[0, 1]:+.3f}  (n={len(p):,})")
    w(f"   same with a random hub as 'upstream':                {np.corrcoef(c.T)[0, 1]:+.3f}  (n={len(c):,})")
    w(f"   a hub's own delay at t vs t+1:                       {np.corrcoef(own.T)[0, 1]:+.3f}  (n={len(own):,})")
    w("   -> upstream hubs predict no better than random hubs")


def congestion(legs, w):
    t0 = legs["od_start_time"].min()
    legs = legs.assign(b=((legs["od_start_time"] - t0).dt.total_seconds() // (6 * 3600)).astype(int))
    legs["load"] = np.log1p(legs.groupby(["source_center", "b"])["delayed"].transform("size"))
    big = legs[legs.groupby("source_center")["delayed"].transform("size") >= 100].copy()
    big["d_res"] = big["delayed"] - big.groupby("source_center")["delayed"].transform("mean")
    big["l_res"] = big["load"] - big.groupby("source_center")["load"].transform("mean")
    w(f"3. congestion: within-hub corr(legs leaving in the same 6h, delayed) = "
      f"{big['d_res'].corr(big['l_res']):+.3f}  ({big['source_center'].nunique()} hubs, {len(big):,} legs)")
    w("   -> no load-delay relationship")


def neighbours(legs, w):
    half = legs["od_start_time"].median()
    first = legs[legs["od_start_time"] <= half].groupby("source_center")["delayed"].agg(["mean", "size"])
    second = legs[legs["od_start_time"] > half].groupby("source_center")["delayed"].agg(["mean", "size"])
    first, second = first[first["size"] >= 10], second[second["size"] >= 10]
    both = first.index.intersection(second.index)
    w(f"4. hub delay rate, first vs second half of the period ({len(both)} hubs): "
      f"corr {first.loc[both, 'mean'].corr(second.loc[both, 'mean']):.2f}")

    rate = legs.groupby("source_center")["delayed"].agg(["mean", "size"])
    rate = rate[rate["size"] >= 10]["mean"]
    e = legs[["source_center", "destination_center"]].drop_duplicates()
    und = pd.concat([e.set_axis(["a", "b"], axis=1), e.set_axis(["b", "a"], axis=1)]).drop_duplicates()
    nb = und[und["b"].isin(rate.index)].groupby("a")["b"].apply(lambda s: rate.loc[s].mean())
    common = rate.index.intersection(nb.index)
    w(f"   corr(hub delay rate, mean rate of its neighbours) = {rate.loc[common].corr(nb.loc[common]):.2f} "
      f"({len(common)} hubs)")
    hubs = pd.read_csv(HUBS_PATH).set_index("center")
    st = rate.to_frame("r").join(hubs["state"]).dropna()
    w(f"   corr(hub delay rate, mean rate of its state)     = {st['r'].corr(st.groupby('state')['r'].transform('mean')):.2f}")
    w("   -> delay-proneness is a stable, regional hub property; this is what a graph model can learn")


def main():
    legs = load()
    lines = [f"{len(legs):,} legs, {legs['od_start_time'].min():%d %b} to {legs['od_start_time'].max():%d %b %Y}\n"]
    w = lines.append
    for fn in (trip_chains, hub_to_hub, congestion, neighbours):
        fn(legs, w)
        w("")
    text = "\n".join(lines)
    with open(OUT, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
