"""Leakage-safe features for the advanced leg-delay model.

Everything that is *learned or counted from data* (hub traffic, graph centrality, smoothed
historical delay rates) is computed from TRAINING legs only. For the training rows themselves the
delay-rate encodings are computed out-of-fold (grouped by trip), so a row never sees its own label.
Static outside information (hub coordinates from the pincode table) is fine for any row.

The same `FeatureContext` is used at training time and by the dashboard predictor, so a single
leg typed into the dashboard goes through exactly the same code path as the test set.
"""
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import GroupKFold

PRIOR_M = 10.0          # pseudo-count of the smoothed delay-rate encoding
PEAK_HOURS = [7, 8, 9, 17, 18, 19]

FEATURES = [
    "dist_km", "osrm_min", "plan_speed", "hav_km", "circuity", "same_state",
    "hour_sin", "hour_cos", "dow", "is_weekend", "is_peak", "is_ftl",
    "src_lat", "src_lon", "dst_lat", "dst_lon",
    "src_out", "src_in", "src_vol", "src_pr", "dst_out", "dst_in", "dst_vol", "dst_pr",
    "te_src", "te_dst", "te_pair", "te_src_state", "te_dst_state", "te_src_route",
    "n_src", "n_dst", "n_pair",
]


def haversine(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def _smooth(sum_, cnt, prior):
    return (sum_ + PRIOR_M * prior) / (cnt + PRIOR_M)


class FeatureContext:
    """Counts and encodings fitted on a training frame."""

    def __init__(self, hubs):
        h = hubs.set_index("center")
        self.hub_lat, self.hub_lon, self.hub_state = h["lat"].to_dict(), h["lon"].to_dict(), h["state"].to_dict()

    # ---------- fitting ----------
    def fit(self, df):
        self.prior = float(df["is_delayed"].mean())
        g = nx.DiGraph()
        for s, d, w in df.groupby(["source_center", "destination_center"]).size().reset_index().values:
            g.add_edge(s, d, weight=float(w))
        pr = nx.pagerank(g, weight="weight") if g.number_of_edges() else {}
        self.pagerank = pr
        self.out_deg = dict(g.out_degree())
        self.in_deg = dict(g.in_degree())
        out_v, in_v = df.groupby("source_center").size(), df.groupby("destination_center").size()
        self.vol = (out_v.add(in_v, fill_value=0)).to_dict()
        self.enc = self._encodings(df)
        return self

    def _keys(self, df):
        ss = df["source_center"].map(self.hub_state).fillna("?")
        ds = df["destination_center"].map(self.hub_state).fillna("?")
        return {
            "src": df["source_center"], "dst": df["destination_center"],
            "pair": df["source_center"] + ">" + df["destination_center"],
            "src_state": ss, "dst_state": ds,
            "src_route": df["source_center"] + "|" + df["route_type"],
        }

    def _encodings(self, df):
        y = df["is_delayed"].to_numpy(float)
        out = {}
        for name, key in self._keys(df).items():
            agg = pd.DataFrame({"k": key.to_numpy(), "y": y}).groupby("k")["y"].agg(["sum", "count"])
            out[name] = (agg["sum"].to_dict(), agg["count"].to_dict())
        return out

    # ---------- transforming ----------
    def _apply_enc(self, df, enc):
        res = {}
        for name, key in self._keys(df).items():
            s, c = enc[name]
            sm = key.map(s).fillna(0.0).to_numpy()
            ct = key.map(c).fillna(0.0).to_numpy()
            res[name] = (_smooth(sm, ct, self.prior), ct)
        return res

    def transform(self, df, train_oof_groups=None, train_df=None):
        """Feature matrix. If train_oof_groups is given, `df` must be the training frame the context
        was fitted on and the delay-rate encodings are computed out-of-fold."""
        df = df.reset_index(drop=True)
        ts = pd.to_datetime(df["od_start_time"], errors="coerce")
        hour = ts.dt.hour.fillna(12)
        out = pd.DataFrame(index=df.index)
        out["dist_km"] = df["actual_distance_to_destination"].astype(float)
        out["osrm_min"] = df["osrm_time_min"].astype(float)
        out["plan_speed"] = out["dist_km"] / out["osrm_min"].replace(0, np.nan)
        s, d = df["source_center"], df["destination_center"]
        for side, col in (("src", s), ("dst", d)):
            out[f"{side}_lat"] = col.map(self.hub_lat).astype(float)
            out[f"{side}_lon"] = col.map(self.hub_lon).astype(float)
        out["hav_km"] = haversine(out.src_lat, out.src_lon, out.dst_lat, out.dst_lon)
        out["circuity"] = out["dist_km"] / out["hav_km"].clip(lower=1.0)
        out["same_state"] = (s.map(self.hub_state) == d.map(self.hub_state)).astype(float)
        out["hour_sin"], out["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
        out["dow"] = ts.dt.dayofweek.fillna(0)
        out["is_weekend"] = (out["dow"] >= 5).astype(float)
        out["is_peak"] = hour.isin(PEAK_HOURS).astype(float)
        out["is_ftl"] = (df["route_type"] == "FTL").astype(float)
        out["src_out"], out["dst_out"] = s.map(self.out_deg), d.map(self.out_deg)
        out["src_in"], out["dst_in"] = s.map(self.in_deg), d.map(self.in_deg)
        out["src_vol"], out["dst_vol"] = s.map(self.vol), d.map(self.vol)
        out["src_pr"], out["dst_pr"] = s.map(self.pagerank), d.map(self.pagerank)
        for c in ["src_out", "src_in", "src_vol", "src_pr", "dst_out", "dst_in", "dst_vol", "dst_pr"]:
            out[c] = out[c].astype(float)

        if train_oof_groups is None:
            enc = self._apply_enc(df, self.enc)
        else:
            enc = {k: (np.zeros(len(df)), np.zeros(len(df))) for k in self._keys(df)}
            for tr, va in GroupKFold(5).split(df, groups=train_oof_groups):
                fold_ctx = FeatureContext.__new__(FeatureContext)
                fold_ctx.__dict__.update(self.__dict__)
                fold_ctx.prior = float(df["is_delayed"].iloc[tr].mean())
                fold_enc = fold_ctx._encodings(df.iloc[tr])
                part = fold_ctx._apply_enc(df.iloc[va], fold_enc)
                for k, (te, ct) in part.items():
                    enc[k][0][va], enc[k][1][va] = te, ct * 1.25  # 4/5 of the data -> rescale counts
        out["te_src"], out["n_src"] = enc["src"]
        out["te_dst"], out["n_dst"] = enc["dst"]
        out["te_pair"], out["n_pair"] = enc["pair"]
        out["te_src_state"], out["te_dst_state"] = enc["src_state"][0], enc["dst_state"][0]
        out["te_src_route"] = enc["src_route"][0]
        return out[FEATURES]


class Platt:
    """Platt scaling: logistic regression on the logit of a score. Monotone, so it never changes
    the ranking (PR/ROC-AUC) of the model, it only fixes the probability scale."""

    def fit(self, p, y):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1e6).fit(self._z(p), y)
        return self

    @staticmethod
    def _z(p):
        p = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def predict(self, p):
        return self.lr.predict_proba(self._z(p))[:, 1]
