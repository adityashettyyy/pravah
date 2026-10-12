"""Inference wrapper around the advanced leg-delay bundle (used by the dashboard and the tests)."""
import os
import numpy as np
import pandas as pd
import joblib

from advanced.features import FEATURES

BUNDLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "leg_model.pkl")
FEATURE_LABELS = {
    "te_pair": "history of this exact hub pair", "te_dst": "destination hub's delay history",
    "te_src": "source hub's delay history", "te_src_route": "source hub on this route type",
    "te_src_state": "source state's delay history", "te_dst_state": "destination state's delay history",
    "hour_sin": "time of day", "hour_cos": "time of day", "osrm_min": "planned drive time",
    "plan_speed": "planned speed", "dist_km": "leg distance", "hav_km": "straight-line distance",
    "circuity": "road circuity", "is_ftl": "full-truck-load route", "is_peak": "peak hour",
    "dow": "weekday", "is_weekend": "weekend", "same_state": "stays within one state",
    "n_pair": "how often this pair is seen", "n_src": "how often the source is seen",
    "n_dst": "how often the destination is seen",
}


class LegPredictor:
    def __init__(self, path=BUNDLE):
        self.b = joblib.load(path)
        self.ctx, self.names = self.b["ctx"], self.b["names"]
        self._explainer = None

    def _prob(self, X):
        b = self.b
        if b["final"] == "Stack":
            z = np.column_stack([b["base"][n].predict_proba(X)[:, 1] for n in self.names])
            z = np.log(np.clip(z, 1e-4, 1 - 1e-4) / (1 - np.clip(z, 1e-4, 1 - 1e-4)))
            raw = b["meta"].predict_proba(z)[:, 1]
        else:
            raw = b["base"]["LightGBM"].predict_proba(X)[:, 1]
        return b["calibrator"].predict(raw)

    def features(self, legs):
        return self.ctx.transform(legs)

    def predict(self, legs):
        """legs: DataFrame with source_center, destination_center, route_type, od_start_time,
        actual_distance_to_destination, osrm_time_min. Returns one row per leg."""
        X = self.features(legs)
        p = self._prob(X)
        q = self.b["conf_q"]
        in1, in0 = (1 - p) <= q[1], p <= q[0]
        verdict = np.where(in1 & ~in0, "delayed", np.where(in0 & ~in1, "on time", "uncertain"))
        r = {a: self.b["cqr"][a].predict(X) for a in (0.05, 0.5, 0.95)}
        osrm = legs["osrm_time_min"].to_numpy(float)
        qh = self.b["cqr_q"]
        mins = lambda z: osrm * (np.exp(z) - 1)
        return pd.DataFrame({"p_delay": p, "flag": p >= self.b["threshold"], "verdict": verdict,
                             "extra_min_lo": np.maximum(mins(r[0.05] - qh), 0), "extra_min_med": np.maximum(mins(r[0.5]), 0),
                             "extra_min_hi": mins(r[0.95] + qh)})

    def explain(self, legs, top=6):
        import shap
        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self.b["base"]["LightGBM"])
        X = self.features(legs.iloc[:1])
        sv = self._explainer.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        order = np.argsort(-np.abs(sv[0]))[:top]
        return [(FEATURE_LABELS.get(FEATURES[i], FEATURES[i]), float(sv[0][i]), float(X.iloc[0, i]))
                for i in order]
