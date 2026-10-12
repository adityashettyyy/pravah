import os
import json
import pytest

from paths import ROOT

ADV = os.path.join(ROOT, "advanced", "artifacts", "advanced_metrics.json")
needs_adv = pytest.mark.skipif(not os.path.exists(ADV), reason="run python -m advanced.train first")


@needs_adv
def test_conformal_coverage_holds_on_test():
    m = json.load(open(ADV))["conformal"]
    assert m["coverage_delayed"] >= 0.87 and m["coverage_ontime"] >= 0.87  # target 0.90, finite-sample slack


@needs_adv
def test_cqr_interval_is_valid_and_better_than_raw():
    m = json.load(open(ADV))["cqr"]
    assert 0.87 <= m["coverage_conformal"] <= 0.97
    assert m["mae_med"] < m["mae_naive"]


@needs_adv
def test_advanced_model_beats_baseline_significantly():
    m = json.load(open(ADV))
    assert m["paired_gain"][1] > 0, "95% CI of the PR-AUC gain should exclude zero"


@needs_adv
def test_leg_predictor_roundtrip():
    import pandas as pd
    from advanced.predictor import LegPredictor
    from ml.problem import load_clean
    leg = LegPredictor()
    legs = load_clean().sample(20, random_state=0)
    out = leg.predict(legs)
    assert out["p_delay"].between(0, 1).all() and set(out["verdict"]) <= {"delayed", "on time", "uncertain"}
    assert (out["extra_min_hi"] >= out["extra_min_lo"]).all()
    assert len(leg.explain(legs)) > 0


def test_dashboard_runs_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(ROOT, "app", "dashboard.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 5
    # change severity and re-run: the whole closed loop must still work
    at.slider[0].set_value(0.5)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
