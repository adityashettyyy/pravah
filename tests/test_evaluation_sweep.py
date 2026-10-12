import numpy as np
import pytest

from app.engine import Engine, POLICY_ORDER
from evaluation import sweep


@pytest.fixture(scope="module")
def eng():
    return Engine()


def test_hub_splits_match_rl_training_split(eng):
    tags = sweep.hub_splits(eng.net)
    assert set(tags.values()) == {"train", "val", "test"}
    assert len(tags) == len(eng.pool)  # every disruptable hub is in exactly one split


def test_select_hubs_filters_by_split_and_caps(eng):
    test_hubs = sweep.select_hubs(eng, "test", 0)
    assert test_hubs and all(s == "test" for _, s in test_hubs)
    assert len(sweep.select_hubs(eng, "all", 4)) == 4


def test_sweep_table_is_consistent(eng):
    hubs = sweep.select_hubs(eng, "test", 2)
    df, lg, ll = sweep.run_sweep(eng, hubs, [0.6, 0.9], [0, 1])
    assert len(df) == 2 * 2 * 2 * len(POLICY_ORDER)
    keep = df[df.policy == "Keep planned route"]
    assert (keep["saving_pct"] == 0).all()
    t = sweep.policy_table(df)
    assert list(t.index) == POLICY_ORDER
    assert (t["overloads_per_ep"] >= 0).all() and t["contained_pct"].between(0, 100).all()
    assert len(lg) == 2 * 2 and len(ll) == 2 * 2 * 2 and (ll > 0).all()


def test_paired_gap_is_antisymmetric(eng):
    hubs = sweep.select_hubs(eng, "test", 2)
    df, _, _ = sweep.run_sweep(eng, hubs, [0.8], [0, 1, 2])
    a, b = "Keep planned route", "Risk-aware rule"
    m1, lo1, hi1 = sweep.paired_gap(df, a, b)
    m2, lo2, hi2 = sweep.paired_gap(df, b, a)
    assert np.isclose(m1, -m2) and lo1 <= m1 <= hi1


def test_outputs_are_written(eng, tmp_path):
    class A:
        split, severities, seeds = "test", [0.8], [0]
    hubs = sweep.select_hubs(eng, "test", 2)
    df, lg, ll = sweep.run_sweep(eng, hubs, A.severities, A.seeds)
    s = sweep.write_outputs(df, lg, ll, A, str(tmp_path))
    for f in ("sweep_raw.csv", "sweep_summary.json", "sweep_report.md"):
        assert (tmp_path / f).exists()
    assert s["scenarios"] == 2 and "gnn_propagation" in s["latency"]
