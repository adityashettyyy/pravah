"""Target definition, split hygiene and, most importantly, leakage tests for the feature builder."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from ml.problem import load_clean, add_target, DELAY_THRESHOLD
from ml.features import get_xy, split_indices
from advanced.features import FeatureContext, FEATURES
from paths import HUBS_PATH


@pytest.fixture(scope="module")
def df():
    return load_clean().reset_index(drop=True)


def test_target_definition():
    d = add_target(pd.DataFrame({"actual_time_min": [100, 120], "osrm_time_min": [40, 40]}))
    assert list(d["is_delayed"]) == [0, 1]
    assert DELAY_THRESHOLD == 2.6


def test_split_has_no_trip_overlap(df):
    _, _, groups = get_xy(df)
    tr, va, te = split_indices(groups)
    sets = [set(groups.iloc[i]) for i in (tr, va, te)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert len(tr) + len(va) + len(te) == len(df)


def test_features_shape_and_unseen_hub(df):
    hubs = pd.read_csv(HUBS_PATH)
    small = df.iloc[:3000].reset_index(drop=True)
    ctx = FeatureContext(hubs).fit(small)
    X = ctx.transform(small, train_oof_groups=small["trip_uuid"].to_numpy())
    assert list(X.columns) == FEATURES and len(X) == len(small)
    unseen = small.iloc[:1].copy()
    unseen["source_center"] = "IND000000ZZZ"
    row = ctx.transform(unseen)
    assert row.shape == (1, len(FEATURES))
    assert row["te_src"].iloc[0] == pytest.approx(ctx.prior, abs=1e-6)  # unseen hub -> falls back to the prior


def test_oof_encoding_does_not_leak_labels(df):
    """With shuffled labels the out-of-fold pair encoding must carry no signal; with real labels it must."""
    hubs = pd.read_csv(HUBS_PATH)
    d = df.iloc[:8000].reset_index(drop=True)
    ctx = FeatureContext(hubs).fit(d)
    X = ctx.transform(d, train_oof_groups=d["trip_uuid"].to_numpy())
    real = roc_auc_score(d["is_delayed"], X["te_pair"])
    shuffled = d.copy()
    shuffled["is_delayed"] = np.random.default_rng(0).permutation(d["is_delayed"].to_numpy())
    ctx2 = FeatureContext(hubs).fit(shuffled)
    X2 = ctx2.transform(shuffled, train_oof_groups=shuffled["trip_uuid"].to_numpy())
    fake = roc_auc_score(shuffled["is_delayed"], X2["te_pair"])
    assert real > 0.60
    assert abs(fake - 0.5) < 0.05, f"out-of-fold encoding leaks: AUC {fake:.3f} with shuffled labels"
