"""Feature engineering, train/val/test split and preprocessing."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# actual_time_min and delay_min are only known after the leg finishes and the
# target is computed from them, so they are never used as features.
# osrm_time_min is the routing estimate, available before dispatch.
NUMERIC_FEATURES = [
    "actual_distance_to_destination",
    "osrm_time_min",
    "hour",
    "dayofweek",
    "is_weekend",
    "is_peak_hour",
    "distance_per_osrm_min",
]
CATEGORICAL_FEATURES = ["route_type", "source_center", "destination_center"]
TARGET = "is_delayed"

# ~870 source and ~1,100 destination hubs, most with very few legs. Only the
# most frequent ones get their own column, the rest share one "infrequent" one.
MAX_HUB_CATEGORIES = 40


def engineer_features(df):
    df = df.copy()
    ts = pd.to_datetime(df["od_start_time"], errors="coerce")
    df["hour"] = ts.dt.hour
    df["dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    # planned average speed: separates highway legs from slow urban ones
    df["distance_per_osrm_min"] = (
        df["actual_distance_to_destination"] / df["osrm_time_min"].replace(0, np.nan)
    )
    return df


def get_xy(df):
    """Returns features, target and trip ids (used to keep legs of one trip together)."""
    df = engineer_features(df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    return X, df[TARGET].copy(), df["trip_uuid"].copy()


def split_indices(groups, test_size=0.15, val_size=0.15, seed=42):
    """70/15/15 split by trip_uuid.

    Legs of the same trip share a day, vehicle and often hubs, so they must
    not end up on both sides of the split.
    """
    idx = np.arange(len(groups))
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val, test = next(outer.split(idx, groups=groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size),
                              random_state=seed)
    train, val = next(inner.split(train_val, groups=groups.iloc[train_val]))
    return train_val[train], train_val[val], test


def build_preprocessor(scale=True):
    """Imputers, encoder and scaler are fit inside the pipeline, so only on training data."""
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                  max_categories=MAX_HUB_CATEGORIES,
                                  sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUMERIC_FEATURES),
        ("cat", cat_pipe, CATEGORICAL_FEATURES),
    ])


def get_feature_names(fitted_preprocessor):
    encoder = fitted_preprocessor.named_transformers_["cat"].named_steps["encoder"]
    return NUMERIC_FEATURES + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))


if __name__ == "__main__":
    from problem import load_clean

    X, y, groups = get_xy(load_clean())
    train, val, test = split_indices(groups)
    for name, ix in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:<5} {len(ix):>6,} legs  delayed {y.iloc[ix].mean():.1%}")

    overlap = set(groups.iloc[train]) & set(groups.iloc[test])
    print(f"trips shared between train and test: {len(overlap)}")

    Xt = build_preprocessor().fit_transform(X.iloc[train])
    print(f"{X.shape[1]} features -> {Xt.shape[1]} columns after preprocessing")
