"""Loading, cleaning and the delay target."""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "processed", "legs.csv")

# A leg is "delayed" if actual_time > DELAY_THRESHOLD * osrm_time.
# OSRM only estimates driving time (no loading, waiting, stops), so the median
# leg already takes ~2x its OSRM estimate. 2.6 is about the 75th percentile of
# actual/osrm, which gives roughly 25% delayed legs.
DELAY_THRESHOLD = 2.6

COLUMNS = [
    "trip_uuid", "source_center", "source_name", "destination_center",
    "destination_name", "route_type", "od_start_time",
    "actual_distance_to_destination", "osrm_time_min", "actual_time_min",
]


def load_legs(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run data/prepare_data.py first.")
    df = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"legs file is missing columns: {missing}")
    return df


def add_target(df, threshold=DELAY_THRESHOLD):
    df = df.copy()
    df["delay_min"] = df["actual_time_min"] - df["osrm_time_min"]
    df["is_delayed"] = (df["actual_time_min"] > threshold * df["osrm_time_min"]).astype(int)
    return df


def clean(df):
    df = df.drop_duplicates(subset=["trip_uuid", "source_center", "destination_center"])
    df = df[(df["osrm_time_min"] > 0)
            & (df["actual_time_min"] > 0)
            & (df["actual_distance_to_destination"] > 0)
            & (df["source_center"] != df["destination_center"])].copy()

    # a few hubs have no name in the raw file, fall back to the center code
    df["source_name"] = df["source_name"].fillna(df["source_center"])
    df["destination_name"] = df["destination_name"].fillna(df["destination_center"])

    df["od_start_time"] = pd.to_datetime(df["od_start_time"], errors="coerce")
    return df.reset_index(drop=True)


def load_clean():
    return clean(add_target(load_legs()))


if __name__ == "__main__":
    df = load_clean()
    print(f"{len(df):,} legs, {df['trip_uuid'].nunique():,} trips")
    print(df["is_delayed"].value_counts().rename({0: "on time", 1: "delayed"}))
    print(f"delayed rate: {df['is_delayed'].mean():.1%}")
