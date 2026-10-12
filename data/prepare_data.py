"""Builds data/processed/legs.csv from the raw Delhivery file.

The raw file has one row per checkpoint. A trip_uuid can cover several legs
(source hub -> destination hub), and actual_time / osrm_time / distance are
cumulative within a leg and restart at 0 when the next leg begins. So the
last (max) value per (trip, source, destination) is that leg's total.
Each leg is one hub-to-hub edge, which is also what the graph model needs.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(HERE, "raw", "delhivery_data.csv")
OUT_PATH = os.path.join(HERE, "processed", "legs.csv")

LEG_KEY = ["trip_uuid", "source_center", "destination_center"]


def main():
    raw = pd.read_csv(RAW_PATH)
    print(f"{len(raw):,} checkpoint rows, {raw['trip_uuid'].nunique():,} trips")

    legs = raw.groupby(LEG_KEY, sort=False).agg(
        source_name=("source_name", "first"),
        destination_name=("destination_name", "first"),
        route_type=("route_type", "first"),
        od_start_time=("od_start_time", "min"),
        od_end_time=("od_end_time", "max"),
        actual_distance_to_destination=("actual_distance_to_destination", "max"),
        osrm_time_min=("osrm_time", "max"),
        actual_time_min=("actual_time", "max"),
    ).reset_index()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    legs.to_csv(OUT_PATH, index=False)
    print(f"{len(legs):,} legs -> {OUT_PATH}")


if __name__ == "__main__":
    main()
