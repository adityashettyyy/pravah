"""Build the hub graph: nodes = hubs, edges = hub pairs that appear as legs.

Coordinates: the Delhivery hub code embeds a 6-digit PIN code
(e.g. IND388121AAA -> 388121). We map that to lat/long using an open
pincode->lat/long table (github.com/sanand0/pincode). Coverage:
  - exact pincode match, else
  - mean of all known pincodes sharing the first 3 digits (same postal
    "sorting district"), else
  - mean of all known pincodes sharing the first 2 digits (postal circle)
A handful of hubs have no digits in their code at all and are dropped.

Road distance: OSRM (or any routing API) is not reachable from this sandbox
(the network allowlist has no routing host), so there is no way to query
real road distance for hub pairs that never appear in the data. For hub
pairs that DO appear as legs, we already have real driving distance
(`actual_distance_to_destination`, OSRM-derived, from the raw data) - we use
that as the edge distance. We do NOT fabricate road distance for pairs with
no data; the graph only has edges where legs actually exist. Haversine
distance (straight-line, from the coordinates above) is computed alongside
as a sanity check and for the RL env's cost model, and is clearly labelled
as straight-line, not road distance.
"""
import os
import re
import sys
import math
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
from problem import load_clean  # noqa: E402

PIN_URL = "https://raw.githubusercontent.com/sanand0/pincode/master/data/IN.csv"
PIN_RAW_PATH = os.path.join(ROOT, "data", "pincode_raw.csv")
OUT_NODES = os.path.join(ROOT, "graph", "nodes.csv")
OUT_EDGES = os.path.join(ROOT, "graph", "edges.csv")
OUT_REPORT = os.path.join(ROOT, "graph", "graph_report.txt")

PIN_RE = re.compile(r"[0-9]{6}")


def extract_pincode(hub_code):
    m = PIN_RE.search(str(hub_code))
    return m.group() if m else None


def load_pincode_table():
    if not os.path.exists(PIN_RAW_PATH):
        raise FileNotFoundError(
            f"{PIN_RAW_PATH} not found. Download it first:\n"
            f"  curl -sL -o {PIN_RAW_PATH} {PIN_URL}"
        )
    pin = pd.read_csv(PIN_RAW_PATH)
    pin["pincode"] = pin["key"].str.replace("IN/", "", regex=False)
    pin = pin.dropna(subset=["latitude", "longitude"])
    return pin


def geocode_hubs(hub_codes, pin):
    exact = pin.set_index("pincode")[["latitude", "longitude"]]
    pin["p3"] = pin["pincode"].str[:3]
    pin["p2"] = pin["pincode"].str[:2]
    p3c = pin.groupby("p3")[["latitude", "longitude"]].mean()
    p2c = pin.groupby("p2")[["latitude", "longitude"]].mean()

    rows = []
    counts = {"exact": 0, "prefix3": 0, "prefix2": 0, "failed": 0}
    for h in hub_codes:
        pincode = extract_pincode(h)
        if pincode is None:
            counts["failed"] += 1
            continue
        if pincode in exact.index:
            lat, lon = exact.loc[pincode, ["latitude", "longitude"]]
            src = "exact"
        elif pincode[:3] in p3c.index:
            lat, lon = p3c.loc[pincode[:3], ["latitude", "longitude"]]
            src = "prefix3"
        elif pincode[:2] in p2c.index:
            lat, lon = p2c.loc[pincode[:2], ["latitude", "longitude"]]
            src = "prefix2"
        else:
            counts["failed"] += 1
            continue
        counts[src] += 1
        rows.append({"hub": h, "pincode": pincode, "latitude": float(lat),
                      "longitude": float(lon), "geocode_source": src})
    return pd.DataFrame(rows), counts


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def build():
    df = load_clean()
    hub_codes = sorted(set(df.source_center) | set(df.destination_center))
    pin = load_pincode_table()
    nodes, counts = geocode_hubs(hub_codes, pin)
    nodes = nodes.set_index("hub")

    # aggregate legs into hub-pair edges (undirected pair key, but keep
    # direction info too since delay can be asymmetric)
    edges = (
        df.groupby(["source_center", "destination_center"])
        .agg(
            n_legs=("actual_time_min", "size"),
            mean_actual_distance_km=("actual_distance_to_destination", "mean"),
            mean_osrm_time_min=("osrm_time_min", "mean"),
            delay_rate=("is_delayed", "mean"),
        )
        .reset_index()
        .rename(columns={"source_center": "src", "destination_center": "dst"})
    )

    # keep only edges where both endpoints geocoded
    edges = edges[edges.src.isin(nodes.index) & edges.dst.isin(nodes.index)].copy()

    hav = []
    for r in edges.itertuples():
        s, d = nodes.loc[r.src], nodes.loc[r.dst]
        hav.append(haversine_km(s.latitude, s.longitude, d.latitude, d.longitude))
    edges["haversine_km"] = hav

    nodes_out = nodes.reset_index()
    nodes_out.to_csv(OUT_NODES, index=False)
    edges.to_csv(OUT_EDGES, index=False)

    # correlation between real road distance and straight-line distance,
    # as a sanity check on the geocoding (and to justify using haversine
    # as an RL cost proxy for hub pairs with no logged leg)
    valid = edges.dropna(subset=["mean_actual_distance_km", "haversine_km"])
    corr = valid["mean_actual_distance_km"].corr(valid["haversine_km"])

    report = []
    report.append(f"Hubs in legs data: {len(hub_codes)}")
    report.append(f"Geocoded: exact pincode {counts['exact']}, "
                   f"3-digit prefix fallback {counts['prefix3']}, "
                   f"2-digit prefix fallback {counts['prefix2']}, "
                   f"failed {counts['failed']}")
    report.append(f"Nodes with coordinates: {len(nodes_out)} "
                   f"({len(nodes_out)/len(hub_codes):.1%} of hubs)")
    report.append(f"Edges (hub pairs with >=1 real leg): {len(edges)}")
    report.append(f"Edges with n_legs>=3: {(edges.n_legs>=3).sum()}")
    report.append(f"corr(real road distance, haversine distance) = {corr:.3f} "
                   f"on {len(valid)} edges")
    report.append("")
    report.append("Road distance note: no routing API is reachable from this "
                   "sandbox. Distance on edges that exist in the data is the "
                   "real OSRM-derived driving distance from the raw file. "
                   "haversine_km is straight-line and is only used where no "
                   "real distance exists (see rl/env.py).")
    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    build()
