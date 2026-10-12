"""Assigns latitude/longitude to every hub -> data/processed/hubs.csv

Delhivery center codes look like IND<pincode><suffix>, so the pincode can be
read straight from the code. The GeoNames pincode list only has ~11k of
India's ~19k pincodes, so about half of the hubs need an estimate:
  exact     pincode is in the list
  prefix5/4/3  median of listed pincodes sharing the first 5/4/3 digits
  name      code has no usable pincode, match the city in the hub name
Hubs that fit none of these are left out of the graph.
"""
import os
import re
import urllib.request
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_URL = "https://raw.githubusercontent.com/sanand0/pincode/master/data/IN.csv"
PIN_PATH = os.path.join(HERE, "raw", "IN_pincodes.csv")
LEGS_PATH = os.path.join(HERE, "processed", "legs.csv")
OUT_PATH = os.path.join(HERE, "processed", "hubs.csv")
REPORT_PATH = os.path.join(os.path.dirname(HERE), "graph", "reports", "geo_report.txt")

# hub-name prefixes that are abbreviations of the city name in the pincode list
MAX_NAME_SPREAD_KM = 100
CITY_ALIASES = {"blr": "bangalore", "bengaluru": "bangalore", "maa": "chennai",
                "cochin": "kochi", "pallakad": "palakkad", "trivandrum": "thiruvananthapuram"}


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = (np.sin((lat2 - lat1) * p / 2) ** 2
         + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def load_pincodes():
    if not os.path.exists(PIN_PATH):
        os.makedirs(os.path.dirname(PIN_PATH), exist_ok=True)
        urllib.request.urlretrieve(PIN_URL, PIN_PATH)
    pin = pd.read_csv(PIN_PATH)
    pin["pincode"] = pin["key"].str.replace("IN/", "", regex=False)
    pin = pin.dropna(subset=["latitude", "longitude"]).drop_duplicates("pincode")
    return pin.reset_index(drop=True)


def hub_table(legs):
    src = legs[["source_center", "source_name"]].set_axis(["center", "name"], axis=1)
    dst = legs[["destination_center", "destination_name"]].set_axis(["center", "name"], axis=1)
    hubs = pd.concat([src, dst]).sort_values("name").drop_duplicates("center").reset_index(drop=True)
    hubs["pincode"] = hubs["center"].str.extract(r"^IND(\d{6})")[0]
    hubs.loc[hubs["pincode"] == "000000", "pincode"] = np.nan
    hubs["state"] = hubs["name"].str.extract(r"\(([^)]+)\)\s*$")[0]
    return hubs


def estimate_from_pincode(pin, pincode):
    """Returns (lat, lon, source) for a pincode, using neighbours if it is not listed."""
    hit = pin[pin["pincode"] == pincode]
    if len(hit):
        return hit["latitude"].iloc[0], hit["longitude"].iloc[0], "exact"
    for k in (5, 4, 3):
        same = pin[pin["pincode"].str[:k] == pincode[:k]]
        if len(same):
            return same["latitude"].median(), same["longitude"].median(), f"prefix{k}"
    return np.nan, np.nan, None


def estimate_from_name(pin, name, state):
    if not isinstance(name, str):
        return np.nan, np.nan, None
    city = re.split(r"[_ ]", name)[0].lower()
    city = CITY_ALIASES.get(city, city)
    cand = pin[pin["place_name"].str.lower().str.contains(city, regex=False, na=False)]
    if isinstance(state, str) and len(cand):
        s = state.lower()
        by_state = cand[cand["admin_name1"].str.lower().apply(lambda a: s in a or a in s)]
        cand = by_state if len(by_state) else cand
    if len(cand) == 0:
        return np.nan, np.nan, None
    lat, lon = cand["latitude"].median(), cand["longitude"].median()
    # repeated place names (e.g. Bilaspur) scatter across states, don't trust those
    spread = haversine_km(cand["latitude"].values, cand["longitude"].values, lat, lon)
    if np.percentile(spread, 75) > MAX_NAME_SPREAD_KM:
        return np.nan, np.nan, None
    return lat, lon, "name"


def locate(hubs, pin):
    out = []
    for h in hubs.itertuples():
        lat, lon, src = np.nan, np.nan, None
        if isinstance(h.pincode, str):
            lat, lon, src = estimate_from_pincode(pin, h.pincode)
        if src is None:
            lat, lon, src = estimate_from_name(pin, h.name, h.state)
        out.append((lat, lon, src))
    hubs[["lat", "lon", "geo_source"]] = pd.DataFrame(out, index=hubs.index)
    return hubs


def tier_errors(pin, hubs):
    """Hides each listed hub pincode and re-estimates it, to size the error per method."""
    listed = hubs[hubs["pincode"].isin(pin["pincode"])].merge(
        pin[["pincode", "latitude", "longitude"]], on="pincode")
    rows = []
    for h in listed.itertuples():
        pool = pin[pin["pincode"] != h.pincode]
        for k in (5, 4, 3):
            same = pool[pool["pincode"].str[:k] == h.pincode[:k]]
            if len(same):
                rows.append((f"prefix{k}", haversine_km(h.latitude, h.longitude,
                             same["latitude"].median(), same["longitude"].median())))
        lat, lon, src = estimate_from_name(pool, h.name, h.state)
        if src:
            rows.append(("name", haversine_km(h.latitude, h.longitude, lat, lon)))
    return pd.DataFrame(rows, columns=["method", "err_km"])


def main():
    legs = pd.read_csv(LEGS_PATH)
    pin = load_pincodes()
    hubs = locate(hub_table(legs), pin)

    traffic = pd.concat([legs["source_center"], legs["destination_center"]]).value_counts()
    hubs["n_legs"] = hubs["center"].map(traffic)
    located = hubs["lat"].notna()
    in_india = hubs["lat"].between(6, 37.5) & hubs["lon"].between(68, 98)
    hubs.loc[~in_india, ["lat", "lon", "geo_source"]] = np.nan

    lines = [f"hubs: {len(hubs):,}   located: {hubs['lat'].notna().sum():,}"]
    lines.append("\nby source (hubs | share of leg endpoints):")
    for src, g in hubs.groupby("geo_source", dropna=False):
        lines.append(f"  {str(src):<8} {len(g):>5,} | {g['n_legs'].sum() / hubs['n_legs'].sum():.1%}")
    lines.append(f"  outside India bounding box, removed: {(located & ~in_india).sum()}")

    err = tier_errors(pin, hub_table(legs).dropna(subset=["pincode"]))
    lines.append("\nerror of each estimate, tested by hiding the pincode of hubs that have one:")
    for method, g in err.groupby("method"):
        lines.append(f"  {method:<8} n={len(g):>4}  median {g['err_km'].median():5.1f} km | "
                     f"p90 {g['err_km'].quantile(.9):6.1f} km")

    # geocoding check: straight-line distance vs the road distance recorded in the data
    xy = hubs.set_index("center")[["lat", "lon"]]
    m = legs.join(xy.add_prefix("s_"), on="source_center").join(xy.add_prefix("d_"), on="destination_center")
    exact = hubs.set_index("center")["geo_source"].eq("exact")
    m = m[m["source_center"].map(exact).fillna(False) & m["destination_center"].map(exact).fillna(False)]
    straight = haversine_km(m["s_lat"], m["s_lon"], m["d_lat"], m["d_lon"])
    ratio = (m["actual_distance_to_destination"] / straight)[straight > 20]
    lines.append(f"\nrecorded leg distance / straight-line distance ({len(ratio):,} legs, exact coordinates):")
    lines.append(f"  median {ratio.median():.2f}. Below 1 is not possible for a full road route, so the")
    lines.append("  distance fields are cut off at the last checkpoint on long legs. Geometry uses coordinates.")

    hubs = hubs.dropna(subset=["lat"])
    hubs[["center", "name", "state", "lat", "lon", "geo_source", "n_legs"]].to_csv(OUT_PATH, index=False)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
