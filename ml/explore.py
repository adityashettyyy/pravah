"""Data summary, cleaning log and EDA. Writes reports/eda_report.txt and fig1-3."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from problem import load_legs, add_target, clean, DELAY_THRESHOLD

REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
INK, AMBER, GREEN = "#1A1815", "#C4622A", "#2F6B4F"

NUM_COLS = ["actual_distance_to_destination", "osrm_time_min",
            "actual_time_min", "delay_min", "is_delayed"]


def summarize(raw, df, w):
    w(f"legs (raw): {len(raw):,}   trips: {raw['trip_uuid'].nunique():,}")
    w(f"legs after cleaning: {len(df):,}  ({len(raw) - len(df):,} removed)")
    start = df["od_start_time"]
    w(f"period: {start.min():%Y-%m-%d} to {start.max():%Y-%m-%d}")
    w(f"source hubs: {df['source_center'].nunique():,}   "
      f"destination hubs: {df['destination_center'].nunique():,}")

    w("\nmissing values (raw):")
    miss = raw.isna().sum()
    for col, n in miss[miss > 0].items():
        w(f"  {col:<32} {n:>5,}")
    if miss.sum() == 0:
        w("  none")

    w(f"\ntarget (actual > {DELAY_THRESHOLD} x osrm):")
    vc = df["is_delayed"].value_counts()
    w(f"  on time  {vc.get(0, 0):>7,}  {vc.get(0, 0) / len(df):.1%}")
    w(f"  delayed  {vc.get(1, 0):>7,}  {vc.get(1, 0) / len(df):.1%}")

    w("\nnumeric summary:")
    w(df[NUM_COLS[:-1]].describe().T.round(1).to_string())

    w("\noutliers (1.5 x IQR, kept in the data):")
    for col in NUM_COLS[:-1]:
        q1, q3 = df[col].quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        w(f"  {col:<32} {n_out:>5,}  ({n_out / len(df):.1%})")


def analyse(df, w):
    w("\ndelay rate by route type:")
    for name, row in df.groupby("route_type")["is_delayed"].agg(["mean", "count"]).iterrows():
        w(f"  {name:<10} {row['mean']:.1%}  (n={int(row['count']):,})")

    w("\ndelay rate by distance quartile:")
    q = pd.qcut(df["actual_distance_to_destination"], 4, labels=["Q1 short", "Q2", "Q3", "Q4 long"])
    for name, row in df.groupby(q, observed=True)["is_delayed"].agg(["mean", "count"]).iterrows():
        w(f"  {name:<9} {row['mean']:.1%}  (n={int(row['count']):,})")

    w("\nhighest delay rate by origin hub (100+ legs):")
    hub = origin_hub_rates(df).head(10)
    for name, row in hub.iterrows():
        w(f"  {name:<40} {row['mean']:.1%}  (n={int(row['count']):,})")

    w("\ncorrelation:")
    w(df[NUM_COLS].corr().round(2).to_string())


def origin_hub_rates(df, min_legs=100):
    hub = df.groupby("source_name")["is_delayed"].agg(["mean", "count"])
    return hub[hub["count"] >= min_legs].sort_values("mean", ascending=False)


def plot_all(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    vc = df["is_delayed"].value_counts().sort_index()
    axes[0].bar(["on time", "delayed"], vc.values, color=[INK, AMBER])
    axes[0].set_title("Target balance")
    axes[0].set_ylabel("legs")

    p99 = df["delay_min"].quantile(0.99)
    axes[1].hist(df.loc[df["delay_min"] <= p99, "delay_min"], bins=60, color=INK)
    axes[1].set_title("Delay (actual - osrm), up to 99th percentile")
    axes[1].set_xlabel("minutes")

    for rt, color in zip(sorted(df["route_type"].unique()), [INK, AMBER]):
        sub = df[df["route_type"] == rt]
        axes[2].scatter(sub["actual_distance_to_destination"], sub["delay_min"],
                        s=3, alpha=0.25, label=rt, color=color)
    axes[2].set_title("Delay vs distance")
    axes[2].set_xlabel("distance (km)")
    axes[2].set_ylabel("delay (min)")
    axes[2].legend(frameon=False, markerscale=4)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig1_distributions.png"), dpi=120)
    plt.close()

    corr = df[NUM_COLS].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(corr, cmap="copper_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUM_COLS)), NUM_COLS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(NUM_COLS)), NUM_COLS, fontsize=8)
    for i in range(len(NUM_COLS)):
        for j in range(len(NUM_COLS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(corr.iloc[i, j]) > 0.5 else INK)
    ax.set_title("Correlation")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig2_correlation.png"), dpi=120)
    plt.close()

    top = origin_hub_rates(df).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index, top["mean"], color=AMBER)
    ax.axvline(df["is_delayed"].mean(), color=INK, ls="--", lw=1.2, label="network average")
    ax.set_xlabel("share of outbound legs delayed")
    ax.set_title("Delay rate by origin hub (hubs with 100+ legs)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "fig3_hub_delay_rate.png"), dpi=120)
    plt.close()


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    raw = add_target(load_legs())
    df = clean(raw)

    lines = []
    w = lines.append
    summarize(raw, df, w)
    analyse(df, w)
    plot_all(df)

    report = "\n".join(lines)
    with open(os.path.join(REPORT_DIR, "eda_report.txt"), "w") as f:
        f.write(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
