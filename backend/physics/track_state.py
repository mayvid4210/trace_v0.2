import numpy as np
import pandas as pd


SECTORS = ["sector_1", "sector_2", "sector_3"]
MIN_LAPS_FOR_DRYING_RATE = 10
INPUT_FILE = "data/processed/wet_track_data.parquet"
OUTPUT_FILE = "data/processed/track_state_data.parquet"


def compute_sector_grip(df):
    """Compute normalized sector grip from measured wetness proxies."""
    df = df.copy()

    for sector in SECTORS:
        wetness_column = f"{sector}_wetness_proxy"
        time_column = f"{sector}_s"
        grip_column = f"{sector}_grip"
        dry_baseline = df[time_column] - df[wetness_column]
        df[grip_column] = 1 - df[wetness_column] / dry_baseline

    return df


def fit_drying_rate(df):
    """Fit per-session wetness slopes using actual-rain laps only."""
    results = {}

    wet = df[df["Rainfall"] == True].copy()
    for sector in SECTORS:
        wetness_column = f"{sector}_wetness_proxy"
        subset = wet.dropna(subset=[wetness_column, "LapNumber"])

        for (grand_prix, session_name), group in subset.groupby(
            ["GrandPrix", "SessionName"]
        ):
            if len(group) < MIN_LAPS_FOR_DRYING_RATE:
                continue

            slope, _ = np.polyfit(
                group["LapNumber"],
                group[wetness_column],
                1,
            )
            key = (grand_prix, session_name, sector)
            results[key] = {
                "drying_rate_s_per_lap": slope,
                "n_laps": len(group),
            }
            print(
                f"{grand_prix} {session_name} {sector}: "
                f"drying_rate={slope:+.4f} s/lap "
                f"(n={len(group)})"
            )

    return results


def build_track_state(df, drying_results):
    """Assemble explicit per-lap track-state fields from measured proxies."""
    df = df.copy()

    for sector in SECTORS:
        def lookup_drying(row, current_sector=sector):
            key = (
                row["GrandPrix"],
                row["SessionName"],
                current_sector,
            )
            return drying_results.get(key, {}).get(
                "drying_rate_s_per_lap",
                np.nan,
            )

        df[f"{sector}_drying_rate"] = df.apply(
            lookup_drying,
            axis=1,
        )

    def make_track_state(row):
        return {
            "temperature": row["TrackTemp"],
            "rainfall": row["Rainfall"],
            "grip": {
                sector: row[f"{sector}_grip"]
                for sector in SECTORS
            },
            "sector_wetness": [
                row[f"{sector}_wetness_proxy"]
                for sector in SECTORS
            ],
            "drying_rate": {
                sector: row[f"{sector}_drying_rate"]
                for sector in SECTORS
            },
            "water_depth": None,
        }

    df["track_state"] = df.apply(make_track_state, axis=1)
    return df


if __name__ == "__main__":
    df = pd.read_parquet(INPUT_FILE)

    print("===== SECTOR GRIP =====")
    df = compute_sector_grip(df)
    print(df[[f"{sector}_grip" for sector in SECTORS]].describe())

    print("\n===== DRYING RATE =====")
    drying_results = fit_drying_rate(df)

    print("\n===== BUILDING TRACK STATE =====")
    df = build_track_state(df, drying_results)

    sample_rows = df[df["Rainfall"] == True]
    if not sample_rows.empty:
        print("\nSample TrackState record (one actual-rain lap):")
        for key, value in sample_rows.iloc[0]["track_state"].items():
            print(f"  {key}: {value}")

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")