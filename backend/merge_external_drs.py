import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

FASTF1_FILE = BASE_DIR / "data" / "processed" / "telemetry.parquet"
EXTERNAL_FILE = BASE_DIR / "data" / "processed" / "external_telemetry.parquet"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "telemetry_with_drs.parquet"

KEYS = [
    "Year",
    "GrandPrix",
    "SessionName",
    "Driver",
    "LapNumber",
]


def main():
    print("Loading FastF1 telemetry...")
    fastf1 = pd.read_parquet(FASTF1_FILE)

    print("Loading external telemetry...")
    external = pd.read_parquet(EXTERNAL_FILE)

    print(f"FastF1 rows:   {len(fastf1):,}")
    print(f"External rows: {len(external):,}")

    # Convert FastF1 timedelta to elapsed seconds.
    fastf1["Time_sec"] = fastf1["Time"].dt.total_seconds()

    # External Time is already in seconds.
    external["Time_sec"] = pd.to_numeric(
        external["Time"],
        errors="coerce"
    )

    # Keep only the fields needed for DRS matching.
    external_drs = external[
        KEYS + ["Time_sec", "DRS"]
    ].copy()

    # Remove invalid rows.
    fastf1 = fastf1.dropna(
        subset=KEYS + ["Time_sec"]
    ).copy()

    external_drs = external_drs.dropna(
        subset=KEYS + ["Time_sec", "DRS"]
    ).copy()

    fastf1["LapNumber"] = pd.to_numeric(
        fastf1["LapNumber"],
        errors="coerce"
    ).fillna(-1).astype("int64")

    external_drs["LapNumber"] = pd.to_numeric(
        external_drs["LapNumber"],
        errors="coerce"
    ).fillna(-1).astype("int64")

    # Sort for merge_asof.
    fastf1 = fastf1.sort_values(
        ["Time_sec"] + KEYS
    ).reset_index(drop=True)

    external_drs = external_drs.sort_values(
        ["Time_sec"] + KEYS
    ).reset_index(drop=True)

    print("\nMatching DRS...")
    print("Tolerance: 0.05 seconds")

    result = pd.merge_asof(
        fastf1,
        external_drs,
        on="Time_sec",
        by=KEYS,
        direction="nearest",
        tolerance=0.05,
        suffixes=("", "_external"),
    )

    # Remove temporary matching column.
    result = result.drop(columns=["Time_sec"])

    # Report matching.
    matched = result["DRS"].notna().sum()
    unmatched = result["DRS"].isna().sum()
    total = len(result)

    print("\n========== MATCH RESULT ==========")
    print(f"Total FastF1 rows: {total:,}")
    print(f"Matched DRS rows:  {matched:,}")
    print(f"Unmatched rows:    {unmatched:,}")

    if total > 0:
        print(f"Match percentage:  {matched / total * 100:.2f}%")

    print("\nDRS distribution:")
    print(result["DRS"].value_counts(dropna=False))

    print("\n========== COVERAGE ==========")
    print(
        result[
            ["Year", "GrandPrix", "SessionName"]
        ]
        .drop_duplicates()
        .sort_values(
            ["GrandPrix", "SessionName"]
        )
        .to_string(index=False)
    )

    print("\nDrivers:")
    print(
        sorted(
            result["Driver"]
            .dropna()
            .unique()
            .tolist()
        )
    )

    # Save new file. Original FastF1 file remains untouched.
    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n========== COMPLETE ==========")
    print(f"Saved to:")
    print(OUTPUT_FILE)
    print(f"Final rows: {len(result):,}")
    print(f"Final columns: {result.columns.tolist()}")


if __name__ == "__main__":
    main()