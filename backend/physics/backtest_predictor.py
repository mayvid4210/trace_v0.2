import numpy as np
import pandas as pd


CAR_MASS_KG = 798.0
TYRE_DEGRADATION = {
    "SOFT": 0.0627,
    "MEDIUM": 0.0447,
    "HARD": 0.1060,
}
FUEL_EFFECT_S_PER_KG = {
    "Bahrain": 0.0357,
    "Canada": 0.0261,
}
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
SECTORS = [1, 2, 3]


def _iqr_clean(df, column):
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[df[column].between(lower, upper)].copy()


def get_base_laptime(track_state_df, circuit):
    """Return an IQR-cleaned median dry sector-sum baseline for a circuit."""
    circuit_df = track_state_df[
        (track_state_df["GrandPrix"] == circuit)
        & (track_state_df["Rainfall"] == False)
        & (track_state_df["TrackStatus"].astype(str) == "1")
    ].copy()

    circuit_df["full_lap_baseline"] = sum(
        circuit_df[f"sector_{sector}_s"]
        - circuit_df[f"sector_{sector}_wetness_proxy"]
        for sector in SECTORS
    )
    clean = _iqr_clean(
        circuit_df.dropna(subset=["full_lap_baseline"]),
        "full_lap_baseline",
    )

    if clean.empty:
        raise ValueError(f"No valid dry baseline rows available for {circuit}")

    return float(clean["full_lap_baseline"].median())


def predict_lap_time(
    base_laptime,
    fuel_effect_rate,
    compound,
    tyre_age,
    fuel_mass_kg,
    sector_wetness=None,
):
    """Predict lap time from validated additive empirical terms."""
    tyre_penalty = TYRE_DEGRADATION.get(compound, 0.0) * tyre_age
    fuel_penalty = fuel_effect_rate * fuel_mass_kg
    wetness_penalty = 0.0
    if sector_wetness is not None:
        wetness_penalty = sum(
            value for value in sector_wetness
            if pd.notna(value) and value > 0
        )

    return base_laptime + tyre_penalty + fuel_penalty + wetness_penalty


def backtest(fuel_df, track_state_df, session_filter="Race"):
    """Backtest every eligible joined lap and report rejection reasons."""
    keys = ["Year", "GrandPrix", "SessionName", "Driver", "LapNumber"]
    track_columns = keys + [
        "sector_1_wetness_proxy",
        "sector_2_wetness_proxy",
        "sector_3_wetness_proxy",
        "Rainfall",
    ]
    track = track_state_df[track_columns].copy()
    track = track.rename(columns={"Rainfall": "track_rainfall"})
    merged = fuel_df.merge(track, on=keys, how="inner")

    rejection_counts = {}
    results = []
    base_laptimes = {
        circuit: get_base_laptime(track_state_df, circuit)
        for circuit in merged["GrandPrix"].dropna().unique()
    }

    for _, row in merged.iterrows():
        if session_filter and row["SessionName"] != session_filter:
            rejection_counts["session_filter"] = (
                rejection_counts.get("session_filter", 0) + 1
            )
            continue
        if row["Compound"] not in set(TYRE_DEGRADATION) | WET_COMPOUNDS:
            rejection_counts["unrecognized_compound"] = (
                rejection_counts.get("unrecognized_compound", 0) + 1
            )
            continue
        required = [
            "LapTime_s", "TyreLife", "total_mass_kg",
        ]
        if any(pd.isna(row[column]) for column in required):
            rejection_counts["missing_required_value"] = (
                rejection_counts.get("missing_required_value", 0) + 1
            )
            continue

        sector_wetness = None
        if row["track_rainfall"] == True:
            sector_wetness = [
                row[f"sector_{sector}_wetness_proxy"]
                for sector in SECTORS
            ]
            if any(pd.isna(value) for value in sector_wetness):
                rejection_counts["missing_wetness_proxy"] = (
                    rejection_counts.get("missing_wetness_proxy", 0) + 1
                )
                continue

        compound = row["Compound"]
        effective_compound = (
            compound if compound in TYRE_DEGRADATION else None
        )
        predicted = predict_lap_time(
            base_laptime=base_laptimes[row["GrandPrix"]],
            fuel_effect_rate=FUEL_EFFECT_S_PER_KG[row["GrandPrix"]],
            compound=effective_compound,
            tyre_age=row["TyreLife"],
            fuel_mass_kg=row["total_mass_kg"] - CAR_MASS_KG,
            sector_wetness=sector_wetness,
        )
        error = predicted - row["LapTime_s"]
        results.append({
            "GrandPrix": row["GrandPrix"],
            "Compound": compound,
            "is_wet": bool(row["track_rainfall"]),
            "actual": row["LapTime_s"],
            "predicted": predicted,
            "error": error,
            "abs_error": abs(error),
        })

    results_df = pd.DataFrame(results)
    print(
        f"Backtested {len(results_df)} laps "
        f"({sum(rejection_counts.values())} rejected)\n"
    )
    print("Rejection reasons:")
    print(rejection_counts or "none")

    if results_df.empty:
        raise ValueError("No eligible laps remained for backtesting")

    print("\n===== OVERALL =====")
    print(f"MAE:  {results_df['abs_error'].mean():.3f}s")
    print(
        f"RMSE: {np.sqrt((results_df['error'] ** 2).mean()):.3f}s"
    )
    print(
        "Mean signed error: "
        f"{results_df['error'].mean():+.3f}s (bias direction)"
    )

    print("\n===== BY COMPOUND =====")
    print(results_df.groupby("Compound")["abs_error"].agg(["mean", "count"]))

    print("\n===== BY CIRCUIT =====")
    print(results_df.groupby("GrandPrix")["abs_error"].agg(["mean", "count"]))

    print("\n===== WET vs DRY =====")
    print(results_df.groupby("is_wet")["abs_error"].agg(["mean", "count"]))

    return results_df


if __name__ == "__main__":
    fuel_df = pd.read_parquet("data/processed/fuel_effect.parquet")
    track_state_df = pd.read_parquet("data/processed/track_state_data.parquet")
    results_df = backtest(fuel_df, track_state_df, session_filter="Race")
    results_df.to_parquet(
        "data/processed/backtest_results.parquet",
        index=False,
    )
    print("\nSaved: data/processed/backtest_results.parquet")
