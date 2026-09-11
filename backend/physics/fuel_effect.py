try:
    import fastf1
except ModuleNotFoundError:
    fastf1 = None
import pandas as pd
import numpy as np

try:
    from backend.physics.fuel_model import CIRCUIT_FUEL_CALIBRATIONS, get_circuit_fuel_calibration
except ModuleNotFoundError:
    from physics.fuel_model import CIRCUIT_FUEL_CALIBRATIONS, get_circuit_fuel_calibration

PHYSICS_DATA_FILE = "data/processed/physics_data.parquet"

# Map your SessionName values to FastF1's session codes
SESSION_CODE_MAP = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Race": "R",
}


def get_real_lap_times(year, grand_prix, session_name):
    """
    Pulls real lap timing and tyre-state fields straight from FastF1.
    """
    session_code = SESSION_CODE_MAP[session_name]
    session = fastf1.get_session(year, grand_prix, session_code)
    
    # Explicitly load laps data while skipping heavy telemetry/weather arrays
    session.load(laps=True, telemetry=False, weather=False)

    laps = session.laps[[
        "Driver", "LapNumber", "LapTime", "PitInTime", "PitOutTime",
        "Compound", "Stint", "TyreLife"
    ]].copy()

    # Drop pit in/out laps — contaminated lap times, not representative of pace
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps = laps.dropna(subset=["LapTime_s"])

    laps["Year"] = year
    laps["GrandPrix"] = grand_prix
    laps["SessionName"] = session_name

    return laps[[
        "Year", "GrandPrix", "SessionName", "Driver", "LapNumber",
        "LapTime_s", "Compound", "Stint", "TyreLife"
    ]]


def get_weather_for_session(year, grand_prix, session_name):
    """Attach the nearest FastF1 weather sample to each lap start."""
    session_code = SESSION_CODE_MAP[session_name]
    session = fastf1.get_session(year, grand_prix, session_code)
    session.load(laps=True, telemetry=False, weather=True)

    laps = session.laps[
        ["Driver", "LapNumber", "LapStartTime"]
    ].copy()
    weather = session.weather_data[
        ["Time", "AirTemp", "TrackTemp", "Rainfall", "Humidity"]
    ].copy()

    laps = laps.dropna(
        subset=["LapStartTime"]
    ).sort_values("LapStartTime")
    weather = weather.sort_values("Time")

    merged = pd.merge_asof(
        laps,
        weather,
        left_on="LapStartTime",
        right_on="Time",
        direction="nearest",
    )
    merged["Year"] = year
    merged["GrandPrix"] = grand_prix
    merged["SessionName"] = session_name

    return merged[
        [
            "Year", "GrandPrix", "SessionName", "Driver", "LapNumber",
            "AirTemp", "TrackTemp", "Rainfall", "Humidity",
        ]
    ]


def main():
    physics = pd.read_parquet(PHYSICS_DATA_FILE)

    # Per-lap mass: mean of total_mass_kg across that lap's frames —
    # derived from real telemetry (fuel_estimate), nothing invented.
    per_lap_mass = (
        physics.groupby(["Year", "GrandPrix", "SessionName", "Driver", "LapNumber"])["total_mass_kg"]
        .mean()
        .reset_index()
    )

    combos = per_lap_mass[["Year", "GrandPrix", "SessionName"]].drop_duplicates()

    all_lap_times = []
    all_weather = []
    failed = []
    
    for _, row in combos.iterrows():
        print(f"Pulling lap times: {row['GrandPrix']} {row['SessionName']} {row['Year']}")
        try:
            lap_times = get_real_lap_times(row["Year"], row["GrandPrix"], row["SessionName"])
            if lap_times.empty:
                print("  SKIPPED — Session loaded but returned no valid lap times.")
                failed.append((row["GrandPrix"], row["SessionName"]))
                continue
            all_lap_times.append(lap_times)

            weather = get_weather_for_session(
                row["Year"],
                row["GrandPrix"],
                row["SessionName"],
            )
            if weather.empty:
                print("  WEATHER SKIPPED — no lap weather matches returned.")
            else:
                all_weather.append(weather)
        except Exception as e:
            print(f"  SKIPPED — Error loading session data: {e}")
            failed.append((row["GrandPrix"], row["SessionName"]))

    if failed:
        print(f"\nSessions skipped due to missing/unloadable lap data: {failed}")

    if not all_lap_times:
        print("\nNo sessions had usable lap time data. Exiting.")
        return

    lap_times = pd.concat(all_lap_times, ignore_index=True)

    weather = None
    if all_weather:
        weather = pd.concat(all_weather, ignore_index=True)

    merged = per_lap_mass.merge(
        lap_times,
        on=["Year", "GrandPrix", "SessionName", "Driver", "LapNumber"],
        how="inner",
    )

    if weather is not None:
        merged = merged.merge(
            weather,
            on=["Year", "GrandPrix", "SessionName", "Driver", "LapNumber"],
            how="left",
        )

    print("\nOutput columns:")
    print(merged.columns.tolist())

    print(f"\nMerged laps with both mass and real lap time: {len(merged)}")

    print("\n===== MULTIVARIATE FUEL SENSITIVITY (per GrandPrix) =====")
    for gp in merged["GrandPrix"].unique():
        result = fit_multivariate_fuel_sensitivity(merged, circuit=gp)
        print(f"\n{gp}:")
        print(f"  Status: {result['status']}")
        print(f"  Fuel sensitivity: {result['slope']:.4f} s/kg")
        if result.get("r2") is not None:
            print(f"  Model R2: {result['r2']:.3f} (samples={result['samples']})")
        if result.get("reason"):
            print(f"  Note: {result['reason']}")

    merged.to_parquet("data/processed/fuel_effect.parquet", index=False)
    print("\nSaved: data/processed/fuel_effect.parquet")


def fit_multivariate_fuel_sensitivity(df, circuit="Bahrain"):
    """Fit fuel sensitivity using multivariate OLS controlling for Compound and TyreLife.

    Restricted strictly to clean Race sessions.
    """
    if df is None or df.empty:
        return {
            "circuit": circuit,
            "slope": CIRCUIT_FUEL_CALIBRATIONS.get(circuit, None).calibrated_fuel_effect_s_per_kg if circuit in CIRCUIT_FUEL_CALIBRATIONS else 0.0300,
            "status": "INSUFFICIENT_DATA_FALLBACK",
            "samples": 0,
            "r2": None,
        }

    race_laps = df[
        (df["GrandPrix"] == circuit)
        & (df["SessionName"] == "Race")
    ].copy()

    if "PitInTime" in race_laps.columns:
        race_laps = race_laps[race_laps["PitInTime"].isna()]
    if "PitOutTime" in race_laps.columns:
        race_laps = race_laps[race_laps["PitOutTime"].isna()]
    if "TrackStatus" in race_laps.columns:
        race_laps = race_laps[race_laps["TrackStatus"].astype(str) == "1"]

    if race_laps.empty or len(race_laps) < 20:
        fallback_val = CIRCUIT_FUEL_CALIBRATIONS[circuit].calibrated_fuel_effect_s_per_kg if circuit in CIRCUIT_FUEL_CALIBRATIONS else 0.0300
        return {
            "circuit": circuit,
            "slope": fallback_val,
            "status": "INSUFFICIENT_DATA_FALLBACK",
            "samples": len(race_laps),
            "r2": None,
        }

    # If Canada, document the drying-track confounding limitation
    if circuit == "Canada":
        return {
            "circuit": "Canada",
            "slope": CIRCUIT_FUEL_CALIBRATIONS["Canada"].calibrated_fuel_effect_s_per_kg,
            "status": "DOCUMENTED_FALLBACK",
            "reason": "Drying track on late dry laps conflates track evolution with fuel mass delta",
            "samples": len(race_laps),
            "r2": None,
        }

    calib = CIRCUIT_FUEL_CALIBRATIONS.get(circuit)
    nominal_burn = calib.nominal_burn_rate_kg_per_lap if calib else 1.807
    starting_scale = calib.nominal_fuel_scale_kg if calib else 100.0

    # Fuel mass proxy under nominal race burn
    race_laps["fuel_kg"] = starting_scale - nominal_burn * (race_laps["LapNumber"] - 1)

    # Feature matrix: One-hot encoded compounds + TyreLife + fuel_kg
    compounds = [c for c in race_laps["Compound"].dropna().unique() if c]
    feature_cols = ["TyreLife", "fuel_kg"]
    for comp in compounds[:-1]:
        col_name = f"is_{comp}"
        race_laps[col_name] = (race_laps["Compound"] == comp).astype(float)
        feature_cols.append(col_name)

    clean_subset = race_laps.dropna(subset=["LapTime_s"] + feature_cols)
    if len(clean_subset) < 20:
        fallback_val = calib.calibrated_fuel_effect_s_per_kg if calib else 0.0357
        return {
            "circuit": circuit,
            "slope": fallback_val,
            "status": "DOCUMENTED_FALLBACK",
            "samples": len(clean_subset),
            "r2": None,
        }

    X = clean_subset[feature_cols].values
    X = np.column_stack([np.ones(len(X)), X])
    y = clean_subset["LapTime_s"].values

    # OLS closed-form fit
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    fuel_coef = float(beta[2])

    y_pred = X @ beta
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "circuit": circuit,
        "slope": fuel_coef,
        "status": "CALIBRATED_MULTIVARIATE",
        "samples": len(clean_subset),
        "r2": float(r2),
    }


if __name__ == "__main__":
    main()
