import fastf1
import pandas as pd
import numpy as np

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
    Pulls real LapTime, Driver, LapNumber straight from FastF1 —
    untouched, real telemetry, same as Speed/RPM/etc.
    """
    session_code = SESSION_CODE_MAP[session_name]
    session = fastf1.get_session(year, grand_prix, session_code)
    
    # Explicitly load laps data while skipping heavy telemetry/weather arrays
    session.load(laps=True, telemetry=False, weather=False)

    laps = session.laps[["Driver", "LapNumber", "LapTime", "PitInTime", "PitOutTime"]].copy()

    # Drop pit in/out laps — contaminated lap times, not representative of pace
    laps = laps[laps["PitInTime"].isna() & laps["PitOutTime"].isna()]

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()
    laps = laps.dropna(subset=["LapTime_s"])

    laps["Year"] = year
    laps["GrandPrix"] = grand_prix
    laps["SessionName"] = session_name

    return laps[["Year", "GrandPrix", "SessionName", "Driver", "LapNumber", "LapTime_s"]]


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
        except Exception as e:
            print(f"  SKIPPED — Error loading session data: {e}")
            failed.append((row["GrandPrix"], row["SessionName"]))

    if failed:
        print(f"\nSessions skipped due to missing/unloadable lap data: {failed}")

    if not all_lap_times:
        print("\nNo sessions had usable lap time data. Exiting.")
        return

    lap_times = pd.concat(all_lap_times, ignore_index=True)

    merged = per_lap_mass.merge(
        lap_times,
        on=["Year", "GrandPrix", "SessionName", "Driver", "LapNumber"],
        how="inner",
    )

    print(f"\nMerged laps with both mass and real lap time: {len(merged)}")

    print("\n===== FUEL MASS vs LAP TIME (per GrandPrix) =====")
    for gp, group in merged.groupby("GrandPrix"):
        if len(group) < 5:
            continue
        slope, intercept = np.polyfit(group["total_mass_kg"], group["LapTime_s"], 1)
        corr = np.corrcoef(group["total_mass_kg"], group["LapTime_s"])[0, 1]
        print(f"\n{gp}  (n={len(group)})")
        print(f"  seconds per kg of fuel: {slope:.4f} s/kg")
        print(f"  correlation (mass vs laptime): {corr:.3f}")

    merged.to_parquet("data/processed/fuel_effect.parquet", index=False)
    print("\nSaved: data/processed/fuel_effect.parquet")


if __name__ == "__main__":
    main()
