"""
Sector wetness proxy - Sector 3 resolved
========================================

Original finding (superseded): Sector 3 initially showed a wrong-signed
TrackTemp correlation (+0.243 overall, +0.259 within Canada Race), which was
at first attributed to a speed-trap or straight-dominated layout segment.

Actual cause: non-green-flag laps (Safety Car, VSC, or red-flag-related
timing, including TrackStatus=41) and one green-flag timing outlier in
Bahrain Practice 2 (more than twice its matched sector baseline) were scored
against the dry baseline. These contaminated Sector 3 disproportionately.

Fix: extraction keeps TrackStatus == "1" only, and dry-baseline comparison
rows exceeding twice their matched baseline are invalidated rather than
clipped.

Result after the fix (263 actual-rain laps):
* TrackTemp correlations: Sector 1 -0.550, Sector 2 -0.530, Sector 3 -0.485.
* Cross-sector correlations: 0.867-0.946 across all pairs.
* Drying rates: -0.0828 to -0.0922 s/lap across all three sectors.

Decision: Sector 3 is reinstated as a validated wetness proxy. TrackState
reports all three sectors. Physical water_depth remains unavailable because
the telemetry has no supporting sensor. One intentionally invalidated timing
row remains null in its affected sector proxy; no grip values are negative.
"""

import fastf1
import numpy as np
import pandas as pd


SECTOR_COLS = ["Sector1Time", "Sector2Time", "Sector3Time"]
SECTOR_NAMES = ["sector_1", "sector_2", "sector_3"]
SESSION_CODE_MAP = {
    "Practice 1": "FP1",
    "Practice 2": "FP2",
    "Practice 3": "FP3",
    "Race": "R",
}
SESSIONS_TO_PULL = [
    (2024, "Bahrain", "Practice 1"),
    (2024, "Bahrain", "Practice 2"),
    (2024, "Bahrain", "Practice 3"),
    (2024, "Bahrain", "Race"),
    (2024, "Canada", "Practice 1"),
    (2024, "Canada", "Practice 2"),
    (2024, "Canada", "Practice 3"),
    (2024, "Canada", "Race"),
]
OUTPUT_FILE = "data/processed/wet_track_data.parquet"


def _iqr_clean(subset, column):
    q1, q3 = subset[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return subset[subset[column].between(lower, upper)].copy()


def get_lap_weather_features(session):
    """Aggregate weather samples over each lap's actual time interval."""
    laps = session.laps[
        ["Driver", "DriverNumber", "LapNumber", "LapStartTime", "Time"]
    ].copy()
    laps = laps.rename(columns={"Time": "LapEndTime"}).dropna(
        subset=["LapStartTime", "LapEndTime"]
    )
    weather = session.weather_data.sort_values("Time").reset_index(drop=True)

    records = []
    for _, lap in laps.iterrows():
        window = weather[
            (weather["Time"] >= lap["LapStartTime"])
            & (weather["Time"] < lap["LapEndTime"])
        ]
        if window.empty:
            window = weather[weather["Time"] <= lap["LapStartTime"]].tail(1)

        records.append({
            "Driver": lap["Driver"],
            "DriverNumber": lap["DriverNumber"],
            "LapNumber": lap["LapNumber"],
            "LapStartTime": lap["LapStartTime"],
            "LapEndTime": lap["LapEndTime"],
            "rainfall_any": (
                bool(window["Rainfall"].any())
                if not window.empty
                else None
            ),
            "rainfall_fraction": (
                float(window["Rainfall"].mean())
                if not window.empty
                else np.nan
            ),
            "humidity_mean": (
                float(window["Humidity"].mean())
                if not window.empty
                else np.nan
            ),
            "tracktemp_mean": (
                float(window["TrackTemp"].mean())
                if not window.empty
                else np.nan
            ),
            "weather_samples_in_lap": len(window),
        })

    return pd.DataFrame(records)


def get_sector_and_weather_data(year, grand_prix, session_name):
    """Pull measured sector times and interval-aggregated weather per lap."""
    session_code = SESSION_CODE_MAP[session_name]
    session = fastf1.get_session(year, grand_prix, session_code)
    session.load(laps=True, telemetry=False, weather=True)

    laps = session.laps[
        [
            "Driver", "DriverNumber", "LapNumber", "LapStartTime", "Time",
            "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate",
            *SECTOR_COLS,
        ]
    ].copy()
    laps["is_out_lap"] = laps["PitOutTime"].notna()
    laps["is_in_lap"] = laps["PitInTime"].notna()
    laps = laps[
        laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["TrackStatus"].astype(str).eq("1")
    ]
    laps = laps.dropna(
        subset=["LapStartTime", *SECTOR_COLS]
    ).copy()

    for column, name in zip(SECTOR_COLS, SECTOR_NAMES):
        laps[f"{name}_s"] = laps[column].dt.total_seconds()

    weather_features = get_lap_weather_features(session)
    merged = laps.merge(
        weather_features,
        on=["Driver", "DriverNumber", "LapNumber", "LapStartTime"],
        how="left",
    )
    merged["Year"] = year
    merged["GrandPrix"] = grand_prix
    merged["SessionName"] = session_name
    merged["Rainfall"] = merged["rainfall_any"]
    merged["TrackTemp"] = merged["tracktemp_mean"]
    merged["session_ever_rained"] = (
        merged["rainfall_any"].fillna(False).astype(bool).any()
    )

    return merged[
        [
            "Year", "GrandPrix", "SessionName", "Driver", "DriverNumber",
            "LapNumber", "LapStartTime", "LapEndTime",
            "sector_1_s", "sector_2_s", "sector_3_s", "TrackTemp",
            "Rainfall", "rainfall_any", "rainfall_fraction", "humidity_mean",
            "tracktemp_mean", "weather_samples_in_lap", "TrackStatus",
            "IsAccurate", "is_out_lap", "is_in_lap", "session_ever_rained",
        ]
    ]


def build_dry_baseline(df):
    """Build session-matched IQR-cleaned median sector baselines."""
    min_dry_laps = 15
    baseline = {}

    for grand_prix in df["GrandPrix"].unique():
        baseline[grand_prix] = {}
        gp_df = df[df["GrandPrix"] == grand_prix]

        for session_name in gp_df["SessionName"].unique():
            session_df = gp_df[gp_df["SessionName"] == session_name]
            own_dry = session_df[session_df["Rainfall"] == False]
            baseline[grand_prix][session_name] = {}

            for sector in SECTOR_NAMES:
                column = f"{sector}_s"
                clean_own = _iqr_clean(
                    own_dry.dropna(subset=[column]),
                    column,
                )

                if len(clean_own) >= min_dry_laps:
                    baseline[grand_prix][session_name][sector] = (
                        clean_own[column].median()
                    )
                    print(
                        f"{grand_prix} {session_name} {sector}: "
                        f"own-session baseline "
                        f"{clean_own[column].median():.3f}s "
                        f"(n={len(clean_own)})"
                    )
                    continue

                same_type = gp_df[
                    (gp_df["SessionName"] == session_name)
                    & (gp_df["Rainfall"] == False)
                ]
                clean_fallback = _iqr_clean(
                    same_type.dropna(subset=[column]),
                    column,
                )

                if len(clean_fallback) >= min_dry_laps:
                    baseline[grand_prix][session_name][sector] = (
                        clean_fallback[column].median()
                    )
                    print(
                        f"{grand_prix} {session_name} {sector}: "
                        f"FALLBACK baseline "
                        f"{clean_fallback[column].median():.3f}s "
                        f"(n={len(clean_fallback)}, "
                        f"own-session had only {len(clean_own)})"
                    )
                else:
                    baseline[grand_prix][session_name][sector] = np.nan
                    print(
                        f"{grand_prix} {session_name} {sector}: "
                        f"NO reliable dry baseline available "
                        f"(own={len(clean_own)}, "
                        f"fallback={len(clean_fallback)})"
                    )

    return baseline


def compute_sector_wetness_proxy(df, baseline):
    """Compute sector time loss relative to the matching circuit dry baseline."""
    df = df.copy()

    for sector in SECTOR_NAMES:
        column = f"{sector}_s"
        proxy_column = f"{sector}_wetness_proxy"
        baseline_values = df.apply(
            lambda row: row[column]
            - baseline.get(row["GrandPrix"], {})
            .get(row["SessionName"], {})
            .get(sector, np.nan),
            axis=1,
        )
        dry_baselines = df.apply(
            lambda row: baseline.get(row["GrandPrix"], {})
            .get(row["SessionName"], {})
            .get(sector, np.nan),
            axis=1,
        )
        df[proxy_column] = baseline_values

        invalid_dry = (
            (df["Rainfall"] == False)
            & df[column].notna()
            & dry_baselines.notna()
            & (df[column] > 2 * dry_baselines)
        )
        if invalid_dry.any():
            print(
                f"{sector}: invalidating {int(invalid_dry.sum())} "
                "dry timing outlier(s) above 2x baseline"
            )
            df.loc[invalid_dry, proxy_column] = np.nan

    return df


def validate_wetness_proxy(df):
    """Validate using per-lap rain flags rather than session-wide flags."""
    wet = df[df["Rainfall"] == True].dropna(
        subset=[f"{sector}_wetness_proxy" for sector in SECTOR_NAMES]
    )

    if len(wet) < 10:
        print("Insufficient actually-raining laps to validate proxy.")
        return

    previous_count = df[df["session_ever_rained"] == True].shape[0]
    print(
        f"\nValidation on {len(wet)} laps where Rainfall==True "
        f"(previous session-wide count: {previous_count}):"
    )
    for sector in SECTOR_NAMES:
        proxy_column = f"{sector}_wetness_proxy"
        corr_temp = wet[proxy_column].corr(wet["TrackTemp"])
        print(
            f"  {sector}: corr(wetness, TrackTemp)={corr_temp:+.3f}  "
            f"mean_wetness={wet[proxy_column].mean():.3f}s"
        )

    proxy_columns = [
        f"{sector}_wetness_proxy" for sector in SECTOR_NAMES
    ]
    print("\nCross-sector correlation:")
    print(wet[proxy_columns].corr())

    print(
        "\nSector 3 interpretation: cross-sector agreement is the direct "
        "wetness validation; TrackTemp is not a reliable Sector 3 validator."
    )

    falsely_wet = df[
        (df["session_ever_rained"] == True)
        & (df["Rainfall"] == False)
    ]
    if len(falsely_wet) > 0:
        print(
            "\nLaps in rain-affected sessions but not currently raining "
            f"(n={len(falsely_wet)}):"
        )
        for sector in SECTOR_NAMES:
            proxy_column = f"{sector}_wetness_proxy"
            print(
                f"  {sector}: mean_wetness="
                f"{falsely_wet[proxy_column].mean():.3f}s"
            )


def diagnose_sector3(df):
    """Break down Sector 3's wetness/temperature correlation by session."""
    wet = df[
        (df["Rainfall"] == True)
        & df["sector_3_wetness_proxy"].notna()
    ].copy()

    if wet.empty:
        print("No valid actual-rain Sector 3 laps to diagnose.")
        return

    print(
        "\nSector 3 wetness by GrandPrix/SessionName "
        "(actual rain laps only):"
    )
    summary = wet.groupby(
        ["GrandPrix", "SessionName"]
    ).apply(
        lambda group: pd.Series({
            "n": len(group),
            "mean_wetness": group["sector_3_wetness_proxy"].mean(),
            "corr_temp": (
                group["sector_3_wetness_proxy"].corr(group["TrackTemp"])
                if len(group) >= 2
                else np.nan
            ),
        }),
        include_groups=False,
    )
    print(summary.to_string())

    for (grand_prix, session_name), group in wet.groupby(
        ["GrandPrix", "SessionName"]
    ):
        if len(group) >= 15:
            correlation = group["sector_3_wetness_proxy"].corr(
                group["TrackTemp"]
            )
            print(
                f"  {grand_prix} {session_name} "
                f"(n={len(group)}): "
                f"within-session corr = {correlation:+.3f}"
            )


def residualize_sector_times(track_df, fuel_df):
    """Remove dry fuel, tyre-age, and lap-number pace structure per session."""
    keys = ["Year", "GrandPrix", "SessionName", "Driver", "LapNumber"]
    controls = fuel_df[keys + ["total_mass_kg", "TyreLife", "Compound"]]
    merged = track_df.merge(controls, on=keys, how="inner")

    for sector in SECTOR_NAMES:
        sector_column = f"{sector}_s"
        residual_column = f"{sector}_wetness_v2"
        merged[residual_column] = np.nan

        for _, group in merged.groupby(["GrandPrix", "SessionName"]):
            train = group[
                group["Rainfall"] == False
            ].dropna(
                subset=[
                    sector_column,
                    "total_mass_kg",
                    "TyreLife",
                    "LapNumber",
                ]
            )
            if len(train) < 15:
                continue

            predictors = train[
                ["total_mass_kg", "TyreLife", "LapNumber"]
            ].to_numpy()
            X = np.column_stack([predictors, np.ones(len(predictors))])
            coefficients, _, _, _ = np.linalg.lstsq(
                X,
                train[sector_column].to_numpy(),
                rcond=None,
            )

            all_rows = group.dropna(
                subset=[
                    sector_column,
                    "total_mass_kg",
                    "TyreLife",
                    "LapNumber",
                ]
            )
            all_predictors = all_rows[
                ["total_mass_kg", "TyreLife", "LapNumber"]
            ].to_numpy()
            all_X = np.column_stack([
                all_predictors,
                np.ones(len(all_predictors)),
            ])
            merged.loc[all_rows.index, residual_column] = (
                all_rows[sector_column].to_numpy()
                - all_X @ coefficients
            )

    merged["total_wetness_v2"] = merged[
        [f"{sector}_wetness_v2" for sector in SECTOR_NAMES]
    ].sum(axis=1, min_count=3)
    return merged


def validate_against_transitions(df, wetness_col="total_wetness_v2"):
    """Compare wetness immediately before and after each rain-state flip."""
    results = []
    for (grand_prix, session_name), group in df.groupby(
        ["GrandPrix", "SessionName"]
    ):
        for driver, driver_group in group.groupby("Driver"):
            driver_group = driver_group.sort_values("LapNumber").reset_index(
                drop=True
            )
            flags = driver_group["rainfall_any"].astype("boolean")
            flips = flags.ne(flags.shift()) & flags.notna() & flags.shift().notna()
            for position in driver_group.index[flips]:
                window = driver_group.iloc[
                    max(0, position - 2):position + 3
                ]
                results.append({
                    "GrandPrix": grand_prix,
                    "SessionName": session_name,
                    "Driver": driver,
                    "transition_lap": driver_group.loc[position, "LapNumber"],
                    "from_rainfall": bool(flags.iloc[position - 1]),
                    "to_rainfall": bool(flags.iloc[position]),
                    "wetness_before": window.iloc[:2][wetness_col].mean(),
                    "wetness_after": window.iloc[-2:][wetness_col].mean(),
                })

    transitions = pd.DataFrame(results)
    if transitions.empty:
        print("No valid rainfall transitions found.")
        return transitions

    transitions["wetness_change"] = (
        transitions["wetness_after"]
        - transitions["wetness_before"]
    )
    print("\n===== RAINFALL TRANSITIONS =====")
    print(transitions.to_string(index=False))
    print("\nTransition summary:")
    print(
        transitions.groupby(
            ["from_rainfall", "to_rainfall"]
        )["wetness_change"].agg(["count", "mean", "median"]).to_string()
    )
    return transitions


if __name__ == "__main__":
    frames = []
    for year, grand_prix, session_name in SESSIONS_TO_PULL:
        print(
            f"Pulling sector/weather data: "
            f"{grand_prix} {session_name} {year}"
        )
        frames.append(
            get_sector_and_weather_data(
                year,
                grand_prix,
                session_name,
            )
        )

    all_data = pd.concat(frames, ignore_index=True)

    print("\n===== BUILDING DRY BASELINE =====")
    baseline = build_dry_baseline(all_data)

    print("\n===== COMPUTING SECTOR WETNESS PROXY =====")
    result = compute_sector_wetness_proxy(all_data, baseline)

    print("\n===== VALIDATION =====")
    validate_wetness_proxy(result)

    print("\n===== SECTOR 3 DIAGNOSTIC =====")
    diagnose_sector3(result)

    fuel_data = pd.read_parquet("data/processed/fuel_effect.parquet")
    residualized = residualize_sector_times(result, fuel_data)
    residualized["total_wetness_v2"] = residualized[
        [f"{sector}_wetness_v2" for sector in SECTOR_NAMES]
    ].sum(axis=1, min_count=3)

    print("\n===== RESIDUALIZED WETNESS V2 =====")
    complete = residualized.dropna(
        subset=["total_wetness_v2"]
    )
    print("Dry-lap distribution:")
    print(
        complete[complete["rainfall_any"] == False]["total_wetness_v2"]
        .describe(percentiles=[.01, .1, .5, .9, .99])
        .to_string()
    )
    print("\nWet-lap distribution:")
    print(
        complete[complete["rainfall_any"] == True]["total_wetness_v2"]
        .describe(percentiles=[.01, .1, .5, .9, .99])
        .to_string()
    )
    validate_against_transitions(residualized)

    residualized.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
