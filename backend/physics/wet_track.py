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


def get_sector_and_weather_data(year, grand_prix, session_name):
    """Pull measured sector times and nearest session weather per lap."""
    session_code = SESSION_CODE_MAP[session_name]
    session = fastf1.get_session(year, grand_prix, session_code)
    session.load(laps=True, telemetry=False, weather=True)

    laps = session.laps[
        [
            "Driver", "LapNumber", "LapStartTime", "PitInTime",
            "PitOutTime", "TrackStatus", *SECTOR_COLS,
        ]
    ].copy()
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

    weather = session.weather_data[
        ["Time", "TrackTemp", "Rainfall"]
    ].copy()
    laps = laps.sort_values("LapStartTime")
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
    merged["session_ever_rained"] = merged["Rainfall"].fillna(False).astype(bool).any()

    return merged[
        [
            "Year", "GrandPrix", "SessionName", "Driver", "LapNumber",
            "sector_1_s", "sector_2_s", "sector_3_s", "TrackTemp",
            "Rainfall", "TrackStatus", "session_ever_rained",
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

    result.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")
