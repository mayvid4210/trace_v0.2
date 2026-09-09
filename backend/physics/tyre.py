"""
Tyre degradation findings.

INTERMEDIATE degradation - not identifiable with available data
================================================================

No tested model specification produced a physically plausible positive
degradation coefficient for INTERMEDIATE tyres, despite three independent,
methodologically distinct attempts.

Attempt 1 - fuel mass only (single predictor)
----------------------------------------------
Race-only ``wear ~ TyreLife`` produced -0.2381 s/lap of age with correlation
-0.454. Fuel burn and stint progression were suspected confounds.

Attempt 2 - fuel mass plus LapNumber
------------------------------------
Race-only ``LapTime_s ~ TyreLife + total_mass_kg + LapNumber`` with n=706
produced -0.1988 s/lap and R^2=0.528. The correlation between total mass and
LapNumber was -1.000 and VIF was infinite, so the result was discarded as
statistically invalid.

Attempt 3 - fuel mass plus TrackTemp
------------------------------------
Race-only ``LapTime_s ~ TyreLife + total_mass_kg + TrackTemp`` with n=706
produced -0.0865 s/lap, fuel +0.1358 s/kg, TrackTemp -0.6903 s/unit, and
R^2=0.655. All VIF values were below 1.6, and TrackTemp was independently
checked against total mass (correlation +0.038). The degradation coefficient
still remained negative.

Conclusion
----------
The persistence of the negative coefficient across increasingly rigorous and
decreasingly confounded specifications makes INTERMEDIATE degradation a
non-identifiable result with this dataset. TyreLife may be proxying for driver
behavior on a drying track rather than tyre wear. Public telemetry does not
provide enough racing-line or grip-evolution information to separate those
processes here.

Decision
--------
INTERMEDIATE degradation is not reported as a physical parameter. WET is
also excluded because it has insufficient data (n=17). Future work should
expand the wet-race dataset before adding more model complexity.

Tyre grip - cross-circuit normalization caveat
===============================================

Finding: ``reference_laptime_s`` in ``compute_tyre_grip()`` is currently
computed as one global median per compound, pooled across all circuits
(Bahrain and Canada). Because compounds are not used evenly across circuits,
each compound's reference lap implicitly skews toward whichever circuit
contributes more of its laps.

Impact: Grip values are self-consistent within a compound (early versus late
stint comparisons are valid), but are not yet safe to compare across
compounds or use as a circuit-specific downstream input. Two compounds at
the same grip value do not necessarily represent the same absolute time loss.

Status: Not fixed yet. If needed, compute ``reference_laptime_s`` per
``(Compound, GrandPrix)`` pair instead of per compound alone.

Cliff probability - tested fix and final conclusion
====================================================

Early-stint filtering (excluding TyreLife below 3 to remove out-lap,
traffic, and settling-in noise) was tested as a fix for the negative trend
in the initial fit. Sample sizes barely changed (for example, MEDIUM went
from 293 to 277 laps), and all trends remained negative or near zero:
MEDIUM -0.0077 to -0.0103, HARD -0.0008 to -0.0010, and INTERMEDIATE
-0.0003 to -0.0004. This rules out early-stint noise as the primary cause.

The remaining explanation is survivorship bias. Tyres in this dataset are
rarely run to an actual end-of-life cliff before pitting, so the event is
underrepresented by racing strategy itself, not by a filtering or model
specification choice. This is a data-availability limit, not a fixable
modeling gap. ``cliff_probability`` is therefore not implemented as a
usable parameter for any compound.
"""

import numpy as np
import pandas as pd


COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
CLIFF_THRESHOLD_SECONDS = 2.0
INPUT_FILE = "data/processed/fuel_effect.parquet"
OUTPUT_FILE = "data/processed/tyre_data.parquet"


def compute_wear_proxy(df):
    """
    Calculate seconds lost relative to the driver's best lap in each stint.
    """
    df = df.copy()

    stint_best = (
        df.groupby(
            ["GrandPrix", "SessionName", "Driver", "Stint"]
        )["LapTime_s"]
        .min()
        .rename("stint_best_laptime_s")
        .reset_index()
    )

    df = df.merge(
        stint_best,
        on=["GrandPrix", "SessionName", "Driver", "Stint"],
        how="left",
    )
    df["wear"] = df["LapTime_s"] - df["stint_best_laptime_s"]
    return df


def _iqr_clean(df, column):
    q1, q3 = df[column].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[df[column].between(lower, upper)].copy()


def compute_vif(df, predictors):
    """Calculate variance inflation factors for the supplied predictors."""
    vifs = {}

    for target in predictors:
        others = [predictor for predictor in predictors if predictor != target]
        X = np.column_stack([
            df[others].to_numpy(),
            np.ones(len(df)),
        ])
        y = df[target].to_numpy()

        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coeffs

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        vifs[target] = (
            1 / (1 - r_squared)
            if r_squared < 1
            else float("inf")
        )

    return vifs


def fit_ridge(X, y, alpha=1.0):
    """Solve Ridge regression without penalizing the final intercept column."""
    penalty = alpha * np.eye(X.shape[1])
    penalty[-1, -1] = 0
    return np.linalg.solve(
        X.T @ X + penalty,
        X.T @ y,
    )


def fit_degradation_rate_multivariate(
    df,
    mass_col="total_mass_kg",
    session_filter="Race",
):
    """
    Fit LapTime_s against tyre age and fuel mass per compound and session.

    Restricting to Race sessions avoids mixing practice run programs with
    race-pace tyre degradation. VIF identifies unstable predictor splits.
    """
    results = {}
    predictors = ["TyreLife", mass_col]

    for compound in COMPOUNDS:
        subset = df[df["Compound"] == compound].copy()

        if session_filter is not None:
            subset = subset[subset["SessionName"] == session_filter]

        available_samples = len(subset)
        subset = subset.dropna(
            subset=predictors + ["LapTime_s"]
        )

        if len(subset) < 30:
            print(
                f"{compound}: insufficient samples "
                f"({len(subset)} available, {available_samples} before NaN drop), "
                "skipping"
            )
            continue

        clean = _iqr_clean(subset, "LapTime_s")

        if len(clean) < 30:
            print(
                f"{compound}: insufficient clean samples after IQR filter "
                f"({len(clean)} from {len(subset)}), skipping"
            )
            continue

        predictor_values = clean[predictors].to_numpy()
        X = np.column_stack([
            predictor_values,
            np.ones(len(predictor_values)),
        ])
        y = clean["LapTime_s"].to_numpy()

        ols_coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        ols_prediction = X @ ols_coeffs
        ols_residual = y - ols_prediction
        ols_r_squared = (
            1 - np.sum(ols_residual**2) / np.sum((y - y.mean()) ** 2)
        )

        vifs = compute_vif(clean, predictors)
        max_vif = max(vifs.values())

        coeffs = ols_coeffs
        model_name = "OLS"
        r_squared = ols_r_squared

        if max_vif > 10:
            model_name = "Ridge(alpha=1.0)"
            ridge_coefficients = {}
            for alpha in [0.1, 1.0, 10.0]:
                ridge = fit_ridge(X, y, alpha=alpha)
                ridge_coefficients[alpha] = ridge
                print(
                    f"  Ridge alpha={alpha:.1f} tyre coefficient: "
                    f"{ridge[0]:+.4f} s/lap"
                )

            coeffs = ridge_coefficients[1.0]
            prediction = X @ coeffs
            residual = y - prediction
            r_squared = 1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2)

        tyre_coef, mass_coef, intercept = coeffs

        results[compound] = {
            "tyre_degradation_s_per_lap": tyre_coef,
            "fuel_s_per_kg": mass_coef,
            "intercept": intercept,
            "r_squared": r_squared,
            "model": model_name,
            "vif": vifs,
            "n_samples": len(clean),
        }

        print(
            f"\n{compound}  (n={len(clean)}, "
            f"session={session_filter or 'ALL'})"
        )
        print(f"  model:              {model_name}")
        print(f"  tyre_degradation:  {tyre_coef:+.4f} s/lap of age")
        print(f"  fuel_effect:       {mass_coef:+.4f} s/kg")
        print(f"  R^2:               {r_squared:.3f}")
        print(f"  VIF TyreLife:       {vifs['TyreLife']:.3f}")
        print(f"  VIF {mass_col}: {vifs[mass_col]:.3f}")

    return results


def fit_degradation_rate_wet(
    df,
    mass_col="total_mass_kg",
    session_filter="Race",
):
    """Fit wet-compound degradation with LapNumber as a drying proxy."""
    results = {}
    predictors = ["TyreLife", mass_col, "LapNumber"]

    for compound in ["INTERMEDIATE", "WET"]:
        subset = df[df["Compound"] == compound].copy()

        if session_filter is not None:
            subset = subset[subset["SessionName"] == session_filter]

        available_samples = len(subset)
        subset = subset.dropna(
            subset=predictors + ["LapTime_s"]
        )

        if len(subset) < 30:

            print(
                f"{compound}: insufficient samples "
                f"({len(subset)} available, {available_samples} before NaN drop), "
                "skipping"
            )
            continue

        clean = _iqr_clean(subset, "LapTime_s")

        if len(clean) < 30:
            print(
                f"{compound}: insufficient clean samples after IQR filter "
                f"({len(clean)} from {len(subset)}), skipping"
            )
            continue

        X = clean[predictors].to_numpy()
        X = np.column_stack([X, np.ones(len(X))])
        y = clean["LapTime_s"].to_numpy()


        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        tyre_coef, mass_coef, lapnum_coef, intercept = coeffs

        y_pred = X @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        vifs = compute_vif(clean, predictors)

        results[compound] = {
            "tyre_degradation_s_per_lap": tyre_coef,
            "fuel_s_per_kg": mass_coef,
            "drying_trend_s_per_lap": lapnum_coef,
            "intercept": intercept,
            "r_squared": r_squared,
            "vif": vifs,
            "n_samples": len(clean),
        }

        print(
            f"\n{compound}  (n={len(clean)}, "
            f"session={session_filter or 'ALL'})"
        )
        print(f"  tyre_degradation: {tyre_coef:+.4f} s/lap")
        print(f"  fuel_effect:      {mass_coef:+.4f} s/kg")
        print(f"  drying_trend:     {lapnum_coef:+.4f} s/lap")
        print(f"  R^2:              {r_squared:.3f}")
        print(
            "  VIF: "
            f"TyreLife={vifs['TyreLife']:.2f} "
            f"{mass_col}={vifs[mass_col]:.2f} "
            f"LapNumber={vifs['LapNumber']:.2f}"
        )

    return results


def check_weather_collinearity(
    df,
    compound="INTERMEDIATE",
    session_filter="Race",
):
    """Report whether weather fields provide independent fit information."""
    subset = df[
        (df["Compound"] == compound)
        & (df["SessionName"] == session_filter)
    ].dropna(
        subset=["TyreLife", "total_mass_kg", "TrackTemp", "Rainfall"]
    )

    print(f"\n{compound} weather-proxy diagnostics (n={len(subset)}):")
    if subset.empty:
        return subset

    print(
        "  corr(TrackTemp, total_mass_kg): "
        f"{subset['TrackTemp'].corr(subset['total_mass_kg']):+.3f}"
    )
    print(
        "  corr(TrackTemp, TyreLife):      "
        f"{subset['TrackTemp'].corr(subset['TyreLife']):+.3f}"
    )
    print(
        "  corr(Rainfall, total_mass_kg):  "
        f"{subset['Rainfall'].corr(subset['total_mass_kg']):+.3f}"
    )
    print(
        f"  TrackTemp range: {subset['TrackTemp'].min():.1f} - "
        f"{subset['TrackTemp'].max():.1f}"
    )
    print(f"  Rainfall unique values: {subset['Rainfall'].unique()}")

    return subset


def fit_degradation_rate_weather(
    df,
    mass_col="total_mass_kg",
    weather_col="TrackTemp",
    compound="INTERMEDIATE",
    session_filter="Race",
):
    """Fit tyre degradation while controlling for a weather proxy."""
    subset = df[df["Compound"] == compound].copy()
    if session_filter is not None:
        subset = subset[subset["SessionName"] == session_filter]

    subset = subset.dropna(
        subset=["TyreLife", mass_col, weather_col, "LapTime_s"]
    )

    if len(subset) < 30:
        print(f"{compound}: insufficient samples ({len(subset)}), skipping")
        return None

    clean = _iqr_clean(subset, "LapTime_s")
    if len(clean) < 30:
        print(
            f"{compound}: insufficient clean samples after IQR filter "
            f"({len(clean)}), skipping"
        )
        return None

    predictors = ["TyreLife", mass_col, weather_col]
    X = clean[predictors].to_numpy()
    X = np.column_stack([X, np.ones(len(X))])
    y = clean["LapTime_s"].to_numpy()

    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    tyre_coef, mass_coef, weather_coef, intercept = coeffs

    y_pred = X @ coeffs
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    vifs = compute_vif(clean, predictors)

    print(f"\n{compound} + {weather_col}  (n={len(clean)})")
    print(f"  tyre_degradation: {tyre_coef:+.4f} s/lap")
    print(f"  fuel_effect:      {mass_coef:+.4f} s/kg")
    print(f"  {weather_col}_effect: {weather_coef:+.4f} s/unit")
    print(f"  R^2:              {r_squared:.3f}")
    print(
        "  VIF: "
        f"TyreLife={vifs['TyreLife']:.2f} "
        f"{mass_col}={vifs[mass_col]:.2f} "
        f"{weather_col}={vifs[weather_col]:.2f}"
    )

    return {
        "tyre_coef": tyre_coef,
        "mass_coef": mass_coef,
        "weather_coef": weather_coef,
        "r_squared": r_squared,
        "vif": vifs,
        "n": len(clean),
    }


VALIDATED_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]


def compute_tyre_grip(df, degradation_results):
    """Compute a 0-1 tyre-grip proxy for validated dry compounds only."""
    df = df.copy()
    df["grip"] = np.nan

    for compound in VALIDATED_COMPOUNDS:
        if compound not in degradation_results:
            print(
                f"{compound}: no validated degradation result available, "
                "skipping grip"
            )
            continue

        mask = (
            (df["Compound"] == compound)
            & (df["SessionName"] == "Race")
        )
        race_rows = df.loc[mask].dropna(
            subset=["LapTime_s", "TyreLife"]
        )
        clean_race_rows = _iqr_clean(race_rows, "LapTime_s")

        if clean_race_rows.empty:
            print(f"{compound}: no clean Race laps available, skipping grip")
            continue

        rate = degradation_results[compound][
            "tyre_degradation_s_per_lap"
        ]
        reference_laptime = clean_race_rows["LapTime_s"].median()
        df.loc[mask, "grip"] = (
            1
            - rate * df.loc[mask, "TyreLife"] / reference_laptime
        )

        print(
            f"{compound}: grip computed using rate={rate:+.4f} s/lap, "
            f"reference_laptime={reference_laptime:.2f}s"
        )

    return df


def attach_ambient_temp_proxy(df):
    """
    Attach ambient and track-temperature proxies from session weather data.

    These are not tyre temperatures. The telemetry source has no actual tyre
    temperature sensor, so these columns must not be treated as a thermal
    state or renamed to generic temperature fields downstream.
    """
    df = df.copy()

    if "AirTemp" not in df.columns or "TrackTemp" not in df.columns:
        raise ValueError(
            "AirTemp/TrackTemp not found - re-run fuel_effect.py to "
            "regenerate the weather-enriched parquet."
        )

    df["ambient_temp_proxy"] = df["AirTemp"]
    df["track_temp_proxy"] = df["TrackTemp"]
    return df


MIN_STINT_LAP_FOR_CLIFF = 3


def fit_cliff_probability(
    df,
    session_filter="Race",
    min_stint_lap=MIN_STINT_LAP_FOR_CLIFF,
):
    """
    Fit the statistical trend of large lap-time jumps by tyre age.

    This is not a physical cliff model. Low sample counts or few events mean
    the data is insufficient to claim a cliff exists or does not exist.
    """
    if session_filter is not None:
        df = df[df["SessionName"] == session_filter].copy()
    else:
        df = df.copy()

    df = df.sort_values(
        ["GrandPrix", "SessionName", "Driver", "Stint", "TyreLife"]
    )
    df["wear_delta"] = df.groupby(
        ["GrandPrix", "SessionName", "Driver", "Stint"]
    )["wear"].diff()
    df["is_cliff_event"] = (
        df["wear_delta"] > CLIFF_THRESHOLD_SECONDS
    ).astype(int)
    df = df[df["TyreLife"] >= min_stint_lap]

    results = {}

    for compound in COMPOUNDS:
        subset = df[df["Compound"] == compound].dropna(
            subset=["TyreLife", "is_cliff_event"]
        )
        event_count = int(subset["is_cliff_event"].sum())

        if len(subset) < 30 or event_count < 5:
            print(
                f"{compound}: insufficient cliff events to fit "
                f"({event_count} events, n={len(subset)}), skipping"
            )
            continue

        bins = pd.cut(
            subset["TyreLife"],
            bins=min(10, subset["TyreLife"].nunique()),
        )
        bin_stats = (
            subset.groupby(bins, observed=True)["is_cliff_event"]
            .agg(["mean", "count"])
        )
        bin_stats = bin_stats[bin_stats["count"] >= 3]

        if len(bin_stats) < 3:
            print(
                f"{compound}: insufficient binned data to fit a trend, "
                "skipping"
            )
            continue

        bin_centers = np.array([
            interval.mid for interval in bin_stats.index
        ])
        probabilities = bin_stats["mean"].to_numpy()
        slope, intercept = np.polyfit(bin_centers, probabilities, 1)

        results[compound] = {
            "cliff_slope_per_lap_age": slope,
            "intercept": intercept,
            "n_cliff_events": event_count,
            "n_samples": len(subset),
        }

        print(
            f"\n{compound}  (n={len(subset)}, "
            f"cliff events={event_count})"
        )
        print(f"  cliff probability trend: {slope:.5f} per lap of age")

    return results


def main():
    df = pd.read_parquet(INPUT_FILE)

    required_columns = {
        "Compound", "Stint", "TyreLife", "LapTime_s",
        "total_mass_kg", "SessionName",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing tyre columns: "
            f"{sorted(missing)}. Run the updated fuel_effect.py first."
        )

    df = compute_wear_proxy(df)

    print("===== STEP 7 (v3): RACE-ONLY TYRE DEGRADATION =====")
    degradation_results = fit_degradation_rate_multivariate(
        df,
        session_filter="Race",
    )
    wet_degradation_results = fit_degradation_rate_wet(
        df,
        session_filter="Race",
    )

    weather_columns = {"TrackTemp", "Rainfall"}
    if weather_columns.issubset(df.columns):
        weather_diag = check_weather_collinearity(
            df,
            compound="INTERMEDIATE",
            session_filter="Race",
        )
        if (
            not weather_diag.empty
            and abs(
                weather_diag["TrackTemp"].corr(
                    weather_diag["total_mass_kg"]
                )
            ) < 0.5
        ):
            weather_degradation_result = fit_degradation_rate_weather(
                df,
                mass_col="total_mass_kg",
                weather_col="TrackTemp",
                compound="INTERMEDIATE",
                session_filter="Race",
            )
        else:
            print("TrackTemp too collinear with fuel mass - skipping fit")
            weather_degradation_result = None
    else:
        print(
            "\nWeather diagnostics skipped: missing columns "
            f"{sorted(weather_columns.difference(df.columns))}"
        )
        weather_degradation_result = None

    print("\n===== STEP 8: TYRE GRIP (SOFT/MEDIUM/HARD only) =====")
    df = compute_tyre_grip(df, degradation_results)
    print(
        df[
            df["Compound"].isin(VALIDATED_COMPOUNDS)
            & df["grip"].notna()
        ][["Compound", "TyreLife", "grip"]]
        .describe()
    )
    print("\nMaximum TyreLife by validated compound:")
    print(
        df[df["Compound"].isin(VALIDATED_COMPOUNDS)]
        .groupby("Compound")["TyreLife"]
        .max()
        .to_string()
    )

    print("\n===== STEP 9: AMBIENT TEMP PROXY =====")
    df = attach_ambient_temp_proxy(df)
    print(
        df[["ambient_temp_proxy", "track_temp_proxy"]]
        .describe()
    )
    print("\nAmbient proxy null counts:")
    print(
        df[["ambient_temp_proxy", "track_temp_proxy"]]
        .isna()
        .sum()
        .to_string()
    )

    print("\n===== STEP 10: CLIFF PROBABILITY (Race only) =====")
    cliff_results = fit_cliff_probability(
        df,
        session_filter="Race",
    )

    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"\nSaved: {OUTPUT_FILE}")

    return {
        "dry_compounds": degradation_results,
        "wet_compounds": wet_degradation_results,
        "weather_compound": weather_degradation_result,
        "cliff_results": cliff_results,
    }


if __name__ == "__main__":
    main()