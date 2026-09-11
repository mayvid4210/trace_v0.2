"""Wet performance and tyre-compound suitability model using XGBoost.

Objective:
    Estimate the incremental lap-time performance delta (wet/damp penalty in seconds
    relative to a theoretical dry reference expectation) of tyre compounds across track
    condition transitions:
        DRY -> DAMP -> WET  and  WET -> DAMP -> DRY

Scientific Formulation:
    Theoretical dry physics baseline:
        T_dry_physics = LapTimePredictor.predict_lap_time(circuit, compound, tyre_age, fuel_mass_kg)

    Dry clean-air reference expectation:
        T_dry_reference = T_dry_physics + CLEAN_AIR_RESIDUAL_REF[(circuit, compound)]

    Observed clean racing lap time:
        T_actual

    Performance Target:
        On dry track (wetness_proxy < 0.1) with dry slicks (HARD, MEDIUM, SOFT):
            wet_delta_s = 0.0
        In damp/wet track conditions or wet tyres (INTERMEDIATE, WET):
            wet_delta_s = max(0.0, T_actual - T_dry_reference)

    This formulation strictly decouples ML-4 from ML-1 (which predicts dry tyre degradation
    residual y1 = T_actual - T_dry_physics), preventing double-counting of dry tyre pace.

    The model learns:
        predicted_lap_delta_s = f(causal_track_and_tyre_features)

Causal Features:
    Strictly causal features observed or estimated at or before the start of the lap:
    - circuit: track identity (BAHRAIN, CANADA)
    - compound: tyre compound fitted (HARD, INTERMEDIATE, MEDIUM, SOFT, WET)
    - tyre_age: completed laps on the current set
    - lap_number: race lap index
    - stint_lap: lap index within the stint
    - fuel_mass_kg: causal estimated fuel mass
    - track_temperature: track temp (deg C) from weather at lap start
    - air_temperature: air temp (deg C) from weather at lap start
    - track_status: race control track status flag ('1' clear, '12' local caution/slippery)
    - rainfall: binary rainfall flag observed at lap start (0.0 or 1.0)
    - wetness_proxy: causal track dampness/wetness index [0.0 to 1.0]

Data Limitations:
    - Public F1 telemetry does not provide track water depth, aquaplaning depth, tyre surface
      temperatures, or physical friction coefficients.
    - Full WET tyre usage is sparse (n=15 clean racing laps in Canada 2024).
    - Dry slicks in heavy rain are under-sampled (teams avoid slicks in rain).
    - INTERMEDIATE performance across damp and wet track has substantial empirical support (n=689).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from backend.physics.fuel_model import get_circuit_fuel_calibration
from backend.physics.lap_time_predictor import (
    CircuitCalibration,
    LapTimePredictor,
)
from backend.ml.tyre_degradation_residual import DEFAULT_CIRCUIT_CALIBRATIONS
from backend.ml.traffic_penalty import CLEAN_AIR_RESIDUAL_REF

WET_PERFORMANCE_FEATURE_COLUMNS: List[str] = [
    "circuit",
    "compound",
    "tyre_age",
    "lap_number",
    "stint_lap",
    "fuel_mass_kg",
    "track_temperature",
    "air_temperature",
    "track_status",
    "rainfall",
    "wetness_proxy",
]

CATEGORICAL_COLUMNS: List[str] = ["circuit", "compound", "track_status"]

CATEGORICAL_DOMAINS: Dict[str, List[str]] = {
    "circuit": ["BAHRAIN", "CANADA"],
    "compound": ["HARD", "INTERMEDIATE", "MEDIUM", "SOFT", "WET"],
    "track_status": ["1", "12", "21", "UNKNOWN"],
}


def compute_causal_wetness_proxy(
    lap_numbers: Sequence[int],
    rainfall_flags: Sequence[bool],
    decay: float = 0.85,
) -> pd.Series:
    """Compute strictly causal track dampness/wetness proxy via exponential decay.

    At lap L:
        W_L = W_{L-1} * decay + rain_L * (1.0 - decay * W_{L-1})
    Bounded strictly in [0.0, 1.0].
    Zero future leakage: depends only on laps 1 to L.
    """
    df = pd.DataFrame({"lap_number": lap_numbers, "rainfall": rainfall_flags})
    unique_laps = sorted(df["lap_number"].unique())

    # Map session-level rainfall at each lap start
    rain_per_lap = df.groupby("lap_number")["rainfall"].any()

    curr_w = 0.0
    w_map: Dict[int, float] = {}
    for l_num in unique_laps:
        is_r = 1.0 if rain_per_lap.get(l_num, False) else 0.0
        curr_w = curr_w * decay + is_r * (1.0 - decay * curr_w)
        curr_w = max(0.0, min(1.0, curr_w))
        w_map[l_num] = float(curr_w)

    return df["lap_number"].map(w_map).fillna(0.0)


def compute_wet_performance_target(
    actual_lap_times: Union[pd.Series, np.ndarray, Sequence[float]],
    dry_reference_lap_times: Union[pd.Series, np.ndarray, Sequence[float]],
    wetness_proxy: Optional[Union[pd.Series, np.ndarray, Sequence[float]]] = None,
    compounds: Optional[Union[pd.Series, np.ndarray, Sequence[str]]] = None,
) -> np.ndarray:
    """Compute incremental wet performance target in seconds.

    Target represents the incremental lap-time penalty caused by wet/damp track conditions
    and compound suitability relative to a dry clean-air reference:
        wet_delta_s = max(0.0, T_actual - T_dry_reference)

    On a dry track (wetness_proxy < 0.1) with dry slick compounds (HARD, MEDIUM, SOFT),
    the wet penalty is strictly 0.0s, eliminating overlap with ML-1 tyre degradation residual.
    """
    actual = np.asarray(actual_lap_times, dtype=np.float64)
    ref = np.asarray(dry_reference_lap_times, dtype=np.float64)
    if actual.shape != ref.shape:
        raise ValueError(
            f"Shape mismatch: actual {actual.shape} vs reference {ref.shape}"
        )
    delta = np.maximum(0.0, actual - ref)
    if wetness_proxy is not None and compounds is not None:
        w = np.asarray(wetness_proxy, dtype=np.float64)
        c = np.asarray([str(x).strip().upper() for x in compounds])
        dry_slicks = np.isin(c, ["HARD", "MEDIUM", "SOFT"])
        dry_track = (w < 0.1)
        delta = np.where(dry_slicks & dry_track, 0.0, delta)
    return delta


def _clean_and_impute_wet_features(df: pd.DataFrame) -> pd.DataFrame:
    """Format numeric features and enforce categorical types for XGBoost."""
    X = pd.DataFrame(index=df.index)

    # Numeric features
    X["tyre_age"] = pd.to_numeric(df.get("tyre_age", 1.0), errors="coerce").fillna(1.0)
    X["lap_number"] = pd.to_numeric(df.get("lap_number", 1), errors="coerce").fillna(1).astype(int)
    X["stint_lap"] = pd.to_numeric(df.get("stint_lap", 1), errors="coerce").fillna(1).astype(int)
    X["fuel_mass_kg"] = pd.to_numeric(df.get("fuel_mass_kg", 50.0), errors="coerce").fillna(50.0)
    X["track_temperature"] = pd.to_numeric(df.get("track_temperature", 25.0), errors="coerce").fillna(25.0)
    X["air_temperature"] = pd.to_numeric(df.get("air_temperature", 20.0), errors="coerce").fillna(20.0)
    X["rainfall"] = pd.to_numeric(df.get("rainfall", 0.0), errors="coerce").fillna(0.0)
    X["wetness_proxy"] = pd.to_numeric(df.get("wetness_proxy", 0.0), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Categorical features
    for col in CATEGORICAL_COLUMNS:
        domain = CATEGORICAL_DOMAINS[col]
        raw = df.get(col, "UNKNOWN")
        if isinstance(raw, pd.Series):
            s = raw.astype(str).str.strip().str.upper()
        else:
            s = pd.Series([str(raw).strip().upper()] * len(df), index=df.index)
        s = s.where(s.isin(domain), domain[-1])
        cat_type = pd.CategoricalDtype(categories=domain, ordered=False)
        X[col] = s.astype(cat_type)

    return X[WET_PERFORMANCE_FEATURE_COLUMNS].copy()


def prepare_wet_performance_data(
    laps_df: Optional[pd.DataFrame] = None,
    predictor: Optional[LapTimePredictor] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    circuits: Sequence[str] = ("Bahrain", "Canada"),
    year: int = 2024,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare clean, causal training data for the wet-performance model.

    Returns:
        X: Feature matrix with WET_PERFORMANCE_FEATURE_COLUMNS and categorical dtypes.
        y: Target lap delta Series (T_actual - T_dry_physics).
        metadata: Tracking metadata (driver, stint_id, stint, circuit, lap_number, T_actual, T_dry_physics).
    """
    if predictor is None:
        predictor = LapTimePredictor(DEFAULT_CIRCUIT_CALIBRATIONS)

    if laps_df is not None:
        raw = laps_df.copy()
        if "LapTime_s" not in raw.columns and "lap_time" in raw.columns:
            raw["LapTime_s"] = raw["lap_time"]
        if "circuit" not in raw.columns and "GrandPrix" in raw.columns:
            raw["circuit"] = raw["GrandPrix"]
        if "driver" not in raw.columns and "Driver" in raw.columns:
            raw["driver"] = raw["Driver"]
        if "compound" not in raw.columns and "Compound" in raw.columns:
            raw["compound"] = raw["Compound"]
        if "lap_number" not in raw.columns and "LapNumber" in raw.columns:
            raw["lap_number"] = raw["LapNumber"]
        if "stint" not in raw.columns and "Stint" in raw.columns:
            raw["stint"] = raw["Stint"]
        if "tyre_age" not in raw.columns and "TyreLife" in raw.columns:
            raw["tyre_age"] = raw["TyreLife"]

        clean = raw.dropna(subset=["LapTime_s"]).copy()
        clean = clean[clean["LapTime_s"] > 0].copy()

        if "PitInTime" in clean.columns:
            clean = clean[clean["PitInTime"].isna()]
        if "PitOutTime" in clean.columns:
            clean = clean[clean["PitOutTime"].isna()]
        if "IsAccurate" in clean.columns:
            clean = clean[clean["IsAccurate"] != False]
        if "TrackStatus" in clean.columns:
            clean = clean[~clean["TrackStatus"].astype(str).str.contains("4")]

        clean["compound"] = clean["compound"].astype(str).str.upper().str.strip()
        clean = clean[clean["compound"].isin(CATEGORICAL_DOMAINS["compound"])].copy()

        if "stint_lap" not in clean.columns:
            clean["stint_lap"] = clean.groupby(["circuit", "driver", "stint"]).cumcount() + 1
        if "fuel_mass_kg" not in clean.columns:
            clean["fuel_mass_kg"] = 50.0
        if "rainfall" not in clean.columns:
            clean["rainfall"] = 0.0
        if "wetness_proxy" not in clean.columns:
            clean["wetness_proxy"] = compute_causal_wetness_proxy(
                clean["lap_number"], clean["rainfall"].astype(bool)
            )

        clean["stint_id"] = (
            clean["circuit"].astype(str)
            + "_"
            + clean["driver"].astype(str)
            + "_S"
            + clean["stint"].astype(str)
        )

        t_physics_list = []
        t_ref_list = []
        for _, r in clean.iterrows():
            c_name = str(r["circuit"]).title()
            comp_name = str(r["compound"]).upper()
            t_phys = predictor.predict_lap_time(
                circuit=c_name,
                compound=comp_name,
                tyre_age=float(r.get("tyre_age", 1.0)),
                fuel_mass_kg=float(r.get("fuel_mass_kg", 50.0)),
            )
            ref_offset = CLEAN_AIR_RESIDUAL_REF.get(
                (str(r["circuit"]).upper(), comp_name),
                CLEAN_AIR_RESIDUAL_REF.get((str(r["circuit"]).upper(), "MEDIUM"), 2.0),
            )
            t_physics_list.append(t_phys)
            t_ref_list.append(t_phys + ref_offset)

        clean["T_dry_physics"] = t_physics_list
        clean["T_dry_reference"] = t_ref_list
        clean["T_actual"] = clean["LapTime_s"]
        clean["wet_delta_s"] = compute_wet_performance_target(
            actual_lap_times=clean["T_actual"],
            dry_reference_lap_times=clean["T_dry_reference"],
            wetness_proxy=clean["wetness_proxy"],
            compounds=clean["compound"],
        )
        clean["lap_delta_s"] = clean["wet_delta_s"]

        X = _clean_and_impute_wet_features(clean)
        y = clean["wet_delta_s"].copy()
        meta_cols = ["driver", "stint_id", "stint", "circuit", "lap_number", "T_actual", "T_dry_physics", "T_dry_reference"]
        metadata = clean[[c for c in meta_cols if c in clean.columns]].copy()
        return X, y, metadata

    # Load from FastF1 local cache
    import fastf1

    raw_dir = cache_dir or (
        Path(__file__).resolve().parent.parent / "data" / "raw"
    )
    fastf1.Cache.enable_cache(str(raw_dir))

    collected_records = []
    for gp in circuits:
        session = fastf1.get_session(year, gp, "R")
        session.load(laps=True, telemetry=False, weather=True)

        laps = session.laps.copy()
        weather = session.weather_data.copy()

        laps = laps.sort_values("LapStartTime")
        weather = weather.sort_values("Time")

        # Causal backward merge of weather
        merged = pd.merge_asof(
            laps,
            weather,
            left_on="LapStartTime",
            right_on="Time",
            direction="backward",
            suffixes=("", "_weather"),
        )

        clean = merged[
            merged["PitInTime"].isna()
            & merged["PitOutTime"].isna()
            & merged["LapTime"].notna()
        ].copy()
        clean["LapTime_s"] = clean["LapTime"].dt.total_seconds()
        clean = clean[clean["LapTime_s"] > 0]
        clean = clean[clean["IsAccurate"] != False]
        # Exclude Safety Car laps where lap times reflect SC delta rather than tyres
        clean = clean[~clean["TrackStatus"].astype(str).str.contains("4")]

        total_laps = int(clean["LapNumber"].max()) if not clean.empty else 57
        burn_rate = 100.0 / total_laps

        # Causal wetness proxy
        lap_rain = clean.groupby("LapNumber")["Rainfall"].any()
        decay = 0.85
        w_proxy = {}
        curr_w = 0.0
        for l_num in sorted(lap_rain.index):
            is_r = 1.0 if lap_rain.loc[l_num] else 0.0
            curr_w = curr_w * decay + is_r * (1.0 - decay * curr_w)
            curr_w = max(0.0, min(1.0, curr_w))
            w_proxy[l_num] = curr_w

        for _, row in clean.iterrows():
            comp = str(row["Compound"]).strip().upper()
            if comp not in CATEGORICAL_DOMAINS["compound"]:
                continue

            tyre_age = float(row["TyreLife"]) if pd.notna(row["TyreLife"]) else 1.0
            lap_num = int(row["LapNumber"])
            stint = int(row["Stint"]) if pd.notna(row["Stint"]) else 1
            driver = str(row["Driver"])
            fuel = max(0.0, 100.0 - (lap_num - 1) * burn_rate)

            t_physics = predictor.predict_lap_time(
                circuit=gp,
                compound=comp,
                tyre_age=tyre_age,
                fuel_mass_kg=fuel,
            )
            ref_offset = CLEAN_AIR_RESIDUAL_REF.get(
                (gp.upper(), comp),
                CLEAN_AIR_RESIDUAL_REF.get((gp.upper(), "MEDIUM"), 2.0),
            )
            t_dry_ref = t_physics + ref_offset
            lap_time = float(row["LapTime_s"])

            track_temp = float(row["TrackTemp"]) if pd.notna(row["TrackTemp"]) else 25.0
            air_temp = float(row["AirTemp"]) if pd.notna(row["AirTemp"]) else 20.0
            track_status = str(row["TrackStatus"]).strip()
            rain = 1.0 if bool(row["Rainfall"]) else 0.0
            wet_val = w_proxy.get(lap_num, 0.0)

            collected_records.append({
                "circuit": gp.upper(),
                "compound": comp,
                "tyre_age": tyre_age,
                "lap_number": lap_num,
                "stint": stint,
                "fuel_mass_kg": fuel,
                "track_temperature": track_temp,
                "air_temperature": air_temp,
                "track_status": track_status,
                "rainfall": rain,
                "wetness_proxy": wet_val,
                "driver": driver,
                "stint_id": f"{gp}_{driver}_S{stint}",
                "T_actual": lap_time,
                "T_dry_physics": t_physics,
                "T_dry_reference": t_dry_ref,
            })

    if not collected_records:
        raise ValueError("No valid clean laps found across specified circuits.")

    all_df = pd.DataFrame(collected_records)
    all_df["stint_lap"] = all_df.groupby(["circuit", "driver", "stint"]).cumcount() + 1
    all_df["wet_delta_s"] = compute_wet_performance_target(
        actual_lap_times=all_df["T_actual"],
        dry_reference_lap_times=all_df["T_dry_reference"],
        wetness_proxy=all_df["wetness_proxy"],
        compounds=all_df["compound"],
    )
    all_df["lap_delta_s"] = all_df["wet_delta_s"]

    X = _clean_and_impute_wet_features(all_df)
    y = all_df["wet_delta_s"].copy()
    metadata = all_df[["driver", "stint_id", "stint", "circuit", "lap_number", "T_actual", "T_dry_physics", "T_dry_reference"]].copy()

    return X, y, metadata


class WetPerformanceModel:
    """Deterministic XGBoost regression model for wet/tyre performance estimation."""

    def __init__(
        self,
        random_state: int = 42,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        subsample: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
    ) -> None:
        self.random_state = random_state
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda

        self.model = xgb.XGBRegressor(
            random_state=self.random_state,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            enable_categorical=True,
            tree_method="hist",
        )
        self.is_fitted: bool = False
        self.feature_names_: List[str] = WET_PERFORMANCE_FEATURE_COLUMNS

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> "WetPerformanceModel":
        """Fit model on causal feature matrix and lap delta target."""
        X_clean = _clean_and_impute_wet_features(X)
        y_clean = np.asarray(y, dtype=np.float64)
        self.model.fit(X_clean, y_clean)
        self.is_fitted = True
        return self

    def predict_lap_delta(
        self,
        X: Union[pd.DataFrame, Dict[str, Any]],
    ) -> np.ndarray:
        """Predict expected lap delta in seconds relative to dry reference baseline."""
        if not self.is_fitted:
            raise RuntimeError("WetPerformanceModel must be fitted before predict_lap_delta().")

        if isinstance(X, dict):
            X_df = pd.DataFrame([X])
        else:
            X_df = X.copy()

        X_clean = _clean_and_impute_wet_features(X_df)
        preds = self.model.predict(X_clean)
        # Wet delta penalty is physically non-negative
        preds = np.maximum(0.0, preds)
        return np.asarray(preds, dtype=np.float64)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        groups: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate model using 5-fold GroupKFold CV across compounds and regimes."""
        y_arr = np.asarray(y, dtype=np.float64)
        X_clean = _clean_and_impute_wet_features(X)

        if groups is not None:
            gkf = GroupKFold(n_splits=min(5, len(np.unique(groups))))
            oof_preds = np.zeros(len(y_arr))
            for trn_idx, val_idx in gkf.split(X_clean, y_arr, groups=groups):
                fold_model = xgb.XGBRegressor(
                    random_state=self.random_state,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    n_estimators=self.n_estimators,
                    subsample=self.subsample,
                    reg_alpha=self.reg_alpha,
                    reg_lambda=self.reg_lambda,
                    enable_categorical=True,
                    tree_method="hist",
                )
                fold_model.fit(X_clean.iloc[trn_idx], y_arr[trn_idx])
                oof_preds[val_idx] = np.maximum(0.0, fold_model.predict(X_clean.iloc[val_idx]))
            eval_preds = oof_preds
        else:
            if not self.is_fitted:
                self.fit(X_clean, y_arr)
            eval_preds = self.predict_lap_delta(X_clean)

        mae = float(mean_absolute_error(y_arr, eval_preds))
        rmse = float(np.sqrt(mean_squared_error(y_arr, eval_preds)))
        r2 = float(r2_score(y_arr, eval_preds))

        # Baseline: compound median
        comp_series = X_clean["compound"].astype(str)
        median_map = pd.Series(y_arr).groupby(comp_series).median()
        base_preds = comp_series.map(median_map).values
        base_mae = float(mean_absolute_error(y_arr, base_preds))
        base_rmse = float(np.sqrt(mean_squared_error(y_arr, base_preds)))
        base_r2 = float(r2_score(y_arr, base_preds))

        # Breakdown by compound
        compound_metrics = {}
        for comp in np.unique(comp_series):
            mask = (comp_series == comp).values
            if mask.sum() > 0:
                compound_metrics[comp] = {
                    "count": int(mask.sum()),
                    "mae": float(mean_absolute_error(y_arr[mask], eval_preds[mask])),
                    "rmse": float(np.sqrt(mean_squared_error(y_arr[mask], eval_preds[mask]))),
                }

        # Breakdown by regime
        wetness = X_clean["wetness_proxy"].values
        regimes = {
            "dry": (wetness < 0.1),
            "damp": ((wetness >= 0.1) & (wetness < 0.5)),
            "wet": (wetness >= 0.5),
        }
        regime_metrics = {}
        for reg_name, mask in regimes.items():
            if mask.sum() > 0:
                regime_metrics[reg_name] = {
                    "count": int(mask.sum()),
                    "mae": float(mean_absolute_error(y_arr[mask], eval_preds[mask])),
                    "rmse": float(np.sqrt(mean_squared_error(y_arr[mask], eval_preds[mask]))),
                    "base_mae": float(mean_absolute_error(y_arr[mask], base_preds[mask])),
                }

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse,
            "baseline_r2": base_r2,
            "mae_improvement": base_mae - mae,
            "mae_improvement_pct": ((base_mae - mae) / base_mae) * 100 if base_mae > 0 else 0.0,
            "rmse_improvement": base_rmse - rmse,
            "by_compound": compound_metrics,
            "by_regime": regime_metrics,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save fitted model and schema metadata to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted WetPerformanceModel.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "random_state": self.random_state,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "n_estimators": self.n_estimators,
                "feature_names": self.feature_names_,
            },
            p,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WetPerformanceModel":
        """Load fitted model artifact from disk."""
        data = joblib.load(path)
        instance = cls(
            random_state=data.get("random_state", 42),
            max_depth=data.get("max_depth", 3),
            learning_rate=data.get("learning_rate", 0.05),
            n_estimators=data.get("n_estimators", 100),
        )
        instance.model = data["model"]
        instance.feature_names_ = data.get("feature_names", WET_PERFORMANCE_FEATURE_COLUMNS)
        instance.is_fitted = True
        return instance
