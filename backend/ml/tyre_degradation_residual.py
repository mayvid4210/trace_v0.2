"""Tyre degradation residual model using XGBoost.

Objective:
    Physics baseline:
        T_physics = LapTimePredictor.predict_lap_time(...)
    Actual telemetry:
        T_actual = observed clean lap time
    Residual:
        target_residual = T_actual - T_physics

    XGBoost learns:
        predicted_residual = f(causal tyre, operational, and track features)

    Composite prediction:
        T_composite = T_physics + predicted_residual

Scientific Interpretation:
    This model provides a physics residual lap-time correction. It does NOT
    measure pure mechanical tyre degradation, because public F1 telemetry
    cannot uniquely separate physical tyre wear from driver pacing, track
    evolution, traffic, or rubbering-in effects.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

try:
    from backend.physics.fuel_model import get_circuit_fuel_calibration
    from backend.physics.lap_time_predictor import CircuitCalibration, LapTimePredictor
except ModuleNotFoundError:
    from physics.fuel_model import get_circuit_fuel_calibration
    from physics.lap_time_predictor import CircuitCalibration, LapTimePredictor

FEATURE_COLUMNS: List[str] = [
    "circuit",
    "compound",
    "tyre_age",
    "stint",
    "stint_lap",
    "lap_number",
    "fuel_mass_kg",
    "track_temperature",
    "air_temperature",
]

CATEGORICAL_COLUMNS: List[str] = ["circuit", "compound"]

NUMERIC_COLUMNS: List[str] = [
    "tyre_age",
    "stint",
    "stint_lap",
    "lap_number",
    "fuel_mass_kg",
    "track_temperature",
    "air_temperature",
]

DEFAULT_CIRCUIT_CALIBRATIONS: Dict[Tuple[str, str], CircuitCalibration] = {
    ("Bahrain", "SOFT"): CircuitCalibration(
        dry_baseline_lap_time_s=92.50, reference_mass_kg=850.0
    ),
    ("Bahrain", "MEDIUM"): CircuitCalibration(
        dry_baseline_lap_time_s=93.10, reference_mass_kg=850.0
    ),
    ("Bahrain", "HARD"): CircuitCalibration(
        dry_baseline_lap_time_s=94.00, reference_mass_kg=850.0
    ),
    ("Bahrain", "__default__"): CircuitCalibration(
        dry_baseline_lap_time_s=93.00, reference_mass_kg=850.0
    ),
    ("Canada", "MEDIUM"): CircuitCalibration(
        dry_baseline_lap_time_s=75.00, reference_mass_kg=850.0
    ),
    ("Canada", "HARD"): CircuitCalibration(
        dry_baseline_lap_time_s=76.00, reference_mass_kg=850.0
    ),
    ("Canada", "__default__"): CircuitCalibration(
        dry_baseline_lap_time_s=75.50, reference_mass_kg=850.0
    ),
}


def calculate_residual(
    actual_lap_times: Union[pd.Series, np.ndarray, Sequence[float]],
    physics_lap_times: Union[pd.Series, np.ndarray, Sequence[float]],
) -> np.ndarray:
    """Calculate physics lap-time residual: T_actual - T_physics."""
    actual = np.asarray(actual_lap_times, dtype=np.float64)
    physics = np.asarray(physics_lap_times, dtype=np.float64)
    if actual.shape != physics.shape:
        raise ValueError(
            f"Shape mismatch: actual {actual.shape} vs physics {physics.shape}"
        )
    return actual - physics


def _clean_and_impute_features(
    df: pd.DataFrame,
    categories_map: Optional[Dict[str, List[Any]]] = None,
) -> pd.DataFrame:
    """Ensure strict causal feature schema, impute missing values, and encode categories."""
    X = pd.DataFrame(index=df.index)

    # Impute and format numeric features
    X["tyre_age"] = pd.to_numeric(df.get("tyre_age", 1.0), errors="coerce").fillna(1.0)
    X["stint"] = pd.to_numeric(df.get("stint", 1), errors="coerce").fillna(1).astype(int)
    X["stint_lap"] = pd.to_numeric(df.get("stint_lap", 1), errors="coerce").fillna(1).astype(int)
    X["lap_number"] = pd.to_numeric(df.get("lap_number", 1), errors="coerce").fillna(1).astype(int)
    X["fuel_mass_kg"] = pd.to_numeric(df.get("fuel_mass_kg", 50.0), errors="coerce").fillna(50.0)

    # Ambient temperatures: sensible defaults if unobserved
    X["track_temperature"] = pd.to_numeric(df.get("track_temperature", 30.0), errors="coerce").fillna(30.0)
    X["air_temperature"] = pd.to_numeric(df.get("air_temperature", 25.0), errors="coerce").fillna(25.0)

    # Categorical features
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            raw_series = df[col].astype(str).str.strip().str.upper()
        else:
            raw_series = pd.Series(["UNKNOWN"] * len(df), index=df.index)
        if categories_map and col in categories_map:
            cat_type = pd.CategoricalDtype(categories=categories_map[col], ordered=False)
            X[col] = raw_series.astype(cat_type)
        else:
            X[col] = raw_series.astype("category")

    return X[FEATURE_COLUMNS].copy()


def prepare_training_data(
    laps_df: Optional[pd.DataFrame] = None,
    predictor: Optional[LapTimePredictor] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    circuits: Sequence[str] = ("Bahrain", "Canada"),
    year: int = 2024,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Prepare clean, causal training data and physics residuals.

    Returns:
        X: Feature matrix with exact FEATURE_COLUMNS and categorical dtypes.
        y: Target residual Series (T_actual - T_physics).
        t_physics: Physics baseline prediction Series.
        metadata: Tracking metadata (stint_id, driver, circuit, stint, lap_number, T_actual).
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

        # Drop corrupted lap times
        clean = raw.dropna(subset=["LapTime_s"]).copy()
        clean = clean[clean["LapTime_s"] > 0].copy()

        # Clean flag filters if columns exist
        if "PitInTime" in clean.columns:
            clean = clean[clean["PitInTime"].isna()]
        if "PitOutTime" in clean.columns:
            clean = clean[clean["PitOutTime"].isna()]
        if "TrackStatus" in clean.columns:
            clean = clean[clean["TrackStatus"].astype(str) == "1"]
        if "IsAccurate" in clean.columns:
            clean = clean[clean["IsAccurate"] != False]

        if "stint_lap" not in clean.columns:
            clean["stint_lap"] = clean.groupby(["circuit", "driver", "stint"]).cumcount() + 1

        if "fuel_mass_kg" not in clean.columns:
            # Estimate fuel causally
            clean["fuel_mass_kg"] = 50.0

        clean["stint_id"] = (
            clean["circuit"].astype(str)
            + "_"
            + clean["driver"].astype(str)
            + "_S"
            + clean["stint"].astype(str)
        )
    else:
        # Load from FastF1 local cache
        import fastf1

        raw_dir = cache_dir or (
            Path(__file__).resolve().parent.parent / "data" / "raw"
        )
        fastf1.Cache.enable_cache(str(raw_dir))

        collected_rows = []
        for gp in circuits:
            session = fastf1.get_session(year, gp, "R")
            session.load(laps=True, telemetry=False, weather=True)

            laps = session.laps.copy()
            weather = session.weather_data.copy()

            # Clean laps: green flag, no pit in/out, valid time, accurate
            clean = laps[
                laps["PitInTime"].isna()
                & laps["PitOutTime"].isna()
                & laps["LapTime"].notna()
            ].copy()

            clean["LapTime_s"] = clean["LapTime"].dt.total_seconds()

            if "TrackStatus" in clean.columns:
                clean = clean[clean["TrackStatus"].astype(str) == "1"].copy()
            if "IsAccurate" in clean.columns:
                clean = clean[clean["IsAccurate"] != False].copy()

            clean["circuit"] = gp
            clean["compound"] = (
                clean["Compound"].astype(str).str.upper().str.strip()
            )
            # Filter for dry compounds
            clean = clean[clean["compound"].isin(["SOFT", "MEDIUM", "HARD"])].copy()

            # Causal fuel mass from validated circuit fuel model
            calib = get_circuit_fuel_calibration(gp)
            total_laps = int(clean["LapNumber"].max()) if not clean.empty else 57
            burn_rate = (
                calib.nominal_burn_rate_kg_per_lap
                if calib
                else 100.0 / float(total_laps)
            )
            clean["fuel_mass_kg"] = (
                100.0 - (clean["LapNumber"] - 1) * burn_rate
            ).clip(lower=0.0)

            # Weather merge_asof backwards
            clean["Time"] = pd.to_timedelta(clean["Time"], errors="coerce")
            weather["Time"] = pd.to_timedelta(weather["Time"], errors="coerce")
            clean = clean.sort_values("Time")
            weather = weather.sort_values("Time")
            clean = pd.merge_asof(
                clean,
                weather[["Time", "TrackTemp", "AirTemp"]],
                on="Time",
                direction="backward",
            )

            clean["track_temperature"] = clean["TrackTemp"].astype(float)
            clean["air_temperature"] = clean["AirTemp"].astype(float)

            clean["stint"] = clean["Stint"].fillna(1).astype(int)
            clean = clean.sort_values(["Driver", "stint", "LapNumber"])
            clean["stint_lap"] = (
                clean.groupby(["Driver", "stint"]).cumcount() + 1
            )
            clean["tyre_age"] = (
                clean["TyreLife"].fillna(clean["stint_lap"]).astype(float)
            )
            clean["lap_number"] = clean["LapNumber"].astype(int)
            clean["driver"] = clean["Driver"].astype(str)
            clean["stint_id"] = (
                clean["circuit"]
                + "_"
                + clean["driver"]
                + "_S"
                + clean["stint"].astype(str)
            )

            collected_rows.append(clean)

        if not collected_rows:
            raise ValueError("No valid clean laps could be loaded.")
        clean = pd.concat(collected_rows, ignore_index=True)

    # Compute physics baseline
    physics_preds = []
    for _, row in clean.iterrows():
        t_phys = predictor.predict_lap_time(
            circuit=str(row["circuit"]),
            compound=str(row["compound"]).upper(),
            tyre_age=float(row["tyre_age"]),
            fuel_mass_kg=float(row["fuel_mass_kg"]),
            sector_wetness=None,
        )
        physics_preds.append(t_phys)

    clean["T_physics"] = physics_preds
    clean["T_actual"] = clean["LapTime_s"].astype(float)
    clean["residual"] = calculate_residual(clean["T_actual"], clean["T_physics"])

    X = _clean_and_impute_features(clean)
    y = clean["residual"].copy().reset_index(drop=True)
    t_physics = clean["T_physics"].copy().reset_index(drop=True)
    metadata = clean[
        ["circuit", "driver", "stint", "stint_id", "lap_number", "T_actual", "T_physics", "residual"]
    ].copy().reset_index(drop=True)

    return X, y, t_physics, metadata


class TyreDegradationResidualModel:
    """Production XGBoost model predicting lap-time physics residuals."""

    def __init__(
        self,
        random_state: int = 42,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        n_estimators: int = 100,
        subsample: float = 0.8,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        **kwargs: Any,
    ):
        self.random_state = random_state
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.kwargs = kwargs

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names_: List[str] = list(FEATURE_COLUMNS)
        self.categories_map_: Dict[str, List[Any]] = {}
        self.is_fitted_: bool = False

    def _init_regressor(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            random_state=self.random_state,
            enable_categorical=True,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            **self.kwargs,
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> "TyreDegradationResidualModel":
        """Fit the XGBoost regressor on causal features."""
        # Record categories mapping for deterministic schema enforcement
        self.categories_map_ = {}
        for col in CATEGORICAL_COLUMNS:
            if col in X.columns:
                if isinstance(X[col].dtype, pd.CategoricalDtype):
                    self.categories_map_[col] = list(X[col].cat.categories)
                else:
                    self.categories_map_[col] = sorted(list(X[col].dropna().unique()))

        X_proc = _clean_and_impute_features(X, self.categories_map_)
        y_arr = np.asarray(y, dtype=np.float64)

        self.model = self._init_regressor()
        self.model.fit(X_proc, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict the physics residual correction in seconds."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("TyreDegradationResidualModel must be fitted before predict.")
        X_proc = _clean_and_impute_features(X, self.categories_map_)
        preds = self.model.predict(X_proc)
        return np.asarray(preds, dtype=np.float64)

    def predict_composite(
        self,
        X: pd.DataFrame,
        t_physics: Union[pd.Series, np.ndarray, float],
    ) -> np.ndarray:
        """Calculate composite prediction: T_physics + predicted_residual."""
        pred_res = self.predict(X)
        phys = np.asarray(t_physics, dtype=np.float64)
        return phys + pred_res

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        t_physics: Union[pd.Series, np.ndarray],
        groups: Optional[Sequence[Any]] = None,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Perform out-of-fold cross-validation on held-out stints.

        Guarantees that entire stints remain together and are evaluated
        strictly on held-out folds.
        """
        y_arr = np.asarray(y, dtype=np.float64)
        t_phys_arr = np.asarray(t_physics, dtype=np.float64)
        t_act_arr = t_phys_arr + y_arr

        if groups is not None and len(np.unique(groups)) >= n_splits:
            splitter = GroupKFold(n_splits=n_splits)
            split_gen = splitter.split(X, y_arr, groups=groups)
            val_method = f"GroupKFold(n_splits={n_splits})"
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_gen = splitter.split(X, y_arr)
            val_method = f"KFold(n_splits={n_splits}, shuffle=True)"

        oof_residuals = np.zeros_like(y_arr)

        for train_idx, val_idx in split_gen:
            X_train, y_train = X.iloc[train_idx], y_arr[train_idx]
            X_val = X.iloc[val_idx]

            fold_model = TyreDegradationResidualModel(
                random_state=self.random_state,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                reg_alpha=self.reg_alpha,
                reg_lambda=self.reg_lambda,
                **self.kwargs,
            )
            fold_model.fit(X_train, y_train)
            oof_residuals[val_idx] = fold_model.predict(X_val)

        oof_composite = t_phys_arr + oof_residuals

        # Baseline Physics Metrics (T_physics vs T_actual)
        base_mae = float(mean_absolute_error(t_act_arr, t_phys_arr))
        base_rmse = float(np.sqrt(mean_squared_error(t_act_arr, t_phys_arr)))
        base_r2 = float(r2_score(t_act_arr, t_phys_arr))

        # Composite Metrics (T_composite vs T_actual)
        comp_mae = float(mean_absolute_error(t_act_arr, oof_composite))
        comp_rmse = float(np.sqrt(mean_squared_error(t_act_arr, oof_composite)))
        comp_r2 = float(r2_score(t_act_arr, oof_composite))

        # Residual Model Metrics (y_pred vs y_actual)
        res_mae = float(mean_absolute_error(y_arr, oof_residuals))
        res_rmse = float(np.sqrt(mean_squared_error(y_arr, oof_residuals)))
        res_r2 = float(r2_score(y_arr, oof_residuals))

        return {
            "n_samples": len(y_arr),
            "n_stints": len(np.unique(groups)) if groups is not None else 0,
            "n_drivers": int(X["driver"].nunique()) if "driver" in X.columns else 0,
            "n_circuits": int(X["circuit"].nunique()) if "circuit" in X.columns else 0,
            "compounds": sorted(list(X["compound"].unique())) if "compound" in X.columns else [],
            "validation_method": val_method,
            "fold_count": n_splits,
            "baseline_physics": {
                "mae": base_mae,
                "rmse": base_rmse,
                "r2": base_r2,
            },
            "composite": {
                "mae": comp_mae,
                "rmse": comp_rmse,
                "r2": comp_r2,
            },
            "residual": {
                "mae": res_mae,
                "rmse": res_rmse,
                "r2": res_r2,
            },
            "improvements": {
                "mae_improvement_s": base_mae - comp_mae,
                "mae_improvement_pct": (base_mae - comp_mae) / base_mae * 100.0 if base_mae > 0 else 0.0,
                "rmse_improvement_s": base_rmse - comp_rmse,
                "rmse_improvement_pct": (base_rmse - comp_rmse) / base_rmse * 100.0 if base_rmse > 0 else 0.0,
                "r2_change": comp_r2 - base_r2,
            },
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted model and metadata schema to a file."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Cannot save an unfitted TyreDegradationResidualModel.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names_,
            "categories_map": self.categories_map_,
            "hyperparameters": {
                "random_state": self.random_state,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "n_estimators": self.n_estimators,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "reg_alpha": self.reg_alpha,
                "reg_lambda": self.reg_lambda,
            },
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TyreDegradationResidualModel":
        """Load a saved TyreDegradationResidualModel from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        payload = joblib.load(path)
        instance = cls(**payload.get("hyperparameters", {}))
        instance.model = payload["model"]
        instance.feature_names_ = payload["feature_names"]
        instance.categories_map_ = payload["categories_map"]
        instance.is_fitted_ = True
        return instance
