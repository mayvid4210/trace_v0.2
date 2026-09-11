"""Traffic and dirty-air lap-time penalty model using XGBoost.

Objective:
    Estimate the additional lap-time penalty attributable to traffic and
    aerodynamic dirty air (wake turbulence) from observable race features.

Scientific Formulation:
    When a car is in clean air (gap_ahead >= 3.0s), wake turbulence is negligible
    and the traffic penalty is 0.0 seconds.
    When a car is in close proximity to a car ahead (gap_ahead < 3.0s), aerodynamic
    downforce loss and inability to use the optimal racing line produce a lap-time
    penalty:
        traffic_penalty_s = max(0.0, residual - clean_residual_ref)
    where clean_residual_ref is the expected baseline residual in clean air for that
    circuit and compound.

Data Limitation:
    In public F1 telemetry, aerodynamic downforce cannot be measured directly.
    This target represents an operational dirty-air lap-time penalty proxy.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

from backend.ml.tyre_degradation_residual import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    _clean_and_impute_features,
    prepare_training_data,
)

TRAFFIC_FEATURE_COLUMNS: List[str] = [
    "circuit",
    "compound",
    "tyre_age",
    "stint",
    "stint_lap",
    "lap_number",
    "fuel_mass_kg",
    "track_temperature",
    "air_temperature",
    "gap_ahead_s",
    "gap_behind_s",
    "position",
]

CLEAN_AIR_RESIDUAL_REF: Dict[Tuple[str, str], float] = {
    ("BAHRAIN", "HARD"): 1.81,
    ("BAHRAIN", "SOFT"): 3.47,
    ("BAHRAIN", "MEDIUM"): 2.20,
    ("CANADA", "HARD"): 0.59,
    ("CANADA", "MEDIUM"): 3.99,
}


def compute_traffic_penalty_target(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.Series:
    """Compute causal operational traffic penalty proxy in seconds.

    Returns:
        Series of traffic penalty in seconds (0.0 in clean air, >= 0.0 in traffic).
    """
    circuits = X["circuit"].astype(str).str.strip().str.upper()
    compounds = X["compound"].astype(str).str.strip().str.upper()
    gaps_ahead = pd.to_numeric(X.get("gap_ahead_s", 60.0), errors="coerce").fillna(60.0)
    residuals = pd.to_numeric(metadata["residual"], errors="coerce").fillna(0.0)

    clean_refs = []
    for c, comp in zip(circuits, compounds):
        clean_refs.append(CLEAN_AIR_RESIDUAL_REF.get((c, comp), 2.0))
    clean_ref_arr = np.array(clean_refs, dtype=np.float64)

    # In clean air (gap_ahead >= 3.0s), traffic penalty is strictly 0.0.
    # In dirty air (gap_ahead < 3.0s), penalty is excess time above clean air expectation.
    penalties = np.where(
        gaps_ahead >= 3.0,
        0.0,
        np.maximum(0.0, residuals - clean_ref_arr),
    )
    # Clip extreme outliers (e.g. yellow flags or off-track excursions)
    penalties = np.clip(penalties, 0.0, 5.0)

    return pd.Series(penalties, index=X.index, name="traffic_penalty_s")


def _impute_traffic_features(
    df: pd.DataFrame,
    categories_map: Optional[Dict[str, List[Any]]] = None,
) -> pd.DataFrame:
    """Ensure exact TRAFFIC_FEATURE_COLUMNS schema and causal numeric imputation."""
    base_clean = _clean_and_impute_features(df, categories_map=categories_map)
    X = base_clean.copy()

    X["gap_ahead_s"] = pd.to_numeric(df.get("gap_ahead_s", 60.0), errors="coerce").fillna(60.0).clip(0.0, 60.0)
    X["gap_behind_s"] = pd.to_numeric(df.get("gap_behind_s", 60.0), errors="coerce").fillna(60.0).clip(0.0, 60.0)
    X["position"] = pd.to_numeric(df.get("position", 10.0), errors="coerce").fillna(10.0).clip(1.0, 20.0)

    return X[TRAFFIC_FEATURE_COLUMNS].copy()


def prepare_traffic_data(
    laps_df: Optional[pd.DataFrame] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    circuits: Sequence[str] = ("Bahrain", "Canada"),
    year: int = 2024,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Prepare clean causal feature matrix, traffic penalty target, and metadata.

    Returns:
        X: Feature DataFrame with TRAFFIC_FEATURE_COLUMNS.
        y_traffic: Operational traffic penalty target Series in seconds.
        metadata: Tracking metadata (stint_id, driver, circuit, lap_number, T_actual, residual).
    """
    X_base, _, _, metadata = prepare_training_data(
        laps_df=laps_df,
        cache_dir=cache_dir,
        circuits=circuits,
        year=year,
    )

    # If laps_df was provided with gap columns, use them directly; otherwise load gaps from FastF1
    if laps_df is not None and "gap_ahead_s" in laps_df.columns:
        X_base["gap_ahead_s"] = laps_df["gap_ahead_s"].values
        X_base["gap_behind_s"] = laps_df.get("gap_behind_s", 60.0)
        X_base["position"] = laps_df.get("position", 10.0)
    else:
        import fastf1

        raw_dir = cache_dir or (
            Path(__file__).resolve().parent.parent / "data" / "raw"
        )
        fastf1.Cache.enable_cache(str(raw_dir))

        laps_list = []
        for gp in circuits:
            session = fastf1.get_session(year, gp, "R")
            session.load(laps=True, telemetry=False, weather=True)
            l = session.laps.copy()
            l = l[l["PitInTime"].isna() & l["PitOutTime"].isna() & l["LapTime"].notna()].copy()
            l["circuit"] = gp
            l["start_time_s"] = l["LapStartTime"].dt.total_seconds()
            l = l.sort_values(["circuit", "LapNumber", "start_time_s"])
            l["gap_ahead_s"] = l.groupby(["circuit", "LapNumber"])["start_time_s"].diff().fillna(60.0).clip(0.0, 60.0)
            l["gap_behind_s"] = (-l.groupby(["circuit", "LapNumber"])["start_time_s"].diff(-1)).fillna(60.0).clip(0.0, 60.0)
            l["position"] = l["Position"].fillna(10.0).astype(float)
            laps_list.append(l[["circuit", "Driver", "LapNumber", "gap_ahead_s", "gap_behind_s", "position"]])

        gaps_df = pd.concat(laps_list, ignore_index=True).rename(
            columns={"Driver": "driver", "LapNumber": "lap_number"}
        )
        merged = metadata.merge(gaps_df, on=["circuit", "driver", "lap_number"], how="left")
        X_base["gap_ahead_s"] = merged["gap_ahead_s"].fillna(60.0).values
        X_base["gap_behind_s"] = merged["gap_behind_s"].fillna(60.0).values
        X_base["position"] = merged["position"].fillna(10.0).values

    X = _impute_traffic_features(X_base)
    y_traffic = compute_traffic_penalty_target(X, metadata)
    return X, y_traffic, metadata


class TrafficPenaltyModel:
    """Production XGBoost model predicting traffic / dirty-air lap-time penalty."""

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
        self.feature_names_: List[str] = list(TRAFFIC_FEATURE_COLUMNS)
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
    ) -> "TrafficPenaltyModel":
        """Fit the traffic penalty regressor on causal features."""
        self.categories_map_ = {}
        for col in CATEGORICAL_COLUMNS:
            if col in X.columns:
                if isinstance(X[col].dtype, pd.CategoricalDtype):
                    self.categories_map_[col] = list(X[col].cat.categories)
                else:
                    self.categories_map_[col] = sorted(list(X[col].dropna().unique()))

        X_proc = _impute_traffic_features(X, self.categories_map_)
        y_arr = np.asarray(y, dtype=np.float64)

        self.model = self._init_regressor()
        self.model.fit(X_proc, y_arr)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict additional lap time in seconds due to traffic / dirty air."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("TrafficPenaltyModel must be fitted before predict.")
        X_proc = _impute_traffic_features(X, self.categories_map_)
        preds = self.model.predict(X_proc)
        return np.clip(np.asarray(preds, dtype=np.float64), 0.0, 5.0)

    def evaluate(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        groups: Optional[Sequence[Any]] = None,
        n_splits: int = 5,
    ) -> Dict[str, Any]:
        """Evaluate out-of-fold predictions and compare against simple baselines."""
        y_arr = np.asarray(y, dtype=np.float64)

        if groups is not None and len(np.unique(groups)) >= n_splits:
            splitter = GroupKFold(n_splits=n_splits)
            split_gen = splitter.split(X, y_arr, groups=groups)
            val_method = f"GroupKFold(n_splits={n_splits})"
        else:
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            split_gen = splitter.split(X, y_arr)
            val_method = f"KFold(n_splits={n_splits}, shuffle=True)"

        oof_preds = np.zeros_like(y_arr)

        for train_idx, val_idx in split_gen:
            X_train, y_train = X.iloc[train_idx], y_arr[train_idx]
            X_val = X.iloc[val_idx]

            fold_model = TrafficPenaltyModel(
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
            oof_preds[val_idx] = fold_model.predict(X_val)

        # Baseline 1: Constant zero (assumes clean air everywhere)
        zero_baseline = np.zeros_like(y_arr)
        base_mae = float(mean_absolute_error(y_arr, zero_baseline))
        base_rmse = float(np.sqrt(mean_squared_error(y_arr, zero_baseline)))

        # Baseline 2: Simple gap heuristic:
        # if gap < 1.0s -> 1.2s; if gap < 2.0s -> 0.6s; else 0.0s
        gaps = pd.to_numeric(X.get("gap_ahead_s", 60.0), errors="coerce").fillna(60.0).values
        heur_baseline = np.where(gaps < 1.0, 1.2, np.where(gaps < 2.0, 0.6, 0.0))
        heur_mae = float(mean_absolute_error(y_arr, heur_baseline))
        heur_rmse = float(np.sqrt(mean_squared_error(y_arr, heur_baseline)))
        heur_r2 = float(r2_score(y_arr, heur_baseline))

        # ML Model Performance
        ml_mae = float(mean_absolute_error(y_arr, oof_preds))
        ml_rmse = float(np.sqrt(mean_squared_error(y_arr, oof_preds)))
        ml_r2 = float(r2_score(y_arr, oof_preds))

        return {
            "n_samples": len(y_arr),
            "n_groups": len(np.unique(groups)) if groups is not None else 0,
            "validation_method": val_method,
            "fold_count": n_splits,
            "zero_baseline": {
                "mae": base_mae,
                "rmse": base_rmse,
            },
            "heuristic_baseline": {
                "mae": heur_mae,
                "rmse": heur_rmse,
                "r2": heur_r2,
            },
            "ml_model": {
                "mae": ml_mae,
                "rmse": ml_rmse,
                "r2": ml_r2,
            },
            "improvements_vs_zero": {
                "mae_improvement_s": base_mae - ml_mae,
                "mae_improvement_pct": (base_mae - ml_mae) / base_mae * 100.0 if base_mae > 0 else 0.0,
                "rmse_improvement_s": base_rmse - ml_rmse,
            },
            "improvements_vs_heuristic": {
                "mae_improvement_s": heur_mae - ml_mae,
                "mae_improvement_pct": (heur_mae - ml_mae) / heur_mae * 100.0 if heur_mae > 0 else 0.0,
                "rmse_improvement_s": heur_rmse - ml_rmse,
                "r2_change": ml_r2 - heur_r2,
            },
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted model and metadata schema to a file."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Cannot save an unfitted TrafficPenaltyModel.")
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
    def load(cls, filepath: Union[str, Path]) -> "TrafficPenaltyModel":
        """Load a saved TrafficPenaltyModel from disk."""
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
