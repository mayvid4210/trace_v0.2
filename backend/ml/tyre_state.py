"""Tyre state estimation model using XGBoost.

Objective:
    Estimate the current effective tyre grip state from observable race and
    telemetry features, without relying on unidentifiable hidden tyre sensors.

Target Definition:
    Effective Tyre Grip (operational proxy in (0, 1]):
        Fuel-corrected lap time neutralizes known fuel mass burn:
            T_fuel_corr = T_actual + (100.0 - fuel_mass_kg) * beta_fuel
        Peak stint condition:
            T_stint_peak = min_{stint} T_fuel_corr
        Effective Grip:
            effective_tyre_grip = T_stint_peak / T_fuel_corr

    Operational Degradation State (seconds lost relative to peak condition):
        tyre_degradation_state_s = T_fuel_corr - T_stint_peak
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

CIRCUIT_FUEL_BETAS: Dict[str, float] = {
    "BAHRAIN": 0.0392,
    "CANADA": 0.0261,
}

VALIDATED_TYRE_RATES: Dict[str, float] = {
    "SOFT": 0.0627,
    "MEDIUM": 0.0447,
    "HARD": 0.1060,
}


def compute_tyre_grip_target(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """Calculate causal effective tyre grip ratio and operational degradation seconds.

    Causal Reference Guarantee:
        For each lap k within a stint, the reference peak pace is strictly computed
        using only laps j <= k in that stint (via cumulative minimum):
            T_causal_peak(k) = min_{j <= k} T_fuel_corr(j)
        Zero future laps from the stint are used.

    Returns:
        target_grip: Series of effective grip ratio in (0, 1].
        target_deg_s: Series of operational degradation seconds >= 0.
    """
    circuits = X["circuit"].astype(str).str.strip().str.upper()
    beta_fuel = circuits.map(CIRCUIT_FUEL_BETAS).fillna(0.03)

    fuel_burned = 100.0 - pd.to_numeric(X["fuel_mass_kg"], errors="coerce").fillna(50.0)
    t_actual = pd.to_numeric(metadata["T_actual"], errors="coerce")
    t_fuel_corr = t_actual + fuel_burned * beta_fuel

    stint_ids = metadata["stint_id"].astype(str)
    # Strictly causal cumulative minimum: only laps j <= k are evaluated
    t_peak = t_fuel_corr.groupby(stint_ids).cummin()

    # Guard against zero or negative division
    t_fuel_corr_safe = np.maximum(t_fuel_corr, 1.0)
    t_peak_safe = np.maximum(t_peak, 1.0)

    target_grip = (t_peak_safe / t_fuel_corr_safe).clip(lower=0.5, upper=1.0)
    target_deg_s = (t_fuel_corr - t_peak).clip(lower=0.0)

    return target_grip, target_deg_s


def prepare_tyre_state_data(
    laps_df: Optional[pd.DataFrame] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    circuits: Sequence[str] = ("Bahrain", "Canada"),
    year: int = 2024,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Prepare clean causal feature matrix, tyre grip target, and metadata.

    Returns:
        X: Feature DataFrame with FEATURE_COLUMNS.
        y_grip: Effective tyre grip target Series.
        y_deg_s: Operational tyre degradation seconds Series.
        metadata: Tracking metadata (stint_id, driver, circuit, lap_number, T_actual).
    """
    X, _, _, metadata = prepare_training_data(
        laps_df=laps_df,
        cache_dir=cache_dir,
        circuits=circuits,
        year=year,
    )
    y_grip, y_deg_s = compute_tyre_grip_target(X, metadata)
    return X, y_grip, y_deg_s, metadata


class TyreStateModel:
    """Production XGBoost model predicting effective tyre grip state."""

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
    ) -> "TyreStateModel":
        """Fit the tyre state regressor on causal features."""
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
        """Predict the effective tyre grip state in (0, 1]."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("TyreStateModel must be fitted before predict.")
        X_proc = _clean_and_impute_features(X, self.categories_map_)
        preds = self.model.predict(X_proc)
        return np.clip(np.asarray(preds, dtype=np.float64), 0.5, 1.0)

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

            fold_model = TyreStateModel(
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

        # Baseline 1: Constant mean grip
        mean_baseline = np.full_like(y_arr, fill_value=np.mean(y_arr))
        base_mae = float(mean_absolute_error(y_arr, mean_baseline))
        base_rmse = float(np.sqrt(mean_squared_error(y_arr, mean_baseline)))
        base_r2 = float(r2_score(y_arr, mean_baseline))

        # Baseline 2: Linear physics degradation rate (1 - rate * (age - 1) / 90.0)
        comp_series = X.get("compound", "HARD").astype(str).str.upper()
        rate_series = comp_series.map(VALIDATED_TYRE_RATES).fillna(0.1060)
        age_series = pd.to_numeric(X.get("tyre_age", 1.0), errors="coerce").fillna(1.0)
        lin_baseline = (1.0 - (rate_series * (age_series - 1.0) / 90.0)).clip(0.5, 1.0)
        lin_mae = float(mean_absolute_error(y_arr, lin_baseline))
        lin_rmse = float(np.sqrt(mean_squared_error(y_arr, lin_baseline)))
        lin_r2 = float(r2_score(y_arr, lin_baseline))

        # ML Model Performance
        ml_mae = float(mean_absolute_error(y_arr, oof_preds))
        ml_rmse = float(np.sqrt(mean_squared_error(y_arr, oof_preds)))
        ml_r2 = float(r2_score(y_arr, oof_preds))

        return {
            "n_samples": len(y_arr),
            "n_groups": len(np.unique(groups)) if groups is not None else 0,
            "validation_method": val_method,
            "fold_count": n_splits,
            "mean_baseline": {
                "mae": base_mae,
                "rmse": base_rmse,
                "r2": base_r2,
            },
            "linear_physics_baseline": {
                "mae": lin_mae,
                "rmse": lin_rmse,
                "r2": lin_r2,
            },
            "ml_model": {
                "mae": ml_mae,
                "rmse": ml_rmse,
                "r2": ml_r2,
            },
            "improvements_vs_mean": {
                "mae_improvement": base_mae - ml_mae,
                "mae_improvement_pct": (base_mae - ml_mae) / base_mae * 100.0 if base_mae > 0 else 0.0,
                "rmse_improvement": base_rmse - ml_rmse,
                "r2_change": ml_r2 - base_r2,
            },
            "improvements_vs_linear_physics": {
                "mae_improvement": lin_mae - ml_mae,
                "mae_improvement_pct": (lin_mae - ml_mae) / lin_mae * 100.0 if lin_mae > 0 else 0.0,
                "rmse_improvement": lin_rmse - ml_rmse,
                "r2_change": ml_r2 - lin_r2,
            },
        }

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted model and metadata schema to a file."""
        if not self.is_fitted_ or self.model is None:
            raise RuntimeError("Cannot save an unfitted TyreStateModel.")
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
    def load(cls, filepath: Union[str, Path]) -> "TyreStateModel":
        """Load a saved TyreStateModel from disk."""
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
