"""Unit tests for ML Model 1: Tyre Degradation Residual (XGBoost).

Covers:
1. Residual calculation (T_actual - T_physics)
2. Feature preparation and schema alignment
3. Missing-value handling/imputation
4. Categorical handling (circuit, compound, driver)
5. Deterministic training and prediction (fixed seed)
6. Prediction shape and numeric type (float64)
7. Leakage protection check (no future/intra-lap columns)
8. Save/load round-trip via model.save() and load()
9. Evaluation output dictionary structure
10. Composite prediction verification (T_physics + predicted_residual)
"""

import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from backend.ml.tyre_degradation_residual import (
    CATEGORICAL_COLUMNS,
    DEFAULT_CIRCUIT_CALIBRATIONS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TyreDegradationResidualModel,
    _clean_and_impute_features,
    calculate_residual,
    prepare_training_data,
)
from backend.physics.lap_time_predictor import CircuitCalibration, LapTimePredictor


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Fixture providing a synthetic, well-formed feature DataFrame."""
    np.random.seed(42)
    n = 60
    return pd.DataFrame(
        {
            "circuit": ["Bahrain"] * 30 + ["Canada"] * 30,
            "compound": (["SOFT", "MEDIUM", "HARD"] * 20)[:n],
            "tyre_age": np.tile(np.arange(1, 21), 3),
            "stint": [1] * 20 + [2] * 20 + [1] * 20,
            "stint_lap": np.tile(np.arange(1, 21), 3),
            "lap_number": list(range(1, 21)) + list(range(21, 41)) + list(range(1, 21)),
            "fuel_mass_kg": np.linspace(100.0, 10.0, n),
            "track_temperature": np.random.uniform(28.0, 42.0, n),
            "air_temperature": np.random.uniform(20.0, 30.0, n),
            "driver": ["VER"] * 20 + ["VER"] * 20 + ["HAM"] * 20,
        }
    )


@pytest.fixture
def sample_targets(sample_features_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """Provide realistic synthetic residuals and physics baseline."""
    n = len(sample_features_df)
    t_physics = pd.Series(92.0 + sample_features_df["tyre_age"] * 0.1 - sample_features_df["fuel_mass_kg"] * 0.03)
    noise = np.random.RandomState(42).normal(0.0, 0.2, n)
    residuals = pd.Series(0.05 * sample_features_df["tyre_age"] + noise)
    return residuals, t_physics


def test_residual_calculation():
    """1. Test residual calculation logic: T_actual - T_physics."""
    actual = np.array([93.5, 94.0, 95.2])
    physics = np.array([92.0, 93.0, 94.0])
    expected = np.array([1.5, 1.0, 1.2])

    res = calculate_residual(actual, physics)
    assert np.allclose(res, expected)
    assert res.dtype == np.float64

    # Test with pandas Series
    s_actual = pd.Series([90.0, 100.0])
    s_physics = pd.Series([88.5, 98.0])
    s_res = calculate_residual(s_actual, s_physics)
    assert np.allclose(s_res, [1.5, 2.0])

    # Test shape mismatch raises ValueError
    with pytest.raises(ValueError, match="Shape mismatch"):
        calculate_residual([90.0, 91.0], [90.0])


def test_feature_preparation_and_schema():
    """2. Verify feature matrix contains exact approved columns in correct order."""
    raw_df = pd.DataFrame(
        {
            "GrandPrix": ["Bahrain", "Bahrain"],
            "Driver": ["VER", "VER"],
            "Compound": ["SOFT", "SOFT"],
            "LapNumber": [1, 2],
            "Stint": [1, 1],
            "TyreLife": [1.0, 2.0],
            "LapTime_s": [93.2, 93.5],
            "fuel_mass_kg": [100.0, 98.2],
            "track_temperature": [35.0, 35.1],
            "air_temperature": [25.0, 25.0],
        }
    )

    X, y, t_phys, meta = prepare_training_data(laps_df=raw_df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert len(X) == 2
    assert len(y) == 2
    assert len(t_phys) == 2
    assert "stint_id" in meta.columns
    assert meta["stint_id"].iloc[0] == "Bahrain_VER_S1"

    for cat_col in CATEGORICAL_COLUMNS:
        assert isinstance(X[cat_col].dtype, pd.CategoricalDtype)


def test_missing_value_imputation():
    """3. Test that missing/NaN values in features are cleanly imputed without failure."""
    dirty_df = pd.DataFrame(
        {
            "circuit": [None, "Bahrain"],
            "compound": ["SOFT", None],
            "tyre_age": [np.nan, 5.0],
            "stint": [None, 2],
            "stint_lap": [np.nan, 3],
            "lap_number": [np.nan, 10],
            "fuel_mass_kg": [np.nan, 45.0],
            "track_temperature": [np.nan, 32.0],
            "air_temperature": [np.nan, 24.0],
            "driver": [None, "NOR"],
        }
    )

    cleaned = _clean_and_impute_features(dirty_df)

    assert list(cleaned.columns) == FEATURE_COLUMNS
    assert not cleaned.isna().any().any()
    # Check default values used
    assert cleaned["tyre_age"].iloc[0] == 1.0
    assert cleaned["fuel_mass_kg"].iloc[0] == 50.0
    assert cleaned["track_temperature"].iloc[0] == 30.0
    assert cleaned["air_temperature"].iloc[0] == 25.0


def test_categorical_handling(sample_features_df):
    """4. Test categorical dtype handling and consistency across train and test."""
    residuals = np.random.RandomState(42).normal(0.0, 0.5, len(sample_features_df))
    model = TyreDegradationResidualModel(n_estimators=10)
    model.fit(sample_features_df, residuals)

    assert model.is_fitted_
    assert "circuit" in model.categories_map_
    assert "compound" in model.categories_map_

    # Predict with matching categories
    preds = model.predict(sample_features_df.iloc[:5])
    assert len(preds) == 5

    # Predict with new unseen category should not raise an error
    test_df = sample_features_df.iloc[:3].copy()
    test_df["circuit"] = "NEW_CIRCUIT"
    preds_new = model.predict(test_df)
    assert len(preds_new) == 3
    assert np.all(np.isfinite(preds_new))


def test_deterministic_training_and_prediction(sample_features_df):
    """5. Verify identical random_state produces bit-for-bit identical predictions."""
    residuals = np.sin(sample_features_df["tyre_age"]) + 0.1 * sample_features_df["fuel_mass_kg"]

    model1 = TyreDegradationResidualModel(random_state=42, n_estimators=30, max_depth=3)
    model1.fit(sample_features_df, residuals)
    preds1 = model1.predict(sample_features_df)

    model2 = TyreDegradationResidualModel(random_state=42, n_estimators=30, max_depth=3)
    model2.fit(sample_features_df, residuals)
    preds2 = model2.predict(sample_features_df)

    assert np.allclose(preds1, preds2, atol=1e-12)


def test_prediction_shape_and_type(sample_features_df):
    """6. Ensure prediction output is a 1D numpy array with float64 dtype and correct shape."""
    residuals = np.zeros(len(sample_features_df))
    model = TyreDegradationResidualModel(n_estimators=10)
    model.fit(sample_features_df, residuals)

    preds = model.predict(sample_features_df)
    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert preds.shape == (len(sample_features_df),)
    assert preds.dtype == np.float64


def test_leakage_protection():
    """7. Strict leakage check: verify no future, intra-lap, or forbidden columns in features."""
    forbidden_keywords = [
        "sector",
        "speed",
        "future",
        "next",
        "pitintime",
        "pitouttime",
        "target",
        "time_s",
        "laptime",
        "crossover",
    ]

    for col in FEATURE_COLUMNS:
        col_lower = col.lower()
        for kw in forbidden_keywords:
            assert kw not in col_lower, f"Forbidden keyword '{kw}' found in feature column '{col}'"

    raw_laps = pd.DataFrame(
        {
            "GrandPrix": ["Bahrain"],
            "Driver": ["VER"],
            "Compound": ["SOFT"],
            "LapNumber": [1],
            "Stint": [1],
            "TyreLife": [1.0],
            "LapTime_s": [93.2],
            "Sector1Time": [30.0],
            "Sector2Time": [40.0],
            "Sector3Time": [23.2],
            "SpeedI1": [310.0],
            "SpeedFL": [290.0],
            "fuel_mass_kg": [100.0],
            "track_temperature": [35.0],
            "air_temperature": [25.0],
        }
    )
    X, _, _, _ = prepare_training_data(laps_df=raw_laps)
    assert "Sector1Time" not in X.columns
    assert "Sector2Time" not in X.columns
    assert "Sector3Time" not in X.columns
    assert "SpeedI1" not in X.columns
    assert "SpeedFL" not in X.columns


def test_model_save_and_load_roundtrip(sample_features_df):
    """8. Test model persistence round-trip and numerical equality of predictions."""
    residuals = np.random.RandomState(42).normal(0.0, 0.5, len(sample_features_df))
    model = TyreDegradationResidualModel(random_state=42, n_estimators=20)
    model.fit(sample_features_df, residuals)

    initial_preds = model.predict(sample_features_df)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_residual_model.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = TyreDegradationResidualModel.load(save_path)
        loaded_preds = loaded_model.predict(sample_features_df)

        assert np.allclose(initial_preds, loaded_preds, atol=1e-12)
        assert loaded_model.feature_names_ == model.feature_names_
        assert loaded_model.is_fitted_ is True


def test_evaluation_metrics_structure(sample_features_df):
    """9. Verify evaluation output structure contains required metrics and keys."""
    n = len(sample_features_df)
    residuals = np.random.RandomState(42).normal(0.0, 0.3, n)
    t_physics = np.full(n, 93.0)
    stints = sample_features_df["circuit"] + "_" + sample_features_df["driver"] + "_S" + sample_features_df["stint"].astype(str)

    model = TyreDegradationResidualModel(random_state=42, n_estimators=10)
    metrics = model.evaluate(
        X=sample_features_df,
        y=residuals,
        t_physics=t_physics,
        groups=stints,
        n_splits=3,
    )

    required_top_keys = [
        "n_samples",
        "n_stints",
        "n_drivers",
        "n_circuits",
        "compounds",
        "validation_method",
        "fold_count",
        "baseline_physics",
        "composite",
        "residual",
        "improvements",
    ]
    for key in required_top_keys:
        assert key in metrics, f"Missing key '{key}' in evaluation metrics"

    for sub_metric in ["baseline_physics", "composite", "residual"]:
        assert "mae" in metrics[sub_metric]
        assert "rmse" in metrics[sub_metric]
        assert "r2" in metrics[sub_metric]

    assert "mae_improvement_s" in metrics["improvements"]
    assert "mae_improvement_pct" in metrics["improvements"]
    assert "rmse_improvement_s" in metrics["improvements"]
    assert "r2_change" in metrics["improvements"]


def test_composite_prediction(sample_features_df):
    """10. Verify composite prediction equals exact sum: T_physics + residual."""
    residuals = np.random.RandomState(42).normal(0.0, 0.4, len(sample_features_df))
    t_physics = np.full(len(sample_features_df), 93.5)

    model = TyreDegradationResidualModel(random_state=42, n_estimators=10)
    model.fit(sample_features_df, residuals)

    pred_residuals = model.predict(sample_features_df)
    composite_preds = model.predict_composite(sample_features_df, t_physics)

    assert np.allclose(composite_preds, t_physics + pred_residuals, atol=1e-12)
