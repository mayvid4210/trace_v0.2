"""Unit tests for ML Model 2: Tyre State Model (XGBoost).

Covers:
1. Feature preparation and schema
2. Target construction (effective tyre grip in (0, 1] and operational degradation s >= 0)
3. Categorical handling (circuit and compound)
4. Deterministic training and prediction (fixed seed)
5. Fit and predict behavior, shape, bounds, and float64 type
6. Leakage protection check (no future/intra-lap/speed-trap columns)
7. Validation logic and comparison against baselines
8. Save and load persistence round-trip
"""

import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from backend.ml.tyre_state import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    TyreStateModel,
    compute_tyre_grip_target,
    prepare_tyre_state_data,
)


@pytest.fixture
def synthetic_stint_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Provide synthetic feature and metadata DataFrames representing 3 stints."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame(
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
        }
    )
    meta = pd.DataFrame(
        {
            "circuit": X["circuit"],
            "driver": ["VER"] * 20 + ["VER"] * 20 + ["HAM"] * 20,
            "stint": X["stint"],
            "stint_id": [f"Bahrain_VER_S1"] * 20 + [f"Bahrain_VER_S2"] * 20 + [f"Canada_HAM_S1"] * 20,
            "lap_number": X["lap_number"],
            # Lap time increases with tyre age minus fuel effect
            "T_actual": 92.0 + X["tyre_age"] * 0.1 - (100.0 - X["fuel_mass_kg"]) * 0.035,
        }
    )
    return X, meta


def test_target_construction(synthetic_stint_data):
    """1. Test that target construction outputs valid grip ratio in (0, 1] and deg >= 0."""
    X, meta = synthetic_stint_data
    grip, deg_s = compute_tyre_grip_target(X, meta)

    assert isinstance(grip, pd.Series)
    assert isinstance(deg_s, pd.Series)
    assert len(grip) == len(X)
    assert len(deg_s) == len(X)

    # Effective grip should be between 0.5 and 1.0
    assert (grip > 0.5).all() and (grip <= 1.0).all()

    # Degradation seconds should be >= 0.0
    assert (deg_s >= 0.0).all()

    # Within each stint, initial clean lap should have grip == 1.0 and deg_s == 0.0
    for stint_id in meta["stint_id"].unique():
        stint_mask = meta["stint_id"] == stint_id
        assert np.isclose(grip[stint_mask].iloc[0], 1.0, atol=1e-5)
        assert np.isclose(deg_s[stint_mask].iloc[0], 0.0, atol=1e-5)


def test_causal_target_construction_no_future_leakage(synthetic_stint_data):
    """Verify that target for lap k is completely unaffected by future laps in the stint."""
    X, meta = synthetic_stint_data
    grip_orig, deg_orig = compute_tyre_grip_target(X, meta)

    # Modify future laps (e.g. laps 15-20 in stint 1) to be dramatically faster
    meta_future = meta.copy()
    meta_future.loc[14:19, "T_actual"] = 70.0

    grip_future, deg_future = compute_tyre_grip_target(X, meta_future)

    # Laps 0 to 13 (prior to lap 14) MUST remain bit-for-bit identical
    assert np.allclose(grip_orig.iloc[:14], grip_future.iloc[:14], atol=1e-12)
    assert np.allclose(deg_orig.iloc[:14], deg_future.iloc[:14], atol=1e-12)


def test_feature_preparation_and_schema():
    """2. Verify feature matrix contains exact approved columns without driver or leakage."""
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
    X, y_grip, y_deg, meta = prepare_tyre_state_data(laps_df=raw_df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert "driver" not in X.columns
    assert len(X) == 2
    assert len(y_grip) == 2
    assert len(y_deg) == 2
    assert "stint_id" in meta.columns


def test_categorical_handling(synthetic_stint_data):
    """3. Test categorical handling and prediction on unseen categories."""
    X, meta = synthetic_stint_data
    grip, _ = compute_tyre_grip_target(X, meta)

    model = TyreStateModel(n_estimators=10)
    model.fit(X, grip)

    assert model.is_fitted_
    assert "circuit" in model.categories_map_
    assert "compound" in model.categories_map_

    # Unseen circuit should predict gracefully without error
    X_unseen = X.iloc[:5].copy()
    X_unseen["circuit"] = "MONZA"
    preds = model.predict(X_unseen)
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


def test_deterministic_behavior(synthetic_stint_data):
    """4. Verify identical random_state produces bit-for-bit identical grip predictions."""
    X, meta = synthetic_stint_data
    grip, _ = compute_tyre_grip_target(X, meta)

    model1 = TyreStateModel(random_state=42, n_estimators=20)
    model1.fit(X, grip)
    preds1 = model1.predict(X)

    model2 = TyreStateModel(random_state=42, n_estimators=20)
    model2.fit(X, grip)
    preds2 = model2.predict(X)

    assert np.allclose(preds1, preds2, atol=1e-12)


def test_fit_predict_shape_and_bounds(synthetic_stint_data):
    """5. Ensure predictions are 1D float64 within valid grip bounds [0.5, 1.0]."""
    X, meta = synthetic_stint_data
    grip, _ = compute_tyre_grip_target(X, meta)

    model = TyreStateModel(n_estimators=10)
    model.fit(X, grip)
    preds = model.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert preds.shape == (len(X),)
    assert preds.dtype == np.float64
    assert np.all((preds >= 0.5) & (preds <= 1.0))


def test_leakage_exclusions():
    """6. Ensure no intra-lap telemetry, speed traps, or completed lap times in features."""
    forbidden = ["sector", "speed", "trap", "laptime", "future", "driver"]
    for col in FEATURE_COLUMNS:
        col_lower = col.lower()
        for kw in forbidden:
            assert kw not in col_lower, f"Forbidden term '{kw}' in feature '{col}'"


def test_validation_and_baseline_comparison(synthetic_stint_data):
    """7. Test out-of-fold validation output structure and baseline comparisons."""
    X, meta = synthetic_stint_data
    grip, _ = compute_tyre_grip_target(X, meta)

    model = TyreStateModel(n_estimators=10)
    metrics = model.evaluate(X, grip, groups=meta["stint_id"], n_splits=3)

    required_keys = [
        "n_samples",
        "n_groups",
        "validation_method",
        "fold_count",
        "mean_baseline",
        "linear_physics_baseline",
        "ml_model",
        "improvements_vs_mean",
        "improvements_vs_linear_physics",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key '{key}' in evaluation output"

    assert "mae" in metrics["ml_model"]
    assert "rmse" in metrics["ml_model"]
    assert "r2" in metrics["ml_model"]


def test_save_and_load_roundtrip(synthetic_stint_data):
    """8. Verify save and load persistence round-trip reproduces predictions exactly."""
    X, meta = synthetic_stint_data
    grip, _ = compute_tyre_grip_target(X, meta)

    model = TyreStateModel(random_state=42, n_estimators=15)
    model.fit(X, grip)
    initial_preds = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_tyre_state.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = TyreStateModel.load(save_path)
        loaded_preds = loaded_model.predict(X)

        assert np.allclose(initial_preds, loaded_preds, atol=1e-12)
        assert loaded_model.is_fitted_ is True
