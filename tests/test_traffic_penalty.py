"""Unit tests for ML Model 3: Traffic / Dirty-Air Penalty Model (XGBoost).

Covers:
1. Target construction (zero in clean air, >= 0 in dirty air)
2. Feature construction and schema alignment
3. Leakage exclusion (no sectors, speed traps, lap time, driver identity)
4. Future information exclusion
5. Categorical handling (circuit and compound)
6. Deterministic training and prediction (fixed seed)
7. Fit and predict behavior, shape, and [0.0, 5.0] bounds
8. Grouped validation and baseline comparisons
9. Save and load persistence round-trip
"""

import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from backend.ml.traffic_penalty import (
    CATEGORICAL_COLUMNS,
    TRAFFIC_FEATURE_COLUMNS,
    TrafficPenaltyModel,
    compute_traffic_penalty_target,
    prepare_traffic_data,
)


@pytest.fixture
def synthetic_traffic_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Provide synthetic feature and metadata DataFrames representing laps in traffic and clean air."""
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
            # Half laps in clean air (> 3.0s), half in dirty air (< 2.0s)
            "gap_ahead_s": [0.8, 1.2, 2.5, 4.0, 10.0, 0.5] * 10,
            "gap_behind_s": [2.0, 5.0, 1.0, 8.0, 15.0, 1.5] * 10,
            "position": np.tile(np.arange(1, 7), 10),
        }
    )
    meta = pd.DataFrame(
        {
            "circuit": X["circuit"],
            "driver": ["VER"] * 20 + ["VER"] * 20 + ["HAM"] * 20,
            "stint": X["stint"],
            "stint_id": ["Bahrain_VER_S1"] * 20 + ["Bahrain_VER_S2"] * 20 + ["Canada_HAM_S1"] * 20,
            "lap_number": X["lap_number"],
            "residual": np.where(X["gap_ahead_s"] < 2.0, 3.5, 1.8),
            "T_actual": 92.0 + np.where(X["gap_ahead_s"] < 2.0, 3.5, 1.8),
        }
    )
    return X, meta


def test_target_construction(synthetic_traffic_data):
    """1. Test target construction: zero in clean air, positive in traffic."""
    X, meta = synthetic_traffic_data
    target = compute_traffic_penalty_target(X, meta)

    assert isinstance(target, pd.Series)
    assert len(target) == len(X)
    assert (target >= 0.0).all()
    assert (target <= 5.0).all()

    # When gap_ahead >= 3.0s, penalty MUST be strictly 0.0
    clean_air = X["gap_ahead_s"] >= 3.0
    assert np.allclose(target[clean_air], 0.0)

    # In close traffic (gap < 1.0s) with high residual, penalty should be positive
    traffic_laps = X["gap_ahead_s"] < 1.0
    assert (target[traffic_laps] > 0.0).all()


def test_feature_construction_and_schema():
    """2. Verify feature matrix contains exact TRAFFIC_FEATURE_COLUMNS without driver."""
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
            "gap_ahead_s": [5.0, 1.2],
            "gap_behind_s": [10.0, 3.0],
            "position": [1.0, 2.0],
        }
    )
    X, y, meta = prepare_traffic_data(laps_df=raw_df)

    assert list(X.columns) == TRAFFIC_FEATURE_COLUMNS
    assert "driver" not in X.columns
    assert len(X) == 2
    assert len(y) == 2
    assert "gap_ahead_s" in X.columns
    assert "gap_behind_s" in X.columns
    assert "position" in X.columns


def test_leakage_exclusion():
    """3. Verify no sector times, speed traps, lap times, or driver identity in features."""
    forbidden = ["sector", "speed", "trap", "laptime", "t_actual", "driver"]
    for col in TRAFFIC_FEATURE_COLUMNS:
        col_lower = col.lower()
        for kw in forbidden:
            assert kw not in col_lower, f"Forbidden term '{kw}' in feature '{col}'"


def test_future_information_exclusion(synthetic_traffic_data):
    """4. Verify modifying future laps does not leak backwards into earlier lap targets."""
    X, meta = synthetic_traffic_data
    target_orig = compute_traffic_penalty_target(X, meta)

    # Modify future laps (laps 15 to 20 in stint 1)
    meta_future = meta.copy()
    meta_future.loc[14:19, "residual"] = 10.0
    meta_future.loc[14:19, "T_actual"] = 110.0

    target_future = compute_traffic_penalty_target(X, meta_future)

    # Earlier laps (0 to 13) MUST remain bit-for-bit identical
    assert np.allclose(target_orig.iloc[:14], target_future.iloc[:14], atol=1e-12)


def test_categorical_handling(synthetic_traffic_data):
    """5. Test categorical handling and prediction on unseen categories."""
    X, meta = synthetic_traffic_data
    target = compute_traffic_penalty_target(X, meta)

    model = TrafficPenaltyModel(n_estimators=10)
    model.fit(X, target)

    assert model.is_fitted_
    assert "circuit" in model.categories_map_
    assert "compound" in model.categories_map_

    X_unseen = X.iloc[:5].copy()
    X_unseen["circuit"] = "SILVERSTONE"
    preds = model.predict(X_unseen)
    assert len(preds) == 5
    assert np.all(np.isfinite(preds))


def test_deterministic_behavior(synthetic_traffic_data):
    """6. Verify identical random_state produces bit-for-bit identical predictions."""
    X, meta = synthetic_traffic_data
    target = compute_traffic_penalty_target(X, meta)

    model1 = TrafficPenaltyModel(random_state=42, n_estimators=20)
    model1.fit(X, target)
    preds1 = model1.predict(X)

    model2 = TrafficPenaltyModel(random_state=42, n_estimators=20)
    model2.fit(X, target)
    preds2 = model2.predict(X)

    assert np.allclose(preds1, preds2, atol=1e-12)


def test_fit_predict_shape_and_bounds(synthetic_traffic_data):
    """7. Ensure predictions are 1D float64 within valid penalty bounds [0.0, 5.0]."""
    X, meta = synthetic_traffic_data
    target = compute_traffic_penalty_target(X, meta)

    model = TrafficPenaltyModel(n_estimators=10)
    model.fit(X, target)
    preds = model.predict(X)

    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert preds.shape == (len(X),)
    assert preds.dtype == np.float64
    assert np.all((preds >= 0.0) & (preds <= 5.0))


def test_grouped_validation_structure(synthetic_traffic_data):
    """8. Test out-of-fold validation output structure and baseline comparisons."""
    X, meta = synthetic_traffic_data
    target = compute_traffic_penalty_target(X, meta)

    model = TrafficPenaltyModel(n_estimators=10)
    metrics = model.evaluate(X, target, groups=meta["stint_id"], n_splits=3)

    required_keys = [
        "n_samples",
        "n_groups",
        "validation_method",
        "fold_count",
        "zero_baseline",
        "heuristic_baseline",
        "ml_model",
        "improvements_vs_zero",
        "improvements_vs_heuristic",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing key '{key}' in evaluation output"

    assert "mae" in metrics["ml_model"]
    assert "rmse" in metrics["ml_model"]
    assert "r2" in metrics["ml_model"]


def test_save_and_load_roundtrip(synthetic_traffic_data):
    """9. Verify save and load persistence round-trip reproduces predictions exactly."""
    X, meta = synthetic_traffic_data
    target = compute_traffic_penalty_target(X, meta)

    model = TrafficPenaltyModel(random_state=42, n_estimators=15)
    model.fit(X, target)
    initial_preds = model.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_traffic_model.joblib"
        model.save(save_path)
        assert save_path.exists()

        loaded_model = TrafficPenaltyModel.load(save_path)
        loaded_preds = loaded_model.predict(X)

        assert np.allclose(initial_preds, loaded_preds, atol=1e-12)
        assert loaded_model.is_fitted_ is True
