"""Unit tests for ML Model 4: Wet Performance / Tyre-Compound Suitability Model.

Covers:
1. Target construction (T_actual - T_dry_physics)
2. Feature preparation and schema alignment
3. Causal wetness proxy handling and exponential decay
4. Leakage exclusion (no sectors, speed traps, lap times, driver identity)
5. Future-information exclusion (causality check)
6. Deterministic predictions (fixed random_state)
7. Fit and predict behavior
8. Grouped validation and metrics calculation
9. Compound handling across dry and wet tyre sets
10. Artifact save and load persistence round-trip
"""

import tempfile
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from backend.ml.wet_performance import (
    CATEGORICAL_COLUMNS,
    WET_PERFORMANCE_FEATURE_COLUMNS,
    WetPerformanceModel,
    compute_causal_wetness_proxy,
    compute_wet_performance_target,
    prepare_wet_performance_data,
)


@pytest.fixture
def synthetic_wet_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Provide synthetic feature, target, and metadata DataFrames covering dry, damp, and wet regimes."""
    np.random.seed(42)
    # 6 stints of 10 laps each across 2 drivers and 2 circuits
    n = 60
    circuits = ["CANADA"] * 40 + ["BAHRAIN"] * 20
    compounds = ["INTERMEDIATE"] * 10 + ["MEDIUM"] * 10 + ["HARD"] * 10 + ["INTERMEDIATE"] * 10 + ["MEDIUM"] * 10 + ["HARD"] * 10
    tyre_age = np.tile(np.arange(1, 11), 6)
    lap_number = list(range(1, 31)) + list(range(1, 31))
    stint = [1] * 10 + [2] * 10 + [3] * 10 + [1] * 10 + [2] * 10 + [3] * 10
    stint_lap = np.tile(np.arange(1, 11), 6)
    fuel_mass_kg = np.tile(np.linspace(100.0, 20.0, 30), 2)
    track_temp = np.random.uniform(20.0, 35.0, n)
    air_temp = np.random.uniform(15.0, 25.0, n)
    track_status = ["12"] * 10 + ["1"] * 50
    rainfall = [1.0] * 5 + [0.0] * 25 + [1.0] * 5 + [0.0] * 25
    # Wetness varying within stints (wet drying out)
    wetness_proxy = list(np.linspace(1.0, 0.4, 10)) + list(np.linspace(0.5, 0.1, 10)) + list(np.linspace(0.1, 0.0, 10)) + list(np.linspace(1.0, 0.4, 10)) + list(np.linspace(0.5, 0.1, 10)) + list(np.linspace(0.1, 0.0, 10))

    # Realistic delta: large in wet/inters, moderate in damp, small in dry slicks
    lap_delta = (
        np.array(wetness_proxy) * 12.0
        + np.where(np.array(compounds) == "INTERMEDIATE", 6.0, 1.5)
        + tyre_age * 0.05
        + np.random.normal(0, 0.1, n)
    )

    X = pd.DataFrame(
        {
            "circuit": circuits,
            "compound": compounds,
            "tyre_age": tyre_age,
            "lap_number": lap_number,
            "stint_lap": stint_lap,
            "fuel_mass_kg": fuel_mass_kg,
            "track_temperature": track_temp,
            "air_temperature": air_temp,
            "track_status": track_status,
            "rainfall": rainfall,
            "wetness_proxy": wetness_proxy,
        }
    )
    y = pd.Series(lap_delta, name="lap_delta_s")
    meta = pd.DataFrame(
        {
            "driver": ["VER"] * 30 + ["HAM"] * 30,
            "stint_id": [
                "CANADA_VER_S1"
            ] * 10 + [
                "CANADA_VER_S2"
            ] * 10 + [
                "BAHRAIN_VER_S3"
            ] * 10 + [
                "CANADA_HAM_S1"
            ] * 10 + [
                "CANADA_HAM_S2"
            ] * 10 + [
                "BAHRAIN_HAM_S3"
            ] * 10,
            "circuit": circuits,
            "stint": stint,
            "lap_number": lap_number,
            "T_actual": 80.0 + lap_delta,
            "T_dry_physics": [80.0] * n,
        }
    )
    return X, y, meta


def test_target_construction():
    """1. Verify target construction: lap_delta_s = max(0.0, T_actual - T_dry_reference)."""
    t_actual = np.array([90.5, 85.0, 78.2])
    t_physics = np.array([76.0, 75.5, 76.2])
    expected = np.array([14.5, 9.5, 2.0])

    target = compute_wet_performance_target(t_actual, t_physics)
    assert np.allclose(target, expected)

    # Negative delta is clipped at 0.0 (physical non-negativity)
    t_faster = np.array([74.0])
    target_clipped = compute_wet_performance_target(t_faster, np.array([75.0]))
    assert np.allclose(target_clipped, [0.0])

    # Shape mismatch error
    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_wet_performance_target(np.array([90.0]), np.array([80.0, 81.0]))


def test_dry_lap_decoupling():
    """Verify dry slicks on dry track have strictly 0.0s wet penalty, decoupling from ML-1."""
    t_act = np.array([78.0, 79.2, 92.0])
    t_ref = np.array([77.0, 77.0, 77.0])
    wetness = np.array([0.0, 0.05, 0.6])
    compounds = np.array(["HARD", "MEDIUM", "INTERMEDIATE"])

    target = compute_wet_performance_target(t_act, t_ref, wetness_proxy=wetness, compounds=compounds)
    # Laps 0 and 1 are dry slicks on dry track -> strictly 0.0s
    assert target[0] == 0.0
    assert target[1] == 0.0
    # Lap 2 is wet track / intermediate tyre -> preserves delta (15.0s)
    assert np.isclose(target[2], 15.0)


def test_feature_preparation_and_schema():
    """2. Verify feature matrix contains exact WET_PERFORMANCE_FEATURE_COLUMNS without driver."""
    raw_df = pd.DataFrame(
        {
            "GrandPrix": ["Canada", "Canada"],
            "Driver": ["VER", "VER"],
            "Compound": ["INTERMEDIATE", "MEDIUM"],
            "LapNumber": [1, 2],
            "Stint": [1, 1],
            "TyreLife": [1.0, 2.0],
            "LapTime_s": [90.0, 85.0],
            "fuel_mass_kg": [100.0, 98.5],
            "track_temperature": [22.0, 22.5],
            "air_temperature": [18.0, 18.0],
            "rainfall": [1.0, 0.0],
            "track_status": ["12", "1"],
        }
    )
    X, y, meta = prepare_wet_performance_data(laps_df=raw_df)

    assert list(X.columns) == WET_PERFORMANCE_FEATURE_COLUMNS
    assert "driver" not in X.columns
    assert len(X) == 2
    assert len(y) == 2
    assert "T_dry_reference" in meta.columns
    assert X["wetness_proxy"].iloc[0] > 0.0


def test_causal_wetness_proxy_handling():
    """3. Verify exponential decay and bounds of causal wetness proxy."""
    # Scenario: dry race
    laps = list(range(1, 11))
    rain_dry = [False] * 10
    dry_proxy = compute_causal_wetness_proxy(laps, rain_dry)
    assert np.allclose(dry_proxy, 0.0)

    # Scenario: rain on laps 1-3, drying on laps 4-10
    rain_event = [True, True, True] + [False] * 7
    proxy = compute_causal_wetness_proxy(laps, rain_event, decay=0.85)

    assert (proxy >= 0.0).all() and (proxy <= 1.0).all()
    # While raining, wetness should be high
    assert proxy.iloc[2] > 0.95
    # After rain stops, it strictly decreases monotonically
    for i in range(3, 9):
        assert proxy.iloc[i + 1] < proxy.iloc[i]


def test_leakage_exclusion():
    """4. Verify forbidden telemetry terms and driver identity are strictly excluded."""
    forbidden = ["sector", "speed", "trap", "laptime", "t_actual", "driver"]
    for col in WET_PERFORMANCE_FEATURE_COLUMNS:
        col_lower = col.lower()
        for kw in forbidden:
            assert kw not in col_lower, f"Forbidden term '{kw}' in feature '{col}'"


def test_future_information_exclusion():
    """5. Verify modifying future laps does not leak backwards into earlier wetness or targets."""
    laps = list(range(1, 21))
    rain_base = [True] * 5 + [False] * 15
    proxy_base = compute_causal_wetness_proxy(laps, rain_base)

    # Modify future rain at lap 15
    rain_future = [True] * 5 + [False] * 9 + [True] + [False] * 5
    proxy_future = compute_causal_wetness_proxy(laps, rain_future)

    # Earlier laps (1 to 14) MUST be bit-for-bit identical
    assert np.allclose(proxy_base.iloc[:14], proxy_future.iloc[:14], atol=1e-12)
    # Lap 15 onwards should differ
    assert proxy_future.iloc[14] > proxy_base.iloc[14]


def test_wetness_causality_and_future_independence():
    """Verify wetness proxy strictly depends only on prior/current lap start information."""
    # Base scenario: 30 laps with rain on laps 1-5
    base_laps = list(range(1, 31))
    base_rain = [True] * 5 + [False] * 25
    proxy_base = compute_causal_wetness_proxy(base_laps, base_rain)

    # 1. Torrential rain added in the future at laps 25-30
    future_rain = [True] * 5 + [False] * 19 + [True] * 6
    proxy_modified_future = compute_causal_wetness_proxy(base_laps, future_rain)

    # Laps 1 to 24 MUST remain bit-for-bit identical
    assert np.allclose(proxy_base.iloc[:24], proxy_modified_future.iloc[:24], atol=1e-12)

    # 2. Appending 20 additional future laps (laps 31 to 50)
    extended_laps = list(range(1, 51))
    extended_rain = base_rain + [True] * 20
    proxy_extended = compute_causal_wetness_proxy(extended_laps, extended_rain)

    # First 30 laps MUST remain bit-for-bit identical
    assert np.allclose(proxy_base, proxy_extended.iloc[:30], atol=1e-12)

    # 3. Future lap times modification does not alter target for earlier laps
    t_actual = pd.Series([90.0, 88.0, 86.0, 84.0, 82.0])
    t_phys = pd.Series([76.0, 76.0, 76.0, 76.0, 76.0])
    target_base = compute_wet_performance_target(t_actual, t_phys)

    t_actual_mod = t_actual.copy()
    t_actual_mod.iloc[4] = 120.0  # Massive change on lap 5
    target_mod = compute_wet_performance_target(t_actual_mod, t_phys)

    # Laps 1 to 4 MUST be identical
    assert np.allclose(target_base[:4], target_mod[:4], atol=1e-12)
    assert target_mod[4] != target_base[4]


def test_deterministic_predictions(synthetic_wet_data):
    """6. Verify model training is strictly deterministic given fixed random_state."""
    X, y, _ = synthetic_wet_data

    m1 = WetPerformanceModel(random_state=42).fit(X, y)
    m2 = WetPerformanceModel(random_state=42).fit(X, y)

    p1 = m1.predict_lap_delta(X)
    p2 = m2.predict_lap_delta(X)

    assert np.allclose(p1, p2, atol=1e-12)


def test_fit_and_predict(synthetic_wet_data):
    """7. Verify fit, predict shape, and dictionary input support."""
    X, y, _ = synthetic_wet_data
    model = WetPerformanceModel(random_state=42)

    # Unfitted error
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict_lap_delta(X)

    model.fit(X, y)
    preds = model.predict_lap_delta(X)
    assert len(preds) == len(X)
    assert preds.dtype == np.float64

    # Dict input support
    single_input = {
        "circuit": "CANADA",
        "compound": "INTERMEDIATE",
        "tyre_age": 5.0,
        "lap_number": 10,
        "stint_lap": 5,
        "fuel_mass_kg": 80.0,
        "track_temperature": 22.0,
        "air_temperature": 18.0,
        "track_status": "1",
        "rainfall": 1.0,
        "wetness_proxy": 0.8,
    }
    single_pred = model.predict_lap_delta(single_input)
    assert single_pred.shape == (1,)
    assert single_pred[0] > 0.0


def test_grouped_validation(synthetic_wet_data):
    """8. Verify evaluate method computes grouped metrics across compounds and regimes."""
    X, y, meta = synthetic_wet_data
    model = WetPerformanceModel(random_state=42)

    metrics = model.evaluate(X, y, groups=meta["stint_id"])

    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "baseline_mae" in metrics
    assert "by_compound" in metrics
    assert "by_regime" in metrics
    assert metrics["mae"] < metrics["baseline_mae"]


def test_compound_handling(synthetic_wet_data):
    """9. Verify all 5 valid compounds are accepted and unknown categories gracefully handled."""
    X, y, _ = synthetic_wet_data
    model = WetPerformanceModel(random_state=42).fit(X, y)

    for comp in ["HARD", "INTERMEDIATE", "MEDIUM", "SOFT", "WET"]:
        row = {
            "circuit": "CANADA",
            "compound": comp,
            "tyre_age": 2.0,
            "lap_number": 5,
            "stint_lap": 2,
            "fuel_mass_kg": 90.0,
            "track_temperature": 25.0,
            "air_temperature": 20.0,
            "track_status": "1",
            "rainfall": 0.0,
            "wetness_proxy": 0.1,
        }
        pred = model.predict_lap_delta(row)
        assert len(pred) == 1
        assert np.isfinite(pred[0])


def test_save_load_roundtrip(synthetic_wet_data):
    """10. Verify artifact persistence round-trip reproduces predictions bit-for-bit."""
    X, y, _ = synthetic_wet_data
    model = WetPerformanceModel(random_state=42).fit(X, y)
    orig_preds = model.predict_lap_delta(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        art_path = Path(tmpdir) / "wet_perf_test.joblib"
        model.save(art_path)

        loaded = WetPerformanceModel.load(art_path)
        loaded_preds = loaded.predict_lap_delta(X)

        assert np.allclose(orig_preds, loaded_preds, atol=1e-12)
