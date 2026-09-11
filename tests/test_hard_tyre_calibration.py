"""Targeted unit tests for Hard compound tyre degradation recalibration and validation."""

import numpy as np
import pandas as pd
import pytest

from backend.physics.lap_time_predictor import (
    CIRCUIT_TYRE_RATES_S_PER_LAP,
    VALIDATED_TYRE_RATES_S_PER_LAP,
    CircuitCalibration,
    LapTimePredictor,
)
from backend.physics.tyre import (
    CIRCUIT_TYRE_CALIBRATIONS,
    PRODUCTION_HARD_DEGRADATION_S_PER_LAP,
    CircuitTyreDegradation,
    calibrate_circuit_hard_degradation,
)


@pytest.fixture
def synthetic_race_data():
    """Synthetic race dataset containing Bahrain and Canada laps with known degradation."""
    laps = []
    # Bahrain clean race laps for Hard (rate = 0.1066 s/lap)
    for driver_idx in range(5):
        driver = f"DRV_{driver_idx}"
        for tyre_life in range(2, 25):
            lap_num = tyre_life + 10
            # Fuel burn effect = 0.0392 * fuel_kg, tyre deg = 0.1066 * (tyre_life - 1)
            fuel_kg = 100.0 - (lap_num - 1) * (100.0 / 57.0)
            base_time = 94.0 + 0.1066 * (tyre_life - 1) + 0.0392 * fuel_kg
            laps.append({
                "GrandPrix": "Bahrain",
                "SessionName": "Race",
                "Driver": driver,
                "Stint": 2,
                "LapNumber": lap_num,
                "Compound": "HARD",
                "TyreLife": float(tyre_life),
                "LapTime_s": base_time,
                "TrackStatus": "1",
                "PitInTime": np.nan,
                "PitOutTime": np.nan,
            })

    # Canada clean race laps for Hard
    for driver_idx in range(3):
        driver = f"DRV_{driver_idx}"
        for tyre_life in range(2, 15):
            lap_num = tyre_life + 50
            laps.append({
                "GrandPrix": "Canada",
                "SessionName": "Race",
                "Driver": driver,
                "Stint": 3,
                "LapNumber": lap_num,
                "Compound": "HARD",
                "TyreLife": float(tyre_life),
                "LapTime_s": 76.0 + 0.04 * tyre_life,
                "TrackStatus": "1",
                "PitInTime": np.nan,
                "PitOutTime": np.nan,
            })

    return pd.DataFrame(laps)


def test_production_hard_rate_value():
    """Verify production Hard degradation value remains 0.1060 s/lap."""
    assert np.isclose(PRODUCTION_HARD_DEGRADATION_S_PER_LAP, 0.1060)
    assert np.isclose(VALIDATED_TYRE_RATES_S_PER_LAP["HARD"], 0.1060)


def test_bahrain_hard_calibration_metadata():
    """Verify pre-computed Bahrain Hard calibration reflects empirical findings."""
    calib = CIRCUIT_TYRE_CALIBRATIONS.get(("Bahrain", "HARD"))
    assert calib is not None
    assert calib.circuit == "Bahrain"
    assert calib.compound == "HARD"
    assert calib.status == "STATISTICALLY_VALIDATED"
    assert np.isclose(calib.degradation_s_per_lap, 0.1060)
    assert calib.uncertainty_s_per_lap is not None
    assert np.isclose(calib.uncertainty_s_per_lap, 0.0042)
    assert calib.sample_count >= 700
    assert calib.n_stints >= 30
    assert calib.n_drivers >= 15


def test_canada_hard_calibration_status():
    """Verify Canada Hard is marked INSUFFICIENTLY_IDENTIFIABLE with production fallback."""
    calib = CIRCUIT_TYRE_CALIBRATIONS.get(("Canada", "HARD"))
    assert calib is not None
    assert calib.circuit == "Canada"
    assert calib.compound == "HARD"
    assert calib.status == "INSUFFICIENTLY_IDENTIFIABLE"
    assert np.isclose(calib.degradation_s_per_lap, 0.1060)
    assert calib.uncertainty_s_per_lap is None
    assert "drying" in calib.notes.lower()


def test_calibrate_circuit_hard_degradation_race_only(synthetic_race_data):
    """Verify calibration ignores non-Race sessions (FP1, FP2, FP3)."""
    # Add dirty practice laps
    practice_laps = synthetic_race_data.copy()
    practice_laps["SessionName"] = "FP1"
    practice_laps["LapTime_s"] += 5.0  # Slow practice runs

    combined = pd.concat([synthetic_race_data, practice_laps], ignore_index=True)
    res = calibrate_circuit_hard_degradation(combined, circuit="Bahrain")

    assert res.status == "STATISTICALLY_VALIDATED"
    # Should only calibrate using the Race rows
    assert res.sample_count == len(synthetic_race_data[synthetic_race_data["GrandPrix"] == "Bahrain"])


def test_calibrate_circuit_hard_degradation_clean_laps_only(synthetic_race_data):
    """Verify pit-in, pit-out, and yellow-flag laps are excluded from calibration."""
    dirty_data = synthetic_race_data.copy()
    # Contaminate one lap with PitInTime
    dirty_data.loc[0, "PitInTime"] = 5000.0
    # Contaminate another with TrackStatus yellow ('2')
    dirty_data.loc[1, "TrackStatus"] = "2"

    res = calibrate_circuit_hard_degradation(dirty_data, circuit="Bahrain")
    assert res.sample_count == len(synthetic_race_data[synthetic_race_data["GrandPrix"] == "Bahrain"]) - 2


def test_calibrate_bahrain_hard_estimated_rate(synthetic_race_data):
    """Verify empirical fit recovers ~0.1066 s/lap and retains 0.1060 production rate."""
    res = calibrate_circuit_hard_degradation(synthetic_race_data, circuit="Bahrain")
    assert res.status == "STATISTICALLY_VALIDATED"
    assert np.isclose(res.degradation_s_per_lap, 0.1060)
    assert res.uncertainty_s_per_lap is not None
    assert res.uncertainty_s_per_lap < 0.010  # Low standard error on clean dataset


def test_calibrate_canada_hard_identifiability(synthetic_race_data):
    """Verify Canada Hard is flagged as INSUFFICIENTLY_IDENTIFIABLE even with input data."""
    res = calibrate_circuit_hard_degradation(synthetic_race_data, circuit="Canada")
    assert res.status == "INSUFFICIENTLY_IDENTIFIABLE"
    assert np.isclose(res.degradation_s_per_lap, 0.1060)
    assert res.uncertainty_s_per_lap is None


def test_no_nan_or_inf_in_tyre_calibrations():
    """Verify all degradation parameters are finite and non-negative."""
    for (circuit, compound), calib in CIRCUIT_TYRE_CALIBRATIONS.items():
        assert not np.isnan(calib.degradation_s_per_lap)
        assert not np.isinf(calib.degradation_s_per_lap)
        assert calib.degradation_s_per_lap > 0.0

    for circuit, rates in CIRCUIT_TYRE_RATES_S_PER_LAP.items():
        for comp, rate in rates.items():
            assert not np.isnan(rate)
            assert not np.isinf(rate)
            assert rate > 0.0


def test_monotonic_tyre_age_effect():
    """Verify tyre degradation strictly increases lap time monotonically with tyre age."""
    calibrations = {
        ("Bahrain", "HARD"): CircuitCalibration(
            dry_baseline_lap_time_s=94.00,
            reference_mass_kg=850.0,
        ),
    }
    predictor = LapTimePredictor(calibrations)

    times = [
        predictor.predict_lap_time(circuit="Bahrain", compound="HARD", tyre_age=age, fuel_mass_kg=50.0)
        for age in range(1, 35)
    ]

    # Monotonically strictly increasing
    for i in range(len(times) - 1):
        assert times[i + 1] > times[i]
        assert np.isclose(times[i + 1] - times[i], 0.1060)


def test_predictor_get_tyre_rate_resolution():
    """Verify LapTimePredictor.get_tyre_rate follows resolution hierarchy."""
    # 1. Circuit + compound calibration
    assert np.isclose(LapTimePredictor.get_tyre_rate("Bahrain", "HARD"), 0.1060)
    assert np.isclose(LapTimePredictor.get_tyre_rate("Bahrain", "SOFT"), 0.0627)

    # 2. Global compound fallback
    assert np.isclose(LapTimePredictor.get_tyre_rate("UnknownCircuit", "HARD"), 0.1060)
    assert np.isclose(LapTimePredictor.get_tyre_rate("UnknownCircuit", "MEDIUM"), 0.0447)

    # 3. Explicit documented fallback for uncalibrated compound
    assert np.isclose(LapTimePredictor.get_tyre_rate("UnknownCircuit", "UNKNOWN"), 0.05)


def test_deterministic_output_without_side_effects(synthetic_race_data):
    """Verify multiple calibration runs produce bitwise identical records."""
    res1 = calibrate_circuit_hard_degradation(synthetic_race_data, circuit="Bahrain")
    res2 = calibrate_circuit_hard_degradation(synthetic_race_data, circuit="Bahrain")

    assert res1.degradation_s_per_lap == res2.degradation_s_per_lap
    assert res1.uncertainty_s_per_lap == res2.uncertainty_s_per_lap
    assert res1.sample_count == res2.sample_count
    assert res1.status == res2.status
