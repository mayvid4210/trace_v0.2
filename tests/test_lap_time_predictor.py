"""Unit tests for the LapTimePredictor empirical interface."""

import numpy as np
import pytest

from backend.physics.lap_time_predictor import (
    FUEL_EFFECT_S_PER_KG,
    VALIDATED_TYRE_RATES_S_PER_LAP,
    CircuitCalibration,
    LapTimePredictor,
)


@pytest.fixture
def predictor():
    """Predictor calibrated with realistic baseline constants for Bahrain and Canada."""
    calibrations = {
        ("Bahrain", "SOFT"): CircuitCalibration(
            dry_baseline_lap_time_s=92.50,
            reference_mass_kg=850.0,
        ),
        ("Bahrain", "MEDIUM"): CircuitCalibration(
            dry_baseline_lap_time_s=93.10,
            reference_mass_kg=850.0,
        ),
        ("Bahrain", "__default__"): CircuitCalibration(
            dry_baseline_lap_time_s=93.00,
            reference_mass_kg=850.0,
        ),
        ("Canada", "__default__"): CircuitCalibration(
            dry_baseline_lap_time_s=75.00,
            reference_mass_kg=850.0,
        ),
    }
    return LapTimePredictor(calibrations)


def test_predict_lap_time_reference_condition(predictor):
    """At reference mass (850kg -> 52kg fuel + 798kg base) and tyre age 1, time is baseline."""
    # 798kg car + 52kg fuel = 850kg total mass
    pred = predictor.predict_lap_time(
        circuit="Bahrain",
        compound="SOFT",
        tyre_age=1,
        fuel_mass_kg=52.0,
        sector_wetness=None,
    )
    assert np.isclose(pred, 92.50)


def test_predict_lap_time_tyre_degradation(predictor):
    """Verify tyre degradation adds expected delta per lap of tyre life."""
    base = predictor.predict_lap_time(
        circuit="Bahrain",
        compound="SOFT",
        tyre_age=1,
        fuel_mass_kg=52.0,
    )
    after_10_laps = predictor.predict_lap_time(
        circuit="Bahrain",
        compound="SOFT",
        tyre_age=11,
        fuel_mass_kg=52.0,
    )
    soft_rate = VALIDATED_TYRE_RATES_S_PER_LAP["SOFT"]
    expected_delta = soft_rate * 10.0  # (11 - 1) laps of age

    assert np.isclose(after_10_laps - base, expected_delta)


def test_predict_lap_time_fuel_burn_effect(predictor):
    """Verify heavier fuel mass increases lap time by circuit slope."""
    fuel_rate = FUEL_EFFECT_S_PER_KG["Bahrain"]  # 0.0392 s/kg validated

    # 100 kg fuel vs 50 kg fuel (delta 50 kg)
    heavy = predictor.predict_lap_time(
        circuit="Bahrain",
        compound="SOFT",
        tyre_age=1,
        fuel_mass_kg=100.0,
    )
    light = predictor.predict_lap_time(
        circuit="Bahrain",
        compound="SOFT",
        tyre_age=1,
        fuel_mass_kg=50.0,
    )

    assert np.isclose(heavy - light, 50.0 * fuel_rate)
    assert np.isclose(heavy - light, 50.0 * 0.0392)


def test_predict_lap_time_wetness_penalty(predictor):
    """Verify sector wetness proxies directly penalize lap time additively."""
    dry = predictor.predict_lap_time(
        circuit="Bahrain",
        compound="SOFT",
        tyre_age=1,
        fuel_mass_kg=52.0,
        sector_wetness=None,
    )
    wet = predictor.predict_lap_time(
        circuit="Bahrain",
        compound="SOFT",
        tyre_age=1,
        fuel_mass_kg=52.0,
        sector_wetness=[1.2, 0.8, 1.5],  # 3.5s total wetness penalty
    )

    assert np.isclose(wet - dry, 3.5)


def test_predict_lap_time_unknown_circuit_raises(predictor):
    """Verify unknown circuit raises KeyError."""
    with pytest.raises(KeyError, match="No calibration available for circuit Monaco"):
        predictor.predict_lap_time(
            circuit="Monaco",
            compound="SOFT",
            tyre_age=1,
            fuel_mass_kg=50.0,
        )


def test_predictor_calibrated_fuel_effect_values(predictor):
    """Verify predictor runtime fuel effect wiring and nominal burn rates."""
    from backend.physics.fuel_model import get_circuit_fuel_calibration

    # a. Bahrain beta fuel = 0.0392
    assert np.isclose(LapTimePredictor.get_fuel_rate("Bahrain"), 0.0392)
    assert np.isclose(FUEL_EFFECT_S_PER_KG["Bahrain"], 0.0392)

    # b. Canada beta fuel = 0.0261
    assert np.isclose(LapTimePredictor.get_fuel_rate("Canada"), 0.0261)
    assert np.isclose(FUEL_EFFECT_S_PER_KG["Canada"], 0.0261)

    # c. Nominal burn rates remain 100/57 (1.7544 kg/lap) and 100/70 (1.4286 kg/lap)
    bah_calib = get_circuit_fuel_calibration("Bahrain")
    can_calib = get_circuit_fuel_calibration("Canada")
    assert np.isclose(bah_calib.nominal_burn_rate_kg_per_lap, 100.0 / 57.0)
    assert np.isclose(can_calib.nominal_burn_rate_kg_per_lap, 100.0 / 70.0)

    # d. Existing predictor API and delta calculation remain intact
    t_heavy = predictor.predict_lap_time(circuit="Bahrain", compound="SOFT", tyre_age=1, fuel_mass_kg=70.0)
    t_light = predictor.predict_lap_time(circuit="Bahrain", compound="SOFT", tyre_age=1, fuel_mass_kg=60.0)
    assert np.isclose(t_heavy - t_light, 10.0 * 0.0392)

