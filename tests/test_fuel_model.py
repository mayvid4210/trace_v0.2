"""Comprehensive unit tests for the telemetry-derived fuel model, circuit calibration,

multivariate fuel sensitivity, and simulator integration.
"""

import numpy as np
import pandas as pd
import pytest

from backend.models.race_state import (
    CarState,
    RaceState,
    TrackCondition,
    TrackStatus,
    TyreCompound,
)
from backend.physics.fuel_model import (
    CIRCUIT_FUEL_CALIBRATIONS,
    CircuitFuelCalibration,
    compute_lap_workload,
    compute_relative_fuel_intensity,
    estimate_lap_burn,
    get_circuit_fuel_calibration,
    update_fuel_state,
)
from backend.physics.fuel_effect import fit_multivariate_fuel_sensitivity
from backend.physics.lap_time_predictor import (
    CircuitCalibration,
    LapTimePredictor,
)
from backend.simulation.simulator import Simulator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_lap_telemetry():
    """Synthetic clean telemetry frames for a single lap."""
    n_samples = 100
    times = pd.to_timedelta(np.linspace(0, 90, n_samples), unit="s")
    return pd.DataFrame({
        "Time": times,
        "Throttle": np.full(n_samples, 75.0),
        "RPM": np.full(n_samples, 11000.0),
        "Speed": np.full(n_samples, 220.0),
        "Distance": np.linspace(0, 5400, n_samples),
        "Brake": np.zeros(n_samples),
    })


@pytest.fixture
def predictor():
    """Predictor calibrated with Bahrain baselines."""
    calibrations = {
        ("Bahrain", "SOFT"): CircuitCalibration(
            dry_baseline_lap_time_s=92.50,
            reference_mass_kg=850.0,
        ),
        ("Bahrain", "MEDIUM"): CircuitCalibration(
            dry_baseline_lap_time_s=93.10,
            reference_mass_kg=850.0,
        ),
        ("Bahrain", "HARD"): CircuitCalibration(
            dry_baseline_lap_time_s=94.00,
            reference_mass_kg=850.0,
        ),
        ("Bahrain", "__default__"): CircuitCalibration(
            dry_baseline_lap_time_s=93.00,
            reference_mass_kg=850.0,
        ),
    }
    return LapTimePredictor(calibrations)


# ---------------------------------------------------------------------------
# 1. Telemetry Workload & Proxy Tests
# ---------------------------------------------------------------------------

def test_compute_lap_workload_accepts_valid_telemetry(sample_lap_telemetry):
    """Verify compute_lap_workload accepts telemetry and produces positive float."""
    workload = compute_lap_workload(sample_lap_telemetry)
    assert workload is not None
    assert isinstance(workload, float)
    assert workload > 0.0


def test_compute_lap_workload_handles_missing_optional_channels(sample_lap_telemetry):
    """Verify compute_lap_workload handles missing RPM by using throttle-only integration."""
    no_rpm = sample_lap_telemetry.drop(columns=["RPM"])
    workload = compute_lap_workload(no_rpm)
    assert workload is not None
    assert workload > 0.0


def test_compute_lap_workload_handles_missing_required_channels():
    """Verify compute_lap_workload returns None when required channels or data are missing."""
    assert compute_lap_workload(None) is None
    assert compute_lap_workload(pd.DataFrame()) is None

    # Missing Throttle
    no_throttle = pd.DataFrame({"Time": [0.1, 0.2, 0.3], "RPM": [10000, 10000, 10000]})
    assert compute_lap_workload(no_throttle) is None

    # Missing Time
    no_time = pd.DataFrame({"Throttle": [100, 100, 100], "RPM": [10000, 10000, 10000]})
    assert compute_lap_workload(no_time) is None


def test_relative_fuel_intensity_deterministic(sample_lap_telemetry):
    """Verify compute_relative_fuel_intensity is deterministic."""
    w1 = compute_lap_workload(sample_lap_telemetry)
    w2 = compute_lap_workload(sample_lap_telemetry)

    i1 = compute_relative_fuel_intensity(w1, reference_work=56.75, track_status="1")
    i2 = compute_relative_fuel_intensity(w2, reference_work=56.75, track_status="1")
    assert np.isclose(i1, i2)


def test_relative_fuel_intensity_no_future_leakage():
    """Verify relative fuel intensity uses only current lap work and reference work."""
    # Lap 1 work = 50.0
    int_1 = compute_relative_fuel_intensity(50.0, reference_work=50.0, track_status="1")
    # Even if subsequent laps vary, Lap 1 intensity is invariant
    assert np.isclose(int_1, 1.0)


def test_green_vs_sc_vsc_intensity_behavior():
    """Verify SC/VSC track status results in lower fuel intensity."""
    # Missing telemetry under Green vs SC
    green_int = compute_relative_fuel_intensity(None, reference_work=50.0, track_status="1")
    sc_int = compute_relative_fuel_intensity(None, reference_work=50.0, track_status="4")
    vsc_int = compute_relative_fuel_intensity(None, reference_work=50.0, track_status="6")

    assert np.isclose(green_int, 1.0)
    assert sc_int < green_int
    assert vsc_int < green_int
    assert np.isclose(sc_int, 0.7242)
    assert np.isclose(vsc_int, 0.7242)


# ---------------------------------------------------------------------------
# 2. Fuel Depletion & State Tests
# ---------------------------------------------------------------------------

def test_fuel_monotonic_depletion():
    """Verify update_fuel_state decreases strictly monotonically for positive burns."""
    fuel = 100.0
    burns = [1.8, 1.7, 1.9, 1.8]
    trajectory = []
    for b in burns:
        fuel = update_fuel_state(fuel, b)
        trajectory.append(fuel)

    for i in range(len(trajectory) - 1):
        assert trajectory[i] > trajectory[i + 1]


def test_fuel_never_negative():
    """Verify update_fuel_state clamps at 0.0 and never returns negative fuel."""
    fuel = 1.0
    excessive_burn = 5.0
    new_fuel = update_fuel_state(fuel, excessive_burn)
    assert new_fuel == 0.0


def test_varying_intensity_changes_consumption():
    """Verify higher workload intensity burns more fuel than lower intensity."""
    calib = CIRCUIT_FUEL_CALIBRATIONS["Bahrain"]
    burn_high = estimate_lap_burn(calib, intensity=1.15)
    burn_low = estimate_lap_burn(calib, intensity=0.85)

    assert burn_high > burn_low


def test_no_synthetic_110_over_laps_in_simulation(predictor):
    """Verify Simulator uses calibrated nominal burn, not synthetic 110.0 / total_laps."""
    car = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    race = RaceState(circuit="Bahrain", total_laps=57, current_lap=1, cars={"VER": car})

    sim = Simulator(predictor=predictor, race_state=race)
    sim.step_lap()

    burned = 100.0 - race.cars["VER"].fuel_kg
    calib = CIRCUIT_FUEL_CALIBRATIONS["Bahrain"]
    expected_burn = calib.nominal_fuel_scale_kg / 57.0  # 100.0 / 57 approx 1.754 kg

    assert np.isclose(burned, expected_burn)
    # Proves it is NOT using 110.0 / 57 (approx 1.930 kg)
    assert not np.isclose(burned, 110.0 / 57.0)


# ---------------------------------------------------------------------------
# 3. Circuit Calibration Tests
# ---------------------------------------------------------------------------

def test_bahrain_and_canada_calibration_independence():
    """Verify Bahrain and Canada calibrations have independent, circuit-specific parameters."""
    bahrain = get_circuit_fuel_calibration("Bahrain")
    canada = get_circuit_fuel_calibration("Canada")

    assert bahrain.circuit == "Bahrain"
    assert canada.circuit == "Canada"
    assert bahrain.total_laps != canada.total_laps
    assert bahrain.track_length_km != canada.track_length_km
    assert bahrain.reference_green_work != canada.reference_green_work
    assert bahrain.nominal_burn_rate_kg_per_lap != canada.nominal_burn_rate_kg_per_lap


def test_uncalibrated_circuit_insufficient_data_behavior():
    """Verify uncalibrated circuits fail clearly unless explicit parameters are supplied."""
    # Unknown circuit without explicit params returns None
    assert get_circuit_fuel_calibration("Silverstone") is None

    # Unknown circuit with explicit total_laps and scale returns configured calibration
    custom = get_circuit_fuel_calibration("Silverstone", total_laps=52, external_fuel_scale_kg=102.0)
    assert custom is not None
    assert custom.circuit == "Silverstone"
    assert custom.total_laps == 52
    assert custom.nominal_fuel_scale_kg == 102.0


def test_no_nan_or_inf_in_calibrations():
    """Verify all calibrated values in registry are valid finite numbers."""
    for name, calib in CIRCUIT_FUEL_CALIBRATIONS.items():
        assert np.isfinite(calib.total_laps)
        assert np.isfinite(calib.track_length_km)
        assert np.isfinite(calib.reference_green_work)
        assert np.isfinite(calib.sc_vsc_work_ratio)
        assert np.isfinite(calib.calibrated_fuel_effect_s_per_kg)
        assert np.isfinite(calib.nominal_fuel_scale_kg)
        assert np.isfinite(calib.nominal_burn_rate_kg_per_lap)


# ---------------------------------------------------------------------------
# 4. Fuel Sensitivity Tests
# ---------------------------------------------------------------------------

def test_fuel_sensitivity_regression_does_not_fit_synthetic_mass():
    """Verify fit_multivariate_fuel_sensitivity controls for Compound and TyreLife."""
    # Create synthetic multi-stint race dataset
    rows = []
    for lap in range(1, 58):
        # 2-stop: stint 1 (1-18 SOFT), stint 2 (19-38 HARD), stint 3 (39-57 SOFT)
        if lap <= 18:
            comp = "SOFT"
            tyre_life = lap
        elif lap <= 38:
            comp = "HARD"
            tyre_life = lap - 18
        else:
            comp = "SOFT"
            tyre_life = lap - 38

        # True underlying physics: 90.0 baseline + 0.10*tyre_life + 0.040*fuel_remaining
        fuel_remaining = 100.0 - 1.75 * (lap - 1)
        comp_penalty = 0.5 if comp == "HARD" else 0.0
        lap_time = 90.0 + comp_penalty + 0.10 * tyre_life + 0.040 * fuel_remaining + np.random.normal(0, 0.05)

        rows.append({
            "GrandPrix": "Bahrain",
            "SessionName": "Race",
            "LapNumber": lap,
            "Compound": comp,
            "TyreLife": tyre_life,
            "LapTime_s": lap_time,
            "PitInTime": np.nan,
            "PitOutTime": np.nan,
            "TrackStatus": "1",
        })

    df = pd.DataFrame(rows)
    result = fit_multivariate_fuel_sensitivity(df, circuit="Bahrain")

    assert result["status"] == "CALIBRATED_MULTIVARIATE"
    assert np.isclose(result["slope"], 0.040, atol=0.015)
    assert result["r2"] > 0.80


def test_canada_fuel_sensitivity_fallback_behavior():
    """Verify Canada returns documented fallback status due to drying track."""
    dummy_df = pd.DataFrame({
        "GrandPrix": ["Canada"] * 30,
        "SessionName": ["Race"] * 30,
        "LapNumber": range(1, 31),
        "Compound": ["MEDIUM"] * 30,
        "TyreLife": range(1, 31),
        "LapTime_s": np.linspace(80, 75, 30),
        "PitInTime": [np.nan] * 30,
        "PitOutTime": [np.nan] * 30,
        "TrackStatus": ["1"] * 30,
    })
    res = fit_multivariate_fuel_sensitivity(dummy_df, circuit="Canada")
    assert res["status"] == "DOCUMENTED_FALLBACK"
    assert res["slope"] == CIRCUIT_FUEL_CALIBRATIONS["Canada"].calibrated_fuel_effect_s_per_kg


# ---------------------------------------------------------------------------
# 5. Simulator Integration & Pit Stop Continuity
# ---------------------------------------------------------------------------

def test_pit_stop_preserves_fuel_continuity(predictor):
    """Verify that a pit stop does NOT reset or alter fuel continuity."""
    car = CarState(driver="VER", position=1, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=100.0)
    race = RaceState(circuit="Bahrain", total_laps=5, current_lap=1, cars={"VER": car})

    pit_schedule = {"VER": {2: TyreCompound.HARD}}
    sim = Simulator(predictor=predictor, race_state=race, pit_stops=pit_schedule)

    # Lap 1: green flag lap
    res1 = sim.step_lap()
    fuel_after_lap1 = race.cars["VER"].fuel_kg
    assert fuel_after_lap1 < 100.0

    # Lap 2: pit stop lap
    res2 = sim.step_lap()
    fuel_after_lap2 = race.cars["VER"].fuel_kg
    assert fuel_after_lap2 < fuel_after_lap1
    assert race.cars["VER"].current_compound == TyreCompound.HARD
    assert race.cars["VER"].tyre_age == 1  # reset to 1 on pit

    # Lap 3: out-lap / continuation
    res3 = sim.step_lap()
    fuel_after_lap3 = race.cars["VER"].fuel_kg
    assert fuel_after_lap3 < fuel_after_lap2


def test_multiple_pit_stops_preserve_fuel_continuity(predictor):
    """Verify 2-stop race preserves monotonic fuel progression across all 3 stints."""
    car = CarState(driver="VER", position=1, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=100.0)
    race = RaceState(circuit="Bahrain", total_laps=6, current_lap=1, cars={"VER": car})

    pit_schedule = {"VER": {2: TyreCompound.HARD, 4: TyreCompound.MEDIUM}}
    sim = Simulator(predictor=predictor, race_state=race, pit_stops=pit_schedule)

    fuel_history = [car.fuel_kg]
    for _ in range(6):
        sim.step_lap()
        fuel_history.append(race.cars["VER"].fuel_kg)

    # Monotonically strictly decreasing
    for i in range(len(fuel_history) - 1):
        assert fuel_history[i] > fuel_history[i + 1]

    # Final fuel is strictly positive
    assert fuel_history[-1] > 0.0


def test_stage3_ordering_and_gaps_preserved_with_fuel_model(predictor):
    """Verify Stage 3 cumulative race ordering and leader gap calculations remain correct."""
    car1 = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    car2 = CarState(driver="HAM", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    race = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car1, "HAM": car2})

    sim = Simulator(predictor=predictor, race_state=race)
    result = sim.run()

    # Classification exists and is ranked by total time
    assert len(result.classification) == 2
    assert result.classification[0].total_time_s <= result.classification[1].total_time_s
    assert result.classification[0].finish_position == 1
    assert result.classification[1].finish_position == 2
    assert (result.classification[1].total_time_s - result.classification[0].total_time_s) >= 0.0
    assert sim.race_state.cars[result.classification[1].driver].gap_to_leader_s >= 0.0
    assert sim.race_state.cars[result.classification[0].driver].gap_to_leader_s == 0.0
