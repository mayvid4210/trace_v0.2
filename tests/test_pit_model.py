"""Targeted unit tests for circuit-specific pit loss calibration and simulator integration."""

import numpy as np
import pytest

from backend.models.race_state import CarState, RaceState, TyreCompound
from backend.physics.lap_time_predictor import CircuitCalibration, LapTimePredictor
from backend.physics.pit_model import (
    CIRCUIT_PIT_LOSS_CALIBRATIONS,
    DEFAULT_PIT_LOSS_S,
    CircuitPitLossCalibration,
    get_circuit_pit_calibration,
    get_circuit_pit_loss,
)
from backend.simulation.simulator import Simulator


@pytest.fixture
def predictor():
    """Predictor calibrated with realistic baseline constants."""
    calibrations = {
        ("Bahrain", "MEDIUM"): CircuitCalibration(
            dry_baseline_lap_time_s=93.00,
            reference_mass_kg=850.0,
        ),
        ("Bahrain", "HARD"): CircuitCalibration(
            dry_baseline_lap_time_s=94.00,
            reference_mass_kg=850.0,
        ),
        ("Canada", "MEDIUM"): CircuitCalibration(
            dry_baseline_lap_time_s=75.00,
            reference_mass_kg=850.0,
        ),
        ("Canada", "HARD"): CircuitCalibration(
            dry_baseline_lap_time_s=76.00,
            reference_mass_kg=850.0,
        ),
        ("Silverstone", "MEDIUM"): CircuitCalibration(
            dry_baseline_lap_time_s=90.00,
            reference_mass_kg=850.0,
        ),
        ("Silverstone", "HARD"): CircuitCalibration(
            dry_baseline_lap_time_s=91.00,
            reference_mass_kg=850.0,
        ),
    }
    return LapTimePredictor(calibrations)


def test_pit_loss_bahrain_calibration():
    """Verify Bahrain pit loss is calibrated to 24.1s with empirical metadata."""
    loss = get_circuit_pit_loss("Bahrain")
    assert np.isclose(loss, 24.1)

    calib = get_circuit_pit_calibration("Bahrain")
    assert calib is not None
    assert calib.circuit == "Bahrain"
    assert calib.sample_count == 34
    assert np.isclose(calib.pit_loss_s, 24.1)
    assert np.isclose(calib.mean_s, 24.370)
    assert np.isclose(calib.std_s, 1.120)
    assert np.isclose(calib.iqr_s, 1.590)
    assert calib.source == "STATISTICALLY_CALIBRATED_FROM_RACE_DATA"


def test_pit_loss_canada_calibration():
    """Verify Canada pit loss is calibrated to 24.7s with empirical metadata."""
    loss = get_circuit_pit_loss("Canada")
    assert np.isclose(loss, 24.7)

    calib = get_circuit_pit_calibration("Canada")
    assert calib is not None
    assert calib.circuit == "Canada"
    assert calib.sample_count == 15
    assert np.isclose(calib.pit_loss_s, 24.7)
    assert np.isclose(calib.mean_s, 24.961)
    assert np.isclose(calib.std_s, 1.023)
    assert np.isclose(calib.iqr_s, 1.837)
    assert calib.source == "STATISTICALLY_CALIBRATED_FROM_RACE_DATA"


def test_pit_loss_uncalibrated_fallback():
    """Verify uncalibrated circuits and None fall back to DEFAULT_PIT_LOSS_S (22.0s)."""
    assert np.isclose(DEFAULT_PIT_LOSS_S, 22.0)
    assert np.isclose(get_circuit_pit_loss("Silverstone"), 22.0)
    assert np.isclose(get_circuit_pit_loss("Monza"), 22.0)
    assert np.isclose(get_circuit_pit_loss(None), 22.0)
    assert np.isclose(get_circuit_pit_loss(""), 22.0)


def test_pit_loss_deterministic_lookup():
    """Verify repeated lookups and case variations produce identical deterministic results."""
    res1 = get_circuit_pit_loss("Bahrain")
    res2 = get_circuit_pit_loss("bahrain")
    res3 = get_circuit_pit_loss("BAHRAIN")
    assert res1 == res2 == res3 == 24.1

    c1 = get_circuit_pit_loss("Canada")
    c2 = get_circuit_pit_loss("canada")
    assert c1 == c2 == 24.7


def test_simulator_uses_circuit_calibration_by_default(predictor):
    """Verify Simulator dynamically applies circuit pit loss when pit_loss_s is None."""
    # 1. Bahrain simulation
    car_bah = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    car_stay_bah = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    race_bah = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car_bah, "NOR": car_stay_bah})

    sim_bah = Simulator(predictor=predictor, race_state=race_bah, pit_stops={"VER": {2: "HARD"}})
    res_bah = sim_bah.run()

    # Lap 2 was pit lap for VER, NOR stayed out
    ver_lap2 = next(l for l in next(d for d in res_bah.classification if d.driver == "VER").laps if l.lap_number == 2)
    nor_lap2 = next(l for l in next(d for d in res_bah.classification if d.driver == "NOR").laps if l.lap_number == 2)
    assert np.isclose(ver_lap2.lap_time_s - nor_lap2.lap_time_s, 24.1)

    # 2. Canada simulation
    car_can = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    car_stay_can = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    race_can = RaceState(circuit="Canada", total_laps=3, current_lap=1, cars={"VER": car_can, "NOR": car_stay_can})

    sim_can = Simulator(predictor=predictor, race_state=race_can, pit_stops={"VER": {2: "HARD"}})
    res_can = sim_can.run()

    ver_can_lap2 = next(l for l in next(d for d in res_can.classification if d.driver == "VER").laps if l.lap_number == 2)
    nor_can_lap2 = next(l for l in next(d for d in res_can.classification if d.driver == "NOR").laps if l.lap_number == 2)
    assert np.isclose(ver_can_lap2.lap_time_s - nor_can_lap2.lap_time_s, 24.7)


def test_simulator_explicit_pit_loss_override(predictor):
    """Verify explicitly supplied pit_loss_s overrides circuit calibration."""
    car1 = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    car2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    race = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car1, "NOR": car2})

    custom_loss = 28.5
    sim = Simulator(predictor=predictor, race_state=race, pit_loss_s=custom_loss, pit_stops={"VER": {2: "HARD"}})
    res = sim.run()

    ver_lap2 = next(l for l in next(d for d in res.classification if d.driver == "VER").laps if l.lap_number == 2)
    nor_lap2 = next(l for l in next(d for d in res.classification if d.driver == "NOR").laps if l.lap_number == 2)
    assert np.isclose(ver_lap2.lap_time_s - nor_lap2.lap_time_s, custom_loss)


def test_pit_loss_changes_cumulative_race_time_correctly(predictor):
    """Verify total cumulative race time increases exactly by pit loss on pit lap."""
    # Simulation without pit stop
    car_nopit = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    race_nopit = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car_nopit})
    sim_nopit = Simulator(predictor=predictor, race_state=race_nopit)
    res_nopit = sim_nopit.run()
    time_nopit = res_nopit.classification[0].total_time_s

    # Simulation with pit stop (compound stays MEDIUM for direct comparison)
    car_pit = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    race_pit = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car_pit})
    sim_pit = Simulator(predictor=predictor, race_state=race_pit, pit_stops={"VER": {2: "MEDIUM"}})
    res_pit = sim_pit.run()
    time_pit = res_pit.classification[0].total_time_s

    # Pitting on lap 2 incurs: 24.1s pit loss - tyre degradation benefit on lap 3 (age 1 vs age 3)
    # Lap 2 pit loss added directly:
    assert (time_pit - time_nopit) >= 23.5


def test_pit_loss_stage3_ordering_and_gaps_preserved(predictor):
    """Verify Stage 3 ordering and gap to leader emerge correctly after circuit-specific pit loss."""
    car1 = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    car2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=100.0)
    race = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car1, "NOR": car2})

    # VER pits on lap 1, losing 24.1s, dropping behind NOR
    sim = Simulator(predictor=predictor, race_state=race, pit_stops={"VER": {1: "HARD"}})
    result = sim.run()

    assert result.classification[0].driver == "NOR"
    assert result.classification[0].finish_position == 1
    assert result.classification[1].driver == "VER"
    assert result.classification[1].finish_position == 2
    assert result.classification[1].total_time_s > result.classification[0].total_time_s

    # Leader gap
    gap = result.classification[1].total_time_s - result.classification[0].total_time_s
    assert np.isclose(gap, sim.race_state.cars["VER"].gap_to_leader_s)
    assert gap >= 24.0
