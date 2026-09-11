"""Unit tests for simulation domain state and result data structures."""

from backend.models.race_state import (
    CarState,
    RaceState,
    TrackCondition,
    TrackStatus,
    TyreCompound,
)
from backend.models.result import (
    DriverResult,
    LapResult,
    SimulationResult,
)


def test_car_and_race_state_creation():
    """Verify CarState and RaceState can be instantiated with valid defaults."""
    car = CarState(
        driver="VER",
        team="Red Bull",
        position=1,
        current_compound=TyreCompound.MEDIUM,
        tyre_age=12,
        stint=1,
        fuel_kg=85.0,
    )

    condition = TrackCondition(
        air_temp_c=28.0,
        track_temp_c=42.0,
        rainfall=False,
        track_status=TrackStatus.CLEAR,
        track_grip=1.0,
    )

    race = RaceState(
        circuit="Bahrain",
        total_laps=57,
        current_lap=13,
        cars={"VER": car},
        track_condition=condition,
    )

    assert race.circuit == "Bahrain"
    assert race.total_laps == 57
    assert race.current_lap == 13
    assert race.cars["VER"].driver == "VER"
    assert race.cars["VER"].fuel_kg == 85.0
    assert race.track_condition.track_status == TrackStatus.CLEAR
    assert race.is_finished is False


def test_simulation_result_serialization():
    """Verify SimulationResult and LapResult serialize cleanly to dict and JSON."""
    lap = LapResult(
        lap_number=1,
        driver="NOR",
        lap_time_s=93.456,
        sector_times_s=[29.1, 38.2, 26.156],
        compound=TyreCompound.HARD,
        tyre_age=1,
        fuel_kg=108.5,
        position=2,
    )

    driver_res = DriverResult(
        driver="NOR",
        team="McLaren",
        finish_position=2,
        grid_position=3,
        total_time_s=5400.123,
        status="Finished",
        points=18,
        pit_stops=1,
        compounds_used=[TyreCompound.MEDIUM, TyreCompound.HARD],
        laps=[lap],
    )

    sim_res = SimulationResult(
        simulation_id="sim_test_001",
        circuit="Bahrain",
        total_laps=57,
        classification=[driver_res],
        lap_history=[lap],
        total_duration_s=5400.123,
    )

    data_dict = sim_res.model_dump()
    assert data_dict["simulation_id"] == "sim_test_001"
    assert data_dict["classification"][0]["driver"] == "NOR"
    assert data_dict["classification"][0]["compounds_used"] == ["MEDIUM", "HARD"]

    json_str = sim_res.model_dump_json()
    assert "sim_test_001" in json_str
    assert "NOR" in json_str
