"""Unit tests for the minimal deterministic Simulator skeleton."""

import pytest
import numpy as np

from backend.models.race_state import (
    CarState,
    RaceState,
    TrackCondition,
    TrackStatus,
    TyreCompound,
)
from backend.models.result import SimulationResult
from backend.physics.lap_time_predictor import (
    CircuitCalibration,
    LapTimePredictor,
)
from backend.simulation.simulator import Simulator


@pytest.fixture
def calibrated_predictor():
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


@pytest.fixture
def initial_race_state():
    """Two-car race state at Bahrain for 3 laps."""
    car1 = CarState(
        driver="VER",
        team="Red Bull",
        position=1,
        current_compound=TyreCompound.SOFT,
        tyre_age=1,
        fuel_kg=110.0,
    )
    car2 = CarState(
        driver="NOR",
        team="McLaren",
        position=2,
        current_compound=TyreCompound.MEDIUM,
        tyre_age=1,
        fuel_kg=110.0,
    )

    track = TrackCondition(
        air_temp_c=25.0,
        track_temp_c=35.0,
        rainfall=False,
        track_status=TrackStatus.CLEAR,
        track_grip=1.0,
    )

    return RaceState(
        circuit="Bahrain",
        total_laps=3,
        current_lap=1,
        cars={"VER": car1, "NOR": car2},
        track_condition=track,
    )


def test_simulator_instantiation(calibrated_predictor, initial_race_state):
    """Verify Simulator can be instantiated with valid predictor and race state."""
    sim = Simulator(predictor=calibrated_predictor, race_state=initial_race_state)
    assert sim.predictor is calibrated_predictor
    assert sim.race_state is initial_race_state


def test_simulator_step_lap(calibrated_predictor, initial_race_state):
    """Verify stepping a single lap updates car states and increments lap number."""
    sim = Simulator(predictor=calibrated_predictor, race_state=initial_race_state)

    lap_results = sim.step_lap()
    assert len(lap_results) == 2
    assert initial_race_state.current_lap == 2

    ver = initial_race_state.cars["VER"]
    assert ver.tyre_age == 2
    assert ver.total_time_s > 0.0
    assert ver.fuel_kg < 110.0


def test_simulator_run_returns_simulation_result(calibrated_predictor, initial_race_state):
    """Verify run executes through race completion and returns a validated SimulationResult."""
    sim = Simulator(predictor=calibrated_predictor, race_state=initial_race_state)
    result = sim.run()

    assert isinstance(result, SimulationResult)
    assert result.circuit == "Bahrain"
    assert result.total_laps == 3
    assert len(result.classification) == 2
    assert len(result.lap_history) == 6  # 2 cars * 3 laps
    assert initial_race_state.is_finished is True

    # P1 finished ahead of P2 in total time
    p1 = result.classification[0]
    p2 = result.classification[1]
    assert p1.total_time_s <= p2.total_time_s


def test_no_pit_stop_preserves_stage1_behavior(calibrated_predictor, initial_race_state):
    """Verify that when no pit stops are scheduled, Stage 1 behavior is strictly preserved."""
    sim = Simulator(predictor=calibrated_predictor, race_state=initial_race_state, pit_stops=None)
    result = sim.run()

    assert all(d.pit_stops == 0 for d in result.classification)
    assert all(not lap.is_pit_lap for lap in result.lap_history)
    # Laps completed match total_laps * cars
    assert len(result.lap_history) == 6


def test_pit_stop_adds_pit_loss(calibrated_predictor):
    """Verify that a pit stop adds exactly the configured pit-lane time loss."""
    car_no_pit = CarState(driver="NOR", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    car_with_pit = CarState(driver="VER", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)

    race = RaceState(
        circuit="Bahrain",
        total_laps=3,
        current_lap=1,
        cars={"NOR": car_no_pit, "VER": car_with_pit},
    )

    custom_pit_loss = 25.0
    pit_schedule = {"VER": {2: TyreCompound.MEDIUM}}
    sim = Simulator(
        predictor=calibrated_predictor,
        race_state=race,
        pit_loss_s=custom_pit_loss,
        pit_stops=pit_schedule,
    )

    # Lap 1: neither car pits
    lap1_res = sim.step_lap()
    ver_lap1 = next(r for r in lap1_res if r.driver == "VER")
    nor_lap1 = next(r for r in lap1_res if r.driver == "NOR")
    assert not ver_lap1.is_pit_lap
    assert np.isclose(ver_lap1.lap_time_s, nor_lap1.lap_time_s)

    # Lap 2: VER pits, NOR does not
    lap2_res = sim.step_lap()
    ver_lap2 = next(r for r in lap2_res if r.driver == "VER")
    nor_lap2 = next(r for r in lap2_res if r.driver == "NOR")
    assert ver_lap2.is_pit_lap
    assert not nor_lap2.is_pit_lap
    # Both entered lap 2 on tyre_age=2 and same fuel, so base times match, plus pit loss
    assert np.isclose(ver_lap2.lap_time_s - nor_lap2.lap_time_s, custom_pit_loss)


def test_pit_stop_compound_change_and_tyre_age_reset(calibrated_predictor):
    """Verify pit stop changes compound to scheduled compound and resets tyre_age to 1."""
    car = CarState(driver="VER", position=1, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=110.0)
    race = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car})

    pit_schedule = {"VER": {2: TyreCompound.HARD}}
    sim = Simulator(predictor=calibrated_predictor, race_state=race, pit_stops=pit_schedule)

    # Lap 1 (SOFT, age 1 -> 2)
    sim.step_lap()
    assert car.current_compound == TyreCompound.SOFT
    assert car.tyre_age == 2
    assert car.pit_stops_count == 0

    # Lap 2 (pits: changes to HARD, tyre_age resets to 1)
    lap2_res = sim.step_lap()
    assert car.current_compound == TyreCompound.HARD
    assert car.tyre_age == 1
    assert car.pit_stops_count == 1
    assert lap2_res[0].is_pit_lap is True

    # Lap 3 (runs on HARD at tyre_age 1, age increments to 2 at end of lap)
    lap3_res = sim.step_lap()
    assert car.current_compound == TyreCompound.HARD
    assert car.tyre_age == 2
    assert lap3_res[0].is_pit_lap is False


def test_pit_stop_fuel_progression_continues(calibrated_predictor):
    """Verify fuel consumption is continuous and unaffected by pit stops."""
    car1 = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    car2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    race = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car1, "NOR": car2})

    sim = Simulator(
        predictor=calibrated_predictor,
        race_state=race,
        fuel_consumption_per_lap=2.0,
        pit_stops={"VER": {2: TyreCompound.HARD}},
    )

    result = sim.run()

    # Both cars should have consumed 3 * 2.0 = 6.0 kg of fuel
    assert np.isclose(car1.fuel_kg, 110.0 - 6.0)
    assert np.isclose(car2.fuel_kg, 110.0 - 6.0)

    # Check compounds used in classification
    ver_res = next(d for d in result.classification if d.driver == "VER")
    assert ver_res.pit_stops == 1
    assert ver_res.compounds_used == [TyreCompound.MEDIUM, TyreCompound.HARD]

    nor_res = next(d for d in result.classification if d.driver == "NOR")
    assert nor_res.pit_stops == 0
    assert nor_res.compounds_used == [TyreCompound.MEDIUM]


def test_race_ordering_normal(calibrated_predictor):
    """Verify normal race ordering and gap expansion when lead car has faster pace."""
    car_fast = CarState(driver="VER", position=1, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=110.0)
    car_slow = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)

    race = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": car_fast, "NOR": car_slow})
    sim = Simulator(predictor=calibrated_predictor, race_state=race)

    # Lap 1
    lap1_res = sim.step_lap()
    ver_lap1 = next(r for r in lap1_res if r.driver == "VER")
    nor_lap1 = next(r for r in lap1_res if r.driver == "NOR")
    assert ver_lap1.position == 1
    assert nor_lap1.position == 2
    assert car_fast.position == 1
    assert car_slow.position == 2
    gap_lap1 = car_slow.gap_to_leader_s
    assert gap_lap1 > 0.0
    assert np.isclose(car_fast.gap_to_leader_s, 0.0)

    # Lap 2: fast car extends gap
    sim.step_lap()
    assert car_fast.position == 1
    assert car_slow.position == 2
    assert car_slow.gap_to_leader_s > gap_lap1


def test_race_ordering_position_change_on_pace(calibrated_predictor):
    """Verify car on faster pace overtakes car on slower pace."""
    # NOR starts P1 on HARD (slower ~94.0s), VER starts P2 on SOFT (faster ~92.5s) with tiny initial gap
    car_leader = CarState(driver="NOR", position=1, current_compound=TyreCompound.HARD, tyre_age=1, fuel_kg=110.0, total_time_s=0.0)
    car_chaser = CarState(driver="VER", position=2, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=110.0, total_time_s=0.5)

    race = RaceState(circuit="Bahrain", total_laps=2, current_lap=1, cars={"NOR": car_leader, "VER": car_chaser})
    sim = Simulator(predictor=calibrated_predictor, race_state=race)

    # After Lap 1, VER runs 92.5s (+0.5s = 93.0s total) and passes NOR (94.0s total)
    lap1_res = sim.step_lap()
    ver_lap1 = next(r for r in lap1_res if r.driver == "VER")
    nor_lap1 = next(r for r in lap1_res if r.driver == "NOR")

    assert ver_lap1.position == 1
    assert nor_lap1.position == 2
    assert car_chaser.position == 1
    assert car_leader.position == 2
    assert car_chaser.total_time_s < car_leader.total_time_s
    assert np.isclose(car_chaser.gap_to_leader_s, 0.0)
    assert np.isclose(car_leader.gap_to_leader_s, car_leader.total_time_s - car_chaser.total_time_s)


def test_race_ordering_gap_to_leader_exact(calibrated_predictor):
    """Verify mathematical exactness of cumulative race times and gaps to leader."""
    car1 = CarState(driver="VER", position=1, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=110.0)
    car2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    car3 = CarState(driver="LEC", position=3, current_compound=TyreCompound.HARD, tyre_age=1, fuel_kg=110.0)

    race = RaceState(circuit="Bahrain", total_laps=2, current_lap=1, cars={"VER": car1, "NOR": car2, "LEC": car3})
    sim = Simulator(predictor=calibrated_predictor, race_state=race)

    sim.step_lap()

    # All active cars sorted by total_time_s
    active = sorted(race.cars.values(), key=lambda c: c.total_time_s)
    leader = active[0]
    p2 = active[1]
    p3 = active[2]

    assert leader.position == 1
    assert leader.gap_to_leader_s == 0.0
    assert leader.gap_to_ahead_s == 0.0

    assert p2.position == 2
    assert np.isclose(p2.gap_to_leader_s, p2.total_time_s - leader.total_time_s)
    assert np.isclose(p2.gap_to_ahead_s, p2.total_time_s - leader.total_time_s)

    assert p3.position == 3
    assert np.isclose(p3.gap_to_leader_s, p3.total_time_s - leader.total_time_s)
    assert np.isclose(p3.gap_to_ahead_s, p3.total_time_s - p2.total_time_s)


def test_race_ordering_pit_loss_affects_order_and_gaps(calibrated_predictor):
    """Verify pit stop time loss drops leader behind chasing car and updates gap accordingly."""
    # Start identical pace on MEDIUM
    car_pits = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0, total_time_s=0.0)
    car_stays = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0, total_time_s=2.0)

    race = RaceState(circuit="Bahrain", total_laps=2, current_lap=1, cars={"VER": car_pits, "NOR": car_stays})
    # VER pits on Lap 1 incurring 22s pit loss
    sim = Simulator(predictor=calibrated_predictor, race_state=race, pit_loss_s=22.0, pit_stops={"VER": {1: "HARD"}})

    sim.step_lap()

    # NOR did not pit, so NOR takes P1, VER drops to P2
    assert car_stays.position == 1
    assert car_pits.position == 2
    assert car_stays.gap_to_leader_s == 0.0
    # VER's gap to leader equals VER total time - NOR total time
    assert car_pits.gap_to_leader_s > 0.0
    assert np.isclose(car_pits.gap_to_leader_s, car_pits.total_time_s - car_stays.total_time_s)


def test_3_driver_multi_stop_and_independent_schedules(calibrated_predictor):
    """Verify 3 drivers with independent pit schedules (2-stop, 1-stop, 0-stop)."""
    c1 = CarState(driver="VER", position=1, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=110.0)
    c2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    c3 = CarState(driver="LEC", position=3, current_compound=TyreCompound.HARD, tyre_age=1, fuel_kg=110.0)

    race = RaceState(circuit="Bahrain", total_laps=5, current_lap=1, cars={"VER": c1, "NOR": c2, "LEC": c3})
    pit_stops = {
        "VER": {2: "HARD", 4: "SOFT"},  # 2-stop
        "NOR": {3: "HARD"},              # 1-stop
        # LEC: 0-stop
    }

    sim = Simulator(predictor=calibrated_predictor, race_state=race, pit_stops=pit_stops)
    result = sim.run()

    # Classification verification
    ver_res = next(d for d in result.classification if d.driver == "VER")
    nor_res = next(d for d in result.classification if d.driver == "NOR")
    lec_res = next(d for d in result.classification if d.driver == "LEC")

    assert ver_res.pit_stops == 2
    assert nor_res.pit_stops == 1
    assert lec_res.pit_stops == 0

    assert ver_res.compounds_used == [TyreCompound.SOFT, TyreCompound.HARD]
    assert nor_res.compounds_used == [TyreCompound.MEDIUM, TyreCompound.HARD]
    assert lec_res.compounds_used == [TyreCompound.HARD]

    # Verify positions emerge strictly from total_time_s
    sorted_times = sorted([ver_res.total_time_s, nor_res.total_time_s, lec_res.total_time_s])
    classification_times = [d.total_time_s for d in result.classification]
    assert sorted_times == classification_times
    assert [d.finish_position for d in result.classification] == [1, 2, 3]


def test_lap_by_lap_history_correctness(calibrated_predictor):
    """Verify lap_history in metadata preserves all 10 required fields and exact values."""
    c1 = CarState(driver="VER", position=1, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=110.0)
    c2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)

    race = RaceState(circuit="Bahrain", total_laps=2, current_lap=1, cars={"VER": c1, "NOR": c2})
    sim = Simulator(predictor=calibrated_predictor, race_state=race, pit_stops={"VER": {1: "HARD"}})
    result = sim.run()

    history = result.metadata.get("lap_history", [])
    assert len(history) == 4  # 2 cars * 2 laps

    # Required fields verification
    required_keys = {
        "lap_number", "driver", "position", "cumulative_time_s",
        "gap_to_leader_s", "gap_to_ahead_s", "compound", "tyre_age",
        "fuel_kg", "is_pit_lap",
    }
    for record in history:
        assert required_keys.issubset(record.keys())
        assert record["cumulative_time_s"] > 0.0
        assert record["fuel_kg"] < 110.0

    # On Lap 1, VER pitted so NOR was leader (position 1)
    lap1_nor = next(r for r in history if r["lap_number"] == 1 and r["driver"] == "NOR")
    lap1_ver = next(r for r in history if r["lap_number"] == 1 and r["driver"] == "VER")

    assert lap1_nor["position"] == 1
    assert lap1_nor["gap_to_leader_s"] == 0.0
    assert lap1_nor["gap_to_ahead_s"] == 0.0
    assert lap1_nor["is_pit_lap"] is False

    assert lap1_ver["position"] == 2
    assert lap1_ver["gap_to_leader_s"] > 0.0
    assert lap1_ver["is_pit_lap"] is True
    assert np.isclose(lap1_ver["gap_to_leader_s"], lap1_ver["cumulative_time_s"] - lap1_nor["cumulative_time_s"])


def test_final_classification_ordering_and_duration(calibrated_predictor):
    """Verify final classification is strictly sorted by finish position and cumulative time."""
    c1 = CarState(driver="VER", position=1, current_compound=TyreCompound.SOFT, tyre_age=1, fuel_kg=110.0)
    c2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    c3 = CarState(driver="LEC", position=3, current_compound=TyreCompound.HARD, tyre_age=1, fuel_kg=110.0)

    race = RaceState(circuit="Bahrain", total_laps=3, current_lap=1, cars={"VER": c1, "NOR": c2, "LEC": c3})
    sim = Simulator(predictor=calibrated_predictor, race_state=race)
    result = sim.run()

    assert len(result.classification) == 3
    # Strict finish position sequence
    assert [d.finish_position for d in result.classification] == [1, 2, 3]

    # Monotonic total time
    times = [d.total_time_s for d in result.classification]
    assert times[0] <= times[1] <= times[2]

    # Winner's time equals total_duration_s
    assert np.isclose(result.total_duration_s, times[0])



