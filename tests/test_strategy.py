"""Unit tests for the deterministic Strategy engine."""

import pytest
import numpy as np

from backend.models.race_state import (
    CarState,
    RaceState,
    TrackCondition,
    TrackStatus,
    TyreCompound,
)
from backend.physics.lap_time_predictor import (
    CircuitCalibration,
    LapTimePredictor,
)
from backend.strategy import (
    PitStop,
    Strategy,
    StrategyEvaluationResult,
    compare_strategies,
    evaluate_strategy,
)


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


@pytest.fixture
def solo_race_state():
    """Single-car race state at Bahrain for 10 laps."""
    car = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    return RaceState(circuit="Bahrain", total_laps=10, current_lap=1, cars={"VER": car})


@pytest.fixture
def multi_car_race_state():
    """Two-car race state at Bahrain for 10 laps."""
    car1 = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    car2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=110.0)
    return RaceState(circuit="Bahrain", total_laps=10, current_lap=1, cars={"VER": car1, "NOR": car2})


def test_strategy_representation_0_1_2_stops():
    """Verify clean strategy creation for 0, 1, and 2+ pit stops."""
    # 0-stop
    s0 = Strategy(starting_compound=TyreCompound.HARD, name="Zero-Stop")
    assert s0.pit_count == 0
    assert s0.to_pit_schedule() == {}

    # 1-stop
    s1 = Strategy(starting_compound="MEDIUM", stops=[(6, "HARD")], name="Med-Hard")
    assert s1.pit_count == 1
    assert s1.starting_compound == TyreCompound.MEDIUM
    assert s1.to_pit_schedule() == {6: TyreCompound.HARD}

    # 2-stop
    s2 = Strategy(
        starting_compound="SOFT",
        stops=[PitStop(lap=4, compound=TyreCompound.MEDIUM), (7, "SOFT")],
        name="Soft-Med-Soft",
    )
    assert s2.pit_count == 2
    assert s2.starting_compound == TyreCompound.SOFT
    assert s2.to_pit_schedule() == {4: TyreCompound.MEDIUM, 7: TyreCompound.SOFT}


def test_strategy_validation_errors():
    """Verify invalid pit stop configurations raise informative ValueErrors."""
    # Lap < 1
    with pytest.raises(ValueError, match="must be >= 1"):
        Strategy(starting_compound="SOFT", stops=[(0, "HARD")])

    # Non-increasing laps
    with pytest.raises(ValueError, match="strictly increasing"):
        Strategy(starting_compound="SOFT", stops=[(5, "HARD"), (5, "MEDIUM")])

    with pytest.raises(ValueError, match="strictly increasing"):
        Strategy(starting_compound="SOFT", stops=[(6, "HARD"), (4, "MEDIUM")])


def test_evaluate_zero_stop_strategy(predictor, solo_race_state):
    """Verify evaluation of a 0-stop strategy."""
    strat = Strategy(starting_compound=TyreCompound.HARD, name="0-Stop Hard")
    eval_res = evaluate_strategy(strategy=strat, predictor=predictor, race_state=solo_race_state, driver="VER")

    assert eval_res.pit_stops == 0
    assert eval_res.compounds_used == [TyreCompound.HARD]
    assert eval_res.final_compound == TyreCompound.HARD
    assert eval_res.final_tyre_age == 11  # 10 laps completed on 1 tyre set (1 -> 11)
    assert len(eval_res.lap_history) == 10
    assert all(not r["is_pit_lap"] for r in eval_res.lap_history)


def test_evaluate_one_stop_strategy(predictor, solo_race_state):
    """Verify evaluation of a 1-stop strategy."""
    strat = Strategy(starting_compound="MEDIUM", stops=[(5, "HARD")], name="1-Stop M-H")
    eval_res = evaluate_strategy(strategy=strat, predictor=predictor, race_state=solo_race_state, driver="VER")

    assert eval_res.pit_stops == 1
    assert eval_res.compounds_used == [TyreCompound.MEDIUM, TyreCompound.HARD]
    assert eval_res.final_compound == TyreCompound.HARD
    assert eval_res.final_tyre_age == 6  # 5 laps after pit stop (laps 6,7,8,9,10: age 1 -> 6)

    # Lap 5 was pit lap
    lap5 = next(r for r in eval_res.lap_history if r["lap_number"] == 5)
    assert lap5["is_pit_lap"] is True


def test_evaluate_two_stop_strategy(predictor, solo_race_state):
    """Verify evaluation of a 2-stop strategy."""
    strat = Strategy(
        starting_compound="SOFT",
        stops=[(3, "MEDIUM"), (7, "SOFT")],
        name="2-Stop S-M-S",
    )
    eval_res = evaluate_strategy(strategy=strat, predictor=predictor, race_state=solo_race_state, driver="VER")

    assert eval_res.pit_stops == 2
    assert eval_res.compounds_used == [TyreCompound.SOFT, TyreCompound.MEDIUM]
    assert eval_res.final_compound == TyreCompound.SOFT
    assert eval_res.final_tyre_age == 4  # 3 laps after 2nd pit stop (laps 8,9,10: age 1 -> 4)

    pit_laps = [r["lap_number"] for r in eval_res.lap_history if r["is_pit_lap"]]
    assert pit_laps == [3, 7]


def test_different_strategies_produce_different_outcomes(predictor, solo_race_state):
    """Verify distinct strategies produce measurably different total race times."""
    s_0stop = Strategy(starting_compound="HARD", name="0-Stop")
    s_1stop = Strategy(starting_compound="MEDIUM", stops=[(5, "HARD")], name="1-Stop")
    s_2stop = Strategy(starting_compound="SOFT", stops=[(3, "MEDIUM"), (7, "SOFT")], name="2-Stop")

    r0 = evaluate_strategy(s_0stop, predictor, solo_race_state, "VER")
    r1 = evaluate_strategy(s_1stop, predictor, solo_race_state, "VER")
    r2 = evaluate_strategy(s_2stop, predictor, solo_race_state, "VER")

    assert r0.total_time_s != r1.total_time_s
    assert r1.total_time_s != r2.total_time_s
    # In a short 10-lap race, extra pit stops incur ~22s penalty each
    assert r0.total_time_s < r1.total_time_s < r2.total_time_s


def test_compare_strategies_ranking(predictor, multi_car_race_state):
    """Verify compare_strategies ranks multiple candidate strategies under identical conditions."""
    s1 = Strategy(starting_compound="HARD", name="0-Stop Hard")
    s2 = Strategy(starting_compound="MEDIUM", stops=[(5, "HARD")], name="1-Stop Med-Hard")
    s3 = Strategy(starting_compound="SOFT", stops=[(3, "MEDIUM"), (7, "SOFT")], name="2-Stop")

    # Opponent NOR does 1 stop on lap 5
    opponent_pits = {"NOR": {5: "HARD"}}

    ranked = compare_strategies(
        strategies=[s2, s3, s1],  # Pass out of order
        predictor=predictor,
        race_state=multi_car_race_state,
        driver="VER",
        opponent_pit_stops=opponent_pits,
    )

    assert len(ranked) == 3
    # Strict monotonically increasing total time
    assert ranked[0].total_time_s <= ranked[1].total_time_s <= ranked[2].total_time_s
    assert ranked[0].strategy_name == "0-Stop Hard"


def test_deterministic_repeatability(predictor, multi_car_race_state):
    """Verify strategy evaluation is 100% deterministically repeatable."""
    strat = Strategy(starting_compound="MEDIUM", stops=[(5, "HARD")], name="TestStrat")

    res1 = evaluate_strategy(strat, predictor, multi_car_race_state, "VER")
    res2 = evaluate_strategy(strat, predictor, multi_car_race_state, "VER")

    assert np.isclose(res1.total_time_s, res2.total_time_s)
    assert res1.finish_position == res2.finish_position
    assert res1.pit_stops == res2.pit_stops
    assert res1.compounds_used == res2.compounds_used
    assert res1.final_tyre_age == res2.final_tyre_age


def test_fair_comparison_does_not_mutate_base_race_state(predictor, multi_car_race_state):
    """Verify base RaceState remains pristine and unmutated after evaluations."""
    initial_lap = multi_car_race_state.current_lap
    initial_ver_time = multi_car_race_state.cars["VER"].total_time_s

    s1 = Strategy(starting_compound="SOFT", stops=[(4, "HARD")])
    s2 = Strategy(starting_compound="HARD")

    compare_strategies([s1, s2], predictor, multi_car_race_state, "VER")

    assert multi_car_race_state.current_lap == initial_lap
    assert multi_car_race_state.cars["VER"].total_time_s == initial_ver_time
    assert multi_car_race_state.is_finished is False
