"""Unit tests for the deterministic Strategy Optimization engine."""

import numpy as np
import pytest

from backend.models.race_state import CarState, RaceState, TyreCompound
from backend.physics.lap_time_predictor import CircuitCalibration, LapTimePredictor
from backend.strategy import (
    OptimizationResult,
    PitStop,
    Strategy,
    compare_strategies,
    evaluate_strategy,
    generate_candidate_strategies,
    optimize_strategy,
    validate_strategy_legality,
)


@pytest.fixture
def predictor():
    """Predictor calibrated with realistic baseline constants."""
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
        ("Canada", "MEDIUM"): CircuitCalibration(
            dry_baseline_lap_time_s=75.00,
            reference_mass_kg=850.0,
        ),
        ("Canada", "HARD"): CircuitCalibration(
            dry_baseline_lap_time_s=76.00,
            reference_mass_kg=850.0,
        ),
        ("Canada", "__default__"): CircuitCalibration(
            dry_baseline_lap_time_s=75.50,
            reference_mass_kg=850.0,
        ),
    }
    return LapTimePredictor(calibrations)


@pytest.fixture
def race_bahrain_20laps():
    """20-lap race state at Bahrain."""
    car = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=50.0)
    return RaceState(circuit="Bahrain", total_laps=20, current_lap=1, cars={"VER": car})


@pytest.fixture
def multi_car_race_20laps():
    """20-lap race state at Bahrain with 2 drivers."""
    car1 = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=50.0)
    car2 = CarState(driver="NOR", position=2, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=50.0)
    return RaceState(circuit="Bahrain", total_laps=20, current_lap=1, cars={"VER": car1, "NOR": car2})


def test_optimizer_deterministic_output(predictor, race_bahrain_20laps):
    """Verify strategy optimizer produces bitwise deterministic output across runs."""
    res1 = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        available_compounds=["MEDIUM", "HARD"],
        max_stops=1,
        min_stint_length=5,
        pit_window_step=2,
    )
    res2 = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        available_compounds=["MEDIUM", "HARD"],
        max_stops=1,
        min_stint_length=5,
        pit_window_step=2,
    )

    assert res1.predicted_total_time_s == res2.predicted_total_time_s
    assert res1.pit_laps == res2.pit_laps
    assert res1.compound_sequence == res2.compound_sequence
    assert res1.strategies_evaluated_count == res2.strategies_evaluated_count
    assert len(res1.ranked_strategies) == len(res2.ranked_strategies)


def test_optimizer_0_stop_strategy_evaluation(predictor, race_bahrain_20laps):
    """Verify 0-stop strategies are correctly evaluated when permitted."""
    res = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        available_compounds=["MEDIUM", "HARD"],
        max_stops=0,
        min_stint_length=5,
        require_different_compounds=False,
    )

    assert res.pit_stop_count == 0
    assert res.pit_laps == []
    assert len(res.compound_sequence) == 1
    assert res.strategies_evaluated_count == 2  # MEDIUM and HARD 0-stop


def test_optimizer_1_stop_search(predictor, race_bahrain_20laps):
    """Verify 1-stop window search evaluates multiple feasible pit laps."""
    res = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        available_compounds=["MEDIUM", "HARD"],
        max_stops=1,
        min_stint_length=5,
        pit_window_step=1,
        require_different_compounds=True,
    )

    assert res.pit_stop_count == 1
    assert len(res.pit_laps) == 1
    # Pit lap must respect min stint length 5: 5 <= L1 <= 15
    assert 5 <= res.pit_laps[0] <= 15
    assert len(res.compound_sequence) == 2
    assert res.strategies_evaluated_count >= 10


def test_optimizer_2_stop_search(predictor, race_bahrain_20laps):
    """Verify 2-stop window search evaluates feasible (L1, L2) combinations."""
    res = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        available_compounds=["SOFT", "HARD"],
        max_stops=2,
        min_stint_length=5,
        pit_window_step=2,
        require_different_compounds=True,
    )

    # In a 20-lap race with 5-lap min stints, 2-stop strategies are feasible (e.g. L6, L12)
    assert res.strategies_evaluated_count > 0
    best = res.best_strategy
    if best.pit_count == 2:
        l1, l2 = best.stops[0].lap, best.stops[1].lap
        assert l1 >= 5
        assert l2 - l1 >= 5
        assert 20 - l2 >= 5


def test_invalid_pit_windows_rejected():
    """Verify illegal strategies violating race distance or boundaries are rejected."""
    # Pit lap outside race distance
    strat_over = Strategy(starting_compound="MEDIUM", stops=[PitStop(lap=25, compound=TyreCompound.HARD)])
    legal, reason = validate_strategy_legality(strat_over, total_laps=20, min_stint_length=5)
    assert not legal
    assert "strictly before total laps" in reason

    # Stint 1 too short
    strat_short = Strategy(starting_compound="MEDIUM", stops=[PitStop(lap=2, compound=TyreCompound.HARD)])
    legal, reason = validate_strategy_legality(strat_short, total_laps=20, min_stint_length=5)
    assert not legal
    assert "Stint 1 length" in reason


def test_minimum_stint_constraint(predictor, race_bahrain_20laps):
    """Verify candidate generator and validator enforce min_stint_length strictly."""
    candidates = generate_candidate_strategies(
        total_laps=20,
        available_compounds=["MEDIUM", "HARD"],
        max_stops=1,
        min_stint_length=8,
        pit_window_step=1,
    )

    # With total_laps=20 and min_stint=8, only L1 in [8, 12] are valid
    for c in candidates:
        for stop in c.stops:
            assert 8 <= stop.lap <= 12


def test_best_strategy_has_minimum_simulated_race_time(predictor, race_bahrain_20laps):
    """Verify the selected best strategy strictly has the lowest cumulative race time."""
    res = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        available_compounds=["MEDIUM", "HARD"],
        max_stops=1,
        min_stint_length=5,
        pit_window_step=1,
    )

    best_time = res.predicted_total_time_s
    for ranked in res.ranked_strategies:
        assert best_time <= ranked.total_time_s


def test_circuit_specific_pit_loss_inherited_from_simulator(predictor):
    """Verify optimizer inherits circuit-specific pit loss (Bahrain 24.1s vs Canada 24.7s)."""
    # 1. Bahrain strategy
    car_bah = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=50.0)
    race_bah = RaceState(circuit="Bahrain", total_laps=10, current_lap=1, cars={"VER": car_bah})
    res_bah = optimize_strategy(
        predictor=predictor,
        race_state=race_bah,
        require_different_compounds=False,
        candidate_strategies=[
            Strategy(starting_compound="MEDIUM", stops=[PitStop(5, "MEDIUM")], name="Bahrain 1-Stop")
        ],
    )

    # 2. Canada strategy
    car_can = CarState(driver="VER", position=1, current_compound=TyreCompound.MEDIUM, tyre_age=1, fuel_kg=50.0)
    race_can = RaceState(circuit="Canada", total_laps=10, current_lap=1, cars={"VER": car_can})
    res_can = optimize_strategy(
        predictor=predictor,
        race_state=race_can,
        require_different_compounds=False,
        candidate_strategies=[
            Strategy(starting_compound="MEDIUM", stops=[PitStop(5, "MEDIUM")], name="Canada 1-Stop")
        ],
    )

    # Baseline 0-stop in each circuit:
    res_bah_0stop = optimize_strategy(
        predictor=predictor,
        race_state=race_bah,
        require_different_compounds=False,
        candidate_strategies=[Strategy(starting_compound="MEDIUM", stops=[], name="0-Stop")],
    )
    res_can_0stop = optimize_strategy(
        predictor=predictor,
        race_state=race_can,
        require_different_compounds=False,
        candidate_strategies=[Strategy(starting_compound="MEDIUM", stops=[], name="0-Stop")],
    )

    bah_delta = res_bah.predicted_total_time_s - res_bah_0stop.predicted_total_time_s
    can_delta = res_can.predicted_total_time_s - res_can_0stop.predicted_total_time_s
    # Lap 5 pit stop adds: pit_loss (24.1 vs 24.7)
    assert np.isclose(can_delta - bah_delta, 24.7 - 24.1)


def test_explicit_pit_loss_override_still_works(predictor, race_bahrain_20laps):
    """Verify explicitly supplied pit_loss_s overrides circuit calibration in optimizer."""
    res_default = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        candidate_strategies=[Strategy(starting_compound="MEDIUM", stops=[PitStop(10, "HARD")])],
    )
    res_override = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        pit_loss_s=30.0,
        candidate_strategies=[Strategy(starting_compound="MEDIUM", stops=[PitStop(10, "HARD")])],
    )

    # 30.0s - 24.1s = 5.9s delta
    assert np.isclose(res_override.predicted_total_time_s - res_default.predicted_total_time_s, 30.0 - 24.1)


def test_opponent_schedules_preserved(predictor, multi_car_race_20laps):
    """Verify opponent pit schedules are preserved and affect target driver's finish position."""
    # If NOR pits on lap 5 (incurring 24.1s pit loss), and VER pits on lap 10:
    res = optimize_strategy(
        predictor=predictor,
        race_state=multi_car_race_20laps,
        driver="VER",
        opponent_pit_stops={"NOR": {5: TyreCompound.HARD}},
        candidate_strategies=[
            Strategy(starting_compound="MEDIUM", stops=[PitStop(10, "HARD")], name="VER 1-Stop")
        ],
    )

    assert res.predicted_finish_position == 1
    # Verify opponent NOR was evaluated in the simulation
    nor_res = next(d for d in res.best_evaluation.simulation_result.classification if d.driver == "NOR")
    assert nor_res.pit_stops == 1
    assert nor_res.compounds_used == [TyreCompound.MEDIUM, TyreCompound.HARD]


def test_ranking_is_deterministic(predictor, race_bahrain_20laps):
    """Verify ranked_strategies is strictly sorted by finish position and total race time."""
    res = optimize_strategy(
        predictor=predictor,
        race_state=race_bahrain_20laps,
        available_compounds=["MEDIUM", "HARD"],
        max_stops=1,
        min_stint_length=5,
        pit_window_step=1,
        top_n=5,
    )

    ranked = res.ranked_strategies
    assert len(ranked) >= 2
    for i in range(len(ranked) - 1):
        assert (
            ranked[i].total_time_s,
            ranked[i].finish_position,
        ) <= (
            ranked[i + 1].total_time_s,
            ranked[i + 1].finish_position,
        )


def test_no_regression_in_existing_strategy_behavior(predictor, race_bahrain_20laps):
    """Verify evaluate_strategy and compare_strategies continue functioning identically."""
    s1 = Strategy(starting_compound="MEDIUM", stops=[PitStop(10, "HARD")], name="Strat 1")
    s2 = Strategy(starting_compound="HARD", stops=[], name="Strat 2")

    r1 = evaluate_strategy(s1, predictor, race_bahrain_20laps, driver="VER")
    assert r1.pit_stops == 1
    assert r1.compounds_used == [TyreCompound.MEDIUM, TyreCompound.HARD]

    comparison = compare_strategies([s1, s2], predictor, race_bahrain_20laps, driver="VER")
    assert len(comparison) == 2
    assert comparison[0].finish_position <= comparison[1].finish_position
