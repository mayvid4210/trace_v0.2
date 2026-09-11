"""Deterministic race strategy optimization engine.

Performs exhaustive/windowed deterministic search over candidate pit strategies
using the existing Strategy -> Simulator -> StrategyEvaluationResult architecture.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from backend.models.race_state import RaceState, TyreCompound
from backend.physics.lap_time_predictor import LapTimePredictor
from backend.strategy.strategy import (
    PitStop,
    Strategy,
    StrategyEvaluationResult,
    _parse_compound,
    evaluate_strategy,
)


@dataclass
class OptimizationResult:
    """Structured result of deterministic strategy optimization."""

    best_strategy: Strategy
    best_evaluation: StrategyEvaluationResult
    predicted_total_time_s: float
    predicted_finish_position: int
    pit_stop_count: int
    pit_laps: List[int]
    compound_sequence: List[TyreCompound]
    final_compound: TyreCompound
    final_tyre_age: int
    fuel_remaining_kg: float
    strategies_evaluated_count: int
    ranked_strategies: List[StrategyEvaluationResult]
    runtime_s: float


def validate_strategy_legality(
    strategy: Strategy,
    total_laps: int,
    min_stint_length: int = 1,
    require_different_compounds: bool = False,
) -> Tuple[bool, Optional[str]]:
    """Validate whether a candidate strategy satisfies physical and regulatory constraints.

    Returns:
        (is_legal, reason_if_illegal)
    """
    if total_laps < 1:
        return False, f"total_laps must be >= 1, got {total_laps}"

    # 0-stop strategy
    if not strategy.stops:
        if total_laps < min_stint_length:
            return False, f"Race length {total_laps} < min stint length {min_stint_length}"
        return True, None

    stops = strategy.stops
    compounds_used = [strategy.starting_compound] + [s.compound for s in stops]

    # Stint 1: Laps 1 to stops[0].lap
    if stops[0].lap < min_stint_length:
        return (
            False,
            f"Stint 1 length {stops[0].lap} is less than min stint length {min_stint_length}",
        )

    # Intermediate stints
    for i in range(len(stops) - 1):
        stint_len = stops[i + 1].lap - stops[i].lap
        if stint_len < min_stint_length:
            return (
                False,
                f"Stint {i + 2} length {stint_len} is less than min stint length {min_stint_length}",
            )

    # Last pit stop must be strictly before total_laps
    if stops[-1].lap >= total_laps:
        return (
            False,
            f"Pit stop lap {stops[-1].lap} must be strictly before total laps {total_laps}",
        )

    # Final stint: Laps stops[-1].lap + 1 to total_laps
    final_stint_len = total_laps - stops[-1].lap
    if final_stint_len < min_stint_length:
        return (
            False,
            f"Final stint length {final_stint_len} is less than min stint length {min_stint_length}",
        )

    # Two-compound regulatory requirement for dry races (if applicable)
    if require_different_compounds and len(set(compounds_used)) < 2:
        return False, "Strategy must use at least two distinct tyre compounds"

    return True, None


def generate_candidate_strategies(
    total_laps: int,
    available_compounds: Sequence[Union[str, TyreCompound]],
    max_stops: int = 2,
    min_stint_length: int = 5,
    pit_window_step: int = 1,
    starting_compounds: Optional[Sequence[Union[str, TyreCompound]]] = None,
    require_different_compounds: bool = True,
    pit_windows: Optional[Dict[int, Tuple[int, int]]] = None,
) -> List[Strategy]:
    """Generate a deterministic set of candidate strategies across feasible pit windows.

    Supports 0-stop, 1-stop, 2-stop (and optionally 3-stop if max_stops >= 3).
    """
    compounds = [_parse_compound(c) for c in available_compounds]
    if not compounds:
        raise ValueError("available_compounds cannot be empty")

    starters = (
        [_parse_compound(c) for c in starting_compounds]
        if starting_compounds
        else compounds
    )

    candidates: List[Strategy] = []
    step = max(1, int(pit_window_step))

    # --- 0-Stop ---
    if total_laps >= min_stint_length:
        for c0 in starters:
            # Note: A 0-stop strategy inherently uses only 1 compound.
            # If require_different_compounds is True, 0-stop is typically illegal in dry F1 races,
            # but we allow generating it if require_different_compounds is False or to test baseline.
            if not require_different_compounds:
                strat = Strategy(
                    starting_compound=c0,
                    stops=[],
                    name=f"0-Stop {c0.value}",
                )
                candidates.append(strat)

    if max_stops < 1:
        return candidates

    # Window bounds helper
    def get_window(stop_idx: int, min_val: int, max_val: int) -> Tuple[int, int]:
        if pit_windows and stop_idx in pit_windows:
            w_min, w_max = pit_windows[stop_idx]
            return max(min_val, w_min), min(max_val, w_max)
        return min_val, max_val

    # --- 1-Stop ---
    # L1: Stint 1 >= min_stint_length, Stint 2 (N - L1) >= min_stint_length
    l1_min = min_stint_length
    l1_max = total_laps - min_stint_length
    w1_min, w1_max = get_window(1, l1_min, l1_max)

    if w1_min <= w1_max:
        for c0 in starters:
            for c1 in compounds:
                if require_different_compounds and c0 == c1:
                    continue
                for l1 in range(w1_min, w1_max + 1, step):
                    candidates.append(
                        Strategy(
                            starting_compound=c0,
                            stops=[PitStop(lap=l1, compound=c1)],
                            name=f"1-Stop {c0.value} L{l1} {c1.value}",
                        )
                    )

    if max_stops < 2:
        return candidates

    # --- 2-Stop ---
    # L1: Stint 1 >= min_stint, L2 - L1 >= min_stint, N - L2 >= min_stint
    # Therefore: L1 in [min_stint, N - 2*min_stint], L2 in [L1 + min_stint, N - min_stint]
    l1_2stop_min = min_stint_length
    l1_2stop_max = total_laps - 2 * min_stint_length
    w1_2s_min, w1_2s_max = get_window(1, l1_2stop_min, l1_2stop_max)

    if w1_2s_min <= w1_2s_max:
        for c0 in starters:
            for c1 in compounds:
                for c2 in compounds:
                    if require_different_compounds and len({c0, c1, c2}) < 2:
                        continue
                    for l1 in range(w1_2s_min, w1_2s_max + 1, step):
                        l2_min = l1 + min_stint_length
                        l2_max = total_laps - min_stint_length
                        w2_min, w2_max = get_window(2, l2_min, l2_max)
                        if w2_min > w2_max:
                            continue
                        for l2 in range(w2_min, w2_max + 1, step):
                            candidates.append(
                                Strategy(
                                    starting_compound=c0,
                                    stops=[
                                        PitStop(lap=l1, compound=c1),
                                        PitStop(lap=l2, compound=c2),
                                    ],
                                    name=f"2-Stop {c0.value} L{l1} {c1.value} L{l2} {c2.value}",
                                )
                            )

    if max_stops < 3:
        return candidates

    # --- 3-Stop (if enabled) ---
    l1_3s_min = min_stint_length
    l1_3s_max = total_laps - 3 * min_stint_length
    w1_3s_min, w1_3s_max = get_window(1, l1_3s_min, l1_3s_max)

    if w1_3s_min <= w1_3s_max:
        for c0 in starters:
            for c1 in compounds:
                for c2 in compounds:
                    for c3 in compounds:
                        if require_different_compounds and len({c0, c1, c2, c3}) < 2:
                            continue
                        for l1 in range(w1_3s_min, w1_3s_max + 1, step):
                            l2_min = l1 + min_stint_length
                            l2_max = total_laps - 2 * min_stint_length
                            w2_min, w2_max = get_window(2, l2_min, l2_max)
                            for l2 in range(w2_min, w2_max + 1, step):
                                l3_min = l2 + min_stint_length
                                l3_max = total_laps - min_stint_length
                                w3_min, w3_max = get_window(3, l3_min, l3_max)
                                for l3 in range(w3_min, w3_max + 1, step):
                                    candidates.append(
                                        Strategy(
                                            starting_compound=c0,
                                            stops=[
                                                PitStop(lap=l1, compound=c1),
                                                PitStop(lap=l2, compound=c2),
                                                PitStop(lap=l3, compound=c3),
                                            ],
                                            name=f"3-Stop {c0.value} L{l1} {c1.value} L{l2} {c2.value} L{l3} {c3.value}",
                                        )
                                    )

    return candidates


def optimize_strategy(
    predictor: LapTimePredictor,
    race_state: RaceState,
    driver: Optional[str] = None,
    available_compounds: Optional[Sequence[Union[str, TyreCompound]]] = None,
    max_stops: int = 2,
    min_stint_length: int = 5,
    pit_window_step: int = 1,
    require_different_compounds: bool = True,
    starting_compounds: Optional[Sequence[Union[str, TyreCompound]]] = None,
    pit_windows: Optional[Dict[int, Tuple[int, int]]] = None,
    candidate_strategies: Optional[Sequence[Strategy]] = None,
    opponent_pit_stops: Optional[Dict[str, Any]] = None,
    pit_loss_s: Optional[float] = None,
    fuel_consumption_per_lap: Optional[float] = None,
    top_n: int = 5,
) -> OptimizationResult:
    """Find the optimal pit strategy for a driver by deterministic simulation search.

    Objective:
    1. Primary: MINIMIZE simulated cumulative race time.
    2. Secondary tie-breaker: Higher finishing position (lower integer).
    3. Tertiary tie-breaker: Fewer pit stops.
    4. Deterministic final tie-breaker: Earlier pit lap numbers.
    """
    start_time = time.perf_counter()

    # Determine candidate strategies
    if candidate_strategies is not None:
        strategies_to_test = list(candidate_strategies)
    else:
        compounds = available_compounds or [
            TyreCompound.SOFT,
            TyreCompound.MEDIUM,
            TyreCompound.HARD,
        ]
        strategies_to_test = generate_candidate_strategies(
            total_laps=race_state.total_laps,
            available_compounds=compounds,
            max_stops=max_stops,
            min_stint_length=min_stint_length,
            pit_window_step=pit_window_step,
            starting_compounds=starting_compounds,
            require_different_compounds=require_different_compounds,
            pit_windows=pit_windows,
        )

    if not strategies_to_test:
        raise ValueError("No candidate strategies available to evaluate.")

    target_driver = driver or next(iter(race_state.cars.keys()))

    evaluations: List[StrategyEvaluationResult] = []
    for strat in strategies_to_test:
        # Validate legality
        is_legal, _ = validate_strategy_legality(
            strategy=strat,
            total_laps=race_state.total_laps,
            min_stint_length=min_stint_length,
            require_different_compounds=require_different_compounds,
        )
        if not is_legal:
            continue

        eval_res = evaluate_strategy(
            strategy=strat,
            predictor=predictor,
            race_state=race_state,
            driver=target_driver,
            opponent_pit_stops=opponent_pit_stops,
            pit_loss_s=pit_loss_s,
            fuel_consumption_per_lap=fuel_consumption_per_lap,
        )
        evaluations.append(eval_res)

    if not evaluations:
        raise ValueError("All candidate strategies failed legality validation.")

    # Sort deterministically:
    # Primary: total_time_s ascending, finish_position ascending
    # Tie-breakers: fewer pit stops, earlier pit lap list
    evaluations.sort(
        key=lambda r: (
            r.total_time_s,
            r.finish_position,
            r.pit_stops,
            [s.lap for s in r.strategy.stops],
        )
    )

    best_eval = evaluations[0]
    best_strat = best_eval.strategy
    elapsed = time.perf_counter() - start_time

    return OptimizationResult(
        best_strategy=best_strat,
        best_evaluation=best_eval,
        predicted_total_time_s=best_eval.total_time_s,
        predicted_finish_position=best_eval.finish_position,
        pit_stop_count=best_eval.pit_stops,
        pit_laps=[s.lap for s in best_strat.stops],
        compound_sequence=[best_strat.starting_compound] + [s.compound for s in best_strat.stops],
        final_compound=best_eval.final_compound,
        final_tyre_age=best_eval.final_tyre_age,
        fuel_remaining_kg=best_eval.final_fuel_kg,
        strategies_evaluated_count=len(evaluations),
        ranked_strategies=evaluations[:top_n],
        runtime_s=elapsed,
    )
