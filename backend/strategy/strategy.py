"""Deterministic race strategy representation and evaluation engine."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from backend.models.race_state import CarState, RaceState, TyreCompound
from backend.models.result import DriverResult, LapResult, SimulationResult
from backend.physics.lap_time_predictor import LapTimePredictor
from backend.simulation.simulator import DEFAULT_PIT_LOSS_S, Simulator


def _parse_compound(comp: Union[str, TyreCompound]) -> TyreCompound:
    """Safely parse string or TyreCompound to TyreCompound enum."""
    if isinstance(comp, TyreCompound):
        return comp
    name = str(comp).strip().upper()
    if name in TyreCompound.__members__:
        return TyreCompound[name]
    return TyreCompound(name)


@dataclass(frozen=True)
class PitStop:
    """A scheduled pit stop event."""

    lap: int
    compound: TyreCompound

    def __post_init__(self):
        if self.lap < 1:
            raise ValueError(f"Pit stop lap must be >= 1, got {self.lap}")
        if not isinstance(self.compound, TyreCompound):
            object.__setattr__(self, "compound", _parse_compound(self.compound))


@dataclass
class Strategy:
    """Representation of a deterministic multi-stint race strategy.

    Supports 0, 1, 2, or more pit stops.
    """

    starting_compound: TyreCompound
    stops: List[PitStop] = field(default_factory=list)
    name: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.starting_compound, TyreCompound):
            self.starting_compound = _parse_compound(self.starting_compound)

        normalized_stops: List[PitStop] = []
        for stop in self.stops:
            if isinstance(stop, PitStop):
                normalized_stops.append(stop)
            elif isinstance(stop, (tuple, list)) and len(stop) == 2:
                normalized_stops.append(
                    PitStop(lap=int(stop[0]), compound=_parse_compound(stop[1]))
                )
            else:
                raise ValueError(f"Invalid pit stop specification: {stop}")

        # Validate strictly increasing pit laps
        for i in range(len(normalized_stops) - 1):
            if normalized_stops[i].lap >= normalized_stops[i + 1].lap:
                raise ValueError(
                    f"Pit stops must be strictly increasing by lap: "
                    f"{normalized_stops[i].lap} >= {normalized_stops[i + 1].lap}"
                )

        self.stops = normalized_stops

    @property
    def pit_count(self) -> int:
        """Number of pit stops in this strategy."""
        return len(self.stops)

    def to_pit_schedule(self) -> Dict[int, TyreCompound]:
        """Convert strategy stops to {lap_number: TyreCompound} for Simulator."""
        return {stop.lap: stop.compound for stop in self.stops}


@dataclass
class StrategyEvaluationResult:
    """Complete evaluation outcome for a single race strategy."""

    strategy_name: str
    strategy: Strategy
    driver: str
    total_time_s: float
    finish_position: int
    pit_stops: int
    compounds_used: List[TyreCompound]
    final_compound: TyreCompound
    final_tyre_age: int
    final_fuel_kg: float
    simulation_result: SimulationResult
    lap_history: List[Dict[str, Any]]


def evaluate_strategy(
    strategy: Strategy,
    predictor: LapTimePredictor,
    race_state: RaceState,
    driver: Optional[str] = None,
    opponent_pit_stops: Optional[Dict[str, Any]] = None,
    pit_loss_s: Optional[float] = None,
    fuel_consumption_per_lap: Optional[float] = None,
) -> StrategyEvaluationResult:
    """Evaluate a single fixed strategy deterministically using Simulator.

    Clones the input RaceState to ensure pure, isolated, non-mutating execution.
    """
    state_copy = race_state.model_copy(deep=True)

    target_driver = driver
    if target_driver is None:
        if not state_copy.cars:
            raise ValueError("RaceState contains no cars to evaluate.")
        target_driver = next(iter(state_copy.cars.keys()))

    if target_driver not in state_copy.cars:
        raise KeyError(f"Driver '{target_driver}' not found in RaceState cars.")

    # Configure target car's starting tyre compound
    car = state_copy.cars[target_driver]
    car.current_compound = strategy.starting_compound
    car.tyre_age = 1
    car.stint = 1
    car.pit_stops_count = 0
    car.total_time_s = 0.0

    # Combine opponent schedules with target strategy schedule
    combined_pit_stops: Dict[str, Any] = {}
    if opponent_pit_stops:
        for d, s in opponent_pit_stops.items():
            if d != target_driver:
                combined_pit_stops[d] = s

    combined_pit_stops[target_driver] = strategy.to_pit_schedule()

    # Execute simulation using existing Simulator as source of truth
    sim = Simulator(
        predictor=predictor,
        race_state=state_copy,
        fuel_consumption_per_lap=fuel_consumption_per_lap,
        pit_loss_s=pit_loss_s,
        pit_stops=combined_pit_stops,
    )
    sim_result = sim.run()

    # Extract target driver classification
    driver_result: Optional[DriverResult] = next(
        (d for d in sim_result.classification if d.driver == target_driver), None
    )
    if driver_result is None:
        raise RuntimeError(f"Driver '{target_driver}' not found in simulation results.")

    driver_history = [
        r for r in sim_result.metadata.get("lap_history", [])
        if r.get("driver") == target_driver
    ]

    strat_name = strategy.name or f"Strategy_{strategy.starting_compound.value}_{len(strategy.stops)}stop"

    return StrategyEvaluationResult(
        strategy_name=strat_name,
        strategy=strategy,
        driver=target_driver,
        total_time_s=driver_result.total_time_s,
        finish_position=driver_result.finish_position,
        pit_stops=driver_result.pit_stops,
        compounds_used=driver_result.compounds_used,
        final_compound=car.current_compound,
        final_tyre_age=car.tyre_age,
        final_fuel_kg=car.fuel_kg,
        simulation_result=sim_result,
        lap_history=driver_history,
    )


def compare_strategies(
    strategies: Sequence[Strategy],
    predictor: LapTimePredictor,
    race_state: RaceState,
    driver: Optional[str] = None,
    opponent_pit_stops: Optional[Dict[str, Any]] = None,
    pit_loss_s: Optional[float] = None,
    fuel_consumption_per_lap: Optional[float] = None,
) -> List[StrategyEvaluationResult]:
    """Evaluate multiple candidate strategies under identical race conditions and rank them.

    Ranking orders by finish position ascending, then total race time ascending.
    """
    if not strategies:
        return []

    evaluated: List[StrategyEvaluationResult] = []
    for strat in strategies:
        result = evaluate_strategy(
            strategy=strat,
            predictor=predictor,
            race_state=race_state,
            driver=driver,
            opponent_pit_stops=opponent_pit_stops,
            pit_loss_s=pit_loss_s,
            fuel_consumption_per_lap=fuel_consumption_per_lap,
        )
        evaluated.append(result)

    # Rank deterministically by finish position, then total duration
    evaluated.sort(key=lambda r: (r.finish_position, r.total_time_s))
    return evaluated
