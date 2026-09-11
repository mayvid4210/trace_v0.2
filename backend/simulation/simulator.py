"""Deterministic race simulation engine skeleton."""

from typing import Any, Dict, List, Optional, Union
import uuid

from backend.models.race_state import CarState, RaceState, TrackCondition, TrackStatus, TyreCompound
from backend.models.result import DriverResult, LapResult, SimulationResult
from backend.physics.fuel_model import get_circuit_fuel_calibration, update_fuel_state
from backend.physics.lap_time_predictor import LapTimePredictor
from backend.physics.pit_model import DEFAULT_PIT_LOSS_S, get_circuit_pit_loss


def _parse_compound(comp: Union[str, TyreCompound]) -> TyreCompound:
    """Safely parse string or TyreCompound to TyreCompound enum."""
    if isinstance(comp, TyreCompound):
        return comp
    name = str(comp).upper()
    if name in TyreCompound.__members__:
        return TyreCompound[name]
    return TyreCompound(name)


def _normalize_pit_schedule(
    schedule: Optional[Dict],
) -> Dict[str, Dict[int, TyreCompound]]:
    """Normalize pit schedule into {driver: {lap_number: TyreCompound}}."""
    if not schedule:
        return {}
    normalized: Dict[str, Dict[int, TyreCompound]] = {}
    for driver, stops in schedule.items():
        normalized[driver] = {}
        if isinstance(stops, dict):
            for lap, comp in stops.items():
                normalized[driver][int(lap)] = _parse_compound(comp)
        elif isinstance(stops, (list, tuple)):
            for item in stops:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    lap, comp = item[0], item[1]
                    normalized[driver][int(lap)] = _parse_compound(comp)
                elif hasattr(item, "lap") and hasattr(item, "compound"):
                    normalized[driver][int(item.lap)] = _parse_compound(item.compound)
    return normalized


class Simulator:
    """Deterministic race simulator skeleton with pit stop support.

    Bridges RaceState and LapTimePredictor to advance cars lap-by-lap,
    execute scheduled pit stops, and produce validated SimulationResult instances.
    """

    def __init__(
        self,
        predictor: LapTimePredictor,
        race_state: Optional[RaceState] = None,
        fuel_consumption_per_lap: Optional[float] = None,
        pit_loss_s: Optional[float] = None,
        pit_stops: Optional[Dict[str, Any]] = None,
    ):
        self.predictor = predictor
        self.race_state = race_state
        self.fuel_consumption_per_lap = fuel_consumption_per_lap
        self.pit_loss_s = pit_loss_s
        self.pit_stops = _normalize_pit_schedule(pit_stops)

    def step_lap(
        self,
        race_state: Optional[RaceState] = None,
        pit_stops: Optional[Dict[str, Any]] = None,
    ) -> List[LapResult]:
        """Advance the race state by one lap for all active cars."""
        state = race_state or self.race_state
        if state is None:
            raise ValueError("RaceState must be provided to step_lap.")
        if state.is_finished:
            return []

        active_pit_stops = (
            _normalize_pit_schedule(pit_stops)
            if pit_stops is not None
            else self.pit_stops
        )

        lap_results: List[LapResult] = []
        burn_rate = self.fuel_consumption_per_lap
        if burn_rate is None:
            calib = get_circuit_fuel_calibration(state.circuit, total_laps=state.total_laps)
            if calib is not None:
                status_str = (
                    state.track_condition.track_status.value
                    if hasattr(state.track_condition.track_status, "value")
                    else str(state.track_condition.track_status)
                )
                is_sc_vsc = ("4" in status_str) or ("6" in status_str)
                intensity = calib.sc_vsc_work_ratio if is_sc_vsc else 1.0
                burn_rate = calib.nominal_burn_rate_kg_per_lap * intensity
            else:
                burn_rate = 1.8  # Documented fallback for uncalibrated circuit

        wetness = (
            state.track_condition.sector_wetness
            if state.track_condition.rainfall
            else None
        )

        for driver, car in state.cars.items():
            if car.is_retired:
                continue

            compound_name = (
                car.current_compound.value
                if hasattr(car.current_compound, "value")
                else str(car.current_compound)
            )

            # Check if pit stop is scheduled for this car on this lap
            driver_stops = active_pit_stops.get(driver, {})
            is_pit = state.current_lap in driver_stops

            # Predict lap time using current physical state
            lap_time = self.predictor.predict_lap_time(
                circuit=state.circuit,
                compound=compound_name,
                tyre_age=car.tyre_age,
                fuel_mass_kg=car.fuel_kg,
                sector_wetness=wetness,
            )

            if is_pit:
                new_compound = driver_stops[state.current_lap]
                pit_loss = (
                    self.pit_loss_s
                    if self.pit_loss_s is not None
                    else get_circuit_pit_loss(state.circuit)
                )
                lap_time += pit_loss
                car.total_time_s += lap_time
                car.current_compound = new_compound
                car.tyre_age = 1
                car.stint += 1
                car.pit_stops_count += 1
                car.in_pit = True
            else:
                car.total_time_s += lap_time
                car.tyre_age += 1
                car.in_pit = False

            car.fuel_kg = update_fuel_state(car.fuel_kg, burn_rate)

            lap_results.append(
                LapResult(
                    lap_number=state.current_lap,
                    driver=driver,
                    lap_time_s=lap_time,
                    compound=car.current_compound,
                    tyre_age=car.tyre_age,
                    fuel_kg=car.fuel_kg,
                    position=car.position,
                    track_status=state.track_condition.track_status.value
                    if hasattr(state.track_condition.track_status, "value")
                    else str(state.track_condition.track_status),
                    is_pit_lap=is_pit,
                )
            )

        # Update running positions and intervals
        active_cars = [c for c in state.cars.values() if not c.is_retired]
        active_cars.sort(key=lambda c: (c.total_time_s, c.position))
        retired_cars = [c for c in state.cars.values() if c.is_retired]
        retired_cars.sort(key=lambda c: c.position)

        leader_time = active_cars[0].total_time_s if active_cars else 0.0
        prev_time = leader_time

        for pos, car in enumerate(active_cars, start=1):
            car.position = pos
            car.gap_to_leader_s = car.total_time_s - leader_time
            car.gap_to_ahead_s = 0.0 if pos == 1 else (car.total_time_s - prev_time)
            prev_time = car.total_time_s

        for pos, car in enumerate(retired_cars, start=len(active_cars) + 1):
            car.position = pos
            car.gap_to_leader_s = 0.0
            car.gap_to_ahead_s = None

        # Synchronize positions in current lap results
        for res in lap_results:
            res.position = state.cars[res.driver].position

        if state.current_lap >= state.total_laps:
            state.is_finished = True
        else:
            state.current_lap += 1

        return lap_results

    def run(
        self,
        race_state: Optional[RaceState] = None,
        simulation_id: Optional[str] = None,
        pit_stops: Optional[Dict[str, Any]] = None,
    ) -> SimulationResult:
        """Run the simulation from current lap through race completion."""
        state = race_state or self.race_state
        if state is None:
            raise ValueError("RaceState must be provided to run.")

        active_pit_stops = (
            _normalize_pit_schedule(pit_stops)
            if pit_stops is not None
            else self.pit_stops
        )

        sim_id = simulation_id or f"sim_{uuid.uuid4().hex[:8]}"
        all_laps: List[LapResult] = []
        lap_history_records: List[Dict[str, Any]] = []

        # Track starting compounds for each driver
        used_compounds: Dict[str, List[TyreCompound]] = {
            d: [c.current_compound] for d, c in state.cars.items()
        }

        while not state.is_finished:
            current_lap_num = state.current_lap
            lap_results = self.step_lap(state, pit_stops=active_pit_stops)
            all_laps.extend(lap_results)

            for d, stops in active_pit_stops.items():
                if current_lap_num in stops:
                    used_compounds[d].append(stops[current_lap_num])

            # Preserve lap-by-lap timing and gap history records for strategy layer inspection
            for car in sorted(state.cars.values(), key=lambda c: c.position):
                lap_res = next((r for r in lap_results if r.driver == car.driver), None)
                if lap_res is not None:
                    lap_history_records.append({
                        "lap_number": current_lap_num,
                        "driver": car.driver,
                        "position": car.position,
                        "cumulative_time_s": car.total_time_s,
                        "gap_to_leader_s": car.gap_to_leader_s,
                        "gap_to_ahead_s": car.gap_to_ahead_s,
                        "compound": car.current_compound.value
                        if hasattr(car.current_compound, "value")
                        else str(car.current_compound),
                        "tyre_age": car.tyre_age,
                        "fuel_kg": car.fuel_kg,
                        "is_pit_lap": lap_res.is_pit_lap,
                    })

        # Build final driver classifications
        classification: List[DriverResult] = []
        sorted_cars = sorted(
            state.cars.values(),
            key=lambda c: (c.is_retired, c.total_time_s, c.position),
        )

        for pos, car in enumerate(sorted_cars, start=1):
            car_laps = [r for r in all_laps if r.driver == car.driver]
            driver_compounds = list(
                dict.fromkeys(used_compounds.get(car.driver, [car.current_compound]))
            )
            classification.append(
                DriverResult(
                    driver=car.driver,
                    team=car.team,
                    finish_position=pos,
                    total_time_s=car.total_time_s,
                    status="Retired" if car.is_retired else "Finished",
                    pit_stops=car.pit_stops_count,
                    compounds_used=driver_compounds,
                    laps=car_laps,
                )
            )

        leader_total_duration = (
            sorted_cars[0].total_time_s if sorted_cars else 0.0
        )

        return SimulationResult(
            simulation_id=sim_id,
            circuit=state.circuit,
            total_laps=state.total_laps,
            classification=classification,
            lap_history=all_laps,
            total_duration_s=leader_total_duration,
            metadata={"lap_history": lap_history_records},
        )
