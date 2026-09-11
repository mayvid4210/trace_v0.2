"""Simulation output and result data structures."""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from backend.models.race_state import TyreCompound


class LapResult(BaseModel):
    """Output metrics for a single completed lap by a single car."""

    lap_number: int
    driver: str
    lap_time_s: float
    sector_times_s: Optional[List[float]] = None
    compound: TyreCompound = TyreCompound.MEDIUM
    tyre_age: int = 1
    fuel_kg: float = 110.0
    position: int = 1
    track_status: str = "1"
    is_pit_lap: bool = False


class DriverResult(BaseModel):
    """Cumulative race outcome for a single driver."""

    driver: str
    team: Optional[str] = None
    finish_position: int
    grid_position: Optional[int] = None
    total_time_s: float
    status: str = "Finished"
    points: int = 0
    pit_stops: int = 0
    compounds_used: List[TyreCompound] = Field(default_factory=list)
    laps: List[LapResult] = Field(default_factory=list)


class SimulationResult(BaseModel):
    """Complete output record for a forward race simulation run."""

    simulation_id: str
    circuit: str
    total_laps: int
    classification: List[DriverResult] = Field(default_factory=list)
    lap_history: List[LapResult] = Field(default_factory=list)
    total_duration_s: float = 0.0
    metadata: Dict = Field(default_factory=dict)
