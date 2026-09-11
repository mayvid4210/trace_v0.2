"""Shared simulation race state data models."""

from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class TrackStatus(str, Enum):
    CLEAR = "1"
    YELLOW = "2"
    SAFETY_CAR = "4"
    RED = "5"
    VSC = "6"


class TyreCompound(str, Enum):
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"


class TrackCondition(BaseModel):
    """Environmental and circuit surface state at a point in time."""

    air_temp_c: float = 25.0
    track_temp_c: float = 35.0
    rainfall: bool = False
    track_status: TrackStatus = TrackStatus.CLEAR
    sector_wetness: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    track_grip: float = 1.0


class CarState(BaseModel):
    """Simulation state for a single car/driver at a given lap."""

    driver: str
    team: Optional[str] = None
    position: int = 1
    current_compound: TyreCompound = TyreCompound.MEDIUM
    tyre_age: int = 0
    stint: int = 1
    fuel_kg: float = 110.0
    total_time_s: float = 0.0
    gap_to_leader_s: float = 0.0
    gap_to_ahead_s: Optional[float] = None
    in_pit: bool = False
    is_retired: bool = False
    pit_stops_count: int = 0


class RaceState(BaseModel):
    """Top-level simulation state for the entire race at a given step."""

    circuit: str
    total_laps: int
    current_lap: int = 1
    cars: Dict[str, CarState] = Field(default_factory=dict)
    track_condition: TrackCondition = Field(default_factory=TrackCondition)
    is_finished: bool = False
