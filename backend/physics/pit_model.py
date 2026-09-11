"""Circuit-specific pit stop loss calibration and configuration.

Replaces the universal 22.0s assumption with statistically calibrated
race-time losses for Bahrain and Canada derived from 2024 Grand Prix telemetry/timing.
"""

from dataclasses import dataclass
from typing import Dict, Optional


DEFAULT_PIT_LOSS_S: float = 22.0


@dataclass(frozen=True)
class CircuitPitLossCalibration:
    """Statistical calibration record for a circuit's race-time pit loss."""

    circuit: str
    pit_loss_s: float
    sample_count: int
    mean_s: float
    std_s: float
    iqr_s: float
    stationary_duration_s: float
    source: str
    limitations: str
    notes: str = ""


CIRCUIT_PIT_LOSS_CALIBRATIONS: Dict[str, CircuitPitLossCalibration] = {
    "Bahrain": CircuitPitLossCalibration(
        circuit="Bahrain",
        pit_loss_s=24.1,
        sample_count=34,
        mean_s=24.370,
        std_s=1.120,
        iqr_s=1.590,
        stationary_duration_s=24.769,
        source="STATISTICALLY_CALIBRATED_FROM_RACE_DATA",
        limitations="Calibrated from 2024 dry race under green flag conditions. Excludes slow stops and yellow flag phases.",
        notes="34 usable clean green-flag pit events after 1.5x IQR filter (median=24.090s, 9 excluded).",
    ),
    "Canada": CircuitPitLossCalibration(
        circuit="Canada",
        pit_loss_s=24.7,
        sample_count=15,
        mean_s=24.961,
        std_s=1.023,
        iqr_s=1.837,
        stationary_duration_s=24.458,
        source="STATISTICALLY_CALIBRATED_FROM_RACE_DATA",
        limitations="Calibrated from 15 green-flag pit events in 2024. Excludes 23 Safety Car stops and 4 outliers.",
        notes="Clean green-flag pit loss is 24.7s. Pitting during SC queues incurs lower relative loss (~12-15s).",
    ),
}


def get_circuit_pit_loss(circuit: Optional[str] = None) -> float:
    """Resolve circuit-specific pit loss in seconds, falling back to DEFAULT_PIT_LOSS_S.

    Resolution order:
    1. Circuit-specific calibration if present in registry.
    2. DEFAULT_PIT_LOSS_S (22.0s) fallback for uncalibrated circuits or None.
    """
    if not circuit:
        return DEFAULT_PIT_LOSS_S

    calib = CIRCUIT_PIT_LOSS_CALIBRATIONS.get(str(circuit))
    if calib is not None:
        return calib.pit_loss_s

    # Circuit name case-insensitive matching fallback
    for key, c in CIRCUIT_PIT_LOSS_CALIBRATIONS.items():
        if key.lower() == str(circuit).lower():
            return c.pit_loss_s

    return DEFAULT_PIT_LOSS_S


def get_circuit_pit_calibration(circuit: Optional[str] = None) -> Optional[CircuitPitLossCalibration]:
    """Retrieve full calibration metadata for a given circuit, if available."""
    if not circuit:
        return None
    for key, c in CIRCUIT_PIT_LOSS_CALIBRATIONS.items():
        if key.lower() == str(circuit).lower():
            return c
    return None
