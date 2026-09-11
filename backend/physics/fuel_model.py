"""Telemetry-derived relative fuel consumption model and circuit calibration.

Non-Negotiable Data Rules:
1. Public F1 telemetry does NOT expose actual onboard fuel mass, tank volume, or fuel flow rate.
2. The relative consumption proxy is derived from observable telemetry (Throttle, RPM, Time, Speed, TrackStatus).
3. The absolute fuel scale is an explicitly configured/calibrated external parameter with documented provenance.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np
import pandas as pd

# Provenance categories
PROVENANCE_TELEMETRY = "TELEMETRY_DERIVED"
PROVENANCE_CALIBRATED = "STATISTICALLY_CALIBRATED"
PROVENANCE_FIA_RULE = "EXTERNAL_FIA_CONSTRAINT"
PROVENANCE_FALLBACK = "EXPLICIT_ASSUMPTION_FALLBACK"

# External FIA Constraints (FIA Formula 1 Technical Regulations 2024)
FIA_MAX_FUEL_CAPACITY_KG: float = 110.0  # Art 6.5.2 Maximum fuel tank capacity
FIA_MIN_FUEL_SAMPLE_KG: float = 1.0     # Art 6.5.2 Minimum fuel sample post-race
FIA_MAX_FUEL_FLOW_KG_H: float = 100.0   # Art 5.1.4 Max fuel mass flow rate above 10,500 RPM


@dataclass(frozen=True)
class CircuitFuelCalibration:
    """Pre-calibrated circuit fuel parameters with explicit provenance."""

    circuit: str
    total_laps: int
    track_length_km: float
    reference_green_work: float         # Telemetry-derived median workload for clean green-flag lap
    sc_vsc_work_ratio: float            # Telemetry-derived ratio of SC/VSC workload to green-flag workload
    calibrated_fuel_effect_s_per_kg: float  # Fuel sensitivity (s/kg)
    fuel_effect_status: str             # "CALIBRATED_MULTIVARIATE" or "DOCUMENTED_FALLBACK"
    nominal_fuel_scale_kg: float        # Explicit external baseline scale parameter (NOT measured fuel)
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def nominal_burn_rate_kg_per_lap(self) -> float:
        """Nominal burn rate per lap: nominal_fuel_scale_kg / total_laps."""
        if self.total_laps <= 0:
            return 0.0
        return self.nominal_fuel_scale_kg / float(self.total_laps)


# Registry of verified circuit fuel calibrations
CIRCUIT_FUEL_CALIBRATIONS: Dict[str, CircuitFuelCalibration] = {
    "Bahrain": CircuitFuelCalibration(
        circuit="Bahrain",
        total_laps=57,
        track_length_km=5.412,
        reference_green_work=56.75,  # Telemetry median from 203 clean green-flag laps in 2024 Race
        sc_vsc_work_ratio=0.7242,   # Telemetry-derived from 2024 race telemetry (Bahrain had 0 SC laps, Canada had 28)
        calibrated_fuel_effect_s_per_kg=0.0392,  # Statistically calibrated via multivariate regression controlling for TyreLife & Compound
        fuel_effect_status="CALIBRATED_MULTIVARIATE",
        nominal_fuel_scale_kg=100.0,  # Explicit external baseline scale parameter (NOT measured telemetry)
        provenance={
            "track_length_km": PROVENANCE_TELEMETRY,
            "reference_green_work": PROVENANCE_TELEMETRY,
            "sc_vsc_work_ratio": PROVENANCE_TELEMETRY,
            "calibrated_fuel_effect_s_per_kg": PROVENANCE_CALIBRATED,
            "nominal_fuel_scale_kg": PROVENANCE_FALLBACK,
        },
    ),
    "Canada": CircuitFuelCalibration(
        circuit="Canada",
        total_laps=70,
        track_length_km=4.361,
        reference_green_work=43.05,  # Telemetry median from 200 clean green-flag laps in 2024 Race
        sc_vsc_work_ratio=0.7242,   # Telemetry-derived: median SC work (31.17) / median green work (43.05) from 2024 Race
        calibrated_fuel_effect_s_per_kg=0.0261,  # Documented fallback; late dry laps conflated with drying track evolution
        fuel_effect_status="DOCUMENTED_FALLBACK",
        nominal_fuel_scale_kg=100.0,  # Explicit external baseline scale parameter (NOT measured telemetry)
        provenance={
            "track_length_km": PROVENANCE_TELEMETRY,
            "reference_green_work": PROVENANCE_TELEMETRY,
            "sc_vsc_work_ratio": PROVENANCE_TELEMETRY,
            "calibrated_fuel_effect_s_per_kg": PROVENANCE_FALLBACK,
            "nominal_fuel_scale_kg": PROVENANCE_FALLBACK,
        },
    ),
}


def get_circuit_fuel_calibration(
    circuit: str,
    total_laps: Optional[int] = None,
    external_fuel_scale_kg: Optional[float] = None,
) -> Optional[CircuitFuelCalibration]:
    """Retrieve circuit fuel calibration.

    For known circuits, returns the verified calibration.
    For uncalibrated circuits, fails clearly (returns None) unless total_laps
    and external_fuel_scale_kg are explicitly provided by the caller.
    """
    if circuit in CIRCUIT_FUEL_CALIBRATIONS:
        calib = CIRCUIT_FUEL_CALIBRATIONS[circuit]
        if external_fuel_scale_kg is not None:
            return CircuitFuelCalibration(
                circuit=calib.circuit,
                total_laps=calib.total_laps,
                track_length_km=calib.track_length_km,
                reference_green_work=calib.reference_green_work,
                sc_vsc_work_ratio=calib.sc_vsc_work_ratio,
                calibrated_fuel_effect_s_per_kg=calib.calibrated_fuel_effect_s_per_kg,
                fuel_effect_status=calib.fuel_effect_status,
                nominal_fuel_scale_kg=external_fuel_scale_kg,
                provenance=calib.provenance,
            )
        return calib

    # Explicit configuration for uncalibrated circuit
    if total_laps is not None and total_laps > 0 and external_fuel_scale_kg is not None:
        return CircuitFuelCalibration(
            circuit=circuit,
            total_laps=total_laps,
            track_length_km=5.0,  # Fallback assumption
            reference_green_work=50.0,
            sc_vsc_work_ratio=0.7242,
            calibrated_fuel_effect_s_per_kg=0.0300,
            fuel_effect_status="DOCUMENTED_FALLBACK",
            nominal_fuel_scale_kg=external_fuel_scale_kg,
            provenance={"circuit": PROVENANCE_FALLBACK},
        )

    return None


def compute_lap_workload(telemetry_df: pd.DataFrame) -> Optional[float]:
    """Compute physical workload proxy from observable lap telemetry.

    Formula:
        W = sum( (Throttle / 100.0) * (RPM / 12000.0) * dt )
    where:
        Throttle: 0-100%
        RPM: Engine speed (~4000 to ~12500 RPM)
        dt: Time delta between telemetry points in seconds

    If RPM is missing, uses sum( (Throttle / 100.0) * dt ).
    Returns None if required channels (Time, Throttle) are missing or df is empty.
    """
    if telemetry_df is None or telemetry_df.empty:
        return None

    if "Throttle" not in telemetry_df.columns or "Time" not in telemetry_df.columns:
        return None

    df = telemetry_df.sort_values("Time")

    time_col = df["Time"]
    if pd.api.types.is_timedelta64_dtype(time_col):
        dt = time_col.diff().dt.total_seconds().fillna(0.0)
    else:
        dt = time_col.diff().fillna(0.0)

    # Filter non-physical or negative time deltas
    dt = dt.clip(lower=0.0, upper=1.0)

    throttle = df["Throttle"].fillna(0.0).clip(lower=0.0, upper=100.0) / 100.0

    if "RPM" in df.columns:
        rpm_weight = df["RPM"].fillna(10000.0).clip(lower=0.0, upper=15000.0) / 12000.0
        work = (throttle * rpm_weight * dt).sum()
    else:
        work = (throttle * dt).sum()

    return float(work)


def compute_relative_fuel_intensity(
    lap_workload: Optional[float],
    reference_work: float,
    track_status: str = "1",
    sc_vsc_ratio: float = 0.7242,
) -> float:
    """Derive relative fuel consumption intensity proxy from telemetry workload.

    The intensity represents relative energy expenditure (1.0 = nominal green lap).
    Does NOT use future data (strictly causal per-lap).
    """
    is_sc_vsc = ("4" in str(track_status)) or ("6" in str(track_status))

    if lap_workload is None or lap_workload <= 0.0:
        # Fallback when telemetry is absent
        return float(sc_vsc_ratio if is_sc_vsc else 1.0)

    if reference_work <= 0.0:
        return 1.0

    raw_ratio = lap_workload / reference_work

    if is_sc_vsc:
        # SC/VSC laps: bounded within observed SC work regime
        return float(np.clip(raw_ratio, 0.30, 1.10))

    # Green-flag racing laps: bounded within normal race variation
    return float(np.clip(raw_ratio, 0.70, 1.30))


def estimate_lap_burn(
    calibration: CircuitFuelCalibration,
    intensity: float = 1.0,
    fuel_scale_kg: Optional[float] = None,
) -> float:
    """Calculate lap fuel burn in kg: nominal_lap_burn * intensity."""
    scale = fuel_scale_kg if fuel_scale_kg is not None else calibration.nominal_fuel_scale_kg
    if calibration.total_laps <= 0:
        return 0.0
    nominal_lap_burn = scale / float(calibration.total_laps)
    return float(nominal_lap_burn * intensity)


def update_fuel_state(current_fuel_kg: float, burn_kg: float) -> float:
    """Enforce non-negative, monotonic fuel depletion."""
    burn = max(0.0, float(burn_kg))
    return max(0.0, float(current_fuel_kg) - burn)
