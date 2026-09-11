from pathlib import Path
import numpy as np
import pandas as pd

try:
    from config import resolve_path, PROCESSED_DATA_DIR
except ModuleNotFoundError:
    try:
        from backend.config import resolve_path, PROCESSED_DATA_DIR
    except ModuleNotFoundError:
        PROJECT_ROOT = Path(__file__).resolve().parents[2]
        PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

        def resolve_path(p):
            p = Path(p)
            return p if p.is_absolute() else PROJECT_ROOT / p


CURVE_FILE = PROCESSED_DATA_DIR / "resistance_curve.parquet"


class VehicleModel:
    """
    Longitudinal resistance model.

    IMPORTANT — what this model actually represents:
    This is an EMPIRICAL combined-resistance curve derived from clean,
    sustained coast-down telemetry (Throttle=0, Brake=False, DRS=0,
    top gear, runs of 5+ consecutive samples with first/last dropped).

    It represents aerodynamic drag + rolling resistance + hybrid
    energy-harvest deceleration COMBINED. We do not have ERS/MGU-K
    telemetry, so these effects cannot be separated. Attempting to fit
    a pure Cd/Crr physics equation to this data produced negative R²
    and a Crr pinned at unphysical values (~0.3 vs a real ~0.01-0.02),
    because harvest deceleration is power-limited (~1/v), not v²-limited
    like aero drag, and no combination of Cd/Crr can reproduce both
    effects at once.

    Curve is reliable roughly 43-82 m/s (~155-295 km/h), the range with
    enough sustained coasting samples (>=5 per bin) to trust the median.
    Above/below that range, the model holds the nearest known value
    flat rather than extrapolating — treat predictions outside this
    range with caution.
    """

    def __init__(self, curve_file=CURVE_FILE, pressure_pa=101325.0, curve_data=None):
        self.pressure_pa = pressure_pa
        self.air_gas_constant = 287.05

        if curve_data is not None:
            curve = curve_data
        else:
            resolved_path = resolve_path(curve_file)
            curve = pd.read_parquet(resolved_path)

        self.curve_speed = curve["speed_ms"].to_numpy()
        self.curve_resistance = curve["resistance_force_n"].to_numpy()
        self.curve_min_speed = self.curve_speed.min()
        self.curve_max_speed = self.curve_speed.max()

    def air_density(self, air_temperature_c):
        temperature_k = air_temperature_c + 273.15
        return self.pressure_pa / (self.air_gas_constant * temperature_k)

    def resistance_force(self, speed_ms):
        # np.interp already holds the boundary values flat outside
        # curve_speed's range - this is intentional, see class docstring.
        return np.interp(speed_ms, self.curve_speed, self.curve_resistance)

    def calculate(self, speed_kmh, air_temperature_c, mass_kg):
        speed_ms = speed_kmh / 3.6
        rho = self.air_density(air_temperature_c)
        resistance = self.resistance_force(speed_ms)

        in_range = bool(self.curve_min_speed <= speed_ms <= self.curve_max_speed)

        return {
            "mass_kg": mass_kg,
            "air_density": rho,
            "resistance_force_n": resistance,
            "in_reliable_range": in_range,
        }