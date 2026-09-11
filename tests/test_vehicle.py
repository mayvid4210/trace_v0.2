"""Unit tests for the VehicleModel interface and resistance calculations."""

import numpy as np
import pandas as pd
import pytest

from backend.physics.vehicle import VehicleModel


@pytest.fixture
def mock_curve_df():
    """Synthetic resistance curve data spanning 40 to 80 m/s."""
    return pd.DataFrame({
        "speed_ms": [40.0, 50.0, 60.0, 70.0, 80.0],
        "resistance_force_n": [500.0, 900.0, 1400.0, 2000.0, 2700.0],
        "samples": [30, 40, 50, 40, 30],
    })


def test_vehicle_air_density():
    """Verify air density physically scales with temperature at 1 atm."""
    # Create model with minimal mock curve
    curve = pd.DataFrame({"speed_ms": [50.0], "resistance_force_n": [1000.0]})
    model = VehicleModel(curve_data=curve)

    rho_cold = model.air_density(15.0)   # 15°C
    rho_hot = model.air_density(35.0)    # 35°C

    # Standard atmospheric density at sea level is ~1.225 kg/m^3 at 15°C
    assert 1.20 < rho_cold < 1.25
    assert 1.10 < rho_hot < 1.18
    # Warmer air is less dense
    assert rho_cold > rho_hot


def test_vehicle_resistance_force_interpolation(mock_curve_df):
    """Verify resistance force interpolates smoothly and clamps at boundaries."""
    model = VehicleModel(curve_data=mock_curve_df)

    # Exact point in curve
    assert np.isclose(model.resistance_force(50.0), 900.0)

    # Midpoint linear interpolation: halfway between 50 m/s (900N) and 60 m/s (1400N)
    assert np.isclose(model.resistance_force(55.0), 1150.0)

    # Below minimum speed (clamps to lowest known value flat)
    assert np.isclose(model.resistance_force(20.0), 500.0)

    # Above maximum speed (clamps to highest known value flat)
    assert np.isclose(model.resistance_force(100.0), 2700.0)


def test_vehicle_calculate_output_structure(mock_curve_df):
    """Verify calculate() returns expected structure, units, and reliability flags."""
    model = VehicleModel(curve_data=mock_curve_df)

    # 180 km/h = 50 m/s (within 40-80 m/s range)
    result = model.calculate(
        speed_kmh=180.0,
        air_temperature_c=25.0,
        mass_kg=850.0,
    )

    assert isinstance(result, dict)
    assert result["mass_kg"] == 850.0
    assert 1.15 < result["air_density"] < 1.22
    assert np.isclose(result["resistance_force_n"], 900.0)
    assert result["in_reliable_range"] is True

    # 100 km/h = 27.8 m/s (below reliable range)
    result_slow = model.calculate(
        speed_kmh=100.0,
        air_temperature_c=25.0,
        mass_kg=850.0,
    )
    assert result_slow["in_reliable_range"] is False
