"""Unit tests for tyre wear proxy, grip, and temperature proxy functions."""

import numpy as np
import pandas as pd
import pytest

from backend.physics.tyre import (
    attach_ambient_temp_proxy,
    compute_tyre_grip,
    compute_wear_proxy,
)


@pytest.fixture
def sample_stint_laps():
    """Synthetic race lap times across two stints."""
    return pd.DataFrame({
        "GrandPrix": ["Bahrain"] * 6,
        "SessionName": ["Race"] * 6,
        "Driver": ["VER"] * 6,
        "Stint": [1, 1, 1, 2, 2, 2],
        "TyreLife": [1, 2, 3, 1, 2, 3],
        "LapTime_s": [92.0, 92.5, 93.1, 91.0, 91.4, 91.9],
        "Compound": ["SOFT"] * 3 + ["MEDIUM"] * 3,
        "total_mass_kg": [890.0, 888.0, 886.0, 850.0, 848.0, 846.0],
        "AirTemp": [24.5] * 6,
        "TrackTemp": [32.0] * 6,
    })


def test_compute_wear_proxy(sample_stint_laps):
    """Verify wear is measured relative to each stint's fastest lap."""
    df = compute_wear_proxy(sample_stint_laps)

    assert "wear" in df.columns
    assert "stint_best_laptime_s" in df.columns

    # Stint 1 best is 92.0s
    stint_1 = df[df["Stint"] == 1]
    assert np.isclose(stint_1.iloc[0]["wear"], 0.0)
    assert np.isclose(stint_1.iloc[1]["wear"], 0.5)
    assert np.isclose(stint_1.iloc[2]["wear"], 1.1)

    # Stint 2 best is 91.0s
    stint_2 = df[df["Stint"] == 2]
    assert np.isclose(stint_2.iloc[0]["wear"], 0.0)
    assert np.isclose(stint_2.iloc[1]["wear"], 0.4)
    assert np.isclose(stint_2.iloc[2]["wear"], 0.9)

    # Wear must always be non-negative
    assert (df["wear"] >= 0.0).all()


def test_compute_tyre_grip(sample_stint_laps):
    """Verify tyre grip decays with tyre life for validated compounds."""
    degradation_results = {
        "SOFT": {"tyre_degradation_s_per_lap": 0.0627},
        "MEDIUM": {"tyre_degradation_s_per_lap": 0.0447},
    }

    df = compute_tyre_grip(sample_stint_laps, degradation_results)
    assert "grip" in df.columns

    soft_laps = df[df["Compound"] == "SOFT"]
    grip_values = soft_laps["grip"].to_numpy()

    # Grip should be close to 1.0 on lap 1
    assert 0.95 < grip_values[0] <= 1.0
    # Grip must strictly decrease as tyre life increases
    assert grip_values[0] > grip_values[1] > grip_values[2]


def test_attach_ambient_temp_proxy(sample_stint_laps):
    """Verify weather temperatures are attached to proxy columns."""
    df = attach_ambient_temp_proxy(sample_stint_laps)

    assert "ambient_temp_proxy" in df.columns
    assert "track_temp_proxy" in df.columns
    assert (df["ambient_temp_proxy"] == 24.5).all()
    assert (df["track_temp_proxy"] == 32.0).all()


def test_attach_ambient_temp_proxy_missing_columns():
    """Verify descriptive error when weather columns are missing."""
    invalid_df = pd.DataFrame({"Driver": ["VER"]})
    with pytest.raises(ValueError, match="AirTemp/TrackTemp not found"):
        attach_ambient_temp_proxy(invalid_df)
