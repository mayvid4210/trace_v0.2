"""Unit tests for sector grip calculation and TrackState structure building."""

import numpy as np
import pandas as pd
import pytest

from backend.physics.track_state import (
    SECTORS,
    build_track_state,
    compute_sector_grip,
)


@pytest.fixture
def sample_track_df():
    """Synthetic sector times and wetness proxies."""
    return pd.DataFrame({
        "GrandPrix": ["Canada", "Canada"],
        "SessionName": ["Race", "Race"],
        "Driver": ["HAM", "HAM"],
        "LapNumber": [1, 2],
        "sector_1_s": [25.0, 24.5],
        "sector_2_s": [28.0, 27.5],
        "sector_3_s": [30.0, 29.5],
        "sector_1_wetness_proxy": [0.0, 0.0],       # Dry sector 1
        "sector_2_wetness_proxy": [2.0, 1.5],       # Wet sector 2
        "sector_3_wetness_proxy": [3.0, 2.0],       # Wet sector 3
        "TrackTemp": [22.0, 22.5],
        "Rainfall": [True, True],
    })


def test_compute_sector_grip(sample_track_df):
    """Verify sector grip index equals 1.0 when dry and is < 1.0 when wet."""
    df = compute_sector_grip(sample_track_df)

    for sector in SECTORS:
        assert f"{sector}_grip" in df.columns

    # Dry sector 1 should have grip == 1.0
    assert np.allclose(df["sector_1_grip"], 1.0)

    # Wet sectors 2 and 3 should have grip < 1.0
    assert (df["sector_2_grip"] < 1.0).all()
    assert (df["sector_3_grip"] < 1.0).all()

    # Lap 2 (drying: wetness dropped from 2.0 to 1.5) should have higher grip than Lap 1
    assert df["sector_2_grip"].iloc[1] > df["sector_2_grip"].iloc[0]


def test_build_track_state(sample_track_df):
    """Verify build_track_state produces complete structured track_state records."""
    df = compute_sector_grip(sample_track_df)

    drying_results = {
        ("Canada", "Race", "sector_2"): {"drying_rate_s_per_lap": -0.085},
        ("Canada", "Race", "sector_3"): {"drying_rate_s_per_lap": -0.090},
    }

    df_state = build_track_state(df, drying_results)
    assert "track_state" in df_state.columns

    record = df_state.iloc[0]["track_state"]
    assert isinstance(record, dict)
    assert record["temperature"] == 22.0
    assert record["rainfall"] is True
    assert "grip" in record
    assert set(record["grip"].keys()) == {"sector_1", "sector_2", "sector_3"}
    assert record["grip"]["sector_1"] == 1.0
    assert len(record["sector_wetness"]) == 3
    assert record["sector_wetness"] == [0.0, 2.0, 3.0]
    assert record["drying_rate"]["sector_2"] == -0.085
    assert np.isnan(record["drying_rate"]["sector_1"])  # No drying rate was fitted for dry sector
    assert record["water_depth"] is None
