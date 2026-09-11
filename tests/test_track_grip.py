"""Unit tests verifying causality, bounds, and session isolation for calculate_track_grip."""

import ast
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Load calculate_track_grip directly from backend/feature_engineering.py source
# to avoid executing top-level pipeline script statements in feature_engineering.py
fe_path = Path(__file__).resolve().parent.parent / "backend" / "feature_engineering.py"
tree = ast.parse(fe_path.read_text(encoding="utf-8"))
func_def = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "calculate_track_grip")
module = ast.Module(body=[func_def], type_ignores=[])
code = compile(module, str(fe_path), "exec")
_ns = {"pd": pd, "np": np}
exec(code, _ns)
calculate_track_grip = _ns["calculate_track_grip"]


@pytest.fixture
def sample_laps():
    """Synthetic lap time dataframe across multiple laps and drivers."""
    return pd.DataFrame({
        "GrandPrix": ["Bahrain"] * 6,
        "SessionName": ["Race"] * 6,
        "Driver": ["VER", "HAM", "VER", "HAM", "VER", "HAM"],
        "LapNumber": [1, 1, 2, 2, 3, 3],
        "LapTimeSeconds": [95.0, 96.0, 94.0, 94.5, 98.0, 99.0],
    })


def test_track_grip_causality():
    """Verify earlier laps' track_grip values are strictly invariant to future lap times.

    If a future lap achieves an extremely fast time, earlier laps must not change.
    """
    initial_laps = pd.DataFrame({
        "GrandPrix": ["Bahrain"] * 4,
        "SessionName": ["Race"] * 4,
        "Driver": ["VER", "HAM", "VER", "HAM"],
        "LapNumber": [1, 1, 2, 2],
        "LapTimeSeconds": [95.0, 96.0, 94.0, 94.5],
    })

    result_initial = calculate_track_grip(initial_laps)
    lap1_grip_before = result_initial[result_initial["LapNumber"] == 1]["track_grip"].values
    lap2_grip_before = result_initial[result_initial["LapNumber"] == 2]["track_grip"].values

    # Now add Lap 3 with a drastically faster time that would have leaked in the old session_best logic
    future_laps = pd.concat([
        initial_laps,
        pd.DataFrame({
            "GrandPrix": ["Bahrain"] * 2,
            "SessionName": ["Race"] * 2,
            "Driver": ["VER", "HAM"],
            "LapNumber": [3, 3],
            "LapTimeSeconds": [70.0, 71.0],  # Super-fast future lap
        })
    ], ignore_index=True)

    result_with_future = calculate_track_grip(future_laps)
    lap1_grip_after = result_with_future[result_with_future["LapNumber"] == 1]["track_grip"].values
    lap2_grip_after = result_with_future[result_with_future["LapNumber"] == 2]["track_grip"].values

    # Laps 1 and 2 must have identical track_grip values regardless of future Lap 3
    assert np.allclose(lap1_grip_before, lap1_grip_after)
    assert np.allclose(lap2_grip_before, lap2_grip_after)


def test_track_grip_bounds_and_values(sample_laps):
    """Verify track_grip stays within [0.0, 1.0] and scales with relative pace."""
    result = calculate_track_grip(sample_laps)

    assert "track_grip" in result.columns
    valid_grip = result["track_grip"].dropna()

    assert (valid_grip >= 0.0).all()
    assert (valid_grip <= 1.0).all()

    # Lap 1: best pace so far is 95.0, lap 1 best is 95.0 -> 1.0
    lap1 = result[result["LapNumber"] == 1]
    assert np.allclose(lap1["track_grip"], 1.0)

    # Lap 2: lap 2 best is 94.0 (new best so far 94.0) -> 94.0 / 94.0 = 1.0
    lap2 = result[result["LapNumber"] == 2]
    assert np.allclose(lap2["track_grip"], 1.0)

    # Lap 3: best lap so far is 94.0, but lap 3 best is 98.0 -> 94.0 / 98.0 approx 0.9592
    lap3 = result[result["LapNumber"] == 3]
    expected_lap3_grip = 94.0 / 98.0
    assert np.allclose(lap3["track_grip"], expected_lap3_grip)


def test_track_grip_session_isolation():
    """Verify track_grip is calculated independently per (GrandPrix, SessionName)."""
    df = pd.DataFrame({
        "GrandPrix": ["Bahrain"] * 2 + ["Canada"] * 2,
        "SessionName": ["FP1"] * 2 + ["Race"] * 2,
        "Driver": ["VER", "VER", "VER", "VER"],
        "LapNumber": [1, 2, 1, 2],
        "LapTimeSeconds": [95.0, 93.0, 78.0, 80.0],
    })

    result = calculate_track_grip(df)

    bahrain = result[result["GrandPrix"] == "Bahrain"]
    canada = result[result["GrandPrix"] == "Canada"]

    assert np.allclose(bahrain[bahrain["LapNumber"] == 1]["track_grip"], 1.0)
    assert np.allclose(bahrain[bahrain["LapNumber"] == 2]["track_grip"], 1.0)

    assert np.allclose(canada[canada["LapNumber"] == 1]["track_grip"], 1.0)
    assert np.allclose(canada[canada["LapNumber"] == 2]["track_grip"], 78.0 / 80.0)


def test_track_grip_missing_lap_time():
    """Verify NaN lap times do not cause failure and result in NaN grip."""
    df = pd.DataFrame({
        "GrandPrix": ["Bahrain"] * 3,
        "SessionName": ["Race"] * 3,
        "Driver": ["VER"] * 3,
        "LapNumber": [1, 2, 3],
        "LapTimeSeconds": [95.0, np.nan, 93.0],
    })

    result = calculate_track_grip(df)
    assert pd.isna(result.loc[result["LapNumber"] == 2, "track_grip"].iloc[0])
    assert result.loc[result["LapNumber"] == 1, "track_grip"].iloc[0] == 1.0
    assert result.loc[result["LapNumber"] == 3, "track_grip"].iloc[0] == 1.0
