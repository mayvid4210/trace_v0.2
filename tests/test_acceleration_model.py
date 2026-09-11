"""Unit tests for the AccelerationPredictor interface and joblib serialization."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from backend.physics.acceleration_model import (
    FEATURES,
    AccelerationPredictor,
    load_model,
    save_model,
)


@pytest.fixture
def fitted_dummy_model():
    """A small lightweight RandomForest fitted on 10 synthetic samples."""
    np.random.seed(42)
    X = pd.DataFrame(
        np.random.randn(10, len(FEATURES)),
        columns=FEATURES,
    )
    y = np.random.randn(10)

    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X, y)
    return model


def test_save_and_load_model(fitted_dummy_model, tmp_path):
    """Verify save_model serializes with joblib and load_model restores it correctly."""
    model_file = tmp_path / "test_model.joblib"

    saved_path = save_model(fitted_dummy_model, filepath=model_file)
    assert saved_path.exists()

    loaded = load_model(saved_path)
    assert hasattr(loaded, "predict")

    # Generate dummy input and check prediction equality
    test_sample = pd.DataFrame(
        np.zeros((1, len(FEATURES))),
        columns=FEATURES,
    )
    orig_pred = fitted_dummy_model.predict(test_sample)
    loaded_pred = loaded.predict(test_sample)
    assert np.allclose(orig_pred, loaded_pred)


def test_acceleration_predictor_with_dataframe(fitted_dummy_model):
    """Verify AccelerationPredictor predicts on a DataFrame input."""
    predictor = AccelerationPredictor(model_or_path=fitted_dummy_model)

    df_input = pd.DataFrame({
        "Speed_ms": [50.0, 60.0],
        "RPM": [10500, 11000],
        "nGear": [6, 7],
        "Throttle": [100.0, 100.0],
        "Brake": [0.0, 0.0],
        "DRS": [1.0, 1.0],
        "total_mass_kg": [850.0, 848.0],
        "track_temperature": [30.0, 30.5],
        "air_temperature": [25.0, 25.2],
        "tyre_age": [5, 6],
        "track_grip": [0.98, 0.97],
    })

    predictions = predictor.predict(df_input)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == 2


def test_acceleration_predictor_with_dict(fitted_dummy_model):
    """Verify AccelerationPredictor predicts on a single dict input."""
    predictor = AccelerationPredictor(model_or_path=fitted_dummy_model)

    sample_dict = {feat: 1.0 for feat in FEATURES}
    prediction = predictor.predict(sample_dict)

    assert isinstance(prediction, np.ndarray)
    assert len(prediction) == 1


def test_acceleration_predictor_missing_features(fitted_dummy_model):
    """Verify AccelerationPredictor raises ValueError on missing features."""
    predictor = AccelerationPredictor(model_or_path=fitted_dummy_model)

    incomplete_dict = {"Speed_ms": 50.0, "RPM": 10500}
    with pytest.raises(ValueError, match="Missing required feature columns"):
        predictor.predict(incomplete_dict)


def test_load_nonexistent_model_raises(tmp_path):
    """Verify FileNotFoundError when target joblib path does not exist."""
    missing_file = tmp_path / "non_existent.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(missing_file)
