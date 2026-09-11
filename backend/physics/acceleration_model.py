from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

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


DEFAULT_MODEL_PATH = PROCESSED_DATA_DIR / "acceleration_model.joblib"

FEATURES = [
    "Speed_ms",
    "RPM",
    "nGear",
    "Throttle",
    "Brake",
    "DRS",
    "total_mass_kg",
    "track_temperature",
    "air_temperature",
    "tyre_age",
    "track_grip",
]


def prepare_data(df):
    df = df.copy()
    df = df.dropna(subset=FEATURES + ["Acceleration_smoothed"])
    return df


def train_model(df, save_path=None):
    X = df[FEATURES]
    y = df["Acceleration_smoothed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    mae = mean_absolute_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    if save_path is not None:
        save_model(model, save_path)

    return model, mae, r2


def save_model(model, filepath=DEFAULT_MODEL_PATH) -> Path:
    """Save a trained acceleration model to disk using joblib."""
    path = resolve_path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(filepath=DEFAULT_MODEL_PATH):
    """Load a persisted acceleration model from disk using joblib."""
    path = resolve_path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Acceleration model not found at {path}")
    return joblib.load(path)


class AccelerationPredictor:
    """Inference interface for the acceleration ML model.

    Allows simulation code to load a pre-trained model and predict
    acceleration without retraining.
    """

    def __init__(self, model_or_path=DEFAULT_MODEL_PATH):
        if isinstance(model_or_path, (str, Path)):
            self.model = load_model(model_or_path)
            self.model_path = resolve_path(model_or_path)
        else:
            self.model = model_or_path
            self.model_path = None

    def predict(self, data: pd.DataFrame | dict) -> np.ndarray:
        """Predict smoothed acceleration (m/s^2) for given input features.

        Accepts either a pandas DataFrame with the required FEATURES columns,
        or a dictionary containing a single feature sample.
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        missing = [col for col in FEATURES if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        X = df[FEATURES]
        return self.model.predict(X)


if __name__ == "__main__":
    data_path = resolve_path("data/processed/physics_data.parquet")
    df = pd.read_parquet(data_path)
    df = prepare_data(df)

    print("\n========== ACCELERATION MODEL ==========")
    print("\nSamples:")
    print(len(df))

    model, mae, r2 = train_model(df, save_path=DEFAULT_MODEL_PATH)

    print("\nMAE:")
    print(f"{mae:.4f} m/s²")

    print("\nR²:")
    print(f"{r2:.4f}")

    print("\nFeature importance:")
    importance = pd.Series(
        model.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)
    print(importance)
    print(f"\nSaved model to {DEFAULT_MODEL_PATH}")