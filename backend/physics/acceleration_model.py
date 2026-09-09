import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


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
    "track_grip"
]


def prepare_data(df):

    df = df.copy()

    df = df.dropna(
        subset=FEATURES + ["Acceleration_smoothed"]
    )

    return df


def train_model(df):

    X = df[FEATURES]
    y = df["Acceleration_smoothed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    return model, mae, r2


if __name__ == "__main__":

    df = pd.read_parquet(
        "data/processed/physics_data.parquet"
    )

    df = prepare_data(df)

    print("\n========== ACCELERATION MODEL ==========")

    print("\nSamples:")
    print(len(df))

    model, mae, r2 = train_model(df)

    print("\nMAE:")
    print(f"{mae:.4f} m/s²")

    print("\nR²:")
    print(f"{r2:.4f}")

    print("\nFeature importance:")

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(
        ascending=False
    )

    print(importance)