import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_PATH = "data/processed/physics_data.parquet"


FEATURES = [
    "Throttle",
    "Brake",
    "nGear",
    "Speed_ms",
    "RPM",
    "total_mass_kg",
    "track_temperature",
    "tyre_age",
    "air_temperature",
    "track_grip",
    "DRS",
]

TARGET = "Acceleration_smoothed"


def main():

    df = pd.read_parquet(
        DATA_PATH
    )

    df = df.dropna(
        subset=FEATURES + [TARGET]
    ).copy()

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        n_jobs=-1,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    prediction = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    print(
        "\n========== ACCELERATION MODEL CHECK =========="
    )

    print(
        f"\nSamples: {len(df)}"
    )

    print(
        f"\nMAE: {mae:.4f} m/s²"
    )

    print(
        f"RMSE: {rmse:.4f} m/s²"
    )

    print(
        f"R²: {r2:.4f}"
    )

    result = X_test.copy()

    result["actual_acceleration"] = y_test.to_numpy()
    result["predicted_acceleration"] = prediction

    result["residual"] = (
        result["actual_acceleration"]
        - result["predicted_acceleration"]
    )

    result["speed_kmh"] = (
        result["Speed_ms"] * 3.6
    )

    print(
        "\n========== RESIDUAL CHECK =========="
    )

    print(
        result["residual"].describe()
    )

    print(
        "\n========== RESIDUAL BY SPEED =========="
    )

    result["speed_bin"] = pd.cut(
        result["speed_kmh"],
        bins=[
            0,
            80,
            120,
            160,
            200,
            240,
            280,
            320,
            400
        ]
    ).astype("string")

    print(
        result.groupby(
            "speed_bin",
            observed=True
        )["residual"]
        .agg(
            ["count", "mean", "std"]
        )
    )

    print(
        "\n========== FEATURE IMPORTANCE =========="
    )

    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(
        ascending=False
    )

    print(
        importance
    )

    result.to_parquet(
        "data/processed/acceleration_predictions.parquet",
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "data/processed/acceleration_predictions.parquet"
    )

    for column in result.select_dtypes(include=["interval", "category"]).columns:
        result[column] = result[column].astype(str)


if __name__ == "__main__":
    main()
