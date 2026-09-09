import pandas as pd

from vehicle import VehicleModel
from prepare_resistance import prepare_resistance_data

INPUT_FILE = "data/processed/physics_data.parquet"


def main():

    raw = pd.read_parquet(INPUT_FILE)

    # Use the SAME clean, sustained-coast filtering that built the curve.
    data = prepare_resistance_data(raw)

    model = VehicleModel()

    predictions = data.apply(
        lambda row: model.calculate(
            row["Speed"],
            row["air_temperature"],
            row["total_mass_kg"]
        ),
        axis=1
    )

    data["predicted_resistance_n"] = predictions.apply(lambda r: r["resistance_force_n"])
    data["in_reliable_range"] = predictions.apply(lambda r: r["in_reliable_range"])

    data["observed_resistance_n"] = data["resistance_force_n"]
    data["resistance_error_n"] = (
        data["observed_resistance_n"] - data["predicted_resistance_n"]
    )

    reliable = data[data["in_reliable_range"]]

    mae = reliable["resistance_error_n"].abs().mean()
    rmse = (reliable["resistance_error_n"].pow(2).mean()) ** 0.5

    print("\n========== RESISTANCE PREDICTION CHECK ==========")
    print(f"\nReliable-range samples: {len(reliable)} / {len(data)}")

    print("\nMAE (reliable range only):")
    print(f"{mae:.2f} N")

    print("\nRMSE (reliable range only):")
    print(f"{rmse:.2f} N")

    print("\nSample comparison:")
    print(
        data[
            [
                "Speed", "total_mass_kg", "observed_resistance_n",
                "predicted_resistance_n", "resistance_error_n", "in_reliable_range"
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()