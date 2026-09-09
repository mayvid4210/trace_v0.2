import pandas as pd

from vehicle import VehicleModel


INPUT_FILE = "data/processed/physics_data.parquet"


def main():

    data = pd.read_parquet(INPUT_FILE)

    model = VehicleModel()

    print("\n========== VEHICLE MODEL CHECK ==========")

    print("\nTesting selected telemetry points:")

    speeds = [150, 200, 250, 300, 340]

    for speed in speeds:

        row = data[
            data["Speed"] >= speed
        ].iloc[0]

        result = model.calculate(
            row["Speed"],
            row["air_temperature"],
            row["total_mass_kg"]
        )

        print("\n--------------------------------")

        print(
            f"Speed: "
            f"{row['Speed']:.1f} km/h"
        )

        print(
            f"Temperature: "
            f"{row['air_temperature']:.1f} C"
        )

        print(
            f"Mass: "
            f"{row['total_mass_kg']:.2f} kg"
        )

        print(
            f"Air density: "
            f"{result['air_density']:.4f} kg/m³"
        )

        print(
            f"Resistance: "
            f"{result['resistance_force_n']:.2f} N"
        )


if __name__ == "__main__":

    main()