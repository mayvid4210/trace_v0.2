import pandas as pd

from vehicle import VehicleModel
from prepare_resistance import prepare_resistance_data

INPUT_FILE = "data/processed/physics_data.parquet"


def main():
    raw = pd.read_parquet(INPUT_FILE)

    # Same clean filter as before, so we're using trustworthy rows only
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
    data["observed_resistance_n"] = data["resistance_force_n"]
    data["resistance_error_n"] = (
        data["observed_resistance_n"] - data["predicted_resistance_n"]
    )

    # Round Distance into 50m bins so we can compare "same place on track"
    # across different laps, even if exact distance values don't line up.
    data["distance_bin"] = (data["Distance"] // 50) * 50

    # For each circuit, check: at each track location (distance_bin),
    # does the error consistently point the same direction (mean far from 0)
    # or does it look random (mean near 0, high spread)?
    print("\n========== GRADIENT PATTERN CHECK ==========\n")

    for grand_prix, group in data.groupby("GrandPrix"):
        summary = (
            group.groupby("distance_bin")["resistance_error_n"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        # Only look at bins with enough samples to trust the pattern
        summary = summary[summary["count"] >= 3]

        if summary.empty:
            continue

        # A bin is "consistent" if the average error is clearly bigger
        # than the noise (std) around it — i.e. it's not just random scatter
        summary["consistent"] = summary["mean"].abs() > summary["std"]

        pct_consistent = summary["consistent"].mean() * 100

        print(f"{grand_prix}:")
        print(f"  Track locations checked: {len(summary)}")
        print(f"  % showing a consistent (non-random) error: {pct_consistent:.1f}%")
        print(f"  Worst spot -> distance_bin={summary.loc[summary['mean'].abs().idxmax(), 'distance_bin']}, "
              f"mean error={summary.loc[summary['mean'].abs().idxmax(), 'mean']:.1f} N\n")


if __name__ == "__main__":
    main()