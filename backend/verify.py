import pandas as pd

laps = pd.read_parquet("data/processed/laps.parquet")

print("Shape:", laps.shape)

print("\nColumns:")
print(laps.columns.tolist())

print("\nCleaning results:")
print(
    laps[
        [
            "Driver",
            "LapNumber",
            "LapStartTime",
            "Time",
            "category",
            "pct_safety_car",
            "pct_vsc",
            "pct_yellow",
            "clean_weight",
            "is_out_lap",
        ]
    ].head(20)
)

print("\nCategory counts:")
print(laps["category"].value_counts(dropna=False))

print("\nNon-clean laps:")
print(
    laps[
        laps["category"].isin(
            ["PARTIAL", "SAFETY_CAR", "YELLOW_FLAG", "INVALID"]
        )
    ][
        [
            "Driver",
            "LapNumber",
            "category",
            "pct_safety_car",
            "pct_vsc",
            "pct_yellow",
            "clean_weight",
        ]
    ].head(20)
)