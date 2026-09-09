import numpy as np
import pandas as pd


def aggregate_telemetry_per_lap(telemetry_df):
    grouped = telemetry_df.groupby(
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber"
        ]
    )

    features = grouped.agg(
        average_speed=("Speed", "mean"),
        throttle_usage=("Throttle", "mean"),
        braking_load=("Brake", "mean"),
        speed_std=("Speed", "std"),
        drs_usage=("DRS", "mean"),
        avg_acceleration=("Acceleration_ms2", "mean"),
        max_acceleration=("Acceleration_ms2", "max"),
        acceleration_std=("Acceleration_ms2", "std"),
    ).reset_index()

    # Speed variability used as a proxy for cornering load
    features["cornering_load"] = features["speed_std"].fillna(0)

    return features.drop(columns=["speed_std"])

if __name__ == "__main__":

    telemetry = pd.read_parquet(
        "data/processed/telemetry_with_acceleration.parquet"
    )

    tel_feats = aggregate_telemetry_per_lap(
        telemetry
    )

    print("\nTelemetry features:")
    print(tel_feats.head())

    print("\nShape:")
    print(tel_feats.shape)

    print("\nNull rates:")
    print(tel_feats.isna().mean())    

def estimate_fuel(df, session):

    df = df.copy()

    total_laps = session.total_laps

    if total_laps is None:
        total_laps = df["LapNumber"].max()

    starting_fuel = 110.0  # kg

    df["fuel_estimate"] = (
        starting_fuel
        * (1 - (df["LapNumber"] - 1) / total_laps)
    )

    df["fuel_estimate"] = df["fuel_estimate"].clip(
        lower=0
    )

    return df

if __name__ == "__main__":

    laps = pd.read_parquet(
        "data/processed/laps.parquet"
    )

    print("\nOriginal laps:")
    print(laps[[
        "Driver",
        "LapNumber"
    ]].head())

    # Temporary test
    class DummySession:
        total_laps = 57

    laps_with_fuel = estimate_fuel(
        laps,
        DummySession()
    )

    print("\nFuel estimate:")
    print(
        laps_with_fuel[[
            "Driver",
            "LapNumber",
            "fuel_estimate"
        ]].head(10)
    )

def attach_weather(laps_df, weather_df):
    laps = laps_df.copy()
    weather = weather_df.copy()

    # Convert time columns
    laps["Time"] = pd.to_timedelta(laps["Time"], errors="coerce")
    weather["Time"] = pd.to_timedelta(weather["Time"], errors="coerce")

    # Keep only required weather columns
    weather = weather[
        [
            "GrandPrix",
            "SessionName",
            "Time",
            "TrackTemp",
            "AirTemp"
        ]
    ].copy()

    # Remove rows where time is unavailable
    laps = laps.dropna(subset=["Time"])
    weather = weather.dropna(subset=["Time"])

    # Sort primarily by TIME for merge_asof
    laps = laps.sort_values(
        ["Time", "GrandPrix", "SessionName"]
    ).reset_index(drop=True)

    weather = weather.sort_values(
        ["Time", "GrandPrix", "SessionName"]
    ).reset_index(drop=True)

    # Session-safe weather alignment
    merged = pd.merge_asof(
        laps,
        weather,
        on="Time",
        by=["GrandPrix", "SessionName"],
        direction="backward"
    )

    merged = merged.rename(
        columns={
            "TrackTemp": "track_temperature",
            "AirTemp": "air_temperature"
        }
    )

    return merged

if __name__ == "__main__":

    laps = pd.read_parquet(
        "data/processed/laps.parquet"
    )

    weather = pd.read_parquet(
        "data/processed/weather.parquet"
    )

    result = attach_weather(
        laps,
        weather
    )

    print("\nWeather attached:")
    print(
        result[
            [
                "Driver",
                "LapNumber",
                "track_temperature",
                "air_temperature"
            ]
        ].head(10)
    )

    print("\nNull rates:")
    print(
        result[
            [
                "track_temperature",
                "air_temperature"
            ]
        ].isna().mean()
    )

def calculate_track_grip(df):

    df = df.copy()

    # Valid lap times only
    valid = df.dropna(
        subset=["LapTimeSeconds"]
    ).copy()

    # Session's final best lap
    session_best = (
        valid.groupby(
            ["GrandPrix", "SessionName"]
        )["LapTimeSeconds"]
        .min()
        .rename("session_best")
        .reset_index()
    )

    # Best lap achieved up to each lap number
    best_by_lap = (
        valid.groupby(
            [
                "GrandPrix",
                "SessionName",
                "LapNumber"
            ]
        )["LapTimeSeconds"]
        .min()
        .reset_index()
        .sort_values(
            [
                "GrandPrix",
                "SessionName",
                "LapNumber"
            ]
        )
    )

    best_by_lap["best_lap_so_far"] = (
        best_by_lap
        .groupby(
            ["GrandPrix", "SessionName"]
        )["LapTimeSeconds"]
        .cummin()
    )

    best_by_lap = best_by_lap.merge(
        session_best,
        on=["GrandPrix", "SessionName"],
        how="left"
    )

    best_by_lap["track_grip"] = (
        best_by_lap["session_best"]
        / best_by_lap["best_lap_so_far"]
    )

    # Attach the value back to every driver/lap
    df = df.merge(
        best_by_lap[
            [
                "GrandPrix",
                "SessionName",
                "LapNumber",
                "track_grip"
            ]
        ],
        on=[
            "GrandPrix",
            "SessionName",
            "LapNumber"
        ],
        how="left"
    )

    return df

if __name__ == "__main__":

    laps = pd.read_parquet(
        "data/processed/laps.parquet"
    )

    result = calculate_track_grip(laps)

    print("\nTrack grip:")
    print("\nValid track grip values:")

    print(
        result[
            [
                "GrandPrix",
                "SessionName",
                "Driver",
                "LapNumber",
                "LapTimeSeconds",
                "track_grip"
            ]
        ]
        .dropna(subset=["track_grip"])
        .head(20)
    )

    print("\nNull rate:")
    print(
        result["track_grip"].isna().mean()
    )

def compute_traffic_proxy(df):

    df = df.copy()

    # Default: no traffic information
    df["traffic"] = np.nan

    # Traffic is calculated only for race sessions
    race = df[df["SessionName"] == "Race"].copy()

    if race.empty:
        return df

    # Remove laps without a valid lap time
    race = race.dropna(
        subset=["LapTimeSeconds"]
    )

    # Sort within each actual race
    race = race.sort_values(
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber"
        ]
    )

    # Calculate cumulative race time for each driver
    race["CumulativeTime"] = (
        race
        .groupby(
            [
                "GrandPrix",
                "SessionName",
                "Driver"
            ]
        )["LapTimeSeconds"]
        .cumsum()
    )

    # Calculate gap to car ahead on the same lap
    race["gap_ahead"] = np.nan

    group_columns = [
        "GrandPrix",
        "SessionName",
        "LapNumber"
    ]

    for _, indices in race.groupby(
        group_columns,
        sort=False
    ).groups.items():

        group = race.loc[indices].sort_values(
            "CumulativeTime"
        )

        gaps = group["CumulativeTime"].diff()

        race.loc[
            group.index,
            "gap_ahead"
        ] = gaps.values

    # Traffic = another car is within 1.5 seconds ahead
    race["traffic"] = (
        race["gap_ahead"] < 1.5
    ).astype(float)

    # Create lookup table
    traffic_lookup = race[
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber",
            "traffic"
        ]
    ].drop_duplicates(
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber"
        ]
    )

    # Merge traffic back into original laps
    df = df.merge(
        traffic_lookup,
        on=[
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber"
        ],
        how="left",
        suffixes=("", "_new")
    )

    # Replace traffic column
    df["traffic"] = df["traffic_new"]

    df.drop(
        columns=["traffic_new"],
        inplace=True
    )

    return df

if __name__ == "__main__":

    laps = pd.read_parquet(
        "data/processed/laps.parquet"
    )

    result = compute_traffic_proxy(laps)

    print("\nTraffic values:")

    print(
        result[
            [
                "GrandPrix",
                "SessionName",
                "Driver",
                "LapNumber",
                "traffic"
            ]
        ]
        .dropna(subset=["traffic"])
        .head(30)
    )

    print("\nTraffic null rate:")
    print(
        result["traffic"].isna().mean()
    )

    print("\nNon-race traffic values:")
    print(
        result.loc[
            result["SessionName"] != "Race",
            "traffic"
        ].notna().sum()
    )

print("\n========== TRAFFIC CHECK ==========")

laps = pd.read_parquet(
    "data/processed/laps.parquet"
)

result = compute_traffic_proxy(laps)

print("\nTraffic counts:")
print(
    result["traffic"].value_counts(
        dropna=False
    )
)

print("\nTraffic = 1 examples:")

print(
    result[
        result["traffic"] == 1
    ][
        [
            "GrandPrix",
            "Driver",
            "LapNumber",
            "LapTimeSeconds",
            "traffic"
        ]
    ].head(20)
)

def add_race_condition_features(df):

    df = df.copy()

    # Yellow flag
    df["yellow_flag"] = (
        df["pct_yellow"]
        .fillna(0)
        > 0
    ).astype(int)

    # Safety car
    df["safety_car"] = (
        df["pct_safety_car"]
        .fillna(0)
        > 0
    ).astype(int)

    return df

print("\n========== RACE CONDITION CHECK ==========")

laps = pd.read_parquet(
    "data/processed/laps.parquet"
)

result = add_race_condition_features(
    laps
)

print("\nRace condition counts:")

print(
    result[
        [
            "yellow_flag",
            "safety_car"
        ]
    ].sum()
)

print("\nExamples:")

print(
    result[
        (
            (result["yellow_flag"] == 1) |
            (result["safety_car"] == 1)
        )
    ][
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber",
            "pct_yellow",
            "pct_safety_car",
            "yellow_flag",
            "safety_car"
        ]
    ].head(20)
)

def add_lap_and_sector_times(df):

    df = df.copy()

    df["lap"] = df["LapNumber"]

    df["lap_time"] = (
        pd.to_timedelta(df["LapTime"], errors="coerce")
        .dt.total_seconds()
    )

    df["sector1_time"] = (
        pd.to_timedelta(
            df["Sector1Time"],
            errors="coerce"
        ).dt.total_seconds()
    )

    df["sector2_time"] = (
        pd.to_timedelta(
            df["Sector2Time"],
            errors="coerce"
        ).dt.total_seconds()
    )

    df["sector3_time"] = (
        pd.to_timedelta(
            df["Sector3Time"],
            errors="coerce"
        ).dt.total_seconds()
    )

    # Remove impossible negative values
    for column in [
        "lap_time",
        "sector1_time",
        "sector2_time",
        "sector3_time"
    ]:
        df.loc[
            df[column] <= 0,
            column
        ] = np.nan

    return df


print("\n========== LAP & SECTOR TIME CHECK ==========")

laps = pd.read_parquet(
    "data/processed/laps.parquet"
)

result = add_lap_and_sector_times(laps)

print(
    result[
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "lap",
            "lap_time",
            "sector1_time",
            "sector2_time",
            "sector3_time"
        ]
    ]
    .dropna(subset=["lap_time"])
    .head(10)
)

print("\nNull rates:")

print(
    result[
        [
            "lap_time",
            "sector1_time",
            "sector2_time",
            "sector3_time"
        ]
    ].isna().mean()
)
def add_tyre_compound(df):

    df = df.copy()

    df["tyre_compound"] = (
        df["Compound"]
        .astype("string")
        .str.upper()
        .str.strip()
    )

    return df

print("\n========== TYRE COMPOUND CHECK ==========")

laps = pd.read_parquet(
    "data/processed/laps.parquet"
)

result = add_tyre_compound(laps)

print("\nTyre compounds:")

print(
    result["tyre_compound"]
    .value_counts(dropna=False)
)

print("\nExamples:")

print(
    result[
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber",
            "Compound",
            "tyre_compound"
        ]
    ]
    .dropna(subset=["tyre_compound"])
    .head(20)
)

def add_tyre_age(df):

    df = df.copy()

    df["tyre_age"] = pd.to_numeric(
        df["TyreLife"],
        errors="coerce"
    )

    return df

print("\n========== TYRE AGE CHECK ==========")

laps = pd.read_parquet(
    "data/processed/laps.parquet"
)

result = add_tyre_age(laps)

print("\nExamples:")

print(
    result[
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber",
            "Stint",
            "Compound",
            "TyreLife",
            "tyre_age",
            "FreshTyre"
        ]
    ]
    .dropna(subset=["tyre_age"])
    .head(30)
)

print("\nTyre age statistics:")

print(
    result["tyre_age"].describe()
)

print("\nTyre age null rate:")

print(
    result["tyre_age"].isna().mean()
)

def add_stint(df):

    df = df.copy()

    df["stint"] = pd.to_numeric(
        df["Stint"],
        errors="coerce"
    )

    return df

print("\n========== STINT CHECK ==========")

laps = pd.read_parquet(
    "data/processed/laps.parquet"
)

result = add_stint(laps)

print("\nStint distribution:")

print(
    result["stint"]
    .value_counts()
    .sort_index()
)

print("\nExamples:")

print(
    result[
        [
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber",
            "stint",
            "Compound",
            "TyreLife"
        ]
    ].head(30)
)

print("\nStint null rate:")

print(
    result["stint"].isna().mean()
)

def estimate_fuel(df, session):

    df = df.copy()

    total_laps = session.total_laps

    if total_laps is None:
        total_laps = df["LapNumber"].max()

    total_laps = float(total_laps)

    starting_fuel = 110.0  # kg, assumed model input

    df["fuel_estimate"] = (
        starting_fuel
        * (
            1
            - (df["LapNumber"] - 1) / total_laps
        )
    )

    df["fuel_estimate"] = df["fuel_estimate"].clip(
        lower=0,
        upper=starting_fuel
    )

    return df

def add_fuel_estimate_by_session(df):

    df = df.copy()

    starting_fuel = 110.0

    df["fuel_estimate"] = np.nan

    for (gp, session_name), group in df.groupby(
        ["GrandPrix", "SessionName"]
    ):

        total_laps = group["LapNumber"].max()

        if pd.isna(total_laps) or total_laps <= 0:
            continue

        fuel = (
            starting_fuel
            * (
                1
                - (group["LapNumber"] - 1)
                / float(total_laps)
            )
        )

        df.loc[group.index, "fuel_estimate"] = fuel.clip(
            lower=0,
            upper=starting_fuel
        )

    return df

def build_lap_features_base(laps_df, weather_df, telemetry_df):

    df = laps_df.copy()

    # -------------------------
    # Timing features
    # -------------------------
    df = add_lap_and_sector_times(df)

    # -------------------------
    # Tyre features
    # -------------------------
    df = add_tyre_compound(df)
    df = add_tyre_age(df)
    df = add_stint(df)

    # -------------------------
    # Fuel estimate
    # -------------------------
    df = add_fuel_estimate_by_session(df)

    # -------------------------
    # Weather
    # -------------------------
    df = attach_weather(df, weather_df)

    # -------------------------
    # Telemetry
    # -------------------------
    telemetry_features = aggregate_telemetry_per_lap(
        telemetry_df
    )

    df = df.merge(
        telemetry_features,
        left_on=[
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber"
        ],
        right_on=[
            "GrandPrix",
            "SessionName",
            "Driver",
            "LapNumber"
        ],
        how="left"
    )

    # -------------------------
    # Race conditions
    # -------------------------
    df = add_race_condition_features(df)

    # -------------------------
    # Track grip
    # -------------------------
    df = calculate_track_grip(df)

    # -------------------------
    # Traffic
    # -------------------------
    df = compute_traffic_proxy(df)

    columns = [
        "GrandPrix",
        "SessionName",
        "Driver",

        "lap",
        "lap_time",
        "sector1_time",
        "sector2_time",
        "sector3_time",

        "tyre_compound",
        "tyre_age",
        "stint",

        "fuel_estimate",

        "track_temperature",
        "air_temperature",
        "track_grip",
        "traffic",

        "average_speed",
        "avg_acceleration",
        "max_acceleration",
        "acceleration_std",
        "braking_load",
        "cornering_load",
        "throttle_usage",
        "drs_usage",

        "yellow_flag",
        "safety_car",
    ]

    return df[columns].copy()

# =========================
# STEP 8: FINAL QUALITY AUDIT
# =========================

if __name__ == "__main__":

    laps = pd.read_parquet(
        "data/processed/laps.parquet"
    )

    weather = pd.read_parquet(
        "data/processed/weather.parquet"
    )

    telemetry = pd.read_parquet(
        "data/processed/telemetry_with_acceleration.parquet"
    )

    features = build_lap_features_base(
        laps,
        weather,
        telemetry
    )

print("\n========== FINAL LAP FEATURES AUDIT ==========")

# 1. Shape
print("\n1. Shape:")
print(features.shape)

# 2. Columns
print("\n2. Columns:")
print(features.columns.tolist())

# 3. Duplicate lap keys
key_cols = [
    "GrandPrix",
    "SessionName",
    "Driver",
    "lap"
]

duplicates = features.duplicated(
    subset=key_cols
).sum()

print("\n3. Duplicate keys:")
print(duplicates)

# 4. Missing values
print("\n4. Missing values:")
print(features.isna().sum())

# 5. Data types
print("\n5. Data types:")
print(features.dtypes)

# 6. Impossible values
print("\n6. Impossible values:")

print("Negative lap times:",
      (features["lap_time"] < 0).sum())

print("Negative sector 1:",
      (features["sector1_time"] < 0).sum())

print("Negative sector 2:",
      (features["sector2_time"] < 0).sum())

print("Negative sector 3:",
      (features["sector3_time"] < 0).sum())

print("Negative fuel:",
      (features["fuel_estimate"] < 0).sum())

print("Negative tyre age:",
      (features["tyre_age"] < 0).sum())

print("Invalid track grip:",
      ((features["track_grip"] < 0) |
       (features["track_grip"] > 1)).sum())

print("Invalid traffic:",
      (~features["traffic"].dropna().isin([0.0, 1.0])).sum())

print("Invalid yellow flag:",
      (~features["yellow_flag"].isin([0, 1])).sum())

print("Invalid safety car:",
      (~features["safety_car"].isin([0, 1])).sum())

# 7. Session consistency
print("\n7. Sessions:")
print(
    features.groupby(
        ["GrandPrix", "SessionName"]
    ).size()
)

# 8. Driver consistency
print("\n8. Drivers:")
print(
    features.groupby(
        ["GrandPrix", "SessionName"]
    )["Driver"].nunique()
)

print("\n========== AUDIT COMPLETE ==========")

features.to_parquet(
    "data/processed/lap_features.parquet",
    index=False
)

print("\nSaved successfully:")
print("data/processed/lap_features.parquet")