from config import CACHE_DIR

import fastf1

session = fastf1.get_session(2024, "Bahrain", "R")

session.load()

#print(session.laps.columns.tolist())

#print(session.laps.head())


# ADD THIS BELOW
import pandas as pd

telemetry = pd.read_parquet("data/processed/telemetry.parquet")

print(telemetry["Time"].dtype)

print(telemetry[[
    "Driver",
    "LapNumber",
    "Time",
    "Speed",
    "Throttle",
    "Brake"
]].head())