"""Test candidate traffic penalty target formulations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache("backend/data/raw")

all_laps = []
for gp in ["Bahrain", "Canada"]:
    s = fastf1.get_session(2024, gp, "R")
    s.load(laps=True, telemetry=False, weather=True)
    l = s.laps.copy()
    # Clean racing laps only (no pit laps, green flag track status 1)
    l = l[l["PitInTime"].isna() & l["PitOutTime"].isna() & l["LapTime"].notna()].copy()
    if "TrackStatus" in l.columns:
        l = l[l["TrackStatus"].astype(str) == "1"].copy()
    l["circuit"] = gp
    l["LapTime_s"] = l["LapTime"].dt.total_seconds()
    all_laps.append(l)

df = pd.concat(all_laps, ignore_index=True)

# Sort strictly by circuit, lap number, and start time
df["start_time_s"] = df["LapStartTime"].dt.total_seconds()
df = df.sort_values(["circuit", "LapNumber", "start_time_s"])

# Causal gap to car ahead at start of lap
df["gap_ahead_s"] = df.groupby(["circuit", "LapNumber"])["start_time_s"].diff().fillna(60.0).clip(lower=0.0, upper=60.0)

# Causal gap to car behind at start of lap
df["gap_behind_s"] = (-df.groupby(["circuit", "LapNumber"])["start_time_s"].diff(-1)).fillna(60.0).clip(lower=0.0, upper=60.0)

# Stint ID
df["stint_id"] = df["circuit"] + "_" + df["Driver"] + "_S" + df["Stint"].astype(str)

# Fuel correction
fuel_betas = {"Bahrain": 0.0392, "Canada": 0.0261}
df["beta_fuel"] = df["circuit"].map(fuel_betas)
total_laps_gp = {"Bahrain": 57, "Canada": 70}
df["total_laps"] = df["circuit"].map(total_laps_gp)
df["burn_rate"] = 100.0 / df["total_laps"]
df["fuel_mass_kg"] = (100.0 - (df["LapNumber"] - 1) * df["burn_rate"]).clip(lower=0.0)
df["fuel_burned_kg"] = 100.0 - df["fuel_mass_kg"]
df["T_fuel_corr"] = df["LapTime_s"] + df["fuel_burned_kg"] * df["beta_fuel"]

# Clean air laps: gap_ahead > 2.5s
clean_air_mask = df["gap_ahead_s"] >= 2.5

# For each stint, compute the clean-air baseline pace
clean_pace = df[clean_air_mask].groupby("stint_id")["T_fuel_corr"].median().rename("clean_pace_stint")
df = df.merge(clean_pace, on="stint_id", how="left")

# Fallback for stints with no clean air laps: overall stint median
df["clean_pace_stint"] = df["clean_pace_stint"].fillna(df.groupby("stint_id")["T_fuel_corr"].transform("median"))

# Operational traffic penalty proxy: excess time lost above clean air pace
df["traffic_penalty_proxy_s"] = (df["T_fuel_corr"] - df["clean_pace_stint"]).clip(lower=0.0)

# When in clean air (gap > 3.0s), traffic penalty should be 0 by definition of clean air
# Or let's see how traffic penalty correlates with gap_ahead:
print("Traffic penalty proxy stats:")
print(df["traffic_penalty_proxy_s"].describe())

print("\nMean traffic penalty proxy by gap_ahead bin:")
df["gap_bin"] = pd.cut(df["gap_ahead_s"], bins=[-0.1, 1.0, 2.0, 3.0, 60.0], labels=["<1s", "1-2s", "2-3s", ">3s"])
print(df.groupby("gap_bin", observed=True)["traffic_penalty_proxy_s"].agg(["mean", "median", "count", "std"]))
