"""Inspect traffic features in cached race telemetry."""

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
    l = l[l["PitInTime"].isna() & l["PitOutTime"].isna() & l["LapTime"].notna()].copy()
    l["circuit"] = gp
    l["LapTime_s"] = l["LapTime"].dt.total_seconds()
    all_laps.append(l)

df = pd.concat(all_laps, ignore_index=True)

df["start_time_s"] = df["LapStartTime"].dt.total_seconds()
df = df.sort_values(["circuit", "LapNumber", "start_time_s"])

# Gap to car ahead at start of lap
df["gap_ahead_s"] = df.groupby(["circuit", "LapNumber"])["start_time_s"].diff().fillna(60.0)

# Gap to car behind at start of lap
df["gap_behind_s"] = (-df.groupby(["circuit", "LapNumber"])["start_time_s"].diff(-1)).fillna(60.0)

print("Gap ahead distribution:")
print(df["gap_ahead_s"].describe())

n_total = len(df)
c1 = (df["gap_ahead_s"] <= 1.0).sum()
c2 = (df["gap_ahead_s"] <= 2.0).sum()
c3 = (df["gap_ahead_s"] > 3.0).sum()
print(f"Total laps: {n_total}")
print(f"Laps within 1.0s (DRS / dirty air): {c1} ({c1/n_total*100:.1f}%)")
print(f"Laps within 2.0s (dirty air): {c2} ({c2/n_total*100:.1f}%)")
print(f"Laps in clean air (> 3.0s): {c3} ({c3/n_total*100:.1f}%)")

# Inspect lap time difference between clean air and dirty air
print("\nMean LapTime_s by gap_ahead bin in Bahrain:")
b_df = df[df["circuit"] == "Bahrain"]
b_df["gap_bin"] = pd.cut(b_df["gap_ahead_s"], bins=[0, 1.0, 2.0, 3.0, 60.0], labels=["<1s", "1-2s", "2-3s", ">3s"])
print(b_df.groupby("gap_bin", observed=True)["LapTime_s"].agg(["mean", "count", "std"]))
