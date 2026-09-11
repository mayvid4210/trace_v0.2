"""Test traffic penalty model data preparation and target construction."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fastf1
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from backend.ml.tyre_degradation_residual import prepare_training_data

fastf1.Cache.enable_cache("backend/data/raw")

print("Loading base training laps...")
X_base, y_res, t_physics, meta = prepare_training_data()

# Load gaps from raw laps
laps_list = []
for gp in ["Bahrain", "Canada"]:
    s = fastf1.get_session(2024, gp, "R")
    s.load(laps=True, telemetry=False, weather=True)
    l = s.laps.copy()
    l = l[l["PitInTime"].isna() & l["PitOutTime"].isna() & l["LapTime"].notna()].copy()
    l["circuit"] = gp
    l["start_time_s"] = l["LapStartTime"].dt.total_seconds()
    l = l.sort_values(["circuit", "LapNumber", "start_time_s"])
    l["gap_ahead_s"] = l.groupby(["circuit", "LapNumber"])["start_time_s"].diff().fillna(60.0).clip(0.0, 60.0)
    l["gap_behind_s"] = (-l.groupby(["circuit", "LapNumber"])["start_time_s"].diff(-1)).fillna(60.0).clip(0.0, 60.0)
    l["position"] = l["Position"].fillna(10.0).astype(int)
    laps_list.append(l[["circuit", "Driver", "LapNumber", "gap_ahead_s", "gap_behind_s", "position"]])

gaps_df = pd.concat(laps_list, ignore_index=True)
gaps_df = gaps_df.rename(columns={"Driver": "driver", "LapNumber": "lap_number"})

df = meta.copy()
df["circuit_raw"] = df["circuit"]
df = df.merge(gaps_df, on=["circuit", "driver", "lap_number"], how="left")
df["gap_ahead_s"] = df["gap_ahead_s"].fillna(60.0)
df["gap_behind_s"] = df["gap_behind_s"].fillna(60.0)
df["position"] = df["position"].fillna(10.0)

# Add causal traffic density indicator: count of cars within 3.0s window
df["in_dirty_air"] = (df["gap_ahead_s"] <= 2.0).astype(float)

# Fuel correction
fuel_betas = {"Bahrain": 0.0392, "Canada": 0.0261}
df["beta_fuel"] = df["circuit"].map(fuel_betas)
total_laps_gp = {"Bahrain": 57, "Canada": 70}
df["burn_rate"] = 100.0 / df["circuit"].map(total_laps_gp)
df["fuel_mass_kg"] = (100.0 - (df["lap_number"] - 1) * df["burn_rate"]).clip(lower=0.0)
df["fuel_burned_kg"] = 100.0 - df["fuel_mass_kg"]
df["T_fuel_corr"] = df["T_actual"] + df["fuel_burned_kg"] * df["beta_fuel"]

# Causal clean-air reference:
# For each stint, the reference clean pace up to lap k:
# If the driver has run laps with gap_ahead > 2.5s, use the cumulative minimum of clean laps.
# If currently in clean air (gap_ahead >= 2.5s), traffic penalty is strictly 0.0!
# If in dirty air (gap_ahead < 2.5s), traffic penalty is excess time lost above the driver's clean-air pace.
df["clean_T_fuel"] = np.where(df["gap_ahead_s"] >= 2.5, df["T_fuel_corr"], np.nan)
df["causal_clean_pace"] = df.groupby("stint_id")["clean_T_fuel"].cummin()

# For laps before any clean air was observed in that stint, fallback to overall circuit/compound clean median
fallback_clean = df[df["gap_ahead_s"] >= 2.5].groupby(["circuit", df["stint_id"].str.extract(r"S(\d+)")[0]])["T_fuel_corr"].transform("median")
df["causal_clean_pace"] = df["causal_clean_pace"].fillna(fallback_clean).fillna(df["T_fuel_corr"])

# Target: traffic_penalty_s
# Strictly 0.0 if gap_ahead >= 2.5s (clean air)
# If gap_ahead < 2.5s: max(0.0, T_fuel_corr - causal_clean_pace)
df["traffic_penalty_proxy_s"] = np.where(
    df["gap_ahead_s"] >= 2.5,
    0.0,
    np.maximum(0.0, df["T_fuel_corr"] - df["causal_clean_pace"]).clip(upper=5.0)
)

print("Target traffic_penalty_proxy_s summary:")
print(df["traffic_penalty_proxy_s"].describe())
print(f"Zero penalty laps (clean air or no deficit): {(df['traffic_penalty_proxy_s'] == 0.0).sum()} / {len(df)} ({(df['traffic_penalty_proxy_s'] == 0.0).mean()*100:.1f}%)")
print(f"Non-zero penalty laps (dirty air deficit): {(df['traffic_penalty_proxy_s'] > 0.0).sum()} / {len(df)} ({(df['traffic_penalty_proxy_s'] > 0.0).mean()*100:.1f}%)")
print("Non-zero penalty stats:")
print(df[df["traffic_penalty_proxy_s"] > 0.0]["traffic_penalty_proxy_s"].describe())

# Check correlation with gap_ahead
print("\nMean penalty by gap_ahead bin:")
df["gap_bin"] = pd.cut(df["gap_ahead_s"], bins=[-0.1, 1.0, 2.0, 3.0, 60.0], labels=["<1s", "1-2s", "2-3s", ">3s"])
print(df.groupby("gap_bin", observed=True)["traffic_penalty_proxy_s"].agg(["mean", "count", "std"]))
