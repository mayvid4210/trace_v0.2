"""Evaluate traffic penalty target formulation and XGBoost model performance."""

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

print("Preparing training data...")
X_base, y_res, t_physics, meta = prepare_training_data()

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
    l["position"] = l["Position"].fillna(10.0).astype(float)
    laps_list.append(l[["circuit", "Driver", "LapNumber", "gap_ahead_s", "gap_behind_s", "position"]])

gaps_df = pd.concat(laps_list, ignore_index=True).rename(columns={"Driver": "driver", "LapNumber": "lap_number"})
df = meta.merge(gaps_df, on=["circuit", "driver", "lap_number"], how="left")
df["gap_ahead_s"] = df["gap_ahead_s"].fillna(60.0)
df["gap_behind_s"] = df["gap_behind_s"].fillna(60.0)
df["position"] = df["position"].fillna(10.0)

# Merge back with X_base features
X = X_base.copy()
X["gap_ahead_s"] = df["gap_ahead_s"]
X["gap_behind_s"] = df["gap_behind_s"]
X["position"] = df["position"]

# Target construction:
# Clean-air reference baseline residual per (circuit, compound) where gap_ahead >= 3.0s
clean_mask = df["gap_ahead_s"] >= 3.0
clean_ref = df[clean_mask].groupby(["circuit", X["compound"]])["residual"].median()

df["compound_str"] = X["compound"].astype(str)
df["circuit_str"] = df["circuit"].astype(str)

clean_ref_map = clean_ref.to_dict()
df["clean_residual_ref"] = df.apply(
    lambda r: clean_ref_map.get((r["circuit"], r["compound_str"]), 2.0),
    axis=1
)

# Target: additional lap time penalty attributable to traffic / dirty air:
# Strictly 0.0 when in clean air (gap_ahead >= 3.0s).
# When gap_ahead < 3.0s, penalty is excess residual above clean air expectation.
df["traffic_penalty_proxy_s"] = np.where(
    df["gap_ahead_s"] >= 3.0,
    0.0,
    np.maximum(0.0, df["residual"] - df["clean_residual_ref"]).clip(upper=5.0)
)
y_traffic = df["traffic_penalty_proxy_s"].copy()

print("Target summary:")
print(y_traffic.describe())
print(f"Zero penalty count: {(y_traffic == 0.0).sum()} / {len(y_traffic)} ({(y_traffic == 0.0).mean()*100:.1f}%)")
print(f"Non-zero penalty count: {(y_traffic > 0.0).sum()} / {len(y_traffic)} ({(y_traffic > 0.0).mean()*100:.1f}%)")

# Baselines:
# Baseline 1: Constant zero (predicting clean air everywhere)
zero_mae = mean_absolute_error(y_traffic, np.zeros_like(y_traffic))
zero_rmse = np.sqrt(mean_squared_error(y_traffic, np.zeros_like(y_traffic)))
print(f"\nBaseline 1 (All Zero): MAE={zero_mae:.4f}s, RMSE={zero_rmse:.4f}s")

# Baseline 2: Simple heuristic based strictly on gap:
# if gap < 1.0s -> 1.2s; if 1-2s -> 0.6s; else 0.0s
heuristic_pred = np.where(X["gap_ahead_s"] < 1.0, 1.2, np.where(X["gap_ahead_s"] < 2.0, 0.6, 0.0))
heur_mae = mean_absolute_error(y_traffic, heuristic_pred)
heur_rmse = np.sqrt(mean_squared_error(y_traffic, heuristic_pred))
heur_r2 = r2_score(y_traffic, heuristic_pred)
print(f"Baseline 2 (Simple Gap Heuristic): MAE={heur_mae:.4f}s, RMSE={heur_rmse:.4f}s, R2={heur_r2:.4f}")

# XGBoost Model
feature_cols = [c for c in X.columns if c != "driver"]
cat_cols = ["circuit", "compound"]
for c in cat_cols:
    X[c] = X[c].astype("category")

print("\n--- EVALUATION: GroupKFold by Stint (85 Stints) ---")
gkf = GroupKFold(n_splits=5)
oof = np.zeros(len(y_traffic))
for tr, val in gkf.split(X, y_traffic, groups=meta["stint_id"]):
    reg = xgb.XGBRegressor(
        random_state=42,
        enable_categorical=True,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    reg.fit(X[feature_cols].iloc[tr], y_traffic.iloc[tr])
    oof[val] = reg.predict(X[feature_cols].iloc[val])

oof_clipped = np.maximum(0.0, oof)
ml_mae = mean_absolute_error(y_traffic, oof_clipped)
ml_rmse = np.sqrt(mean_squared_error(y_traffic, oof_clipped))
ml_r2 = r2_score(y_traffic, oof_clipped)
print(f"XGBoost (Held-Out Stints): MAE={ml_mae:.4f}s, RMSE={ml_rmse:.4f}s, R2={ml_r2:.4f}")

print("\n--- EVALUATION: GroupKFold by Driver (20 Drivers - Stricter) ---")
oof_driver = np.zeros(len(y_traffic))
for tr, val in gkf.split(X, y_traffic, groups=meta["driver"]):
    reg = xgb.XGBRegressor(
        random_state=42,
        enable_categorical=True,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )
    reg.fit(X[feature_cols].iloc[tr], y_traffic.iloc[tr])
    oof_driver[val] = reg.predict(X[feature_cols].iloc[val])

oof_driver_clipped = np.maximum(0.0, oof_driver)
drv_mae = mean_absolute_error(y_traffic, oof_driver_clipped)
drv_rmse = np.sqrt(mean_squared_error(y_traffic, oof_driver_clipped))
drv_r2 = r2_score(y_traffic, oof_driver_clipped)
print(f"XGBoost (Held-Out Drivers): MAE={drv_mae:.4f}s, RMSE={drv_rmse:.4f}s, R2={drv_r2:.4f}")

# Feature importances
reg_full = xgb.XGBRegressor(
    random_state=42,
    enable_categorical=True,
    max_depth=3,
    learning_rate=0.05,
    n_estimators=100,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
)
reg_full.fit(X[feature_cols], y_traffic)
b = reg_full.get_booster()
gains = b.get_score(importance_type="gain")
weights = b.get_score(importance_type="weight")
imp = pd.DataFrame([{"feature": c, "gain": gains.get(c, 0.0), "weight": weights.get(c, 0.0)} for c in feature_cols]).sort_values("gain", ascending=False)
print("\nFeature Importances:")
print(imp.to_string(index=False))
