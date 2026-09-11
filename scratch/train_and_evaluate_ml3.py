"""Train, evaluate, and persist ML Model 3 (Traffic / Dirty-Air Penalty Model)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from backend.ml.traffic_penalty import (
    TRAFFIC_FEATURE_COLUMNS,
    TrafficPenaltyModel,
    prepare_traffic_data,
)


def main():
    print("Preparing traffic data from cached race telemetry...")
    X, y_traffic, metadata = prepare_traffic_data()

    print(f"Total samples: {len(X)}")
    print(f"Total stints: {metadata['stint_id'].nunique()}")
    print(f"Drivers: {metadata['driver'].nunique()}")
    print(f"Circuits: {metadata['circuit'].value_counts().to_dict()}")
    print(f"Target traffic_penalty stats: mean={y_traffic.mean():.4f}s, std={y_traffic.std():.4f}s, min={y_traffic.min():.4f}s, max={y_traffic.max():.4f}s")
    print(f"Clean air (zero penalty) proportion: {(y_traffic == 0.0).mean()*100:.1f}%")
    print(f"Traffic-affected (positive penalty) proportion: {(y_traffic > 0.0).mean()*100:.1f}%")

    model = TrafficPenaltyModel(
        random_state=42,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    print("\n--- EVALUATION 1: GroupKFold on Held-Out Stints (85 Stints) ---")
    stint_metrics = model.evaluate(
        X=X,
        y=y_traffic,
        groups=metadata["stint_id"],
        n_splits=5,
    )
    print(f"Validation Method: {stint_metrics['validation_method']}")
    print("Zero Baseline (Predict Clean Air Everywhere):")
    print(f"  MAE:  {stint_metrics['zero_baseline']['mae']:.4f} s")
    print(f"  RMSE: {stint_metrics['zero_baseline']['rmse']:.4f} s")

    print("Heuristic Baseline (Simple Gap Cutoffs):")
    print(f"  MAE:  {stint_metrics['heuristic_baseline']['mae']:.4f} s")
    print(f"  RMSE: {stint_metrics['heuristic_baseline']['rmse']:.4f} s")
    print(f"  R^2:  {stint_metrics['heuristic_baseline']['r2']:.4f}")

    print("TrafficPenaltyModel (XGBoost ML-3):")
    print(f"  MAE:  {stint_metrics['ml_model']['mae']:.4f} s")
    print(f"  RMSE: {stint_metrics['ml_model']['rmse']:.4f} s")
    print(f"  R^2:  {stint_metrics['ml_model']['r2']:.4f}")

    print("Improvements vs Heuristic Baseline:")
    print(f"  Delta MAE:  -{stint_metrics['improvements_vs_heuristic']['mae_improvement_s']:.4f} s (-{stint_metrics['improvements_vs_heuristic']['mae_improvement_pct']:.2f}%)")
    print(f"  Delta RMSE: -{stint_metrics['improvements_vs_heuristic']['rmse_improvement_s']:.4f} s")
    print(f"  Delta R^2:  +{stint_metrics['improvements_vs_heuristic']['r2_change']:.4f}")

    print("\n--- EVALUATION 2 (STRICTER): GroupKFold on Held-Out Drivers (20 Drivers) ---")
    driver_metrics = model.evaluate(
        X=X,
        y=y_traffic,
        groups=metadata["driver"],
        n_splits=5,
    )
    print(f"Validation Method: {driver_metrics['validation_method']}")
    print("TrafficPenaltyModel (XGBoost ML-3):")
    print(f"  MAE:  {driver_metrics['ml_model']['mae']:.4f} s")
    print(f"  RMSE: {driver_metrics['ml_model']['rmse']:.4f} s")
    print(f"  R^2:  {driver_metrics['ml_model']['r2']:.4f}")
    print("Improvements vs Heuristic Baseline:")
    print(f"  Delta MAE:  -{driver_metrics['improvements_vs_heuristic']['mae_improvement_s']:.4f} s (-{driver_metrics['improvements_vs_heuristic']['mae_improvement_pct']:.2f}%)")
    print(f"  Delta RMSE: -{driver_metrics['improvements_vs_heuristic']['rmse_improvement_s']:.4f} s")
    print(f"  Delta R^2:  +{driver_metrics['improvements_vs_heuristic']['r2_change']:.4f}")

    print("\nFitting final TrafficPenaltyModel on full dataset...")
    model.fit(X, y_traffic)

    booster = model.model.get_booster()
    score_gain = booster.get_score(importance_type="gain")
    score_weight = booster.get_score(importance_type="weight")

    importance_df = pd.DataFrame(
        [
            {
                "feature": col,
                "gain": score_gain.get(col, 0.0),
                "weight": score_weight.get(col, 0.0),
            }
            for col in TRAFFIC_FEATURE_COLUMNS
        ]
    ).sort_values("gain", ascending=False)

    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))

    # Persist model
    artifacts_dir = Path("backend/ml/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "traffic_penalty_model.joblib"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save summary json
    summary = {
        "metrics_held_out_stints": stint_metrics,
        "metrics_held_out_drivers": driver_metrics,
        "feature_importances": importance_df.to_dict(orient="records"),
        "dataset_summary": {
            "total_clean_laps": len(X),
            "total_stints": int(metadata["stint_id"].nunique()),
            "drivers": int(metadata["driver"].nunique()),
            "circuits": metadata["circuit"].value_counts().to_dict(),
            "target_stats": {
                "mean": float(y_traffic.mean()),
                "std": float(y_traffic.std()),
                "zero_pct": float((y_traffic == 0.0).mean() * 100),
                "traffic_affected_pct": float((y_traffic > 0.0).mean() * 100),
            },
        },
    }
    summary_path = artifacts_dir / "traffic_penalty_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
