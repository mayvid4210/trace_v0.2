"""Train, evaluate, and persist ML Model 2 (Tyre State Model)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from backend.ml.tyre_state import (
    FEATURE_COLUMNS,
    TyreStateModel,
    prepare_tyre_state_data,
)


def main():
    print("Preparing tyre state data from race telemetry...")
    X, y_grip, y_deg_s, metadata = prepare_tyre_state_data()

    print(f"Total samples: {len(X)}")
    print(f"Total stints: {metadata['stint_id'].nunique()}")
    print(f"Drivers: {metadata['driver'].nunique()}")
    print(f"Circuits: {metadata['circuit'].value_counts().to_dict()}")
    print(f"Grip target stats: mean={y_grip.mean():.4f}, std={y_grip.std():.4f}, min={y_grip.min():.4f}, max={y_grip.max():.4f}")
    print(f"Degradation seconds stats: mean={y_deg_s.mean():.4f}s, std={y_deg_s.std():.4f}s, max={y_deg_s.max():.4f}s")

    model = TyreStateModel(
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
        y=y_grip,
        groups=metadata["stint_id"],
        n_splits=5,
    )
    print(f"Validation Method: {stint_metrics['validation_method']}")
    print("Constant Mean Baseline:")
    print(f"  MAE:  {stint_metrics['mean_baseline']['mae']:.5f}")
    print(f"  RMSE: {stint_metrics['mean_baseline']['rmse']:.5f}")
    print(f"  R^2:  {stint_metrics['mean_baseline']['r2']:.4f}")

    print("Linear Physics Degradation Baseline:")
    print(f"  MAE:  {stint_metrics['linear_physics_baseline']['mae']:.5f}")
    print(f"  RMSE: {stint_metrics['linear_physics_baseline']['rmse']:.5f}")
    print(f"  R^2:  {stint_metrics['linear_physics_baseline']['r2']:.4f}")

    print("TyreStateModel (XGBoost):")
    print(f"  MAE:  {stint_metrics['ml_model']['mae']:.5f}")
    print(f"  RMSE: {stint_metrics['ml_model']['rmse']:.5f}")
    print(f"  R^2:  {stint_metrics['ml_model']['r2']:.4f}")

    print("Improvements vs Linear Physics Baseline:")
    print(f"  Delta MAE:  -{stint_metrics['improvements_vs_linear_physics']['mae_improvement']:.5f} (-{stint_metrics['improvements_vs_linear_physics']['mae_improvement_pct']:.2f}%)")
    print(f"  Delta RMSE: -{stint_metrics['improvements_vs_linear_physics']['rmse_improvement']:.5f}")
    print(f"  Delta R^2:  +{stint_metrics['improvements_vs_linear_physics']['r2_change']:.4f}")

    print("\n--- EVALUATION 2 (STRICTER): GroupKFold on Held-Out Drivers (20 Drivers) ---")
    driver_metrics = model.evaluate(
        X=X,
        y=y_grip,
        groups=metadata["driver"],
        n_splits=5,
    )
    print(f"Validation Method: {driver_metrics['validation_method']}")
    print("TyreStateModel (XGBoost):")
    print(f"  MAE:  {driver_metrics['ml_model']['mae']:.5f}")
    print(f"  RMSE: {driver_metrics['ml_model']['rmse']:.5f}")
    print(f"  R^2:  {driver_metrics['ml_model']['r2']:.4f}")
    print("Improvements vs Linear Physics Baseline:")
    print(f"  Delta MAE:  -{driver_metrics['improvements_vs_linear_physics']['mae_improvement']:.5f} (-{driver_metrics['improvements_vs_linear_physics']['mae_improvement_pct']:.2f}%)")
    print(f"  Delta RMSE: -{driver_metrics['improvements_vs_linear_physics']['rmse_improvement']:.5f}")
    print(f"  Delta R^2:  +{driver_metrics['improvements_vs_linear_physics']['r2_change']:.4f}")

    print("\nFitting final TyreStateModel on full dataset...")
    model.fit(X, y_grip)

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
            for col in FEATURE_COLUMNS
        ]
    ).sort_values("gain", ascending=False)

    print("\nFeature Importances:")
    print(importance_df.to_string(index=False))

    # Persist model
    artifacts_dir = Path("backend/ml/artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "tyre_state_model.joblib"
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
            "compounds": X["compound"].value_counts().to_dict(),
            "target_grip_stats": {
                "mean": float(y_grip.mean()),
                "std": float(y_grip.std()),
                "min": float(y_grip.min()),
                "max": float(y_grip.max()),
            },
        },
    }
    summary_path = artifacts_dir / "tyre_state_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
