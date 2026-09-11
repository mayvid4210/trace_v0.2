import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd

from backend.ml.tyre_degradation_residual import (
    FEATURE_COLUMNS,
    TyreDegradationResidualModel,
    prepare_training_data,
)


def main():
    print("Preparing training data from cached race telemetry...")
    X, y, t_physics, metadata = prepare_training_data()

    print(f"Total samples: {len(X)}")
    print(f"Total stints: {metadata['stint_id'].nunique()}")
    print(f"Drivers: {metadata['driver'].nunique()}")
    print(f"Circuits: {metadata['circuit'].value_counts().to_dict()}")
    print(f"Compounds: {X['compound'].value_counts().to_dict()}")

    model = TyreDegradationResidualModel(
        random_state=42,
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
    )

    print("\nEvaluating 5-fold GroupKFold on held-out stints...")
    eval_metrics = model.evaluate(
        X=X,
        y=y,
        t_physics=t_physics,
        groups=metadata["stint_id"],
        n_splits=5,
    )

    print("\n--- EVALUATION 1: GroupKFold on Held-Out Stints (85 Stints) ---")
    print(f"Validation Method: {eval_metrics['validation_method']}")
    print("Baseline Physics:")
    print(f"  MAE:  {eval_metrics['baseline_physics']['mae']:.4f} s")
    print(f"  RMSE: {eval_metrics['baseline_physics']['rmse']:.4f} s")
    print(f"  R^2:  {eval_metrics['baseline_physics']['r2']:.4f}")

    print("Composite (Physics + XGBoost Residual):")
    print(f"  MAE:  {eval_metrics['composite']['mae']:.4f} s")
    print(f"  RMSE: {eval_metrics['composite']['rmse']:.4f} s")
    print(f"  R^2:  {eval_metrics['composite']['r2']:.4f}")

    print("Residual Model:")
    print(f"  MAE:  {eval_metrics['residual']['mae']:.4f} s")
    print(f"  RMSE: {eval_metrics['residual']['rmse']:.4f} s")
    print(f"  R^2:  {eval_metrics['residual']['r2']:.4f}")

    print("Delta vs Baseline:")
    print(f"  Delta MAE:  -{eval_metrics['improvements']['mae_improvement_s']:.4f} s (-{eval_metrics['improvements']['mae_improvement_pct']:.2f}%)")
    print(f"  Delta RMSE: -{eval_metrics['improvements']['rmse_improvement_s']:.4f} s (-{eval_metrics['improvements']['rmse_improvement_pct']:.2f}%)")
    print(f"  Delta R^2:  +{eval_metrics['improvements']['r2_change']:.4f}")

    print("\n--- EVALUATION 2 (STRICTER): GroupKFold on Held-Out Drivers (20 Drivers) ---")
    strict_metrics = model.evaluate(
        X=X,
        y=y,
        t_physics=t_physics,
        groups=metadata["driver"],
        n_splits=5,
    )
    print(f"Validation Method: {strict_metrics['validation_method']}")
    print("Composite (Physics + XGBoost Residual):")
    print(f"  MAE:  {strict_metrics['composite']['mae']:.4f} s")
    print(f"  RMSE: {strict_metrics['composite']['rmse']:.4f} s")
    print(f"  R^2:  {strict_metrics['composite']['r2']:.4f}")
    print("Delta vs Baseline:")
    print(f"  Delta MAE:  -{strict_metrics['improvements']['mae_improvement_s']:.4f} s (-{strict_metrics['improvements']['mae_improvement_pct']:.2f}%)")
    print(f"  Delta RMSE: -{strict_metrics['improvements']['rmse_improvement_s']:.4f} s (-{strict_metrics['improvements']['rmse_improvement_pct']:.2f}%)")
    print(f"  Delta R^2:  +{strict_metrics['improvements']['r2_change']:.4f}")

    print("\nFitting final model on full dataset...")
    model.fit(X, y)

    # Feature importances
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
    model_path = artifacts_dir / "tyre_residual_model.joblib"
    model.save(model_path)
    print(f"\nModel saved to {model_path}")

    # Save summary json
    summary = {
        "metrics_held_out_stints": eval_metrics,
        "metrics_held_out_drivers": strict_metrics,
        "feature_importances": importance_df.to_dict(orient="records"),
        "dataset_summary": {
            "total_clean_laps": len(X),
            "total_stints": int(metadata["stint_id"].nunique()),
            "drivers": int(metadata["driver"].nunique()),
            "circuits": metadata["circuit"].value_counts().to_dict(),
            "compounds": X["compound"].value_counts().to_dict(),
        },
    }
    summary_path = artifacts_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
