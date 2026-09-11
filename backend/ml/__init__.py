"""Machine learning modules for TRACE."""

from backend.ml.tyre_degradation_residual import (
    TyreDegradationResidualModel,
    calculate_residual,
    prepare_training_data,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
)

from backend.ml.tyre_state import (
    TyreStateModel,
    compute_tyre_grip_target,
    prepare_tyre_state_data,
)

from backend.ml.traffic_penalty import (
    TrafficPenaltyModel,
    compute_traffic_penalty_target,
    prepare_traffic_data,
    TRAFFIC_FEATURE_COLUMNS,
)

from backend.ml.wet_performance import (
    WetPerformanceModel,
    compute_causal_wetness_proxy,
    compute_wet_performance_target,
    prepare_wet_performance_data,
    WET_PERFORMANCE_FEATURE_COLUMNS,
)

__all__ = [
    "TyreDegradationResidualModel",
    "calculate_residual",
    "prepare_training_data",
    "FEATURE_COLUMNS",
    "CATEGORICAL_COLUMNS",
    "NUMERIC_COLUMNS",
    "TyreStateModel",
    "compute_tyre_grip_target",
    "prepare_tyre_state_data",
    "TrafficPenaltyModel",
    "compute_traffic_penalty_target",
    "prepare_traffic_data",
    "TRAFFIC_FEATURE_COLUMNS",
    "WetPerformanceModel",
    "compute_causal_wetness_proxy",
    "compute_wet_performance_target",
    "prepare_wet_performance_data",
    "WET_PERFORMANCE_FEATURE_COLUMNS",
]

