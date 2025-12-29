"""Simulation infrastructure for running prediction market experiments."""

from .runner import Simulation, SimulationConfig, SimulationResult, run_batch
from .metrics import (
    # Primary metrics (use these!)
    calculate_estimation_error,
    calculate_outcome_error,
    calculate_brier_score,
    calculate_calibration_curve,
    # Comparison metrics
    calculate_baseline_comparisons,
    calculate_information_ratio,
    calculate_aggregation_efficiency,
    # Analysis metrics
    calculate_convergence_speed,
    calculate_welfare_distribution,
    calculate_trading_activity,
    # Summary
    summarize_batch,
    # Deprecated (for backward compatibility)
    calculate_price_error,
    calculate_price_error_vs_aggregate,
)

__all__ = [
    # Simulation runner
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "run_batch",
    # Primary metrics
    "calculate_estimation_error",
    "calculate_outcome_error",
    "calculate_brier_score",
    "calculate_calibration_curve",
    # Comparison metrics
    "calculate_baseline_comparisons",
    "calculate_information_ratio",
    "calculate_aggregation_efficiency",
    # Analysis metrics
    "calculate_convergence_speed",
    "calculate_welfare_distribution",
    "calculate_trading_activity",
    # Summary
    "summarize_batch",
    # Deprecated
    "calculate_price_error",
    "calculate_price_error_vs_aggregate",
]
