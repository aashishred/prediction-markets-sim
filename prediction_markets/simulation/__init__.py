"""Simulation infrastructure for running prediction market experiments."""

from .runner import Simulation, SimulationConfig, SimulationResult
from .metrics import (
    calculate_price_error,
    calculate_brier_score,
    calculate_information_ratio,
    calculate_welfare_distribution,
)

__all__ = [
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "calculate_price_error",
    "calculate_brier_score",
    "calculate_information_ratio",
    "calculate_welfare_distribution",
]
