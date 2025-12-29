"""
Prediction Markets Simulation Framework

An agent-based simulation for testing theoretical claims about prediction markets,
specifically examining the distinction between knowledge aggregation and discovery.

Quick Start:
    from prediction_markets import (
        LMSRMarket, BinaryContract,
        InformedAgent, NoiseTrader,
        UnifiedEnvironment, UnifiedConfig,
        Simulation, SimulationConfig,
    )
"""

__version__ = "0.3.0"

# Convenient top-level imports
from .markets import (
    Market,
    LMSRMarket,
    BinaryContract,
    MultinomialContract,
    Contract,
)
from .agents import (
    Agent,
    InformedAgent,
    NoiseTrader,
    NoiseTraderType,
    DiscovererAgent,
)
from .environments import (
    Environment,
    EnvironmentType,
    UnifiedEnvironment,
    UnifiedConfig,
    HayekianEnvironment,
    DiscoverableEnvironment,
)
from .simulation import (
    Simulation,
    SimulationConfig,
    SimulationResult,
    run_batch,
    calculate_estimation_error,
    calculate_brier_score,
    summarize_batch,
)

__all__ = [
    # Version
    "__version__",
    # Markets
    "Market",
    "LMSRMarket",
    "BinaryContract",
    "MultinomialContract",
    "Contract",
    # Agents
    "Agent",
    "InformedAgent",
    "NoiseTrader",
    "NoiseTraderType",
    "DiscovererAgent",
    # Environments
    "Environment",
    "EnvironmentType",
    "UnifiedEnvironment",
    "UnifiedConfig",
    "HayekianEnvironment",
    "DiscoverableEnvironment",
    # Simulation
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "run_batch",
    "calculate_estimation_error",
    "calculate_brier_score",
    "summarize_batch",
]
