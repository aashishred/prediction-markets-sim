"""Agent implementations for prediction market simulation."""

from .base import Agent, AgentState
from .informed import InformedAgent
from .noise import NoiseTrader, NoiseTraderType
from .discovery import (
    Signal,
    DiscoveryOpportunity,
    DiscoveryModel,
    DiscoveryRecord,
)
from .discoverer import DiscovererAgent, AdaptiveDiscoverer

__all__ = [
    # Base classes
    "Agent",
    "AgentState",
    # Agent types
    "InformedAgent",
    "NoiseTrader",
    "NoiseTraderType",
    "DiscovererAgent",
    "AdaptiveDiscoverer",
    # Discovery infrastructure
    "Signal",
    "DiscoveryOpportunity",
    "DiscoveryModel",
    "DiscoveryRecord",
]
