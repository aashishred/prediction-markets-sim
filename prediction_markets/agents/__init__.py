"""Agent implementations for prediction market simulation."""

from .base import Agent, AgentState
from .informed import InformedAgent
from .noise import NoiseTrader
from .discovery import (
    Signal,
    DiscoveryOpportunity,
    DiscoveryModel,
    DiscoveryRecord,
)
from .discoverer import DiscovererAgent, AdaptiveDiscoverer

__all__ = [
    "Agent",
    "AgentState",
    "InformedAgent",
    "NoiseTrader",
    "Signal",
    "DiscoveryOpportunity",
    "DiscoveryModel",
    "DiscoveryRecord",
    "DiscovererAgent",
    "AdaptiveDiscoverer",
]
