"""
Base environment class for prediction market simulations.

An environment defines:
1. The true state of the world (what will actually happen)
2. How information about that state is distributed to agents
3. The dynamics of information revelation over time

Different environments test different aspects of the aggregation vs discovery thesis:
- Knightian: No information available - pure uncertainty
- Hayekian: Information distributed as tacit knowledge - tests aggregation
- Discoverable: Information available at cost - tests discovery incentives
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np

from ..agents.base import Agent
from ..markets.contracts import Contract


class EnvironmentType(Enum):
    """Types of information environments."""
    KNIGHTIAN = auto()    # No information leakage
    HAYEKIAN = auto()     # Distributed tacit knowledge
    DISCOVERABLE = auto()  # Information available at cost


@dataclass
class EnvironmentConfig:
    """
    Configuration for an environment.

    Attributes:
        n_outcomes: Number of possible outcomes
        true_outcome: The actual outcome (if predetermined)
        random_seed: Random seed for reproducibility
        leakiness: How much information leaks to agents (0 = none, 1 = full)
    """
    n_outcomes: int = 2
    true_outcome: int | float | None = None
    random_seed: int | None = None
    leakiness: float = 0.5


@dataclass
class Environment(ABC):
    """
    Abstract base class for simulation environments.

    An environment controls:
    - The ground truth (what will actually happen)
    - How signals are distributed to agents
    - What information can be discovered

    The key theoretical variable is 'leakiness': how much information about
    the true state leaks into the present through various channels.
    """
    config: EnvironmentConfig
    contract: Contract = field(default=None)
    rng: np.random.Generator = field(default=None)

    # State
    true_value: Any = field(default=None)  # The ground truth
    is_resolved: bool = False
    current_step: int = 0

    def __post_init__(self):
        if self.rng is None:
            seed = self.config.random_seed
            self.rng = np.random.default_rng(seed)

        # Generate true value if not specified
        if self.true_value is None:
            self.true_value = self._generate_true_value()

    @property
    @abstractmethod
    def environment_type(self) -> EnvironmentType:
        """Return the type of this environment."""
        pass

    @abstractmethod
    def _generate_true_value(self) -> Any:
        """
        Generate the true outcome/value for this environment.

        Called at initialization if true_value not specified.
        """
        pass

    @abstractmethod
    def generate_signals(self, agents: list[Agent]) -> dict[str, dict[str, Any]]:
        """
        Generate signals to distribute to agents.

        This is the key method that defines how information flows.

        Args:
            agents: List of agents to potentially receive signals

        Returns:
            Dictionary mapping agent_id to signal dict
        """
        pass

    def distribute_signals(self, agents: list[Agent]) -> None:
        """
        Distribute signals to agents according to the environment's rules.

        Args:
            agents: List of agents to receive signals
        """
        signals = self.generate_signals(agents)
        for agent in agents:
            if agent.agent_id in signals:
                agent.receive_signal(signals[agent.agent_id])

    def step(self) -> None:
        """Advance the environment by one timestep."""
        self.current_step += 1

    def resolve(self) -> Any:
        """
        Resolve the environment and return the true outcome.

        Returns:
            The true value/outcome
        """
        if self.contract is not None:
            self.contract.resolve(self.true_value)
        self.is_resolved = True
        return self.true_value

    def get_true_probabilities(self) -> np.ndarray:
        """
        Get the true probability distribution over outcomes.

        For deterministic true values, this is a one-hot vector.
        For stochastic environments, this may be a distribution.

        Returns:
            Array of true probabilities
        """
        if self.contract is None:
            raise ValueError("No contract associated with environment")

        n = self.contract.n_outcomes
        probs = np.zeros(n)

        if isinstance(self.true_value, (int, np.integer)):
            # Deterministic outcome
            probs[self.true_value] = 1.0
        else:
            # For continuous, return point mass at true value
            # Subclasses can override for more complex cases
            probs[0] = self.true_value  # For continuous contracts

        return probs

    def get_optimal_price(self) -> np.ndarray:
        """
        Get the price that a perfectly informed market should reach.

        Returns:
            Array of optimal prices (= true probabilities for discrete outcomes)
        """
        return self.get_true_probabilities()

    def calculate_aggregate_information(self, agents: list[Agent]) -> np.ndarray:
        """
        Calculate what the aggregated beliefs of all agents should be.

        This is useful for comparing market prices to the theoretical
        optimum if all private information were perfectly aggregated.

        Args:
            agents: List of all agents with their beliefs

        Returns:
            Aggregated belief distribution
        """
        if not agents:
            return np.ones(self.config.n_outcomes) / self.config.n_outcomes

        # Simple average of beliefs (could be weighted by wealth/confidence)
        beliefs = np.array([a.beliefs for a in agents if a.beliefs is not None])
        if len(beliefs) == 0:
            return np.ones(self.config.n_outcomes) / self.config.n_outcomes

        return np.mean(beliefs, axis=0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"type={self.environment_type.name}, "
            f"true_value={self.true_value}, "
            f"leakiness={self.config.leakiness})"
        )
