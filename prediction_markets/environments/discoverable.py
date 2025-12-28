"""
Discoverable environment where information exists but must be paid for.

This models the case where information has "leaked into the environment"
but not directly into anyone's mind. Agents can pay discovery costs to
learn it, then trade on what they've discovered.

This is distinct from:
- Knightian: No information exists to discover
- Hayekian: Information is already in minds (tacit knowledge), just needs aggregation

The key hypothesis: Markets that incentivise discovery create more value
than markets that only aggregate existing knowledge.

Examples of discoverable information:
- Scientific data requiring analysis
- Investigative journalism uncovering facts
- Due diligence research on a company
- Experiments that reveal product-market fit
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Environment, EnvironmentConfig, EnvironmentType
from ..agents.base import Agent
from ..agents.discovery import DiscoveryOpportunity, Signal


@dataclass
class DiscoverableConfig(EnvironmentConfig):
    """
    Configuration for Discoverable environment.

    Attributes:
        n_hidden_signals: Number of discoverable pieces of information
        signal_weights: How much each signal contributes to true value
        discovery_cost: Base cost to attempt discovery
        success_probability: Base probability of successful discovery
        signal_validity: How accurate discovered signals are
        validity_uncertainty: Uncertainty in signal accuracy
        bearing_known: Whether agents know how important each signal is
        cost_varies: Whether different signals have different costs
        diminishing_returns: Whether later discoveries are less valuable
    """
    n_hidden_signals: int = 10
    signal_weights: np.ndarray | None = None
    discovery_cost: float = 5.0
    success_probability: float = 0.8
    signal_validity: float = 0.9
    validity_uncertainty: float = 0.1
    bearing_known: bool = False  # If False, agents don't know signal weights
    cost_varies: bool = False     # If True, some signals cost more
    diminishing_returns: bool = True  # If True, later discoveries less valuable (DEFAULT ON)
    time_limit: int | None = None  # If set, discovery only allowed until this step


@dataclass
class DiscoverableEnvironment(Environment):
    """
    Environment where information exists but must be discovered at a cost.

    The true outcome is determined by hidden signals that agents can discover
    by paying costs. This models the case where market value comes from
    incentivising research and discovery, not just aggregating existing knowledge.

    Key features:
    - Information exists objectively (unlike Knightian uncertainty)
    - But it's not in anyone's mind yet (unlike Hayekian tacit knowledge)
    - Agents must pay to discover it
    - Discovery is uncertain (may fail, may be noisy)
    - Multiple signals can be discovered, each with different importance
    """
    config: DiscoverableConfig = field(default_factory=DiscoverableConfig)

    # Hidden state (the ground truth that can be discovered)
    _hidden_signals: np.ndarray = field(default=None, repr=False)
    _signal_weights: np.ndarray = field(default=None, repr=False)
    _signal_costs: np.ndarray = field(default=None, repr=False)

    # Discovery tracking
    _discovery_opportunities: list[DiscoveryOpportunity] = field(default_factory=list)
    _discovered_by: dict[int, list[str]] = field(default_factory=dict)  # signal_idx -> agent_ids

    def __post_init__(self):
        # Initialize rng first
        if self.rng is None:
            seed = self.config.random_seed
            self.rng = np.random.default_rng(seed)

        # Set up signal weights
        if self.config.signal_weights is not None:
            self._signal_weights = self.config.signal_weights
        else:
            n = self.config.n_hidden_signals
            self._signal_weights = np.ones(n) / n

        # Generate hidden signals
        self._generate_hidden_signals()

        # Set up discovery opportunities
        self._setup_discovery_opportunities()

        super().__post_init__()

    @property
    def environment_type(self) -> EnvironmentType:
        return EnvironmentType.DISCOVERABLE

    def _generate_hidden_signals(self) -> None:
        """Generate the hidden signals that determine the true value."""
        n = self.config.n_hidden_signals

        if self.config.n_outcomes == 2:
            # Binary: signals are 0 or 1
            self._hidden_signals = self.rng.binomial(1, 0.5, size=n)
        else:
            # Multinomial: signals vote for outcomes
            self._hidden_signals = self.rng.integers(0, self.config.n_outcomes, size=n)

        # Set up costs per signal
        if self.config.cost_varies:
            # Some signals more expensive (correlated with importance?)
            self._signal_costs = self.config.discovery_cost * (0.5 + self.rng.random(n))
        else:
            self._signal_costs = np.full(n, self.config.discovery_cost)

    def _setup_discovery_opportunities(self) -> None:
        """Create discovery opportunities for each hidden signal."""
        self._discovery_opportunities = []
        self._discovered_by = {i: [] for i in range(self.config.n_hidden_signals)}

        for i in range(self.config.n_hidden_signals):
            # Bearing: how much this signal matters
            true_bearing = self._signal_weights[i]

            # What agents expect the bearing to be
            if self.config.bearing_known:
                expected_bearing = true_bearing
                bearing_uncertainty = 0.01
            else:
                # Agents don't know - use average as expectation
                expected_bearing = 1.0 / self.config.n_hidden_signals
                bearing_uncertainty = 0.1

            # Diminishing returns: later signals less valuable
            if self.config.diminishing_returns:
                success_prob = self.config.success_probability * (0.5 + 0.5 * (1 - i / self.config.n_hidden_signals))
            else:
                success_prob = self.config.success_probability

            opportunity = DiscoveryOpportunity(
                cost=self._signal_costs[i],
                success_probability=success_prob,
                expected_validity=self.config.signal_validity,
                expected_bearing=expected_bearing,
                validity_uncertainty=self.config.validity_uncertainty,
                bearing_uncertainty=bearing_uncertainty,
            )
            self._discovery_opportunities.append(opportunity)

    def _generate_true_value(self) -> int:
        """Generate true value from hidden signals."""
        if self.config.n_outcomes == 2:
            prob = np.average(self._hidden_signals, weights=self._signal_weights)
            return int(self.rng.random() < prob)
        else:
            votes = np.zeros(self.config.n_outcomes)
            for i, signal in enumerate(self._hidden_signals):
                votes[signal] += self._signal_weights[i]
            return int(np.argmax(votes))

    def get_discovery_opportunities(self, agent_id: str) -> list[tuple[int, DiscoveryOpportunity]]:
        """
        Get available discovery opportunities for an agent.

        Returns list of (signal_index, opportunity) tuples for signals
        the agent hasn't already discovered.
        """
        available = []
        for i, opportunity in enumerate(self._discovery_opportunities):
            if agent_id not in self._discovered_by[i]:
                available.append((i, opportunity))
        return available

    def attempt_discovery(
        self,
        agent_id: str,
        signal_index: int
    ) -> tuple[Any, float]:
        """
        Get the true signal and bearing for a discovery attempt.

        The agent's DiscoveryModel will add noise; this just provides ground truth.

        Args:
            agent_id: ID of attempting agent
            signal_index: Which signal to discover

        Returns:
            Tuple of (true_signal, true_bearing)
        """
        if signal_index < 0 or signal_index >= self.config.n_hidden_signals:
            raise ValueError(f"Invalid signal index: {signal_index}")

        true_signal = self._hidden_signals[signal_index]
        true_bearing = self._signal_weights[signal_index]

        return true_signal, true_bearing

    def record_discovery(self, agent_id: str, signal_index: int) -> None:
        """Record that an agent has discovered a signal."""
        if agent_id not in self._discovered_by[signal_index]:
            self._discovered_by[signal_index].append(agent_id)

    def generate_signals(self, agents: list[Agent]) -> dict[str, dict[str, Any]]:
        """
        In Discoverable environment, signals are NOT distributed passively.

        Agents must actively discover them. This method returns empty
        for all agents - they start with no information.
        """
        # No passive signals - agents must discover
        return {}

    def get_hidden_state(self) -> dict[str, Any]:
        """
        Get the hidden state for analysis.

        This reveals ground truth - only for post-simulation analysis.
        """
        return {
            "n_signals": self.config.n_hidden_signals,
            "hidden_signals": self._hidden_signals.tolist(),
            "signal_weights": self._signal_weights.tolist(),
            "signal_costs": self._signal_costs.tolist(),
            "true_value": self.true_value,
            "discovery_counts": {
                i: len(agents) for i, agents in self._discovered_by.items()
            },
        }

    def theoretical_aggregate(self) -> np.ndarray:
        """
        Calculate what perfect aggregation of all hidden signals would give.

        This is the "oracle" price if someone knew everything.
        """
        if self.config.n_outcomes == 2:
            prob = np.average(self._hidden_signals, weights=self._signal_weights)
            return np.array([1 - prob, prob])
        else:
            votes = np.zeros(self.config.n_outcomes)
            for i, signal in enumerate(self._hidden_signals):
                votes[signal] += self._signal_weights[i]
            return votes / votes.sum()

    def total_discovery_cost(self) -> float:
        """Total cost to discover all signals."""
        return float(np.sum(self._signal_costs))

    def information_discovered_fraction(self) -> float:
        """Fraction of total information that has been discovered by anyone."""
        total_weight = np.sum(self._signal_weights)
        discovered_weight = sum(
            self._signal_weights[i]
            for i, agents in self._discovered_by.items()
            if len(agents) > 0
        )
        return discovered_weight / total_weight if total_weight > 0 else 0.0

    def __repr__(self) -> str:
        discovered = sum(1 for agents in self._discovered_by.values() if agents)
        return (
            f"DiscoverableEnvironment("
            f"n_hidden={self.config.n_hidden_signals}, "
            f"discovered={discovered}, "
            f"cost={self.config.discovery_cost}, "
            f"true_value={self.true_value})"
        )
