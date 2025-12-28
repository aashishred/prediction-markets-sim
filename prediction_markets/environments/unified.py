"""
Unified environment with continuous leakiness parameter.

This replaces the discrete three-scenario system (Knightian, Hayekian, Discoverable)
with a continuous leakiness spectrum from 0.0 to 1.0:

LEAKINESS SPECTRUM:
- 0.0: Pure Knightian uncertainty - no information available anywhere
- 0.3: Low leakiness - minimal distributed tacit knowledge
- 0.5: Medium leakiness - substantial distributed knowledge (classic Hayekian)
- 0.7: High leakiness - most information accessible
- 1.0: Perfect leakiness - all information freely available

The leakiness parameter controls:
1. Fraction of total information that "leaks" into the present
2. Distribution: low leakiness → centralized, high leakiness → discoverable
3. Access cost: low leakiness → free (in agent minds), high leakiness → costly discovery

MECHANISM:
- True value determined by hidden signals (as in Hayekian/Discoverable)
- Leakiness determines what % of signals leak into the environment
- Of leaked signals:
  - Fraction goes directly to agent minds (Hayekian-style, no cost)
  - Fraction requires discovery (Discoverable-style, costs money)
- Higher leakiness → more signals available, but more require discovery
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Environment, EnvironmentConfig, EnvironmentType
from ..agents.base import Agent
from ..agents.discovery import DiscoveryOpportunity


@dataclass
class UnifiedConfig(EnvironmentConfig):
    """
    Configuration for unified leakiness-based environment.

    Attributes:
        leakiness: Core parameter (0.0 to 1.0) controlling information availability
        n_signals: Total number of signals that determine true value
        signal_weights: How much each signal contributes (default: equal)
        signal_precision: How accurately agents observe their signals
        signals_per_agent: How many free signals each agent receives
        discovery_base_cost: Base cost to discover a signal
        discovery_success_prob: Base probability of successful discovery
        signal_validity: How accurate discovered signals are
    """
    leakiness: float = 0.5  # Core parameter: 0 (Knightian) to 1 (full disclosure)
    n_signals: int = 20
    signal_weights: np.ndarray | None = None
    signal_precision: float = 0.85
    signals_per_agent: float = 2.0
    discovery_base_cost: float = 5.0
    discovery_success_prob: float = 0.8
    signal_validity: float = 0.9

    def __post_init__(self):
        """Validate leakiness parameter."""
        if not 0.0 <= self.leakiness <= 1.0:
            raise ValueError(f"leakiness must be in [0, 1], got {self.leakiness}")


@dataclass
class UnifiedEnvironment(Environment):
    """
    Unified environment with continuous leakiness parameter.

    The leakiness parameter determines the information regime:

    LOW LEAKINESS (0.0 - 0.3): Knightian-like
    - Very little information available
    - What exists is mostly in agent minds (cheap)
    - Little opportunity for discovery
    - Market has limited value-add

    MEDIUM LEAKINESS (0.3 - 0.7): Hayekian-like
    - Substantial distributed tacit knowledge
    - Information split between free (in minds) and discoverable
    - Market aggregates and incentivizes discovery
    - Classic prediction market use case

    HIGH LEAKINESS (0.8 - 1.0): Discoverable-like
    - Most/all information exists but requires discovery
    - Discovery is the primary mechanism
    - Market value comes from incentivizing research

    MECHANISM:
    1. Generate n_signals that determine true value
    2. Fraction (leakiness) of signals "leak" into present
    3. Of leaked signals:
       - free_fraction = (1 - leakiness) → distributed to agents (Hayekian)
       - discovery_fraction = leakiness → available for discovery
    4. Discovery cost scales inversely with leakiness (more leaky = cheaper)
    """
    config: UnifiedConfig = field(default_factory=UnifiedConfig)

    # Hidden state
    _signals: np.ndarray = field(default=None, repr=False)
    _signal_weights: np.ndarray = field(default=None, repr=False)

    # What information is available where
    _leaked_signals: set[int] = field(default_factory=set, repr=False)  # Which signals leaked
    _free_signals: set[int] = field(default_factory=set, repr=False)  # Which can be distributed free
    _discoverable_signals: set[int] = field(default_factory=set, repr=False)  # Which require discovery

    # Agent assignments
    _agent_signals: dict[str, list[int]] = field(default_factory=dict, repr=False)

    # Discovery tracking
    _discovery_opportunities: dict[int, DiscoveryOpportunity] = field(default_factory=dict, repr=False)
    _discovered_by: dict[int, list[str]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Initialize rng first
        if self.rng is None:
            seed = self.config.random_seed
            self.rng = np.random.default_rng(seed)

        # Set up signal weights
        if self.config.signal_weights is not None:
            self._signal_weights = self.config.signal_weights
        else:
            n = self.config.n_signals
            self._signal_weights = np.ones(n) / n

        # Generate signals and partition by leakiness
        self._generate_signals()
        self._partition_signals_by_leakiness()
        self._setup_discovery_opportunities()

        super().__post_init__()

    @property
    def environment_type(self) -> EnvironmentType:
        """
        Return environment type based on leakiness.

        This maintains backward compatibility with the discrete types.
        """
        if self.config.leakiness < 0.3:
            return EnvironmentType.KNIGHTIAN
        elif self.config.leakiness < 0.7:
            return EnvironmentType.HAYEKIAN
        else:
            return EnvironmentType.DISCOVERABLE

    def _generate_signals(self) -> None:
        """Generate the hidden signals that determine true value."""
        n = self.config.n_signals

        if self.config.n_outcomes == 2:
            # Binary: signals are 0 or 1
            self._signals = self.rng.binomial(1, 0.5, size=n)
        else:
            # Multinomial: signals vote for outcomes
            self._signals = self.rng.integers(0, self.config.n_outcomes, size=n)

    def _partition_signals_by_leakiness(self) -> None:
        """
        Partition signals based on leakiness parameter.

        ALGORITHM:
        1. Determine how many signals leak: n_leaked = leakiness * n_signals
        2. Of leaked signals, partition into free vs discoverable:
           - free_fraction = (1 - leakiness)  # Low leakiness → more free
           - discovery_fraction = leakiness    # High leakiness → more discovery
        3. Remaining signals stay hidden (Knightian uncertainty)
        """
        n = self.config.n_signals
        leakiness = self.config.leakiness

        # How many signals leak into the present?
        n_leaked = int(leakiness * n)

        # Randomly select which signals leak
        all_indices = np.arange(n)
        self.rng.shuffle(all_indices)
        self._leaked_signals = set(all_indices[:n_leaked].tolist())

        # Partition leaked signals into free vs discoverable
        leaked_list = list(self._leaked_signals)
        self.rng.shuffle(leaked_list)

        # Lower leakiness → more signals are free (in agent minds)
        # Higher leakiness → more signals require discovery
        free_fraction = (1 - leakiness)
        n_free = int(len(leaked_list) * free_fraction)

        self._free_signals = set(leaked_list[:n_free])
        self._discoverable_signals = set(leaked_list[n_free:])

    def _setup_discovery_opportunities(self) -> None:
        """Create discovery opportunities for discoverable signals."""
        self._discovery_opportunities = {}
        self._discovered_by = {i: [] for i in self._discoverable_signals}

        for i in self._discoverable_signals:
            # Discovery cost scales inversely with leakiness
            # High leakiness = cheaper discovery (info is "more leaked")
            cost_multiplier = 2.0 - self.config.leakiness  # 2.0 at leakiness=0, 1.0 at leakiness=1
            cost = self.config.discovery_base_cost * cost_multiplier

            # Bearing: how much this signal matters
            true_bearing = self._signal_weights[i]
            expected_bearing = 1.0 / self.config.n_signals

            opportunity = DiscoveryOpportunity(
                cost=cost,
                success_probability=self.config.discovery_success_prob,
                expected_validity=self.config.signal_validity,
                expected_bearing=expected_bearing,
                validity_uncertainty=0.1,
                bearing_uncertainty=0.1,
            )
            self._discovery_opportunities[i] = opportunity

    def _generate_true_value(self) -> int:
        """Generate true value from ALL signals (including hidden ones)."""
        if self.config.n_outcomes == 2:
            prob = np.average(self._signals, weights=self._signal_weights)
            return int(self.rng.random() < prob)
        else:
            votes = np.zeros(self.config.n_outcomes)
            for i, signal in enumerate(self._signals):
                votes[signal] += self._signal_weights[i]
            return int(np.argmax(votes))

    def _assign_free_signals_to_agents(self, agents: list[Agent]) -> None:
        """
        Assign free signals to agents (Hayekian-style distribution).

        Only free signals (those in agent minds) are distributed.
        Discoverable signals must be actively acquired.
        """
        n_agents = len(agents)
        if n_agents == 0 or not self._free_signals:
            self._agent_signals = {agent.agent_id: [] for agent in agents}
            return

        self._agent_signals = {agent.agent_id: [] for agent in agents}

        # Distribute free signals randomly
        free_list = list(self._free_signals)

        # Each agent gets signals_per_agent signals on average
        for agent in agents:
            n_to_assign = max(0, int(self.rng.poisson(self.config.signals_per_agent)))
            n_to_assign = min(n_to_assign, len(free_list))

            if n_to_assign > 0:
                assigned = self.rng.choice(free_list, size=n_to_assign, replace=False)
                self._agent_signals[agent.agent_id] = assigned.tolist()

    def generate_signals(self, agents: list[Agent]) -> dict[str, dict[str, Any]]:
        """
        Generate signals to distribute to agents.

        Only FREE signals are distributed. Discoverable signals must be
        actively discovered through the discovery mechanism.
        """
        # Assign free signals if not already done
        if not self._agent_signals or set(self._agent_signals.keys()) != {a.agent_id for a in agents}:
            self._assign_free_signals_to_agents(agents)

        result = {}

        for agent in agents:
            signal_indices = self._agent_signals.get(agent.agent_id, [])
            if not signal_indices:
                continue

            # For binary outcomes
            if self.config.n_outcomes == 2:
                observed_signals = []
                observed_weights = []

                for idx in signal_indices:
                    true_signal = self._signals[idx]
                    # Observe with some precision
                    if self.rng.random() < self.config.signal_precision:
                        observed = true_signal
                    else:
                        observed = 1 - true_signal
                    observed_signals.append(observed)
                    observed_weights.append(self._signal_weights[idx])

                # Implied probability from observed signals
                if observed_weights:
                    weight_sum = sum(observed_weights)
                    implied_prob = sum(s * w for s, w in zip(observed_signals, observed_weights)) / weight_sum
                else:
                    implied_prob = 0.5

                result[agent.agent_id] = {
                    "outcome": 1 if implied_prob > 0.5 else 0,
                    "precision": self.config.signal_precision,
                    "strength": len(signal_indices) / self.config.n_signals,
                    "implied_probability": implied_prob,
                    "n_signals": len(signal_indices),
                }
            else:
                # Multinomial outcomes
                votes = np.zeros(self.config.n_outcomes)
                for idx in signal_indices:
                    true_signal = self._signals[idx]
                    if self.rng.random() < self.config.signal_precision:
                        observed = true_signal
                    else:
                        observed = self.rng.integers(0, self.config.n_outcomes)
                    votes[observed] += self._signal_weights[idx]

                result[agent.agent_id] = {
                    "outcome": int(np.argmax(votes)),
                    "precision": self.config.signal_precision,
                    "strength": len(signal_indices) / self.config.n_signals,
                    "vote_distribution": votes / votes.sum() if votes.sum() > 0 else None,
                    "n_signals": len(signal_indices),
                }

        return result

    def get_discovery_opportunities(self, agent_id: str) -> list[tuple[int, DiscoveryOpportunity]]:
        """
        Get available discovery opportunities for an agent.

        Returns list of (signal_index, opportunity) tuples for discoverable
        signals the agent hasn't already discovered.
        """
        available = []
        for signal_idx, opportunity in self._discovery_opportunities.items():
            if agent_id not in self._discovered_by[signal_idx]:
                available.append((signal_idx, opportunity))
        return available

    def attempt_discovery(self, agent_id: str, signal_index: int) -> tuple[Any, float]:
        """
        Get the true signal and bearing for a discovery attempt.

        The agent's DiscoveryModel will add noise; this just provides ground truth.

        Args:
            agent_id: ID of attempting agent
            signal_index: Which signal to discover

        Returns:
            Tuple of (true_signal, true_bearing)
        """
        if signal_index not in self._discoverable_signals:
            raise ValueError(f"Signal {signal_index} is not discoverable")

        true_signal = self._signals[signal_index]
        true_bearing = self._signal_weights[signal_index]

        return true_signal, true_bearing

    def record_discovery(self, agent_id: str, signal_index: int) -> None:
        """Record that an agent has discovered a signal."""
        if signal_index in self._discovered_by and agent_id not in self._discovered_by[signal_index]:
            self._discovered_by[signal_index].append(agent_id)

    def theoretical_aggregate(self) -> np.ndarray:
        """
        Calculate what perfect aggregation of ALL LEAKED signals would give.

        This is the oracle price if someone knew everything that leaked,
        but NOT the hidden signals (those that didn't leak).
        """
        # Only aggregate leaked signals
        leaked_indices = list(self._leaked_signals)

        if not leaked_indices:
            # No information leaked → return prior
            return np.ones(self.config.n_outcomes) / self.config.n_outcomes

        leaked_signals = self._signals[leaked_indices]
        leaked_weights = self._signal_weights[leaked_indices]

        if self.config.n_outcomes == 2:
            prob = np.average(leaked_signals, weights=leaked_weights)
            return np.array([1 - prob, prob])
        else:
            votes = np.zeros(self.config.n_outcomes)
            for i, signal in enumerate(leaked_signals):
                votes[signal] += leaked_weights[i]
            return votes / votes.sum()

    def get_information_structure(self) -> dict[str, Any]:
        """
        Get detailed breakdown of information structure.

        Useful for understanding what's available where.
        """
        total_weight = np.sum(self._signal_weights)
        leaked_weight = np.sum([self._signal_weights[i] for i in self._leaked_signals])
        free_weight = np.sum([self._signal_weights[i] for i in self._free_signals])
        discoverable_weight = np.sum([self._signal_weights[i] for i in self._discoverable_signals])
        hidden_weight = total_weight - leaked_weight

        return {
            "leakiness": self.config.leakiness,
            "n_signals": {
                "total": self.config.n_signals,
                "leaked": len(self._leaked_signals),
                "free": len(self._free_signals),
                "discoverable": len(self._discoverable_signals),
                "hidden": self.config.n_signals - len(self._leaked_signals),
            },
            "weight_distribution": {
                "total": float(total_weight),
                "leaked": float(leaked_weight),
                "free": float(free_weight),
                "discoverable": float(discoverable_weight),
                "hidden": float(hidden_weight),
            },
            "percentages": {
                "leaked": float(leaked_weight / total_weight * 100) if total_weight > 0 else 0,
                "free": float(free_weight / total_weight * 100) if total_weight > 0 else 0,
                "discoverable": float(discoverable_weight / total_weight * 100) if total_weight > 0 else 0,
                "hidden": float(hidden_weight / total_weight * 100) if total_weight > 0 else 0,
            },
            "true_value": self.true_value,
        }

    def information_discovered_fraction(self) -> float:
        """Fraction of discoverable information that has been discovered."""
        if not self._discoverable_signals:
            return 1.0  # Nothing to discover

        total_weight = sum(self._signal_weights[i] for i in self._discoverable_signals)
        discovered_weight = sum(
            self._signal_weights[i]
            for i, agents in self._discovered_by.items()
            if len(agents) > 0
        )
        return discovered_weight / total_weight if total_weight > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"UnifiedEnvironment("
            f"leakiness={self.config.leakiness:.2f}, "
            f"n_signals={self.config.n_signals}, "
            f"leaked={len(self._leaked_signals)}, "
            f"free={len(self._free_signals)}, "
            f"discoverable={len(self._discoverable_signals)}, "
            f"true_value={self.true_value})"
        )
