"""
Hayekian environment where information is distributed as tacit knowledge.

This models the classic Hayekian case from "The Use of Knowledge in Society":
- True value is determined by aggregation of dispersed information
- Each agent has access to a piece of the information
- No agent knows the full picture
- The market should aggregate this distributed knowledge

This tests the pure aggregation case. If prediction markets only aggregate,
their value is limited by how much dispersed knowledge exists to aggregate.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Environment, EnvironmentConfig, EnvironmentType
from ..agents.base import Agent


@dataclass
class HayekianConfig(EnvironmentConfig):
    """
    Configuration for Hayekian environment.

    Attributes:
        n_signals: Number of independent signals that determine true value
        signal_weights: How much each signal contributes (default: equal)
        signal_precision: How accurately agents observe their signals
        signals_per_agent: How many signals each agent receives
        signal_distribution: How signals are distributed ('uniform', 'random', 'clustered')
    """
    n_signals: int = 10
    signal_weights: np.ndarray | None = None
    signal_precision: float = 0.9
    signals_per_agent: float | int = 1  # Can be fractional (probability)
    signal_distribution: str = "uniform"


@dataclass
class HayekianEnvironment(Environment):
    """
    Environment where the true value emerges from distributed signals.

    The true outcome is determined by a weighted combination of signals.
    Each agent observes some subset of these signals with some precision.
    A perfect aggregation of all signals would reveal the true outcome.

    This models the Hayekian insight that local knowledge is distributed
    across many minds and cannot be centrally collected, but can be
    aggregated through market prices.
    """
    config: HayekianConfig = field(default_factory=HayekianConfig)

    # Signal state
    _signals: np.ndarray = field(default=None, repr=False)
    _signal_weights: np.ndarray = field(default=None, repr=False)
    _agent_signals: dict[str, list[int]] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # Initialize rng first (before parent which calls _generate_true_value)
        if self.rng is None:
            seed = self.config.random_seed
            self.rng = np.random.default_rng(seed)

        # Set up signal weights
        if self.config.signal_weights is not None:
            self._signal_weights = self.config.signal_weights
        else:
            # Equal weights, normalized
            self._signal_weights = np.ones(self.config.n_signals) / self.config.n_signals

        # Generate signals before calling parent __post_init__
        self._generate_signals()

        super().__post_init__()

    @property
    def environment_type(self) -> EnvironmentType:
        return EnvironmentType.HAYEKIAN

    def _generate_signals(self) -> None:
        """
        Generate the underlying signals that determine the true value.

        For binary outcomes: signals are 0 or 1, and we aggregate to get
        a probability, then sample the outcome.

        For multinomial: signals vote for outcomes.
        """
        n = self.config.n_signals

        if self.config.n_outcomes == 2:
            # Binary: signals are 0 or 1
            # The weighted mean gives the probability of outcome 1
            self._signals = self.rng.binomial(1, 0.5, size=n)
        else:
            # Multinomial: signals vote for outcomes
            self._signals = self.rng.integers(0, self.config.n_outcomes, size=n)

    def _generate_true_value(self) -> int:
        """
        Generate true value from signals.

        The true outcome is determined by the weighted combination of signals.
        """
        if self.config.n_outcomes == 2:
            # Probability of outcome 1 = weighted mean of binary signals
            prob = np.average(self._signals, weights=self._signal_weights)
            return int(self.rng.random() < prob)
        else:
            # For multinomial, count votes for each outcome
            votes = np.zeros(self.config.n_outcomes)
            for i, signal in enumerate(self._signals):
                votes[signal] += self._signal_weights[i]
            # Winner is outcome with most votes
            return int(np.argmax(votes))

    def _assign_signals_to_agents(self, agents: list[Agent]) -> None:
        """
        Assign signals to agents based on distribution strategy.
        """
        n_agents = len(agents)
        n_signals = self.config.n_signals
        distribution = self.config.signal_distribution

        self._agent_signals = {agent.agent_id: [] for agent in agents}

        if distribution == "uniform":
            # Each signal goes to signals_per_agent agents (on average)
            for signal_idx in range(n_signals):
                # Each agent has a probability of receiving this signal
                prob = min(1.0, self.config.signals_per_agent / n_agents * n_signals / n_signals)
                for agent in agents:
                    if self.rng.random() < prob:
                        self._agent_signals[agent.agent_id].append(signal_idx)

        elif distribution == "random":
            # Randomly assign each signal to one agent
            for signal_idx in range(n_signals):
                agent = self.rng.choice(agents)
                self._agent_signals[agent.agent_id].append(signal_idx)

        elif distribution == "clustered":
            # Signals are clustered - nearby signals go to same agent
            signals_per = max(1, n_signals // n_agents)
            for i, agent in enumerate(agents):
                start = (i * signals_per) % n_signals
                for j in range(int(self.config.signals_per_agent)):
                    idx = (start + j) % n_signals
                    self._agent_signals[agent.agent_id].append(idx)

    def generate_signals(self, agents: list[Agent]) -> dict[str, dict[str, Any]]:
        """
        Generate signals to send to each agent.

        Each agent receives a noisy observation of their assigned signals.
        """
        # Assign signals if not already done
        if not self._agent_signals or set(self._agent_signals.keys()) != {a.agent_id for a in agents}:
            self._assign_signals_to_agents(agents)

        result = {}

        for agent in agents:
            signal_indices = self._agent_signals.get(agent.agent_id, [])
            if not signal_indices:
                continue

            # For binary, compute weighted belief update based on signals
            if self.config.n_outcomes == 2:
                # Agent observes their signals with some precision
                observed_signals = []
                observed_weights = []
                for idx in signal_indices:
                    true_signal = self._signals[idx]
                    # With probability = precision, observe correctly
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
                    "strength": len(signal_indices) / self.config.n_signals,  # More signals = stronger update
                    "implied_probability": implied_prob,
                    "n_signals": len(signal_indices),
                }
            else:
                # Multinomial: vote based on observed signals
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

    def get_signal_distribution(self) -> dict[str, Any]:
        """
        Get summary of how signals are distributed.

        Useful for analyzing whether aggregation is possible.
        """
        return {
            "n_signals": self.config.n_signals,
            "signals": self._signals.tolist(),
            "weights": self._signal_weights.tolist(),
            "true_value": self.true_value,
            "agent_assignments": {
                aid: len(signals) for aid, signals in self._agent_signals.items()
            }
        }

    def theoretical_aggregate(self) -> np.ndarray:
        """
        Calculate the theoretical aggregate of all signals.

        This is what a perfectly efficient market should converge to.
        """
        if self.config.n_outcomes == 2:
            prob = np.average(self._signals, weights=self._signal_weights)
            return np.array([1 - prob, prob])
        else:
            votes = np.zeros(self.config.n_outcomes)
            for i, signal in enumerate(self._signals):
                votes[signal] += self._signal_weights[i]
            return votes / votes.sum()

    def __repr__(self) -> str:
        return (
            f"HayekianEnvironment("
            f"n_signals={self.config.n_signals}, "
            f"precision={self.config.signal_precision}, "
            f"true_value={self.true_value})"
        )
