"""
General discovery model for prediction market agents.

This implements the fully general discovery procedure discussed in the theoretical
framework. Discovery is not a simple binary "learn or don't learn" — it's a
probabilistic process where:

1. Agent pays a cost to attempt information acquisition
2. With some probability, agent obtains a signal
3. The signal has uncertain validity (may be false positive/negative)
4. The signal has unclear bearing on contract value (weight/relevance uncertain)
5. Agent can repeatedly pay to accumulate signals (incremental learning)

Special cases emerge from parameter choices:
- Binary discovery: success_prob=1, validity=1, bearing=1 → certain knowledge
- Noisy signal: validity<1, bearing=1 → knows relevance but signal may be wrong
- Unclear relevance: validity=1, bearing<1 → signal is accurate but weight unknown
- Full uncertainty: validity<1, bearing<1 → both signal accuracy and relevance uncertain

This models the spectrum from Knightian uncertainty (nothing discoverable) to
perfect information (everything knowable at zero cost).
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Signal:
    """
    A piece of information that an agent has acquired.

    Attributes:
        content: The actual content of the signal (e.g., 0 or 1 for binary)
        true_content: What the signal would be if perfectly observed (for analysis)
        validity: Agent's estimate of P(signal is correct)
        bearing: Agent's estimate of how much this signal matters for the contract
        cost_paid: How much the agent paid to acquire this signal
        timestamp: When the signal was acquired
    """
    content: Any
    true_content: Any
    validity: float  # P(content == true_content)
    bearing: float   # Weight of this signal in determining contract value
    cost_paid: float
    timestamp: int

    def is_correct(self) -> bool:
        """Check if the observed content matches the true content."""
        return self.content == self.true_content


@dataclass
class DiscoveryOpportunity:
    """
    Represents an opportunity to discover information.

    This is what the environment offers to agents. The agent decides whether
    to pay the cost to attempt discovery.

    Attributes:
        cost: Cost to attempt discovery
        success_probability: P(obtaining a signal | paying cost)
        expected_validity: Expected accuracy of signal if obtained
        expected_bearing: Expected relevance of signal to contract value
        validity_uncertainty: Std dev of validity (agent doesn't know exact accuracy)
        bearing_uncertainty: Std dev of bearing (agent doesn't know exact relevance)
    """
    cost: float
    success_probability: float = 1.0
    expected_validity: float = 0.8
    expected_bearing: float = 0.1  # Each signal worth ~10% of total info
    validity_uncertainty: float = 0.1
    bearing_uncertainty: float = 0.05


@dataclass
class DiscoveryModel:
    """
    The general discovery model that governs how agents acquire information.

    This sits between the environment (which has the true information) and
    agents (who want to learn). It handles:
    - Whether discovery attempts succeed
    - How much noise is added to signals
    - How signals update agent beliefs
    """

    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def attempt_discovery(
        self,
        opportunity: DiscoveryOpportunity,
        true_signal: Any,
        true_bearing: float
    ) -> Signal | None:
        """
        Attempt to discover information.

        Args:
            opportunity: The discovery opportunity being pursued
            true_signal: The actual signal value (from environment)
            true_bearing: The actual importance of this signal

        Returns:
            Signal if discovery succeeded, None if failed
        """
        # Step 1: Does the attempt succeed at all?
        if self.rng.random() > opportunity.success_probability:
            return None

        # Step 2: What does the agent observe? (may be corrupted)
        # The validity determines P(observed == true)
        actual_validity = np.clip(
            self.rng.normal(opportunity.expected_validity, opportunity.validity_uncertainty),
            0.01, 0.99
        )

        if self.rng.random() < actual_validity:
            observed_content = true_signal
        else:
            # Observe the wrong thing
            if isinstance(true_signal, (int, np.integer)) and true_signal in (0, 1):
                observed_content = 1 - true_signal
            else:
                # For non-binary, add noise
                observed_content = true_signal + self.rng.normal(0, 0.5)

        # Step 3: What bearing does the agent think this has?
        # Agent doesn't know true_bearing exactly
        perceived_bearing = np.clip(
            self.rng.normal(opportunity.expected_bearing, opportunity.bearing_uncertainty),
            0.01, 1.0
        )

        return Signal(
            content=observed_content,
            true_content=true_signal,
            validity=actual_validity,
            bearing=perceived_bearing,
            cost_paid=opportunity.cost,
            timestamp=0  # Will be set by caller
        )

    def update_beliefs_from_signal(
        self,
        prior_beliefs: np.ndarray,
        signal: Signal,
        n_outcomes: int = 2
    ) -> np.ndarray:
        """
        Update beliefs based on a discovered signal.

        Uses Bayesian updating with the signal's validity as the likelihood.
        The bearing determines how strongly to update.

        Args:
            prior_beliefs: Current probability distribution over outcomes
            signal: The discovered signal
            n_outcomes: Number of possible outcomes

        Returns:
            Updated probability distribution
        """
        if n_outcomes != 2:
            # For now, only handle binary case fully
            # Multinomial would need more complex updating
            return prior_beliefs

        # For binary: signal content is 0 or 1
        # Validity is P(signal correct | true outcome)
        # Bearing scales the strength of update

        signal_points_to = int(signal.content) if signal.content in (0, 1) else (1 if signal.content > 0.5 else 0)

        # Likelihood ratio
        # P(signal | outcome=signal_points_to) / P(signal | outcome=other)
        # = validity / (1 - validity)
        likelihood_ratio = signal.validity / (1 - signal.validity)

        # Scale by bearing (weaker bearing = weaker update)
        # bearing=1 means full update, bearing=0 means no update
        effective_lr = 1 + (likelihood_ratio - 1) * signal.bearing

        # Bayesian update
        posterior = prior_beliefs.copy()
        if signal_points_to == 1:
            # Signal points to outcome 1
            posterior[1] = prior_beliefs[1] * effective_lr
            posterior[0] = prior_beliefs[0]
        else:
            # Signal points to outcome 0
            posterior[0] = prior_beliefs[0] * effective_lr
            posterior[1] = prior_beliefs[1]

        # Normalize
        posterior = posterior / posterior.sum()

        return posterior

    def expected_value_of_discovery(
        self,
        opportunity: DiscoveryOpportunity,
        current_beliefs: np.ndarray,
        market_prices: np.ndarray
    ) -> float:
        """
        Estimate the expected value of pursuing a discovery opportunity.

        This helps agents decide whether to pay for discovery.
        Value comes from being able to trade more profitably after learning.

        Args:
            opportunity: The discovery opportunity
            current_beliefs: Agent's current beliefs
            market_prices: Current market prices

        Returns:
            Expected value of discovery minus cost
        """
        # Simplified calculation:
        # EV = P(success) * E[trading_profit | signal] - cost

        # If we get a signal, it might move our beliefs toward 0 or 1
        # The value is in being able to trade more aggressively

        # Current edge (absolute)
        current_edge = np.max(np.abs(current_beliefs - market_prices))

        # Expected edge after signal (rough approximation)
        # Signal with validity v moves beliefs by factor of v/(1-v)
        # This increases our confidence, potentially increasing edge
        expected_validity = opportunity.expected_validity
        expected_bearing = opportunity.expected_bearing

        # Very rough: expected edge increase proportional to bearing
        expected_edge_increase = expected_bearing * (expected_validity - 0.5)
        expected_new_edge = current_edge + expected_edge_increase

        # Value is proportional to edge squared (roughly, from Kelly)
        # But we only get it with probability success_probability
        expected_value = (
            opportunity.success_probability *
            (expected_new_edge ** 2 - current_edge ** 2) * 100  # Scale factor
        )

        return expected_value - opportunity.cost


@dataclass
class DiscoveryRecord:
    """
    Record of all discovery activity for analysis.
    """
    attempts: int = 0
    successes: int = 0
    total_cost: float = 0.0
    signals_acquired: list[Signal] = field(default_factory=list)

    def record_attempt(self, cost: float, signal: Signal | None) -> None:
        """Record a discovery attempt."""
        self.attempts += 1
        self.total_cost += cost
        if signal is not None:
            self.successes += 1
            self.signals_acquired.append(signal)

    @property
    def success_rate(self) -> float:
        """Fraction of attempts that succeeded."""
        return self.successes / self.attempts if self.attempts > 0 else 0.0

    @property
    def cost_per_signal(self) -> float:
        """Average cost per successful signal."""
        return self.total_cost / self.successes if self.successes > 0 else float('inf')

    @property
    def signal_accuracy(self) -> float:
        """Fraction of acquired signals that were correct."""
        if not self.signals_acquired:
            return 0.0
        correct = sum(1 for s in self.signals_acquired if s.is_correct())
        return correct / len(self.signals_acquired)
