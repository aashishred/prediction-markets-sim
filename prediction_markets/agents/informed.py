"""
Informed agent that receives private signals about the true outcome.

Informed agents trade to profit from their private information. They update
their beliefs based on signals and trade when market prices diverge from
their beliefs sufficiently to compensate for risk.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Agent
from ..markets.base import Market


@dataclass
class InformedAgent(Agent):
    """
    An agent with private information about the true outcome.

    The agent receives signals that update their beliefs. They trade when
    the expected profit (based on belief vs price difference) exceeds
    their risk-adjusted threshold.

    Attributes:
        signal_precision: How accurate the agent's signals are (0 to 1)
        trade_threshold: Minimum expected profit to trigger a trade
        max_position: Maximum shares to hold in any outcome
        kelly_fraction: Fraction of Kelly criterion to use (0 to 1)
    """
    signal_precision: float = 0.8  # Probability signal is correct
    trade_threshold: float = 0.05  # Min price-belief gap to trade
    max_position: float = 100.0    # Maximum position size
    kelly_fraction: float = 0.5   # Conservative Kelly betting

    # Private signal state
    _signal_received: bool = field(default=False, repr=False)
    _signal_outcome: int | None = field(default=None, repr=False)

    def receive_signal(self, signal: dict[str, Any]) -> None:
        """
        Receive a signal about the true outcome.

        Expected signal format:
        {
            "outcome": int,           # The outcome the signal points to
            "precision": float,       # Override default precision (optional)
            "strength": float,        # How strong the signal is (0-1)
        }
        """
        if "outcome" not in signal:
            return

        self._signal_received = True
        self._signal_outcome = signal["outcome"]

        # Use signal-specific precision if provided
        precision = signal.get("precision", self.signal_precision)
        strength = signal.get("strength", 1.0)

        # Update beliefs using Bayesian updating
        # P(outcome | signal) ∝ P(signal | outcome) * P(outcome)
        n_outcomes = len(self.beliefs)

        # Likelihood: signal points to this outcome with probability 'precision'
        # Otherwise, signal is wrong and points uniformly to others
        likelihoods = np.ones(n_outcomes) * (1 - precision) / (n_outcomes - 1)
        likelihoods[self._signal_outcome] = precision

        # Apply strength (interpolate between prior and full update)
        effective_likelihood = strength * likelihoods + (1 - strength) * np.ones(n_outcomes) / n_outcomes

        # Bayesian update
        posterior = effective_likelihood * self.beliefs
        posterior /= posterior.sum()

        self.beliefs = posterior

    def decide_trade(self, market: Market) -> tuple[int | None, float]:
        """
        Decide whether to trade based on belief-price divergence.

        Uses a modified Kelly criterion to size bets, with risk aversion.
        """
        if self.beliefs is None or len(self.beliefs) == 0:
            return None, 0.0

        prices = market.get_prices()

        # Find the outcome with the largest positive edge
        edges = self.beliefs - prices  # Positive = underpriced
        best_outcome = np.argmax(edges)
        best_edge = edges[best_outcome]

        # Also check for overpriced outcomes to potentially short
        worst_outcome = np.argmin(edges)
        worst_edge = edges[worst_outcome]  # Most negative = most overpriced

        # Decide whether to buy the best or sell the worst
        if best_edge > abs(worst_edge) and best_edge > self.trade_threshold:
            # Buy the underpriced outcome
            outcome_to_trade = best_outcome
            direction = 1  # Buy
            edge = best_edge
            price = prices[best_outcome]
        elif abs(worst_edge) > best_edge and abs(worst_edge) > self.trade_threshold:
            # Sell the overpriced outcome
            outcome_to_trade = worst_outcome
            direction = -1  # Sell
            edge = abs(worst_edge)
            price = prices[worst_outcome]
        else:
            # No trade - edges too small
            return None, 0.0

        # Calculate position size using Kelly criterion
        # Kelly: f* = (p*b - q) / b where p = prob of winning, b = odds, q = 1-p
        # For prediction markets: f* = edge / (1 - price) for buying
        if direction > 0:
            # Buying: edge = belief - price, potential profit per share = 1 - price
            kelly_fraction_calc = edge / (1 - price) if price < 1 else 0
        else:
            # Selling: edge = price - belief, profit per share = price (if outcome doesn't happen)
            kelly_fraction_calc = edge / price if price > 0 else 0

        # Apply fractional Kelly and risk aversion
        position_fraction = kelly_fraction_calc * self.kelly_fraction / self.risk_aversion

        # Convert to shares (as fraction of wealth)
        target_shares = position_fraction * self.wealth * direction

        # Enforce position limits
        current_position = self.holdings[outcome_to_trade] if self.holdings is not None else 0
        if direction > 0:
            # Buying - check we don't exceed max
            max_additional = self.max_position - current_position
            target_shares = min(target_shares, max_additional)
        else:
            # Selling - check we don't go too negative
            min_additional = -self.max_position - current_position
            target_shares = max(target_shares, min_additional)

        # Don't make tiny trades
        if abs(target_shares) < 0.1:
            return None, 0.0

        return outcome_to_trade, target_shares

    def get_edge(self, market: Market) -> np.ndarray:
        """
        Calculate the edge (belief - price) for each outcome.

        Returns:
            Array of edges, positive = underpriced, negative = overpriced
        """
        if self.beliefs is None:
            return np.array([])
        prices = market.get_prices()
        return self.beliefs - prices

    def has_signal(self) -> bool:
        """Check if agent has received a signal."""
        return self._signal_received

    def __repr__(self) -> str:
        signal_info = f", signal={self._signal_outcome}" if self._signal_received else ""
        return (
            f"InformedAgent(id={self.agent_id}, wealth={self.wealth:.2f}, "
            f"precision={self.signal_precision}{signal_info})"
        )
