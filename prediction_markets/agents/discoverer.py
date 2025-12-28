"""
Discoverer agent that can pay to acquire information.

This implements the general discovery model where agents can:
1. Pay costs to attempt information acquisition
2. Receive signals with uncertain validity
3. Signals have unclear bearing on contract value
4. Accumulate signals over time (incremental learning)

The key difference from InformedAgent is that Discoverers must actively
decide to acquire information at a cost, modeling the economic tradeoff
between discovery costs and trading profits.

This tests the hypothesis that markets create value through discovery
incentives, not just aggregation of existing knowledge.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import Agent
from .discovery import (
    DiscoveryModel,
    DiscoveryOpportunity,
    DiscoveryRecord,
    Signal,
)
from ..markets.base import Market


@dataclass
class DiscovererAgent(Agent):
    """
    An agent that can pay to discover information and then trade on it.

    Unlike InformedAgent (who receives signals passively), Discoverer must
    actively decide whether the expected value of discovery exceeds the cost.

    Attributes:
        discovery_budget: Maximum to spend on discovery per simulation
        discovery_threshold: Minimum expected value to attempt discovery
        max_attempts_per_step: Maximum discovery attempts per timestep
        discovery_model: The model governing how discovery works
        kelly_fraction: Fraction of Kelly criterion for position sizing
        trade_threshold: Minimum edge to trade
    """
    discovery_budget: float = 100.0
    discovery_threshold: float = 0.0  # Attempt if EV > 0
    max_attempts_per_step: int = 1
    discovery_model: DiscoveryModel = field(default_factory=DiscoveryModel)
    kelly_fraction: float = 0.3
    trade_threshold: float = 0.02
    max_position: float = 100.0

    # State
    discovery_record: DiscoveryRecord = field(default_factory=DiscoveryRecord)
    accumulated_signals: list[Signal] = field(default_factory=list)
    discovery_spent: float = 0.0

    # Available opportunities (set by environment)
    available_opportunities: list[DiscoveryOpportunity] = field(default_factory=list)

    def set_opportunities(self, opportunities: list[DiscoveryOpportunity]) -> None:
        """Set the discovery opportunities available to this agent."""
        self.available_opportunities = opportunities

    def attempt_discovery(
        self,
        opportunity: DiscoveryOpportunity,
        true_signal: Any,
        true_bearing: float,
        timestamp: int
    ) -> Signal | None:
        """
        Attempt to discover information.

        Args:
            opportunity: The opportunity to pursue
            true_signal: The true signal value (from environment)
            true_bearing: The true importance of this signal
            timestamp: Current simulation time

        Returns:
            Signal if successful, None otherwise
        """
        # Check budget
        if self.discovery_spent + opportunity.cost > self.discovery_budget:
            return None

        # Pay the cost regardless of outcome
        self.discovery_spent += opportunity.cost
        self.wealth -= opportunity.cost

        # Attempt discovery
        signal = self.discovery_model.attempt_discovery(
            opportunity, true_signal, true_bearing
        )

        if signal is not None:
            signal.timestamp = timestamp
            self.accumulated_signals.append(signal)

            # Update beliefs based on new signal
            self.beliefs = self.discovery_model.update_beliefs_from_signal(
                self.beliefs, signal, n_outcomes=len(self.beliefs)
            )

        # Record the attempt
        self.discovery_record.record_attempt(opportunity.cost, signal)

        return signal

    def should_discover(
        self,
        opportunity: DiscoveryOpportunity,
        market: Market
    ) -> bool:
        """
        Decide whether to pursue a discovery opportunity.

        Args:
            opportunity: The opportunity to consider
            market: Current market state

        Returns:
            True if agent should attempt discovery
        """
        # Check budget
        if self.discovery_spent + opportunity.cost > self.discovery_budget:
            return False

        # Check wealth
        if opportunity.cost > self.wealth * 0.1:  # Don't spend more than 10% of wealth on one attempt
            return False

        # Calculate expected value
        ev = self.discovery_model.expected_value_of_discovery(
            opportunity,
            self.beliefs,
            market.get_prices()
        )

        return ev > self.discovery_threshold

    def decide_trade(self, market: Market) -> tuple[int | None, float]:
        """
        Decide whether to trade based on current beliefs.

        Similar to InformedAgent but beliefs come from accumulated signals.
        """
        if self.beliefs is None or len(self.beliefs) == 0:
            return None, 0.0

        prices = market.get_prices()

        # Find best trading opportunity
        edges = self.beliefs - prices
        best_outcome = np.argmax(edges)
        best_edge = edges[best_outcome]

        worst_outcome = np.argmin(edges)
        worst_edge = edges[worst_outcome]

        # Decide direction
        if best_edge > abs(worst_edge) and best_edge > self.trade_threshold:
            outcome_to_trade = best_outcome
            direction = 1
            edge = best_edge
            price = prices[best_outcome]
        elif abs(worst_edge) > best_edge and abs(worst_edge) > self.trade_threshold:
            outcome_to_trade = worst_outcome
            direction = -1
            edge = abs(worst_edge)
            price = prices[worst_outcome]
        else:
            return None, 0.0

        # Position sizing (Kelly criterion)
        if direction > 0:
            kelly = edge / (1 - price) if price < 1 else 0
        else:
            kelly = edge / price if price > 0 else 0

        position_fraction = kelly * self.kelly_fraction
        target_shares = position_fraction * self.wealth * direction

        # Position limits
        current_position = self.holdings[outcome_to_trade] if self.holdings is not None else 0
        if direction > 0:
            max_additional = self.max_position - current_position
            target_shares = min(target_shares, max_additional)
        else:
            min_additional = -self.max_position - current_position
            target_shares = max(target_shares, min_additional)

        if abs(target_shares) < 0.1:
            return None, 0.0

        return outcome_to_trade, target_shares

    def receive_signal(self, signal: dict[str, Any]) -> None:
        """
        Receive a signal from the environment.

        For Discoverer agents, this is typically called after a successful
        discovery attempt. Can also receive passive signals like InformedAgent.
        """
        if "outcome" not in signal:
            return

        # Convert dict signal to Signal object
        s = Signal(
            content=signal["outcome"],
            true_content=signal.get("true_outcome", signal["outcome"]),
            validity=signal.get("precision", 0.8),
            bearing=signal.get("strength", 0.1),
            cost_paid=0,  # Passive signal, no cost
            timestamp=0
        )

        self.accumulated_signals.append(s)

        # Update beliefs
        self.beliefs = self.discovery_model.update_beliefs_from_signal(
            self.beliefs, s, n_outcomes=len(self.beliefs)
        )

    def get_discovery_stats(self) -> dict[str, Any]:
        """Get statistics about discovery activity."""
        return {
            "attempts": self.discovery_record.attempts,
            "successes": self.discovery_record.successes,
            "success_rate": self.discovery_record.success_rate,
            "total_cost": self.discovery_record.total_cost,
            "cost_per_signal": self.discovery_record.cost_per_signal,
            "signal_accuracy": self.discovery_record.signal_accuracy,
            "signals_accumulated": len(self.accumulated_signals),
            "budget_remaining": self.discovery_budget - self.discovery_spent,
        }

    def __repr__(self) -> str:
        n_signals = len(self.accumulated_signals)
        return (
            f"DiscovererAgent(id={self.agent_id}, wealth={self.wealth:.2f}, "
            f"signals={n_signals}, spent={self.discovery_spent:.2f})"
        )


@dataclass
class AdaptiveDiscoverer(DiscovererAgent):
    """
    A more sophisticated discoverer that adapts strategy based on results.

    This agent:
    - Increases discovery effort when signals are profitable
    - Decreases effort when signals don't lead to profits
    - Balances exploration (more discovery) vs exploitation (trading on existing knowledge)
    """

    exploration_rate: float = 0.2  # Fraction of actions devoted to exploration
    learning_rate: float = 0.1    # How fast to update strategy

    # Learned parameters
    _estimated_signal_value: float = field(default=1.0, repr=False)

    def should_discover(
        self,
        opportunity: DiscoveryOpportunity,
        market: Market
    ) -> bool:
        """
        Adaptive discovery decision.

        Uses learned estimate of signal value to decide whether to discover.
        """
        # Exploration: sometimes discover even if EV seems low
        if self.discovery_model.rng.random() < self.exploration_rate:
            if self.discovery_spent + opportunity.cost <= self.discovery_budget:
                return True

        # Exploitation: use learned signal value estimate
        estimated_ev = (
            opportunity.success_probability *
            self._estimated_signal_value *
            opportunity.expected_bearing
        ) - opportunity.cost

        return estimated_ev > 0 and super().should_discover(opportunity, market)

    def update_signal_value_estimate(self, profit_from_signal: float) -> None:
        """
        Update the estimated value of signals based on realized profits.

        Args:
            profit_from_signal: Profit (or loss) attributed to a signal
        """
        self._estimated_signal_value = (
            (1 - self.learning_rate) * self._estimated_signal_value +
            self.learning_rate * profit_from_signal
        )
