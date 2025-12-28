"""
Abstract base class for market mechanisms.

Provides a common interface for different market implementations (LMSR, order book, etc.)
so that agents and simulations can interact with markets uniformly.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .contracts import Contract


@dataclass
class Trade:
    """
    Record of a single trade executed in the market.
    """
    timestamp: int
    agent_id: str
    outcome_index: int
    shares: float  # Positive = buy, negative = sell
    price: float   # Price per share at execution
    cost: float    # Total cost (positive = paid, negative = received)


@dataclass
class MarketState:
    """
    Snapshot of market state at a point in time.
    """
    timestamp: int
    prices: np.ndarray          # Current prices for each outcome
    quantities: np.ndarray      # Total shares outstanding for each outcome
    total_volume: float         # Cumulative trading volume
    subsidy_spent: float        # Total subsidy consumed (for LMSR)


@dataclass
class Market(ABC):
    """
    Abstract base class for market mechanisms.

    A market provides a mechanism for trading shares in a contract's outcomes.
    Different implementations (LMSR, order book) have different liquidity
    and price discovery properties.
    """
    contract: Contract
    history: list[Trade] = field(default_factory=list)
    state_history: list[MarketState] = field(default_factory=list)
    current_timestamp: int = 0

    def __post_init__(self):
        # Record initial state
        self._record_state()

    @abstractmethod
    def get_price(self, outcome_index: int) -> float:
        """
        Get the current price for a single share of the given outcome.

        Args:
            outcome_index: Index of the outcome

        Returns:
            Current price in [0, 1]
        """
        pass

    def get_prices(self) -> np.ndarray:
        """
        Get current prices for all outcomes.

        Returns:
            Array of prices, one per outcome
        """
        return np.array([
            self.get_price(i) for i in range(self.contract.n_outcomes)
        ])

    @abstractmethod
    def get_cost(self, outcome_index: int, shares: float) -> float:
        """
        Calculate the cost to buy (positive shares) or sell (negative shares)
        the given number of shares of an outcome.

        Args:
            outcome_index: Index of the outcome
            shares: Number of shares (positive = buy, negative = sell)

        Returns:
            Cost (positive = pay, negative = receive)
        """
        pass

    @abstractmethod
    def execute_trade(
        self,
        agent_id: str,
        outcome_index: int,
        shares: float
    ) -> Trade:
        """
        Execute a trade in the market.

        Args:
            agent_id: Identifier of the trading agent
            outcome_index: Index of the outcome to trade
            shares: Number of shares (positive = buy, negative = sell)

        Returns:
            Trade record with execution details
        """
        pass

    @abstractmethod
    def get_quantities(self) -> np.ndarray:
        """
        Get the current quantity of shares outstanding for each outcome.

        Returns:
            Array of quantities, one per outcome
        """
        pass

    def get_volume(self) -> float:
        """
        Get total trading volume (sum of absolute share amounts).

        Returns:
            Total volume traded
        """
        return sum(abs(trade.shares) for trade in self.history)

    def get_current_state(self) -> MarketState:
        """
        Get the current market state.

        Returns:
            Current MarketState snapshot
        """
        return MarketState(
            timestamp=self.current_timestamp,
            prices=self.get_prices(),
            quantities=self.get_quantities(),
            total_volume=self.get_volume(),
            subsidy_spent=self._get_subsidy_spent()
        )

    @abstractmethod
    def _get_subsidy_spent(self) -> float:
        """Get total subsidy consumed (relevant for LMSR)."""
        pass

    def _record_state(self) -> None:
        """Record current state to history."""
        self.state_history.append(self.get_current_state())

    def advance_time(self, steps: int = 1) -> None:
        """
        Advance the market timestamp.

        Args:
            steps: Number of timesteps to advance
        """
        self.current_timestamp += steps
        self._record_state()

    def resolve(self, outcome: Any) -> dict[str, float]:
        """
        Resolve the market and calculate payouts.

        Args:
            outcome: The resolution outcome (passed to contract)

        Returns:
            Dictionary mapping agent_id to net payout
        """
        self.contract.resolve(outcome)

        # Calculate each agent's holdings and payouts
        agent_holdings: dict[str, np.ndarray] = {}
        agent_costs: dict[str, float] = {}

        for trade in self.history:
            if trade.agent_id not in agent_holdings:
                agent_holdings[trade.agent_id] = np.zeros(self.contract.n_outcomes)
                agent_costs[trade.agent_id] = 0.0

            agent_holdings[trade.agent_id][trade.outcome_index] += trade.shares
            agent_costs[trade.agent_id] += trade.cost

        # Calculate payouts
        payouts: dict[str, float] = {}
        for agent_id, holdings in agent_holdings.items():
            payout = sum(
                holdings[i] * self.contract.payout(i)
                for i in range(self.contract.n_outcomes)
            )
            # Net P&L = payout - cost
            payouts[agent_id] = payout - agent_costs[agent_id]

        return payouts

    def get_price_history(self) -> np.ndarray:
        """
        Get the price history over time.

        Returns:
            Array of shape (n_timestamps, n_outcomes)
        """
        return np.array([state.prices for state in self.state_history])

    def get_trade_history(self) -> list[Trade]:
        """Get all trades executed in this market."""
        return self.history.copy()
