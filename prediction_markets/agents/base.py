"""
Base agent class for prediction market participants.

Agents interact with markets by observing prices and deciding whether to trade.
Different agent types (informed, noise, discoverer) have different information
and decision-making processes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..markets.base import Market, Trade


@dataclass
class AgentState:
    """
    Snapshot of an agent's state at a point in time.
    """
    timestamp: int
    wealth: float
    holdings: np.ndarray
    beliefs: np.ndarray  # Believed probabilities for each outcome
    total_cost: float    # Total spent on trades


@dataclass
class Agent(ABC):
    """
    Abstract base class for market participants.

    An agent has:
    - Wealth (capital available for trading)
    - Holdings (shares owned in each outcome)
    - Beliefs (subjective probabilities for each outcome)
    - A strategy for deciding when and how much to trade
    """
    agent_id: str
    initial_wealth: float = 1000.0
    risk_aversion: float = 1.0  # Higher = more conservative

    # State tracking
    wealth: float = field(default=None)
    holdings: np.ndarray = field(default=None)
    beliefs: np.ndarray = field(default=None)
    total_cost: float = 0.0
    trade_history: list[Trade] = field(default_factory=list)
    state_history: list[AgentState] = field(default_factory=list)

    def __post_init__(self):
        if self.wealth is None:
            self.wealth = self.initial_wealth

    def initialise(self, n_outcomes: int, prior: np.ndarray | None = None) -> None:
        """
        Initialise agent for a market with n outcomes.

        Args:
            n_outcomes: Number of possible outcomes
            prior: Prior probability distribution (default: uniform)
        """
        self.holdings = np.zeros(n_outcomes)
        if prior is None:
            self.beliefs = np.ones(n_outcomes) / n_outcomes
        else:
            self.beliefs = prior.copy()
        self._record_state(0)

    @abstractmethod
    def decide_trade(self, market: Market) -> tuple[int | None, float]:
        """
        Decide whether and how much to trade.

        Args:
            market: The market to potentially trade in

        Returns:
            Tuple of (outcome_index, shares) or (None, 0) if no trade
            Positive shares = buy, negative = sell
        """
        pass

    def execute_trade(self, market: Market, outcome_index: int, shares: float) -> Trade | None:
        """
        Execute a trade in the market and update agent state.

        Args:
            market: The market to trade in
            outcome_index: Which outcome to trade
            shares: Number of shares (positive = buy, negative = sell)

        Returns:
            Trade record, or None if trade couldn't be executed
        """
        if shares == 0:
            return None

        # Check if agent can afford the trade
        cost = market.get_cost(outcome_index, shares)
        if cost > self.wealth:
            # Can't afford this trade - try to buy as much as possible
            if shares > 0:
                # Binary search for affordable amount
                max_shares = self._find_affordable_shares(market, outcome_index, shares)
                if max_shares <= 0:
                    return None
                shares = max_shares
                cost = market.get_cost(outcome_index, shares)
            else:
                # Selling generates money, should always be affordable
                pass

        # Execute the trade
        trade = market.execute_trade(self.agent_id, outcome_index, shares)

        # Update agent state
        self.wealth -= trade.cost
        self.holdings[outcome_index] += trade.shares
        self.total_cost += trade.cost
        self.trade_history.append(trade)

        return trade

    def _find_affordable_shares(
        self,
        market: Market,
        outcome_index: int,
        desired_shares: float
    ) -> float:
        """Find the maximum shares the agent can afford."""
        if desired_shares <= 0:
            return desired_shares

        low, high = 0.0, desired_shares
        for _ in range(50):  # Binary search
            mid = (low + high) / 2
            cost = market.get_cost(outcome_index, mid)
            if cost <= self.wealth:
                low = mid
            else:
                high = mid
            if high - low < 0.01:
                break
        return low

    def update_beliefs(self, new_beliefs: np.ndarray) -> None:
        """
        Update the agent's beliefs about outcome probabilities.

        Args:
            new_beliefs: New probability distribution (must sum to 1)
        """
        if abs(np.sum(new_beliefs) - 1.0) > 1e-6:
            raise ValueError("Beliefs must sum to 1")
        self.beliefs = new_beliefs.copy()

    def receive_signal(self, signal: dict[str, Any]) -> None:
        """
        Receive and process a signal (information).

        Override in subclasses to implement signal processing.

        Args:
            signal: Signal information (structure depends on environment)
        """
        pass

    def get_expected_value(self, market: Market) -> float:
        """
        Calculate the expected value of current holdings under agent's beliefs.

        Returns:
            Expected payout value
        """
        return np.sum(self.holdings * self.beliefs)

    def get_pnl(self) -> float:
        """
        Calculate current profit/loss (excluding resolution payout).

        Returns:
            Current P&L = expected value - total cost
        """
        return self.get_expected_value_of_holdings() - self.total_cost

    def get_expected_value_of_holdings(self) -> float:
        """
        Expected value of holdings under own beliefs.
        """
        return np.sum(self.holdings * self.beliefs)

    def _record_state(self, timestamp: int) -> None:
        """Record current state to history."""
        state = AgentState(
            timestamp=timestamp,
            wealth=self.wealth,
            holdings=self.holdings.copy() if self.holdings is not None else np.array([]),
            beliefs=self.beliefs.copy() if self.beliefs is not None else np.array([]),
            total_cost=self.total_cost
        )
        self.state_history.append(state)

    def step(self, market: Market) -> Trade | None:
        """
        Perform one step of agent decision-making and trading.

        Args:
            market: The market to interact with

        Returns:
            Trade executed, or None if no trade
        """
        outcome_index, shares = self.decide_trade(market)
        if outcome_index is not None and shares != 0:
            trade = self.execute_trade(market, outcome_index, shares)
            self._record_state(market.current_timestamp)
            return trade
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.agent_id}, wealth={self.wealth:.2f})"
