"""
Logarithmic Market Scoring Rule (LMSR) market maker.

The LMSR, developed by Robin Hanson, is an automated market maker that always
provides liquidity. It uses a cost function based on the log-sum-exp of
outstanding shares to determine prices.

Key properties:
- Always ready to trade (no need for counterparty)
- Bounded worst-case loss for market maker (= subsidy)
- Prices adjust smoothly with trading volume
- Liquidity parameter b controls price sensitivity

Cost function: C(q) = b * log(sum(exp(q_i / b)))
Price for outcome i: p_i = exp(q_i / b) / sum(exp(q_j / b))

This is equivalent to the softmax function, ensuring prices are always in (0,1)
and sum to 1 for multinomial contracts.
"""

from dataclasses import dataclass, field

import numpy as np

from .base import Market, Trade
from .contracts import Contract


def logsumexp(x: np.ndarray) -> float:
    """
    Numerically stable log-sum-exp computation.

    log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))


@dataclass
class LSMRMarket(Market):
    """
    Logarithmic Market Scoring Rule (LMSR) automated market maker.

    The liquidity parameter b controls how much capital is needed to move prices:
    - Small b: Prices move quickly, small trades have large impact
    - Large b: Prices move slowly, need large trades to shift prices

    The market maker subsidises trading by accepting bounded losses. The maximum
    loss is b * log(n) where n is the number of outcomes.
    """
    liquidity: float = 100.0  # b parameter
    quantities: np.ndarray = field(default=None)
    initial_cost: float = field(default=None)

    def __post_init__(self):
        # Initialise quantities to zero for each outcome
        if self.quantities is None:
            self.quantities = np.zeros(self.contract.n_outcomes)

        # Calculate initial cost (this is what the market maker "invests")
        if self.initial_cost is None:
            self.initial_cost = self._cost_function(self.quantities)

        # Initialise history lists
        if not hasattr(self, 'history') or self.history is None:
            self.history = []
        if not hasattr(self, 'state_history') or self.state_history is None:
            self.state_history = []
        if not hasattr(self, 'current_timestamp'):
            self.current_timestamp = 0

        # Record initial state
        super().__post_init__()

    def _cost_function(self, q: np.ndarray) -> float:
        """
        Compute the LMSR cost function.

        C(q) = b * log(sum(exp(q_i / b)))

        Args:
            q: Vector of quantities for each outcome

        Returns:
            Cost value
        """
        return self.liquidity * logsumexp(q / self.liquidity)

    def get_price(self, outcome_index: int) -> float:
        """
        Get the current price for one share of the given outcome.

        Price is the derivative of the cost function:
        p_i = dC/dq_i = exp(q_i / b) / sum(exp(q_j / b))

        This is the softmax function, ensuring prices in (0,1) that sum to 1.
        """
        if outcome_index < 0 or outcome_index >= self.contract.n_outcomes:
            raise ValueError(f"Invalid outcome index: {outcome_index}")

        # Softmax for numerical stability
        scaled = self.quantities / self.liquidity
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        return exp_scaled[outcome_index] / np.sum(exp_scaled)

    def get_cost(self, outcome_index: int, shares: float) -> float:
        """
        Calculate the cost to buy (positive) or sell (negative) shares.

        Cost = C(q') - C(q) where q' is the new quantity vector after trade.
        """
        if outcome_index < 0 or outcome_index >= self.contract.n_outcomes:
            raise ValueError(f"Invalid outcome index: {outcome_index}")

        new_quantities = self.quantities.copy()
        new_quantities[outcome_index] += shares

        return self._cost_function(new_quantities) - self._cost_function(self.quantities)

    def get_cost_for_target_price(
        self,
        outcome_index: int,
        target_price: float
    ) -> tuple[float, float]:
        """
        Calculate shares needed to move price to target, and the cost.

        Useful for agents who want to move the price to their believed value.

        Args:
            outcome_index: Outcome to trade
            target_price: Desired price after trade

        Returns:
            Tuple of (shares_needed, cost)
        """
        if target_price <= 0 or target_price >= 1:
            raise ValueError("Target price must be in (0, 1)")

        current_price = self.get_price(outcome_index)
        if abs(current_price - target_price) < 1e-10:
            return 0.0, 0.0

        # For LMSR with 2 outcomes, we can solve analytically
        # p = exp(q/b) / (exp(q/b) + exp(q'/b))
        # For n outcomes, we use numerical approximation

        # Binary search for the right number of shares
        # Start with an estimate based on linear approximation
        price_sensitivity = current_price * (1 - current_price) / self.liquidity
        shares_estimate = (target_price - current_price) / price_sensitivity

        # Refine with binary search
        low, high = -1e6, 1e6
        if shares_estimate > 0:
            low = 0
            high = max(shares_estimate * 10, 1000)
        else:
            low = min(shares_estimate * 10, -1000)
            high = 0

        for _ in range(100):  # Binary search iterations
            mid = (low + high) / 2
            new_q = self.quantities.copy()
            new_q[outcome_index] += mid
            new_price = self._price_at_quantities(new_q, outcome_index)

            if abs(new_price - target_price) < 1e-8:
                break
            if new_price < target_price:
                low = mid
            else:
                high = mid

        shares = (low + high) / 2
        cost = self.get_cost(outcome_index, shares)
        return shares, cost

    def _price_at_quantities(self, q: np.ndarray, outcome_index: int) -> float:
        """Calculate price for an outcome at given quantities."""
        scaled = q / self.liquidity
        max_scaled = np.max(scaled)
        exp_scaled = np.exp(scaled - max_scaled)
        return exp_scaled[outcome_index] / np.sum(exp_scaled)

    def execute_trade(
        self,
        agent_id: str,
        outcome_index: int,
        shares: float
    ) -> Trade:
        """
        Execute a trade in the LMSR market.

        The trade is always executed (LMSR always provides liquidity).
        """
        if outcome_index < 0 or outcome_index >= self.contract.n_outcomes:
            raise ValueError(f"Invalid outcome index: {outcome_index}")

        # Calculate cost before updating quantities
        cost = self.get_cost(outcome_index, shares)
        price_before = self.get_price(outcome_index)

        # Update quantities
        self.quantities[outcome_index] += shares

        # Record trade
        trade = Trade(
            timestamp=self.current_timestamp,
            agent_id=agent_id,
            outcome_index=outcome_index,
            shares=shares,
            price=price_before,
            cost=cost
        )
        self.history.append(trade)

        return trade

    def get_quantities(self) -> np.ndarray:
        """Get current outstanding shares for each outcome."""
        return self.quantities.copy()

    def _get_subsidy_spent(self) -> float:
        """
        Calculate how much of the market maker's subsidy has been "spent".

        This is the difference between current cost and initial cost.
        Maximum possible loss is b * log(n).
        """
        current_cost = self._cost_function(self.quantities)
        return current_cost - self.initial_cost

    def max_loss(self) -> float:
        """
        Calculate the maximum possible loss for the market maker.

        This is the subsidy required to guarantee liquidity.
        Maximum loss = b * log(n) where n = number of outcomes.
        """
        return self.liquidity * np.log(self.contract.n_outcomes)

    def __repr__(self) -> str:
        prices = self.get_prices()
        price_str = ", ".join(f"{p:.3f}" for p in prices)
        return (
            f"LSMRMarket(contract={self.contract.name}, "
            f"liquidity={self.liquidity}, prices=[{price_str}])"
        )
