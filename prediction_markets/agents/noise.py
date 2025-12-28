"""
Noise trader agent that trades for reasons unrelated to information.

Noise traders are essential for market function (per Milgrom-Stokey no-trade theorem).
They provide liquidity and create opportunities for informed traders to profit.
Noise trading can represent:
- Entertainment/gambling motivation
- Hedging needs
- Liquidity needs
- Uniformed speculation
- Expressive trading (betting on preferred outcomes)
"""

from dataclasses import dataclass
from enum import Enum, auto
import numpy as np

from .base import Agent
from ..markets.base import Market


class NoiseTraderType(Enum):
    """Types of noise trading behavior."""
    RANDOM = auto()        # Pure random trades
    MOMENTUM = auto()      # Follow price trends
    CONTRARIAN = auto()    # Trade against trends
    BIASED = auto()        # Systematically favor certain outcomes
    LIQUIDITY = auto()     # Trade based on liquidity needs


@dataclass
class NoiseTrader(Agent):
    """
    An agent that trades for non-informational reasons.

    Noise traders create the profit opportunities that incentivise informed
    traders to participate. Without them, the Milgrom-Stokey no-trade theorem
    implies that rational informed traders won't trade.

    Attributes:
        trader_type: What kind of noise trading behavior
        trade_probability: Probability of trading each step
        trade_size_mean: Mean trade size (as fraction of wealth)
        trade_size_std: Std dev of trade size
        bias_outcome: For BIASED type, which outcome to favor
        bias_strength: How strongly to favor the biased outcome (0-1)
        momentum_lookback: For MOMENTUM/CONTRARIAN, how many periods to look back
    """
    trader_type: NoiseTraderType = NoiseTraderType.RANDOM
    trade_probability: float = 0.3
    trade_size_mean: float = 0.05   # 5% of wealth
    trade_size_std: float = 0.02
    bias_outcome: int | None = None
    bias_strength: float = 0.5
    momentum_lookback: int = 5

    # Internal state
    _rng: np.random.Generator = None

    def __post_init__(self):
        super().__post_init__()
        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def decide_trade(self, market: Market) -> tuple[int | None, float]:
        """
        Decide whether to make a noise trade.

        The decision depends on the trader type and random factors.
        """
        # First, decide whether to trade at all
        if self._rng.random() > self.trade_probability:
            return None, 0.0

        # Determine trade size
        size = self._rng.normal(self.trade_size_mean, self.trade_size_std)
        size = max(0.01, size)  # Ensure positive
        shares = size * self.wealth

        # Determine direction and outcome based on trader type
        n_outcomes = market.contract.n_outcomes

        if self.trader_type == NoiseTraderType.RANDOM:
            outcome, direction = self._random_trade(n_outcomes)

        elif self.trader_type == NoiseTraderType.MOMENTUM:
            outcome, direction = self._momentum_trade(market, n_outcomes)

        elif self.trader_type == NoiseTraderType.CONTRARIAN:
            outcome, direction = self._contrarian_trade(market, n_outcomes)

        elif self.trader_type == NoiseTraderType.BIASED:
            outcome, direction = self._biased_trade(n_outcomes)

        elif self.trader_type == NoiseTraderType.LIQUIDITY:
            outcome, direction = self._liquidity_trade(n_outcomes)

        else:
            outcome, direction = self._random_trade(n_outcomes)

        return outcome, shares * direction

    def _random_trade(self, n_outcomes: int) -> tuple[int, int]:
        """Pure random: pick random outcome and direction."""
        outcome = self._rng.integers(0, n_outcomes)
        direction = 1 if self._rng.random() > 0.5 else -1
        return outcome, direction

    def _momentum_trade(self, market: Market, n_outcomes: int) -> tuple[int, int]:
        """
        Momentum: buy outcomes whose prices have been rising.
        """
        if len(market.state_history) < 2:
            return self._random_trade(n_outcomes)

        # Calculate price changes over lookback period
        lookback = min(self.momentum_lookback, len(market.state_history) - 1)
        current_prices = market.get_prices()
        past_prices = market.state_history[-lookback - 1].prices

        price_changes = current_prices - past_prices

        # Buy the outcome with the largest positive change
        # Or sell the one with largest negative change
        if np.max(price_changes) > abs(np.min(price_changes)):
            outcome = np.argmax(price_changes)
            direction = 1
        else:
            outcome = np.argmin(price_changes)
            direction = -1

        return outcome, direction

    def _contrarian_trade(self, market: Market, n_outcomes: int) -> tuple[int, int]:
        """
        Contrarian: buy outcomes whose prices have been falling.
        """
        if len(market.state_history) < 2:
            return self._random_trade(n_outcomes)

        lookback = min(self.momentum_lookback, len(market.state_history) - 1)
        current_prices = market.get_prices()
        past_prices = market.state_history[-lookback - 1].prices

        price_changes = current_prices - past_prices

        # Buy the outcome with the largest negative change (contrarian)
        if abs(np.min(price_changes)) > np.max(price_changes):
            outcome = np.argmin(price_changes)
            direction = 1  # Buy the loser
        else:
            outcome = np.argmax(price_changes)
            direction = -1  # Sell the winner

        return outcome, direction

    def _biased_trade(self, n_outcomes: int) -> tuple[int, int]:
        """
        Biased: systematically favor a particular outcome.

        This represents expressive trading - betting on what you want to happen.
        """
        if self.bias_outcome is None:
            self.bias_outcome = 0

        # With probability bias_strength, trade the biased outcome
        if self._rng.random() < self.bias_strength:
            outcome = self.bias_outcome
            direction = 1  # Always buy the preferred outcome
        else:
            # Otherwise, trade randomly
            outcome = self._rng.integers(0, n_outcomes)
            direction = 1 if self._rng.random() > 0.5 else -1

        return outcome, direction

    def _liquidity_trade(self, n_outcomes: int) -> tuple[int, int]:
        """
        Liquidity: trade to meet cash needs, regardless of prices.

        More likely to sell when holdings are high, buy when low.
        """
        if self.holdings is None or np.sum(np.abs(self.holdings)) == 0:
            # No holdings, buy something
            outcome = self._rng.integers(0, n_outcomes)
            return outcome, 1

        # Probability of selling proportional to holdings
        total_holdings = np.sum(np.abs(self.holdings))
        holdings_ratio = np.abs(self.holdings) / total_holdings

        # More likely to sell outcomes we hold more of
        if self._rng.random() < 0.7:  # 70% chance to sell existing holdings
            outcome = self._rng.choice(n_outcomes, p=holdings_ratio)
            direction = -1 if self.holdings[outcome] > 0 else 1
        else:
            outcome = self._rng.integers(0, n_outcomes)
            direction = 1

        return outcome, direction

    def __repr__(self) -> str:
        return (
            f"NoiseTrader(id={self.agent_id}, type={self.trader_type.name}, "
            f"wealth={self.wealth:.2f}, trade_prob={self.trade_probability})"
        )
