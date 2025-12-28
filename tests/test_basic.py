"""Basic tests to verify core components work."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_markets.markets.contracts import BinaryContract, MultinomialContract
from prediction_markets.markets.lmsr import LSMRMarket
from prediction_markets.agents.informed import InformedAgent
from prediction_markets.agents.noise import NoiseTrader, NoiseTraderType


class TestContracts:
    """Test contract implementations."""

    def test_binary_contract_creation(self):
        contract = BinaryContract(name="Test", description="Will X happen?")
        assert contract.n_outcomes == 2
        assert contract.outcome_names == ["No", "Yes"]

    def test_binary_contract_resolution(self):
        contract = BinaryContract(name="Test")
        contract.resolve(True)
        assert contract.is_resolved()
        assert contract.payout(1) == 1.0  # Yes wins
        assert contract.payout(0) == 0.0  # No loses

    def test_multinomial_contract(self):
        contract = MultinomialContract(
            name="Election",
            outcomes=["A", "B", "C", "D"]
        )
        assert contract.n_outcomes == 4
        contract.resolve("B")
        assert contract.payout(1) == 1.0
        assert contract.payout(0) == 0.0


class TestLSMR:
    """Test LMSR market maker."""

    def test_initial_prices(self):
        contract = BinaryContract(name="Test")
        market = LSMRMarket(contract=contract, liquidity=100)

        prices = market.get_prices()
        assert len(prices) == 2
        assert np.isclose(prices[0], 0.5)
        assert np.isclose(prices[1], 0.5)
        assert np.isclose(sum(prices), 1.0)

    def test_buy_increases_price(self):
        contract = BinaryContract(name="Test")
        market = LSMRMarket(contract=contract, liquidity=100)

        initial_price = market.get_price(1)
        market.execute_trade("agent1", outcome_index=1, shares=10)
        new_price = market.get_price(1)

        assert new_price > initial_price
        assert np.isclose(sum(market.get_prices()), 1.0)

    def test_sell_decreases_price(self):
        contract = BinaryContract(name="Test")
        market = LSMRMarket(contract=contract, liquidity=100)

        initial_price = market.get_price(1)
        market.execute_trade("agent1", outcome_index=1, shares=-10)
        new_price = market.get_price(1)

        assert new_price < initial_price
        assert np.isclose(sum(market.get_prices()), 1.0)

    def test_cost_calculation(self):
        contract = BinaryContract(name="Test")
        market = LSMRMarket(contract=contract, liquidity=100)

        cost = market.get_cost(outcome_index=1, shares=10)
        assert cost > 0  # Buying costs money

        sell_cost = market.get_cost(outcome_index=1, shares=-10)
        assert sell_cost < 0  # Selling gives money

    def test_market_resolution(self):
        contract = BinaryContract(name="Test")
        market = LSMRMarket(contract=contract, liquidity=100)

        # Agent buys Yes shares
        market.execute_trade("agent1", outcome_index=1, shares=20)
        # Agent buys No shares
        market.execute_trade("agent2", outcome_index=0, shares=10)

        # Resolve as Yes
        payouts = market.resolve(True)

        # Agent1 should profit, Agent2 should lose
        assert payouts["agent1"] > 0
        assert payouts["agent2"] < 0


class TestAgents:
    """Test agent implementations."""

    def test_informed_agent_creation(self):
        agent = InformedAgent(
            agent_id="test",
            initial_wealth=1000,
            signal_precision=0.8
        )
        assert agent.wealth == 1000
        assert agent.signal_precision == 0.8

    def test_informed_agent_signal_update(self):
        agent = InformedAgent(agent_id="test")
        agent.initialise(n_outcomes=2)

        # Initially uniform beliefs
        assert np.allclose(agent.beliefs, [0.5, 0.5])

        # Receive signal pointing to outcome 1
        agent.receive_signal({"outcome": 1, "precision": 0.9, "strength": 1.0})

        # Beliefs should now favor outcome 1
        assert agent.beliefs[1] > agent.beliefs[0]

    def test_noise_trader_random(self):
        agent = NoiseTrader(
            agent_id="noise",
            trader_type=NoiseTraderType.RANDOM,
            trade_probability=1.0  # Always trade
        )
        agent.initialise(n_outcomes=2)
        agent.set_seed(42)

        contract = BinaryContract(name="Test")
        market = LSMRMarket(contract=contract, liquidity=100)

        outcome, shares = agent.decide_trade(market)
        assert outcome in [0, 1, None]
        if outcome is not None:
            assert shares != 0


class TestIntegration:
    """Integration tests combining components."""

    def test_simple_trading_cycle(self):
        # Setup
        contract = BinaryContract(name="Test")
        market = LSMRMarket(contract=contract, liquidity=100)

        agent = InformedAgent(agent_id="informed", signal_precision=0.9)
        agent.initialise(n_outcomes=2)
        agent.receive_signal({"outcome": 1, "precision": 0.9, "strength": 1.0})

        # Agent should want to buy outcome 1 (believes it's underpriced at 0.5)
        outcome, shares = agent.decide_trade(market)

        if outcome is not None and shares != 0:
            trade = agent.execute_trade(market, outcome, shares)
            assert trade is not None
            assert agent.wealth < 1000  # Spent money
            assert agent.holdings[outcome] > 0  # Owns shares


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
