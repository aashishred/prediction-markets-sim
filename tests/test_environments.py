"""Tests for environment implementations."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_markets.environments.unified import UnifiedEnvironment, UnifiedConfig
from prediction_markets.environments.hayekian import HayekianEnvironment, HayekianConfig
from prediction_markets.environments.base import EnvironmentType
from prediction_markets.agents.informed import InformedAgent


class TestUnifiedEnvironment:
    """Test the unified leakiness-based environment."""

    def test_leakiness_validation(self):
        """Leakiness must be in [0, 1]."""
        with pytest.raises(ValueError, match="leakiness must be in"):
            UnifiedConfig(leakiness=1.5)

        with pytest.raises(ValueError, match="leakiness must be in"):
            UnifiedConfig(leakiness=-0.1)

    def test_zero_leakiness_knightian(self):
        """Zero leakiness should produce Knightian-like environment."""
        config = UnifiedConfig(leakiness=0.0, n_signals=20, random_seed=42)
        env = UnifiedEnvironment(config=config)

        info = env.get_information_structure()

        # At leakiness=0, nothing should leak
        assert info["n_signals"]["leaked"] == 0
        assert info["n_signals"]["free"] == 0
        assert info["n_signals"]["discoverable"] == 0
        assert info["n_signals"]["hidden"] == 20

        # Should classify as Knightian
        assert env.environment_type == EnvironmentType.KNIGHTIAN

    def test_full_leakiness_discoverable(self):
        """Full leakiness should make all signals discoverable."""
        config = UnifiedConfig(leakiness=1.0, n_signals=20, random_seed=42)
        env = UnifiedEnvironment(config=config)

        info = env.get_information_structure()

        # At leakiness=1, all signals should leak and be discoverable
        assert info["n_signals"]["leaked"] == 20
        assert info["n_signals"]["free"] == 0  # free_fraction = 1 - leakiness = 0
        assert info["n_signals"]["discoverable"] == 20
        assert info["n_signals"]["hidden"] == 0

        # Should classify as Discoverable
        assert env.environment_type == EnvironmentType.DISCOVERABLE

    def test_medium_leakiness_hayekian(self):
        """Medium leakiness should produce Hayekian-like environment."""
        config = UnifiedConfig(leakiness=0.5, n_signals=20, random_seed=42)
        env = UnifiedEnvironment(config=config)

        info = env.get_information_structure()

        # At leakiness=0.5:
        # - 50% of signals leak (10)
        # - Of those, 50% are free (5), 50% discoverable (5)
        assert info["n_signals"]["leaked"] == 10
        assert info["n_signals"]["free"] == 5
        assert info["n_signals"]["discoverable"] == 5
        assert info["n_signals"]["hidden"] == 10

        # Should classify as Hayekian
        assert env.environment_type == EnvironmentType.HAYEKIAN

    def test_signal_distribution(self):
        """Signals should be distributed to agents correctly."""
        config = UnifiedConfig(
            leakiness=0.5,
            n_signals=20,
            signals_per_agent=3.0,
            random_seed=42,
        )
        env = UnifiedEnvironment(config=config)

        # Create test agents
        agents = [
            InformedAgent(agent_id=f"agent_{i}")
            for i in range(5)
        ]
        for agent in agents:
            agent.initialise(n_outcomes=2)

        # Distribute signals
        signals = env.generate_signals(agents)

        # At least some agents should receive signals
        agents_with_signals = sum(1 for a in agents if a.agent_id in signals)
        assert agents_with_signals > 0

    def test_theoretical_aggregate(self):
        """Theoretical aggregate should only consider leaked signals."""
        config = UnifiedConfig(leakiness=0.5, n_signals=20, random_seed=42)
        env = UnifiedEnvironment(config=config)

        aggregate = env.theoretical_aggregate()

        # Should be a valid probability distribution
        assert len(aggregate) == 2
        assert np.isclose(aggregate.sum(), 1.0)
        assert all(0 <= p <= 1 for p in aggregate)

    def test_discovery_opportunities(self):
        """Discovery opportunities should be available for discoverable signals."""
        config = UnifiedConfig(leakiness=0.7, n_signals=20, random_seed=42)
        env = UnifiedEnvironment(config=config)

        opportunities = env.get_discovery_opportunities("test_agent")

        # At leakiness=0.7, should have some discoverable signals
        info = env.get_information_structure()
        expected_discoverable = info["n_signals"]["discoverable"]

        assert len(opportunities) == expected_discoverable

    def test_discovery_cost_scaling(self):
        """Discovery cost should scale inversely with leakiness."""
        low_leaky = UnifiedConfig(leakiness=0.3, n_signals=20, random_seed=42)
        high_leaky = UnifiedConfig(leakiness=0.9, n_signals=20, random_seed=42)

        env_low = UnifiedEnvironment(config=low_leaky)
        env_high = UnifiedEnvironment(config=high_leaky)

        # Get discovery opportunities
        opps_low = env_low.get_discovery_opportunities("agent")
        opps_high = env_high.get_discovery_opportunities("agent")

        if opps_low and opps_high:
            # Low leakiness should have higher costs
            cost_low = opps_low[0][1].cost
            cost_high = opps_high[0][1].cost
            assert cost_low > cost_high

    def test_true_value_generation(self):
        """True value should be generated from all signals."""
        config = UnifiedConfig(leakiness=0.5, n_signals=20, random_seed=42)
        env = UnifiedEnvironment(config=config)

        # True value should be 0 or 1 for binary
        assert env.true_value in (0, 1)

    def test_reproducibility(self):
        """Same seed should produce same environment."""
        config1 = UnifiedConfig(leakiness=0.5, n_signals=20, random_seed=123)
        config2 = UnifiedConfig(leakiness=0.5, n_signals=20, random_seed=123)

        env1 = UnifiedEnvironment(config=config1)
        env2 = UnifiedEnvironment(config=config2)

        assert env1.true_value == env2.true_value
        assert np.array_equal(env1.theoretical_aggregate(), env2.theoretical_aggregate())


class TestHayekianEnvironment:
    """Test the Hayekian distributed knowledge environment."""

    def test_creation(self):
        """Should create a valid environment."""
        config = HayekianConfig(n_signals=10, random_seed=42)
        env = HayekianEnvironment(config=config)

        assert env.environment_type == EnvironmentType.HAYEKIAN
        assert env.true_value in (0, 1)

    def test_signal_distribution(self):
        """Signals should be distributed to agents."""
        config = HayekianConfig(
            n_signals=20,
            signals_per_agent=4.0,
            random_seed=42,
        )
        env = HayekianEnvironment(config=config)

        agents = [
            InformedAgent(agent_id=f"agent_{i}")
            for i in range(5)
        ]
        for agent in agents:
            agent.initialise(n_outcomes=2)

        signals = env.generate_signals(agents)

        # Should have signals for at least some agents
        assert len(signals) > 0

    def test_theoretical_aggregate(self):
        """Should compute theoretical aggregate of all signals."""
        config = HayekianConfig(n_signals=20, random_seed=42)
        env = HayekianEnvironment(config=config)

        aggregate = env.theoretical_aggregate()

        assert len(aggregate) == 2
        assert np.isclose(aggregate.sum(), 1.0)


class TestEnvironmentReproducibility:
    """Test that environments are reproducible with seeds."""

    def test_unified_different_seeds_different_results(self):
        """Different seeds should produce different environments (with high probability)."""
        # Use seeds that are known to produce different results
        # Test across multiple seed pairs to reduce flakiness
        different_found = False
        seed_pairs = [(1, 100), (42, 999), (123, 456)]

        for seed1, seed2 in seed_pairs:
            env1 = UnifiedEnvironment(
                config=UnifiedConfig(leakiness=0.5, n_signals=20, random_seed=seed1)
            )
            env2 = UnifiedEnvironment(
                config=UnifiedConfig(leakiness=0.5, n_signals=20, random_seed=seed2)
            )

            aggregate1 = env1.theoretical_aggregate()
            aggregate2 = env2.theoretical_aggregate()

            if not np.allclose(aggregate1, aggregate2) or env1.true_value != env2.true_value:
                different_found = True
                break

        # At least one pair should differ (extremely unlikely all match)
        assert different_found, "All seed pairs produced identical results - very unlikely"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
