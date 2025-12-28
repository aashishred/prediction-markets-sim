"""
Experiment 2: Discovery vs Aggregation (Testing Hypothesis H2)

This experiment tests the core hypothesis: Markets that incentivise discovery
create more value than markets that only aggregate existing knowledge.

We compare:
1. Hayekian environment (aggregation-only): Signals pre-distributed to agents
2. Discoverable environment (discovery-enabled): Agents must pay to discover

Key questions:
- Does discovery lead to more accurate prices?
- Is the cost of discovery justified by the accuracy improvement?
- How does discovery activity affect market efficiency?
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_markets.markets.contracts import BinaryContract
from prediction_markets.markets.lmsr import LSMRMarket
from prediction_markets.agents.informed import InformedAgent
from prediction_markets.agents.noise import NoiseTrader, NoiseTraderType
from prediction_markets.agents.discoverer import DiscovererAgent
from prediction_markets.agents.discovery import DiscoveryModel
from prediction_markets.environments.hayekian import HayekianEnvironment, HayekianConfig
from prediction_markets.environments.discoverable import DiscoverableEnvironment, DiscoverableConfig


@dataclass
class ExperimentResult:
    """Results from a single simulation run."""
    environment_type: str
    final_price: float
    true_value: int
    theoretical_aggregate: float
    price_error_vs_truth: float
    price_error_vs_aggregate: float
    information_ratio: float
    total_trades: int
    discovery_attempts: int = 0
    discovery_successes: int = 0
    discovery_cost_spent: float = 0.0
    information_discovered_fraction: float = 0.0


def run_hayekian_simulation(
    n_signals: int = 10,
    n_informed: int = 5,
    n_noise: int = 3,
    signal_precision: float = 0.85,
    liquidity: float = 100.0,
    n_steps: int = 50,
    seed: int = None
) -> ExperimentResult:
    """Run simulation in Hayekian (aggregation-only) environment."""

    contract = BinaryContract(name="Test Event", description="Will the event occur?")

    env_config = HayekianConfig(
        n_outcomes=2,
        n_signals=n_signals,
        signal_precision=signal_precision,
        signals_per_agent=n_signals / n_informed,
        signal_distribution="uniform",
        random_seed=seed
    )
    environment = HayekianEnvironment(config=env_config, contract=contract)

    market = LSMRMarket(contract=contract, liquidity=liquidity)

    # Create agents
    agents = []
    for i in range(n_informed):
        agent = InformedAgent(
            agent_id=f"informed_{i}",
            initial_wealth=1000.0,
            signal_precision=signal_precision,
            trade_threshold=0.02,
            kelly_fraction=0.3,
            max_position=50.0
        )
        agents.append(agent)

    for i in range(n_noise):
        agent = NoiseTrader(
            agent_id=f"noise_{i}",
            initial_wealth=500.0,
            trader_type=NoiseTraderType.RANDOM,
            trade_probability=0.2,
            trade_size_mean=0.03,
            trade_size_std=0.01
        )
        agent.set_seed(1000 + i + (seed or 0))
        agents.append(agent)

    # Initialize and distribute signals
    for agent in agents:
        agent.initialise(n_outcomes=2)
    environment.distribute_signals(agents)

    # Run simulation
    rng = np.random.default_rng(seed)
    total_trades = 0

    for step in range(n_steps):
        shuffled = agents.copy()
        rng.shuffle(shuffled)

        for agent in shuffled:
            outcome_idx, shares = agent.decide_trade(market)
            if outcome_idx is not None and abs(shares) > 0.1:
                trade = agent.execute_trade(market, outcome_idx, shares)
                if trade is not None:
                    total_trades += 1

        market.advance_time()

    # Calculate results
    final_prices = market.get_prices()
    theoretical = environment.theoretical_aggregate()
    true_probs = np.array([1 - environment.true_value, environment.true_value])

    price_error_truth = abs(final_prices[1] - true_probs[1])
    price_error_aggregate = abs(final_prices[1] - theoretical[1])
    prior_error = abs(0.5 - true_probs[1])
    info_ratio = 1 - price_error_truth / prior_error if prior_error > 0 else 0

    return ExperimentResult(
        environment_type="Hayekian (Aggregation)",
        final_price=final_prices[1],
        true_value=environment.true_value,
        theoretical_aggregate=theoretical[1],
        price_error_vs_truth=price_error_truth,
        price_error_vs_aggregate=price_error_aggregate,
        information_ratio=info_ratio,
        total_trades=total_trades,
        information_discovered_fraction=1.0  # All info is "discovered" (pre-distributed)
    )


def run_discoverable_simulation(
    n_signals: int = 10,
    n_discoverers: int = 5,
    n_noise: int = 3,
    discovery_cost: float = 5.0,
    success_probability: float = 0.8,
    signal_validity: float = 0.85,
    liquidity: float = 100.0,
    n_steps: int = 50,
    seed: int = None
) -> ExperimentResult:
    """Run simulation in Discoverable environment with discoverer agents."""

    contract = BinaryContract(name="Test Event", description="Will the event occur?")

    env_config = DiscoverableConfig(
        n_outcomes=2,
        n_hidden_signals=n_signals,
        discovery_cost=discovery_cost,
        success_probability=success_probability,
        signal_validity=signal_validity,
        validity_uncertainty=0.1,
        bearing_known=False,
        random_seed=seed
    )
    environment = DiscoverableEnvironment(config=env_config, contract=contract)

    market = LSMRMarket(contract=contract, liquidity=liquidity)

    # Create agents
    agents = []
    discoverers = []

    for i in range(n_discoverers):
        discovery_model = DiscoveryModel(rng=np.random.default_rng((seed or 0) + i + 100))
        agent = DiscovererAgent(
            agent_id=f"discoverer_{i}",
            initial_wealth=1000.0,
            discovery_budget=100.0,
            discovery_threshold=-1.0,  # Willing to discover even with slightly negative EV
            max_attempts_per_step=3,   # More attempts per step
            discovery_model=discovery_model,
            kelly_fraction=0.3,
            trade_threshold=0.02,
            max_position=50.0
        )
        agents.append(agent)
        discoverers.append(agent)

    for i in range(n_noise):
        agent = NoiseTrader(
            agent_id=f"noise_{i}",
            initial_wealth=500.0,
            trader_type=NoiseTraderType.RANDOM,
            trade_probability=0.2,
            trade_size_mean=0.03,
            trade_size_std=0.01
        )
        agent.set_seed(1000 + i + (seed or 0))
        agents.append(agent)

    # Initialize agents
    for agent in agents:
        agent.initialise(n_outcomes=2)

    # Run simulation with discovery phase
    rng = np.random.default_rng(seed)
    total_trades = 0
    total_discovery_attempts = 0
    total_discovery_successes = 0
    total_discovery_cost = 0.0

    for step in range(n_steps):
        # Discovery phase: discoverers attempt to discover
        for discoverer in discoverers:
            opportunities = environment.get_discovery_opportunities(discoverer.agent_id)

            # Set opportunities for the agent
            opp_list = [opp for _, opp in opportunities]
            discoverer.set_opportunities(opp_list)

            # Try to discover
            for signal_idx, opportunity in opportunities[:discoverer.max_attempts_per_step]:
                if discoverer.should_discover(opportunity, market):
                    true_signal, true_bearing = environment.attempt_discovery(
                        discoverer.agent_id, signal_idx
                    )

                    signal = discoverer.attempt_discovery(
                        opportunity, true_signal, true_bearing, step
                    )

                    total_discovery_attempts += 1
                    total_discovery_cost += opportunity.cost

                    if signal is not None:
                        total_discovery_successes += 1
                        environment.record_discovery(discoverer.agent_id, signal_idx)

        # Trading phase
        shuffled = agents.copy()
        rng.shuffle(shuffled)

        for agent in shuffled:
            outcome_idx, shares = agent.decide_trade(market)
            if outcome_idx is not None and abs(shares) > 0.1:
                trade = agent.execute_trade(market, outcome_idx, shares)
                if trade is not None:
                    total_trades += 1

        market.advance_time()

    # Calculate results
    final_prices = market.get_prices()
    theoretical = environment.theoretical_aggregate()
    true_probs = np.array([1 - environment.true_value, environment.true_value])

    price_error_truth = abs(final_prices[1] - true_probs[1])
    price_error_aggregate = abs(final_prices[1] - theoretical[1])
    prior_error = abs(0.5 - true_probs[1])
    info_ratio = 1 - price_error_truth / prior_error if prior_error > 0 else 0

    return ExperimentResult(
        environment_type="Discoverable (Discovery)",
        final_price=final_prices[1],
        true_value=environment.true_value,
        theoretical_aggregate=theoretical[1],
        price_error_vs_truth=price_error_truth,
        price_error_vs_aggregate=price_error_aggregate,
        information_ratio=info_ratio,
        total_trades=total_trades,
        discovery_attempts=total_discovery_attempts,
        discovery_successes=total_discovery_successes,
        discovery_cost_spent=total_discovery_cost,
        information_discovered_fraction=environment.information_discovered_fraction()
    )


def run_comparison_experiment(
    n_runs: int = 30,
    n_signals: int = 10,
    n_agents: int = 5,
    n_noise: int = 3,
    base_seed: int = 42
) -> tuple[list[ExperimentResult], list[ExperimentResult]]:
    """Run matched comparison experiments."""

    hayekian_results = []
    discoverable_results = []

    print(f"Running {n_runs} matched simulations...")
    print()

    for i in range(n_runs):
        seed = base_seed + i

        # Run Hayekian
        h_result = run_hayekian_simulation(
            n_signals=n_signals,
            n_informed=n_agents,
            n_noise=n_noise,
            signal_precision=0.85,
            liquidity=100.0,
            n_steps=50,
            seed=seed
        )
        hayekian_results.append(h_result)

        # Run Discoverable with same seed
        # Lower discovery cost to make discovery worthwhile
        d_result = run_discoverable_simulation(
            n_signals=n_signals,
            n_discoverers=n_agents,
            n_noise=n_noise,
            discovery_cost=0.5,  # Lower cost to incentivize discovery
            success_probability=0.9,  # Higher success rate
            signal_validity=0.85,
            liquidity=100.0,
            n_steps=50,
            seed=seed
        )
        discoverable_results.append(d_result)

        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_runs} runs")

    return hayekian_results, discoverable_results


def analyze_results(
    hayekian: list[ExperimentResult],
    discoverable: list[ExperimentResult]
) -> dict[str, Any]:
    """Analyze and compare results from both environments."""

    h_errors = [r.price_error_vs_truth for r in hayekian]
    d_errors = [r.price_error_vs_truth for r in discoverable]

    h_info_ratios = [r.information_ratio for r in hayekian]
    d_info_ratios = [r.information_ratio for r in discoverable]

    h_trades = [r.total_trades for r in hayekian]
    d_trades = [r.total_trades for r in discoverable]

    d_attempts = [r.discovery_attempts for r in discoverable]
    d_successes = [r.discovery_successes for r in discoverable]
    d_costs = [r.discovery_cost_spent for r in discoverable]
    d_discovered = [r.information_discovered_fraction for r in discoverable]

    analysis = {
        "n_runs": len(hayekian),
        "hayekian": {
            "mean_error": np.mean(h_errors),
            "std_error": np.std(h_errors),
            "mean_info_ratio": np.mean(h_info_ratios),
            "std_info_ratio": np.std(h_info_ratios),
            "mean_trades": np.mean(h_trades),
        },
        "discoverable": {
            "mean_error": np.mean(d_errors),
            "std_error": np.std(d_errors),
            "mean_info_ratio": np.mean(d_info_ratios),
            "std_info_ratio": np.std(d_info_ratios),
            "mean_trades": np.mean(d_trades),
            "mean_discovery_attempts": np.mean(d_attempts),
            "mean_discovery_successes": np.mean(d_successes),
            "mean_discovery_cost": np.mean(d_costs),
            "mean_info_discovered": np.mean(d_discovered),
        },
        "comparison": {
            "error_difference": np.mean(h_errors) - np.mean(d_errors),
            "info_ratio_difference": np.mean(d_info_ratios) - np.mean(h_info_ratios),
            "discovery_better_pct": sum(1 for h, d in zip(h_errors, d_errors) if d < h) / len(h_errors) * 100,
        }
    }

    return analysis


def print_results(analysis: dict[str, Any]) -> None:
    """Print formatted results."""

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 RESULTS: Discovery vs Aggregation")
    print("=" * 70)

    print(f"\nNumber of runs: {analysis['n_runs']}")

    print("\n" + "-" * 40)
    print("HAYEKIAN ENVIRONMENT (Aggregation Only)")
    print("-" * 40)
    h = analysis["hayekian"]
    print(f"  Mean price error vs truth:  {h['mean_error']:.3f} (+/- {h['std_error']:.3f})")
    print(f"  Mean information ratio:     {h['mean_info_ratio']:.3f} (+/- {h['std_info_ratio']:.3f})")
    print(f"  Mean trades per simulation: {h['mean_trades']:.1f}")

    print("\n" + "-" * 40)
    print("DISCOVERABLE ENVIRONMENT (Discovery Enabled)")
    print("-" * 40)
    d = analysis["discoverable"]
    print(f"  Mean price error vs truth:  {d['mean_error']:.3f} (+/- {d['std_error']:.3f})")
    print(f"  Mean information ratio:     {d['mean_info_ratio']:.3f} (+/- {d['std_info_ratio']:.3f})")
    print(f"  Mean trades per simulation: {d['mean_trades']:.1f}")
    print(f"\n  Discovery activity:")
    print(f"    Mean attempts: {d['mean_discovery_attempts']:.1f}")
    print(f"    Mean successes: {d['mean_discovery_successes']:.1f}")
    print(f"    Mean cost spent: {d['mean_discovery_cost']:.1f}")
    print(f"    Mean info discovered: {d['mean_info_discovered']*100:.1f}%")

    print("\n" + "-" * 40)
    print("COMPARISON")
    print("-" * 40)
    c = analysis["comparison"]
    print(f"  Error difference (H - D): {c['error_difference']:.3f}")
    if c['error_difference'] > 0:
        print(f"    -> Discovery environment has LOWER error (better)")
    else:
        print(f"    -> Aggregation environment has LOWER error (better)")

    print(f"  Info ratio difference (D - H): {c['info_ratio_difference']:.3f}")
    if c['info_ratio_difference'] > 0:
        print(f"    -> Discovery environment has HIGHER info ratio (better)")
    else:
        print(f"    -> Aggregation environment has HIGHER info ratio (better)")

    print(f"  Discovery better in {c['discovery_better_pct']:.0f}% of runs")

    print("\n" + "=" * 70)


def create_visualizations(
    hayekian: list[ExperimentResult],
    discoverable: list[ExperimentResult],
    output_path: Path
) -> None:
    """Create comparison visualizations."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Error distributions
    ax1 = axes[0, 0]
    h_errors = [r.price_error_vs_truth for r in hayekian]
    d_errors = [r.price_error_vs_truth for r in discoverable]

    positions = [1, 2]
    bp = ax1.boxplot([h_errors, d_errors], positions=positions, widths=0.6, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')

    ax1.set_xticklabels(['Hayekian\n(Aggregation)', 'Discoverable\n(Discovery)'])
    ax1.set_ylabel('Price Error vs Truth')
    ax1.set_title('Price Accuracy Comparison')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Prior error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Information ratios
    ax2 = axes[0, 1]
    h_info = [r.information_ratio for r in hayekian]
    d_info = [r.information_ratio for r in discoverable]

    bp2 = ax2.boxplot([h_info, d_info], positions=positions, widths=0.6, patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightblue')
    bp2['boxes'][1].set_facecolor('lightgreen')

    ax2.set_xticklabels(['Hayekian\n(Aggregation)', 'Discoverable\n(Discovery)'])
    ax2.set_ylabel('Information Ratio')
    ax2.set_title('Market Value-Add Comparison')
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No value-add')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Paired comparison
    ax3 = axes[1, 0]
    runs = range(len(hayekian))
    ax3.scatter(h_errors, d_errors, alpha=0.6, s=50)
    ax3.plot([0, 1], [0, 1], 'r--', label='Equal performance')
    ax3.set_xlabel('Hayekian Error')
    ax3.set_ylabel('Discoverable Error')
    ax3.set_title('Paired Run Comparison\n(points below line = discovery better)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)

    # Count and annotate
    discovery_wins = sum(1 for h, d in zip(h_errors, d_errors) if d < h)
    ax3.annotate(f'Discovery better: {discovery_wins}/{len(hayekian)}',
                 xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='left', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 4: Discovery activity vs accuracy
    ax4 = axes[1, 1]
    d_discovered = [r.information_discovered_fraction for r in discoverable]

    ax4.scatter(d_discovered, d_errors, alpha=0.6, s=50, c='green', label='Discoverable')

    # Add trend line only if there's variance in the data
    if len(set(d_discovered)) > 1:
        try:
            z = np.polyfit(d_discovered, d_errors, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(d_discovered), max(d_discovered), 100)
            ax4.plot(x_line, p(x_line), 'g--', alpha=0.8, label='Trend')
        except np.linalg.LinAlgError:
            pass  # Skip trend line if fitting fails

    ax4.set_xlabel('Fraction of Information Discovered')
    ax4.set_ylabel('Price Error vs Truth')
    ax4.set_title('Discovery Activity vs Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nVisualization saved to: {output_path}")


def main():
    """Run the full experiment."""

    print("=" * 70)
    print("EXPERIMENT 2: Discovery vs Aggregation (Hypothesis H2)")
    print("=" * 70)
    print()
    print("Testing whether markets that incentivise discovery outperform")
    print("markets that only aggregate existing knowledge.")
    print()

    # Run comparison
    hayekian_results, discoverable_results = run_comparison_experiment(
        n_runs=30,
        n_signals=10,
        n_agents=5,
        n_noise=3,
        base_seed=42
    )

    # Analyze
    analysis = analyze_results(hayekian_results, discoverable_results)

    # Print
    print_results(analysis)

    # Visualize
    output_path = Path(__file__).parent / "experiment_02_discovery_comparison.png"
    create_visualizations(hayekian_results, discoverable_results, output_path)

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    c = analysis["comparison"]

    if c['error_difference'] > 0.02:
        print("""
The Discovery environment shows meaningfully LOWER error than pure Aggregation.
This supports Hypothesis H2: markets that incentivise discovery create more value.

When agents must pay to discover information (rather than receiving it passively),
the market still achieves similar or better accuracy. This suggests that the
incentive structure of markets can successfully motivate costly information
production.
""")
    elif c['error_difference'] < -0.02:
        print("""
The Aggregation environment shows LOWER error than Discovery.
This appears to contradict Hypothesis H2.

However, consider:
1. In Hayekian, signals are pre-distributed (free information)
2. In Discoverable, agents must pay for information (costly discovery)
3. The discovery cost may exceed the value of information acquired

This might actually support the Coasean constraint (H3): when discovery costs
are high relative to information value, markets fail to beat simpler mechanisms.
""")
    else:
        print("""
Results are similar between environments (within noise).

This is actually an interesting finding: even when information must be costly
discovered (rather than freely distributed), markets can achieve comparable
accuracy. This suggests discovery incentives ARE working.

The Grossman-Stiglitz insight applies: markets can't be fully efficient
(which would eliminate discovery incentives), but they can be efficient enough
to justify discovery costs.
""")

    return analysis


if __name__ == "__main__":
    analysis = main()
