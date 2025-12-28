"""
Experiment 1: Testing Aggregation in Hayekian Environment

This experiment tests the core claim that markets aggregate dispersed information.
We compare:
1. Market final price vs true probability
2. Market accuracy vs simple averaging of agent beliefs
3. How accuracy varies with number of agents and signal quality

Hypothesis (H1): Aggregation alone has limited value when transaction costs
are considered. Market improvement over simple averaging is small.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# Add parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_markets.markets.contracts import BinaryContract
from prediction_markets.markets.lmsr import LSMRMarket
from prediction_markets.agents.informed import InformedAgent
from prediction_markets.agents.noise import NoiseTrader, NoiseTraderType
from prediction_markets.environments.hayekian import HayekianEnvironment, HayekianConfig
from prediction_markets.simulation.runner import Simulation, SimulationConfig, run_batch
from prediction_markets.simulation.metrics import (
    calculate_price_error,
    calculate_information_ratio,
    calculate_aggregation_efficiency,
    summarize_batch,
)


def create_basic_simulation(
    n_informed: int = 10,
    n_noise: int = 5,
    n_signals: int = 10,
    signal_precision: float = 0.8,
    liquidity: float = 100.0,
    n_steps: int = 100,
    seed: int | None = None
) -> Simulation:
    """
    Create a simulation with informed agents in a Hayekian environment.
    """
    # Create contract
    contract = BinaryContract(
        name="Test Event",
        description="Will the event occur?"
    )

    # Create environment
    env_config = HayekianConfig(
        n_outcomes=2,
        n_signals=n_signals,
        signal_precision=signal_precision,
        signals_per_agent=n_signals / n_informed if n_informed > 0 else 1,
        random_seed=seed
    )
    environment = HayekianEnvironment(config=env_config, contract=contract)

    # Create market
    market = LSMRMarket(contract=contract, liquidity=liquidity)

    # Create agents
    agents = []

    # Informed agents
    for i in range(n_informed):
        agent = InformedAgent(
            agent_id=f"informed_{i}",
            initial_wealth=1000.0,
            signal_precision=signal_precision,
            trade_threshold=0.02,
            kelly_fraction=0.3
        )
        agents.append(agent)

    # Noise traders
    for i in range(n_noise):
        agent = NoiseTrader(
            agent_id=f"noise_{i}",
            initial_wealth=500.0,
            trader_type=NoiseTraderType.RANDOM,
            trade_probability=0.2
        )
        agents.append(agent)

    # Create simulation
    sim_config = SimulationConfig(
        n_steps=n_steps,
        random_seed=seed
    )

    return Simulation(
        environment=environment,
        market=market,
        agents=agents,
        config=sim_config
    )


def run_experiment_vary_agents(n_runs: int = 50) -> dict:
    """
    Experiment: How does market accuracy vary with number of informed agents?
    """
    agent_counts = [2, 5, 10, 20, 50]
    results = {}

    for n_agents in agent_counts:
        print(f"Running with {n_agents} informed agents...")

        def create_sim():
            return create_basic_simulation(
                n_informed=n_agents,
                n_noise=max(2, n_agents // 2),
                n_signals=n_agents * 2,  # 2 signals per agent on average
                n_steps=100
            )

        batch_results = run_batch(create_sim, n_runs)
        results[n_agents] = summarize_batch(batch_results)

    return results


def run_experiment_vary_precision(n_runs: int = 50) -> dict:
    """
    Experiment: How does market accuracy vary with signal precision?
    """
    precisions = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    results = {}

    for precision in precisions:
        print(f"Running with signal precision {precision}...")

        def create_sim():
            return create_basic_simulation(
                n_informed=10,
                n_noise=5,
                n_signals=20,
                signal_precision=precision,
                n_steps=100
            )

        batch_results = run_batch(create_sim, n_runs)
        results[precision] = summarize_batch(batch_results)

    return results


def run_experiment_vary_liquidity(n_runs: int = 50) -> dict:
    """
    Experiment: How does liquidity (LMSR parameter) affect outcomes?

    Lower liquidity = prices move more easily = less subsidy needed
    Higher liquidity = prices more stable = more subsidy, harder to move
    """
    liquidities = [10, 50, 100, 200, 500]
    results = {}

    for liq in liquidities:
        print(f"Running with liquidity {liq}...")

        def create_sim():
            return create_basic_simulation(
                n_informed=10,
                n_noise=5,
                liquidity=liq,
                n_steps=100
            )

        batch_results = run_batch(create_sim, n_runs)
        results[liq] = summarize_batch(batch_results)

    return results


def plot_results(results: dict, x_label: str, title: str, filename: str):
    """Plot experiment results."""
    x_values = list(results.keys())
    mean_errors = [results[x]["mean_price_error"] for x in x_values]
    std_errors = [results[x]["std_price_error"] for x in x_values]
    info_ratios = [results[x]["mean_information_ratio"] for x in x_values]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Price error plot
    ax1 = axes[0]
    ax1.errorbar(x_values, mean_errors, yerr=std_errors, marker='o', capsize=5)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Mean Price Error")
    ax1.set_title(f"{title}: Price Error")
    ax1.grid(True, alpha=0.3)

    # Information ratio plot
    ax2 = axes[1]
    ax2.plot(x_values, info_ratios, marker='s', color='green')
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Information Ratio")
    ax2.set_title(f"{title}: Information Ratio")
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No improvement over prior')
    ax2.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Perfect information')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved plot to {filename}")


def main():
    """Run all experiments."""
    print("=" * 60)
    print("Experiment 1: Testing Aggregation in Hayekian Environment")
    print("=" * 60)

    # Single run demo first
    print("\n--- Single Run Demo ---")
    sim = create_basic_simulation(n_informed=10, n_noise=5, seed=42)
    result = sim.run()

    print(f"True outcome: {result.true_value}")
    print(f"Final prices: {result.final_prices}")
    print(f"Price error: {calculate_price_error(result):.4f}")
    print(f"Information ratio: {calculate_information_ratio(result):.4f}")
    if result.theoretical_aggregate is not None:
        print(f"Theoretical aggregate: {result.theoretical_aggregate}")
        print(f"Aggregation efficiency: {calculate_aggregation_efficiency(result):.4f}")

    # Run experiments
    print("\n--- Experiment 1a: Vary Number of Agents ---")
    results_agents = run_experiment_vary_agents(n_runs=30)
    for n, stats in results_agents.items():
        print(f"  {n} agents: error={stats['mean_price_error']:.4f} (±{stats['std_price_error']:.4f}), "
              f"info_ratio={stats['mean_information_ratio']:.4f}")

    print("\n--- Experiment 1b: Vary Signal Precision ---")
    results_precision = run_experiment_vary_precision(n_runs=30)
    for p, stats in results_precision.items():
        print(f"  precision={p}: error={stats['mean_price_error']:.4f} (±{stats['std_price_error']:.4f}), "
              f"info_ratio={stats['mean_information_ratio']:.4f}")

    print("\n--- Experiment 1c: Vary Liquidity ---")
    results_liquidity = run_experiment_vary_liquidity(n_runs=30)
    for liq, stats in results_liquidity.items():
        print(f"  liquidity={liq}: error={stats['mean_price_error']:.4f} (±{stats['std_price_error']:.4f}), "
              f"info_ratio={stats['mean_information_ratio']:.4f}")

    # Generate plots
    print("\n--- Generating Plots ---")
    plot_results(results_agents, "Number of Informed Agents",
                 "Aggregation vs Agents", "experiments/plot_agents.png")
    plot_results(results_precision, "Signal Precision",
                 "Aggregation vs Precision", "experiments/plot_precision.png")
    plot_results(results_liquidity, "LMSR Liquidity Parameter",
                 "Aggregation vs Liquidity", "experiments/plot_liquidity.png")

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
