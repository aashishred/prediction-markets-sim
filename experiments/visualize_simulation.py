"""
Diagnostic Visualization: Understanding What's Happening in the Simulation

This script runs a single simulation with detailed output and visualizations
to help understand the mechanics of how agents trade and prices evolve.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_markets.markets.contracts import BinaryContract
from prediction_markets.markets.lmsr import LSMRMarket
from prediction_markets.agents.informed import InformedAgent
from prediction_markets.agents.noise import NoiseTrader, NoiseTraderType
from prediction_markets.environments.hayekian import HayekianEnvironment, HayekianConfig
from prediction_markets.simulation.runner import Simulation, SimulationConfig
from prediction_markets.simulation.metrics import (
    calculate_price_error,
    calculate_information_ratio,
    calculate_aggregation_efficiency,
)


def run_diagnostic_simulation():
    """Run a single simulation with detailed diagnostics."""

    print("=" * 70)
    print("DIAGNOSTIC SIMULATION: Understanding Prediction Market Dynamics")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Create the environment
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: ENVIRONMENT SETUP")
    print("=" * 70)

    contract = BinaryContract(
        name="Test Event",
        description="Will the event occur?"
    )

    n_signals = 20
    n_informed = 10

    env_config = HayekianConfig(
        n_outcomes=2,
        n_signals=n_signals,
        signal_precision=0.85,  # 85% chance of correct signal
        signals_per_agent=n_signals / n_informed,  # 2 signals per agent on average
        signal_distribution="uniform",
        random_seed=42
    )

    environment = HayekianEnvironment(config=env_config, contract=contract)

    print(f"\nEnvironment created:")
    print(f"  - Number of signals: {n_signals}")
    print(f"  - Signal precision: {env_config.signal_precision}")
    print(f"  - Signals per agent: {env_config.signals_per_agent}")

    # Show the actual signals
    print(f"\nUnderlying signals (ground truth):")
    print(f"  - Raw signals: {environment._signals}")
    print(f"  - Signal weights: {environment._signal_weights}")

    # What probability do these signals imply?
    theoretical = environment.theoretical_aggregate()
    print(f"\nTheoretical aggregate (perfect aggregation would give):")
    print(f"  - P(No) = {theoretical[0]:.3f}, P(Yes) = {theoretical[1]:.3f}")

    print(f"\nTrue outcome (sampled from signals): {environment.true_value}")
    print(f"  (This is the actual resolution - what will happen)")

    # =========================================================================
    # STEP 2: Create the market
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: MARKET SETUP (LMSR)")
    print("=" * 70)

    liquidity = 100.0
    market = LSMRMarket(contract=contract, liquidity=liquidity)

    print(f"\nMarket created:")
    print(f"  - Mechanism: LMSR (Logarithmic Market Scoring Rule)")
    print(f"  - Liquidity parameter (b): {liquidity}")
    print(f"  - Maximum possible loss for market maker: £{market.max_loss():.2f}")
    print(f"  - Initial prices: P(No)={market.get_price(0):.3f}, P(Yes)={market.get_price(1):.3f}")

    # =========================================================================
    # STEP 3: Create agents
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: AGENT SETUP")
    print("=" * 70)

    agents = []

    # Informed agents
    for i in range(n_informed):
        agent = InformedAgent(
            agent_id=f"informed_{i}",
            initial_wealth=1000.0,
            signal_precision=0.85,
            trade_threshold=0.02,  # Trade if edge > 2%
            kelly_fraction=0.3,    # Conservative Kelly betting
            max_position=50.0
        )
        agents.append(agent)

    # Noise traders
    n_noise = 5
    for i in range(n_noise):
        agent = NoiseTrader(
            agent_id=f"noise_{i}",
            initial_wealth=500.0,
            trader_type=NoiseTraderType.RANDOM,
            trade_probability=0.2,
            trade_size_mean=0.03,
            trade_size_std=0.01
        )
        agent.set_seed(100 + i)
        agents.append(agent)

    print(f"\nAgents created:")
    print(f"  - {n_informed} informed agents (wealth=£1000 each)")
    print(f"  - {n_noise} noise traders (wealth=£500 each)")

    # =========================================================================
    # STEP 4: Initialize and distribute signals
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: SIGNAL DISTRIBUTION")
    print("=" * 70)

    # Initialize agents
    for agent in agents:
        agent.initialise(n_outcomes=2)

    # Distribute signals
    environment.distribute_signals(agents)

    print("\nSignal distribution to agents:")
    signal_dist = environment.get_signal_distribution()

    informed_agents = [a for a in agents if a.agent_id.startswith("informed")]
    for agent in informed_agents:
        n_signals_received = signal_dist["agent_assignments"].get(agent.agent_id, 0)
        print(f"  - {agent.agent_id}: received {n_signals_received} signals, "
              f"beliefs now P(Yes)={agent.beliefs[1]:.3f}")

    # =========================================================================
    # STEP 5: Run simulation step by step
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: SIMULATION (watching trades happen)")
    print("=" * 70)

    n_steps = 50
    price_history = [market.get_prices().copy()]
    trade_log = []

    print(f"\nRunning {n_steps} steps...")
    print(f"Initial price: P(Yes) = {market.get_price(1):.3f}")
    print()

    for step in range(n_steps):
        step_trades = []

        # Shuffle agent order
        np.random.shuffle(agents)

        for agent in agents:
            outcome_idx, shares = agent.decide_trade(market)

            if outcome_idx is not None and abs(shares) > 0.1:
                # Get price before trade
                price_before = market.get_price(outcome_idx)

                # Execute trade
                trade = agent.execute_trade(market, outcome_idx, shares)

                if trade is not None:
                    price_after = market.get_price(outcome_idx)

                    step_trades.append({
                        "agent": agent.agent_id,
                        "outcome": "Yes" if outcome_idx == 1 else "No",
                        "shares": shares,
                        "cost": trade.cost,
                        "price_before": price_before,
                        "price_after": price_after
                    })

                    # Print first 10 trades in detail
                    if len(trade_log) < 10:
                        direction = "BUY" if shares > 0 else "SELL"
                        print(f"  Trade {len(trade_log)+1}: {agent.agent_id} {direction} "
                              f"{abs(shares):.1f} {step_trades[-1]['outcome']} @ £{price_before:.3f} "
                              f"-> £{price_after:.3f} (cost: £{trade.cost:.2f})")

        trade_log.extend(step_trades)
        price_history.append(market.get_prices().copy())
        market.advance_time()

    print(f"\n... (simulation complete)")
    print(f"Total trades executed: {len(trade_log)}")

    # =========================================================================
    # STEP 6: Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 6: RESULTS")
    print("=" * 70)

    final_prices = market.get_prices()
    true_probs = np.array([1 - environment.true_value, environment.true_value])

    print(f"\nFinal market state:")
    print(f"  - Final prices: P(No)={final_prices[0]:.3f}, P(Yes)={final_prices[1]:.3f}")
    print(f"  - True outcome: {'Yes' if environment.true_value == 1 else 'No'}")
    print(f"  - Theoretical aggregate: P(Yes)={theoretical[1]:.3f}")

    # Calculate errors
    price_error = np.abs(final_prices[1] - true_probs[1])
    aggregate_error = np.abs(final_prices[1] - theoretical[1])
    prior_error = np.abs(0.5 - true_probs[1])

    print(f"\nError analysis:")
    print(f"  - Price error vs truth: {price_error:.3f}")
    print(f"  - Price error vs theoretical aggregate: {aggregate_error:.3f}")
    print(f"  - Prior error (0.5) vs truth: {prior_error:.3f}")
    print(f"  - Information ratio: {1 - price_error/prior_error:.3f}")

    # Agent P&L
    print(f"\nAgent profit/loss (if resolved now):")
    contract.resolve(environment.true_value)

    total_informed_pnl = 0
    total_noise_pnl = 0

    for agent in agents:
        holdings = agent.holdings
        payout = holdings[environment.true_value]  # Only winning shares pay out
        pnl = payout - agent.total_cost

        if agent.agent_id.startswith("informed"):
            total_informed_pnl += pnl
        else:
            total_noise_pnl += pnl

    print(f"  - Total informed trader P&L: £{total_informed_pnl:.2f}")
    print(f"  - Total noise trader P&L: £{total_noise_pnl:.2f}")
    print(f"  - Market maker P&L: £{-market._get_subsidy_spent():.2f}")

    # =========================================================================
    # STEP 7: Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 7: GENERATING VISUALIZATIONS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Price evolution
    ax1 = axes[0, 0]
    price_history_arr = np.array(price_history)
    steps = range(len(price_history_arr))

    ax1.plot(steps, price_history_arr[:, 1], 'b-', linewidth=2, label='Market P(Yes)')
    ax1.axhline(y=theoretical[1], color='g', linestyle='--', linewidth=2,
                label=f'Theoretical Aggregate ({theoretical[1]:.2f})')
    ax1.axhline(y=true_probs[1], color='r', linestyle=':', linewidth=2,
                label=f'True Probability ({true_probs[1]:.0f})')
    ax1.axhline(y=0.5, color='gray', linestyle='-.', alpha=0.5, label='Prior (0.5)')

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Probability of Yes')
    ax1.set_title('Price Evolution Over Time')
    ax1.legend(loc='best')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Agent beliefs vs market price
    ax2 = axes[0, 1]
    informed_beliefs = [a.beliefs[1] for a in agents if a.agent_id.startswith("informed")]

    ax2.bar(range(len(informed_beliefs)), informed_beliefs, alpha=0.7, label='Agent beliefs P(Yes)')
    ax2.axhline(y=final_prices[1], color='b', linewidth=2, label=f'Final market price ({final_prices[1]:.2f})')
    ax2.axhline(y=theoretical[1], color='g', linestyle='--', linewidth=2,
                label=f'Theoretical aggregate ({theoretical[1]:.2f})')
    ax2.axhline(y=true_probs[1], color='r', linestyle=':', linewidth=2,
                label=f'True probability ({true_probs[1]:.0f})')

    ax2.set_xlabel('Informed Agent')
    ax2.set_ylabel('P(Yes)')
    ax2.set_title('Agent Beliefs vs Market Price')
    ax2.legend(loc='best')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Trade activity over time
    ax3 = axes[1, 0]
    if trade_log:
        trade_times = []
        trade_sizes = []
        trade_colors = []

        current_step = 0
        trades_per_step = {i: 0 for i in range(n_steps)}

        for i, trade in enumerate(trade_log):
            trade_sizes.append(abs(trade["shares"]))
            trade_colors.append('green' if trade["shares"] > 0 else 'red')

        ax3.scatter(range(len(trade_log)), trade_sizes, c=trade_colors, alpha=0.6, s=50)
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Trade Size (shares)')
        ax3.set_title('Trade Activity (green=buy, red=sell)')
        ax3.grid(True, alpha=0.3)

    # Plot 4: The gap between price and aggregate
    ax4 = axes[1, 1]

    # Show what the market "should" have learned
    price_vs_aggregate = price_history_arr[:, 1] - theoretical[1]
    price_vs_truth = price_history_arr[:, 1] - true_probs[1]

    ax4.plot(steps, price_vs_aggregate, 'g-', linewidth=2, label='Price - Aggregate')
    ax4.plot(steps, price_vs_truth, 'r-', linewidth=2, label='Price - Truth')
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax4.fill_between(steps, price_vs_aggregate, 0, alpha=0.3, color='green')

    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Error')
    ax4.set_title('Price Error Over Time')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / "diagnostic_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")

    plt.close()

    # =========================================================================
    # STEP 8: Diagnosis
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 8: DIAGNOSIS")
    print("=" * 70)

    print("\nKey observations:")

    # Check if agents are trading
    if len(trade_log) < 10:
        print("  [!] Very few trades happening - agents may not be acting on signals")
    else:
        print(f"  [OK] {len(trade_log)} trades executed")

    # Check if price moved
    price_movement = abs(final_prices[1] - 0.5)
    if price_movement < 0.05:
        print(f"  [!] Price barely moved from 0.5 (only {price_movement:.3f})")
    else:
        print(f"  [OK] Price moved {price_movement:.3f} from prior")

    # Check if price moved toward aggregate
    if aggregate_error < abs(0.5 - theoretical[1]):
        print(f"  [OK] Price moved toward theoretical aggregate")
    else:
        print(f"  [!] Price did NOT move toward aggregate")

    # Check agent belief diversity
    belief_std = np.std(informed_beliefs)
    print(f"  Agent belief std dev: {belief_std:.3f}")

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    return {
        "price_history": price_history_arr,
        "trade_log": trade_log,
        "final_prices": final_prices,
        "theoretical": theoretical,
        "true_value": environment.true_value,
        "agent_beliefs": informed_beliefs
    }


if __name__ == "__main__":
    results = run_diagnostic_simulation()
