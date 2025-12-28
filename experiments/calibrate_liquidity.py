"""
Calibrate LMSR Liquidity Parameter

Problem: With b=100, prices barely move (0.04 change with informed trades).
Goal: Find b such that typical informed trades move price ~5-10 percentage points.

This script tests different liquidity values and finds the optimal one.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_markets.markets.contracts import BinaryContract
from prediction_markets.markets.lmsr import LSMRMarket
from prediction_markets.agents.informed import InformedAgent
from prediction_markets.environments.hayekian import HayekianEnvironment, HayekianConfig


def test_liquidity_parameter(liquidity: float, seed: int = 42) -> dict:
    """Test a single liquidity parameter value."""

    contract = BinaryContract(name="Test", description="Test")
    env_config = HayekianConfig(
        n_outcomes=2,
        n_signals=20,
        signal_precision=0.85,
        signals_per_agent=2.0,
        random_seed=seed
    )
    environment = HayekianEnvironment(config=env_config, contract=contract)
    market = LSMRMarket(contract=contract, liquidity=liquidity)

    # Create one informed agent with very low threshold to ensure trading
    agent = InformedAgent(
        agent_id="test",
        initial_wealth=1000.0,
        signal_precision=0.85,
        trade_threshold=0.001,  # Very low threshold to ensure trading
        kelly_fraction=0.3,
        max_position=50.0
    )

    agent.initialise(n_outcomes=2)

    # Manually set belief to create clear edge (avoid equality issue)
    agent.beliefs = np.array([0.30, 0.70])  # Believe outcome 1 is 70% likely

    # Track price movements
    initial_price = market.get_price(1)
    belief = agent.beliefs[1]
    edge = abs(belief - initial_price)

    # Execute one trade
    outcome_idx, shares = agent.decide_trade(market)

    if outcome_idx is not None and abs(shares) > 0.1:
        price_before = market.get_price(outcome_idx)
        trade = agent.execute_trade(market, outcome_idx, shares)

        if trade is not None:
            price_after = market.get_price(outcome_idx)
            price_impact = abs(price_after - price_before)

            return {
                "liquidity": liquidity,
                "belief": belief,
                "initial_price": initial_price,
                "edge": edge,
                "shares_traded": abs(shares),
                "price_impact": price_impact,
                "impact_per_share": price_impact / abs(shares) if shares != 0 else 0,
            }

    return {
        "liquidity": liquidity,
        "belief": belief,
        "initial_price": initial_price,
        "edge": edge,
        "shares_traded": 0,
        "price_impact": 0,
        "impact_per_share": 0,
    }


def main():
    print("\n" + "="*70)
    print("LMSR LIQUIDITY CALIBRATION")
    print("="*70)
    print("\nGoal: Find liquidity parameter where typical trades move price 5-10%")
    print()

    # Test range of liquidity values
    liquidities = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    results = []

    for liq in liquidities:
        result = test_liquidity_parameter(liq)
        results.append(result)

        print(f"Liquidity={liq:3.0f}: "
              f"Edge={result['edge']:.3f}, "
              f"Shares={result['shares_traded']:5.1f}, "
              f"Impact={result['price_impact']:.4f} "
              f"({result['price_impact']*100:.1f}%)")

    # Find optimal liquidity
    target_impact = 0.075  # Target 7.5% price movement
    best_idx = min(range(len(results)),
                   key=lambda i: abs(results[i]['price_impact'] - target_impact))
    optimal = results[best_idx]

    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print(f"\nOptimal liquidity parameter: {optimal['liquidity']:.0f}")
    print(f"  Expected price impact: {optimal['price_impact']*100:.1f}%")
    print(f"  Shares per trade: {optimal['shares_traded']:.1f}")
    print(f"  Impact per share: {optimal['impact_per_share']:.5f}")

    print(f"\nCurrent (b=100): {results[-2]['price_impact']*100:.1f}% impact")
    print(f"Recommended (b={optimal['liquidity']:.0f}): {optimal['price_impact']*100:.1f}% impact")
    print(f"Improvement: {(optimal['price_impact'] / results[-2]['price_impact']):.1f}x more price movement")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Price impact vs liquidity
    ax1 = axes[0]
    liq_vals = [r['liquidity'] for r in results]
    impacts = [r['price_impact'] * 100 for r in results]

    ax1.plot(liq_vals, impacts, 'b-o', linewidth=2, markersize=8)
    ax1.axhline(y=target_impact*100, color='green', linestyle='--', linewidth=2,
                label=f'Target: {target_impact*100:.1f}%')
    ax1.axvline(x=optimal['liquidity'], color='red', linestyle='--', linewidth=2,
                label=f"Optimal: b={optimal['liquidity']:.0f}")
    ax1.axvline(x=100, color='gray', linestyle=':', linewidth=2,
                label='Current: b=100')

    ax1.set_xlabel('Liquidity Parameter (b)')
    ax1.set_ylabel('Price Impact (%)')
    ax1.set_title('Price Impact vs Liquidity Parameter')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Impact per share
    ax2 = axes[1]
    impact_per_share = [r['impact_per_share'] * 100 for r in results]

    ax2.plot(liq_vals, impact_per_share, 'g-o', linewidth=2, markersize=8)
    ax2.axvline(x=optimal['liquidity'], color='red', linestyle='--', linewidth=2,
                label=f"Optimal: b={optimal['liquidity']:.0f}")
    ax2.axvline(x=100, color='gray', linestyle=':', linewidth=2,
                label='Current: b=100')

    ax2.set_xlabel('Liquidity Parameter (b)')
    ax2.set_ylabel('Price Impact per Share (%)')
    ax2.set_title('Marginal Price Impact')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path(__file__).parent / "liquidity_calibration.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    plt.close()

    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print(f"""
Update all experiments to use b={optimal['liquidity']:.0f} instead of b=100.

This will:
1. Make price movements more visible
2. Allow beliefs to translate into prices more efficiently
3. Better match real prediction market behavior

Files to update:
- experiments/experiment_01_aggregation.py
- experiments/experiment_02_discovery.py
- experiments/comprehensive_visualization.py
- Any other experiments using LSMRMarket
""")


if __name__ == "__main__":
    main()
