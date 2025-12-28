"""
Test unified environment across the leakiness spectrum.

This demonstrates how the continuous leakiness parameter works,
showing the transition from Knightian → Hayekian → Discoverable.
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_markets.environments.unified import UnifiedEnvironment, UnifiedConfig
from prediction_markets.markets.contracts import BinaryContract


def test_leakiness_level(leakiness: float, seed: int = 42) -> dict:
    """Test a single leakiness level and return information structure."""

    contract = BinaryContract(name="Test", description="Test")
    config = UnifiedConfig(
        n_outcomes=2,
        n_signals=20,
        leakiness=leakiness,
        signal_precision=0.85,
        signals_per_agent=2.0,
        discovery_base_cost=5.0,
        random_seed=seed
    )

    env = UnifiedEnvironment(config=config, contract=contract)

    # Get information structure
    info = env.get_information_structure()

    return {
        "leakiness": leakiness,
        "env_type": env.environment_type.name,
        "n_leaked": info["n_signals"]["leaked"],
        "n_free": info["n_signals"]["free"],
        "n_discoverable": info["n_signals"]["discoverable"],
        "n_hidden": info["n_signals"]["hidden"],
        "pct_leaked": info["percentages"]["leaked"],
        "pct_free": info["percentages"]["free"],
        "pct_discoverable": info["percentages"]["discoverable"],
        "pct_hidden": info["percentages"]["hidden"],
        "theoretical_aggregate": env.theoretical_aggregate(),
        "true_value": env.true_value,
    }


def main():
    print("\n" + "="*80)
    print("UNIFIED ENVIRONMENT: CONTINUOUS LEAKINESS SPECTRUM")
    print("="*80)
    print("\nTesting how information structure changes with leakiness parameter...")
    print()

    # Test across the full spectrum
    leakiness_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"{'Leak':>5} | {'Type':^12} | {'Leaked':>7} | {'Free':>7} | {'Discov':>7} | {'Hidden':>7}")
    print("-" * 80)

    results = []
    for leakiness in leakiness_values:
        result = test_leakiness_level(leakiness)
        results.append(result)

        print(f"{result['leakiness']:>5.2f} | {result['env_type']:^12} | "
              f"{result['pct_leaked']:>6.1f}% | {result['pct_free']:>6.1f}% | "
              f"{result['pct_discoverable']:>6.1f}% | {result['pct_hidden']:>6.1f}%")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("""
The leakiness parameter controls information availability:

KNIGHTIAN REGIME (leakiness < 0.3):
- Very little information leaks into the present
- What exists is mostly FREE (in agent minds)
- Little opportunity for discovery
- Market has limited value-add (aggregation only)

HAYEKIAN REGIME (0.3 <= leakiness < 0.7):
- Moderate information leakage
- Mix of FREE (distributed knowledge) and DISCOVERABLE
- Market aggregates tacit knowledge AND incentivizes discovery
- Classic prediction market use case

DISCOVERABLE REGIME (leakiness >= 0.7):
- Most/all information has leaked
- Most leaked info requires DISCOVERY (costs money)
- Market value comes primarily from incentivizing research
- Discovery dominates aggregation

KEY INSIGHT:
As leakiness increases:
1. More total information becomes available (Leaked % increases)
2. BUT more of it requires costly discovery (Discoverable % increases)
3. Less is freely distributed (Free % decreases)
4. The regime shifts from aggregation to discovery
""")

    print("\n" + "="*80)
    print("EXAMPLE: Medium Leakiness (0.5)")
    print("="*80)

    medium = [r for r in results if r["leakiness"] == 0.5][0]
    print(f"""
At leakiness = 0.5 (classic Hayekian case):
- {medium['pct_leaked']:.1f}% of information has leaked into the present
  - {medium['pct_free']:.1f}% is FREE (distributed as tacit knowledge)
  - {medium['pct_discoverable']:.1f}% requires DISCOVERY (costs money)
- {medium['pct_hidden']:.1f}% remains hidden (Knightian uncertainty)

The market must:
1. Aggregate the {medium['pct_free']:.1f}% of free distributed knowledge
2. Incentivize discovery of the {medium['pct_discoverable']:.1f}% discoverable information
3. Accept {medium['pct_hidden']:.1f}% fundamental uncertainty

This is the OPTIMAL use case for prediction markets:
- Enough distributed knowledge to aggregate
- Enough discoverable info to incentivize research
- Not so much leakiness that discovery is trivial
""")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Create experiments that sweep across the leakiness spectrum
2. Measure market performance at each level
3. Find the optimal leakiness range for prediction markets
4. Compare market value-add across the spectrum
5. Test hypothesis: markets excel at medium leakiness (0.3-0.7)
""")


if __name__ == "__main__":
    main()
