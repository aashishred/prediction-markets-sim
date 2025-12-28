"""
Comprehensive Visualization Dashboard

This creates detailed visualizations to understand:
1. What agents are doing (beliefs, trades, discovery)
2. How the market behaves (price evolution, liquidity)
3. How well aggregation works (vs baselines)
4. The role of parameters (liquidity, signal precision, etc.)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
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


def run_detailed_hayekian_simulation(seed=42):
    """Run a single Hayekian simulation with detailed tracking."""

    print("\n" + "="*70)
    print("HAYEKIAN ENVIRONMENT: Aggregation Simulation")
    print("="*70)

    contract = BinaryContract(name="Test Event", description="Will the event occur?")

    env_config = HayekianConfig(
        n_outcomes=2,
        n_signals=20,
        signal_precision=0.85,
        signals_per_agent=2.0,
        signal_distribution="uniform",
        random_seed=seed
    )
    environment = HayekianEnvironment(config=env_config, contract=contract)

    market = LSMRMarket(contract=contract, liquidity=100.0)

    # Create agents
    n_informed = 10
    n_noise = 5
    agents = []

    for i in range(n_informed):
        agent = InformedAgent(
            agent_id=f"informed_{i}",
            initial_wealth=1000.0,
            signal_precision=0.85,
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
        agent.set_seed(1000 + i + seed)
        agents.append(agent)

    # Initialize
    for agent in agents:
        agent.initialise(n_outcomes=2)

    environment.distribute_signals(agents)

    # Get initial state
    informed_agents = [a for a in agents if a.agent_id.startswith("informed")]
    initial_beliefs = np.array([a.beliefs[1] for a in informed_agents])

    theoretical = environment.theoretical_aggregate()
    true_value = environment.true_value

    print(f"\nEnvironment setup:")
    print(f"  Latent probability (from signals): P(Yes) = {theoretical[1]:.3f}")
    print(f"  Realized outcome: {'Yes' if true_value == 1 else 'No'}")
    print(f"  Agent beliefs: mean={initial_beliefs.mean():.3f}, std={initial_beliefs.std():.3f}")

    # Tracking
    price_history = [market.get_prices().copy()]
    belief_history = [initial_beliefs.copy()]
    trade_log = []

    # Baselines
    mean_belief_history = [initial_beliefs.mean()]

    n_steps = 50
    rng = np.random.default_rng(seed)

    for step in range(n_steps):
        shuffled = agents.copy()
        rng.shuffle(shuffled)

        for agent in shuffled:
            outcome_idx, shares = agent.decide_trade(market)

            if outcome_idx is not None and abs(shares) > 0.1:
                price_before = market.get_price(outcome_idx)
                trade = agent.execute_trade(market, outcome_idx, shares)

                if trade is not None:
                    price_after = market.get_price(outcome_idx)

                    # Calculate belief-price gap
                    agent_belief = agent.beliefs[outcome_idx] if agent.beliefs is not None else 0.5
                    belief_gap = agent_belief - price_before

                    trade_log.append({
                        'step': step,
                        'agent': agent.agent_id,
                        'agent_type': 'informed' if 'informed' in agent.agent_id else 'noise',
                        'outcome': outcome_idx,
                        'shares': shares,
                        'price_before': price_before,
                        'price_after': price_after,
                        'price_impact': price_after - price_before,
                        'belief_gap': belief_gap,
                        'cost': trade.cost
                    })

        price_history.append(market.get_prices().copy())
        current_beliefs = np.array([a.beliefs[1] for a in informed_agents])
        belief_history.append(current_beliefs.copy())
        mean_belief_history.append(current_beliefs.mean())

        market.advance_time()

    return {
        'environment_type': 'Hayekian',
        'price_history': np.array(price_history),
        'belief_history': belief_history,
        'mean_belief_history': mean_belief_history,
        'trade_log': trade_log,
        'theoretical': theoretical,
        'true_value': true_value,
        'initial_beliefs': initial_beliefs,
        'agents': agents,
        'market': market,
        'n_steps': n_steps
    }


def run_detailed_discoverable_simulation(seed=42):
    """Run a single Discoverable simulation with detailed tracking."""

    print("\n" + "="*70)
    print("DISCOVERABLE ENVIRONMENT: Discovery Simulation")
    print("="*70)

    contract = BinaryContract(name="Test Event", description="Will the event occur?")

    env_config = DiscoverableConfig(
        n_outcomes=2,
        n_hidden_signals=20,
        discovery_cost=0.5,
        success_probability=0.9,
        signal_validity=0.85,
        validity_uncertainty=0.1,
        bearing_known=False,
        random_seed=seed
    )
    environment = DiscoverableEnvironment(config=env_config, contract=contract)

    market = LSMRMarket(contract=contract, liquidity=100.0)

    # Create agents
    n_discoverers = 10
    n_noise = 5
    agents = []
    discoverers = []

    for i in range(n_discoverers):
        discovery_model = DiscoveryModel(rng=np.random.default_rng(seed + i + 100))
        agent = DiscovererAgent(
            agent_id=f"discoverer_{i}",
            initial_wealth=1000.0,
            discovery_budget=100.0,
            discovery_threshold=-1.0,
            max_attempts_per_step=3,
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
        agent.set_seed(1000 + i + seed)
        agents.append(agent)

    # Initialize
    for agent in agents:
        agent.initialise(n_outcomes=2)

    theoretical = environment.theoretical_aggregate()
    true_value = environment.true_value

    print(f"\nEnvironment setup:")
    print(f"  Latent probability (from hidden signals): P(Yes) = {theoretical[1]:.3f}")
    print(f"  Realized outcome: {'Yes' if true_value == 1 else 'No'}")

    # Tracking
    price_history = [market.get_prices().copy()]
    belief_history = []
    discovery_log = []
    trade_log = []
    signals_discovered_history = []

    n_steps = 50
    rng = np.random.default_rng(seed)

    for step in range(n_steps):
        # Discovery phase
        for discoverer in discoverers:
            opportunities = environment.get_discovery_opportunities(discoverer.agent_id)
            opp_list = [opp for _, opp in opportunities]
            discoverer.set_opportunities(opp_list)

            for signal_idx, opportunity in opportunities[:discoverer.max_attempts_per_step]:
                if discoverer.should_discover(opportunity, market):
                    true_signal, true_bearing = environment.attempt_discovery(
                        discoverer.agent_id, signal_idx
                    )

                    signal = discoverer.attempt_discovery(
                        opportunity, true_signal, true_bearing, step
                    )

                    discovery_log.append({
                        'step': step,
                        'agent': discoverer.agent_id,
                        'signal_idx': signal_idx,
                        'success': signal is not None,
                        'cost': opportunity.cost,
                        'true_signal': true_signal,
                        'observed_signal': signal.content if signal else None,
                        'validity': signal.validity if signal else None,
                        'bearing': signal.bearing if signal else None
                    })

                    if signal is not None:
                        environment.record_discovery(discoverer.agent_id, signal_idx)

        # Trading phase
        shuffled = agents.copy()
        rng.shuffle(shuffled)

        for agent in shuffled:
            outcome_idx, shares = agent.decide_trade(market)

            if outcome_idx is not None and abs(shares) > 0.1:
                price_before = market.get_price(outcome_idx)
                trade = agent.execute_trade(market, outcome_idx, shares)

                if trade is not None:
                    price_after = market.get_price(outcome_idx)
                    agent_belief = agent.beliefs[outcome_idx] if agent.beliefs is not None else 0.5

                    trade_log.append({
                        'step': step,
                        'agent': agent.agent_id,
                        'agent_type': 'discoverer' if 'discoverer' in agent.agent_id else 'noise',
                        'outcome': outcome_idx,
                        'shares': shares,
                        'price_before': price_before,
                        'price_after': price_after,
                        'price_impact': price_after - price_before,
                        'belief_gap': agent_belief - price_before,
                        'cost': trade.cost
                    })

        price_history.append(market.get_prices().copy())
        current_beliefs = np.array([a.beliefs[1] for a in discoverers])
        belief_history.append(current_beliefs.copy())
        signals_discovered_history.append(environment.information_discovered_fraction())

        market.advance_time()

    return {
        'environment_type': 'Discoverable',
        'price_history': np.array(price_history),
        'belief_history': belief_history,
        'trade_log': trade_log,
        'discovery_log': discovery_log,
        'signals_discovered_history': signals_discovered_history,
        'theoretical': theoretical,
        'true_value': true_value,
        'agents': agents,
        'discoverers': discoverers,
        'market': market,
        'n_steps': n_steps
    }


def create_comprehensive_dashboard(hayekian_result, discoverable_result, output_path):
    """Create a comprehensive multi-panel visualization."""

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Color scheme
    color_latent = '#2ecc71'  # Green for latent probability
    color_market = '#3498db'  # Blue for market price
    color_realized = '#e74c3c'  # Red for realized outcome
    color_belief = '#9b59b6'  # Purple for beliefs

    # --- ROW 1: Price Evolution Comparison ---

    # Panel 1: Hayekian price evolution
    ax1 = fig.add_subplot(gs[0, 0:2])
    h_steps = range(len(hayekian_result['price_history']))
    ax1.plot(h_steps, hayekian_result['price_history'][:, 1],
             color=color_market, linewidth=2, label='Market Price')
    ax1.axhline(y=hayekian_result['theoretical'][1], color=color_latent,
                linestyle='--', linewidth=2, label=f"Latent P(Yes)={hayekian_result['theoretical'][1]:.2f}")
    ax1.axhline(y=hayekian_result['true_value'], color=color_realized,
                linestyle=':', linewidth=2, label=f"Realized: {'Yes' if hayekian_result['true_value']==1 else 'No'}")
    ax1.plot(h_steps, hayekian_result['mean_belief_history'],
             color=color_belief, linestyle='-.', alpha=0.7, label='Mean Agent Belief')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Probability')
    ax1.set_title('HAYEKIAN: Price Evolution (Aggregation)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Panel 2: Discoverable price evolution
    ax2 = fig.add_subplot(gs[0, 2:4])
    d_steps = range(len(discoverable_result['price_history']))
    ax2.plot(d_steps, discoverable_result['price_history'][:, 1],
             color=color_market, linewidth=2, label='Market Price')
    ax2.axhline(y=discoverable_result['theoretical'][1], color=color_latent,
                linestyle='--', linewidth=2, label=f"Latent P(Yes)={discoverable_result['theoretical'][1]:.2f}")
    ax2.axhline(y=discoverable_result['true_value'], color=color_realized,
                linestyle=':', linewidth=2, label=f"Realized: {'Yes' if discoverable_result['true_value']==1 else 'No'}")

    # Discovery progress
    ax2_twin = ax2.twinx()
    ax2_twin.plot(d_steps[1:], discoverable_result['signals_discovered_history'],
                  color='orange', linestyle='-.', alpha=0.6, label='Info Discovered')
    ax2_twin.set_ylabel('Fraction Discovered', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2_twin.set_ylim(0, 1.1)

    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Probability')
    ax2.set_title('DISCOVERABLE: Price Evolution (Discovery)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # --- ROW 2: Trading Activity ---

    # Panel 3: Hayekian trade impact
    ax3 = fig.add_subplot(gs[1, 0])
    h_trades = hayekian_result['trade_log']
    if h_trades:
        informed_trades = [t for t in h_trades if t['agent_type'] == 'informed']
        noise_trades = [t for t in h_trades if t['agent_type'] == 'noise']

        if informed_trades:
            ax3.scatter([t['step'] for t in informed_trades],
                       [abs(t['price_impact']) for t in informed_trades],
                       c='blue', s=30, alpha=0.6, label='Informed')
        if noise_trades:
            ax3.scatter([t['step'] for t in noise_trades],
                       [abs(t['price_impact']) for t in noise_trades],
                       c='gray', s=20, alpha=0.4, label='Noise')

    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('|Price Impact|')
    ax3.set_title('Hayekian: Trade Impact', fontsize=10, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Hayekian belief-price gaps
    ax4 = fig.add_subplot(gs[1, 1])
    if h_trades:
        informed_trades = [t for t in h_trades if t['agent_type'] == 'informed']
        if informed_trades:
            ax4.scatter([t['step'] for t in informed_trades],
                       [t['belief_gap'] for t in informed_trades],
                       c='blue', s=30, alpha=0.6)
            ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Belief - Price')
    ax4.set_title('Hayekian: Belief-Price Gap', fontsize=10, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    # Panel 5: Discoverable trade impact
    ax5 = fig.add_subplot(gs[1, 2])
    d_trades = discoverable_result['trade_log']
    if d_trades:
        disc_trades = [t for t in d_trades if t['agent_type'] == 'discoverer']
        noise_trades = [t for t in d_trades if t['agent_type'] == 'noise']

        if disc_trades:
            ax5.scatter([t['step'] for t in disc_trades],
                       [abs(t['price_impact']) for t in disc_trades],
                       c='green', s=30, alpha=0.6, label='Discoverer')
        if noise_trades:
            ax5.scatter([t['step'] for t in noise_trades],
                       [abs(t['price_impact']) for t in noise_trades],
                       c='gray', s=20, alpha=0.4, label='Noise')

    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('|Price Impact|')
    ax5.set_title('Discoverable: Trade Impact', fontsize=10, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Panel 6: Discovery activity
    ax6 = fig.add_subplot(gs[1, 3])
    d_disc = discoverable_result['discovery_log']
    if d_disc:
        successful = [d for d in d_disc if d['success']]
        failed = [d for d in d_disc if not d['success']]

        if successful:
            ax6.scatter([d['step'] for d in successful],
                       [d['signal_idx'] for d in successful],
                       c='green', s=50, alpha=0.7, marker='o', label='Success')
        if failed:
            ax6.scatter([d['step'] for d in failed],
                       [d['signal_idx'] for d in failed],
                       c='red', s=30, alpha=0.5, marker='x', label='Failed')

    ax6.set_xlabel('Time Step')
    ax6.set_ylabel('Signal Index')
    ax6.set_title('Discovery Attempts', fontsize=10, fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)

    # --- ROW 3: Performance Metrics ---

    # Panel 7: Error decomposition (Hayekian)
    ax7 = fig.add_subplot(gs[2, 0])
    h_final_price = hayekian_result['price_history'][-1, 1]
    h_latent = hayekian_result['theoretical'][1]
    h_realized = hayekian_result['true_value']

    errors_h = {
        'Price vs\nLatent': abs(h_final_price - h_latent),
        'Price vs\nRealized': abs(h_final_price - h_realized),
        'Prior vs\nRealized': abs(0.5 - h_realized)
    }

    bars = ax7.bar(errors_h.keys(), errors_h.values(),
                   color=['blue', 'red', 'gray'], alpha=0.7)
    ax7.set_ylabel('Absolute Error')
    ax7.set_title('Hayekian: Error Breakdown', fontsize=10, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')

    # Add values on bars
    for bar in bars:
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Panel 8: Error decomposition (Discoverable)
    ax8 = fig.add_subplot(gs[2, 1])
    d_final_price = discoverable_result['price_history'][-1, 1]
    d_latent = discoverable_result['theoretical'][1]
    d_realized = discoverable_result['true_value']

    errors_d = {
        'Price vs\nLatent': abs(d_final_price - d_latent),
        'Price vs\nRealized': abs(d_final_price - d_realized),
        'Prior vs\nRealized': abs(0.5 - d_realized)
    }

    bars = ax8.bar(errors_d.keys(), errors_d.values(),
                   color=['blue', 'red', 'gray'], alpha=0.7)
    ax8.set_ylabel('Absolute Error')
    ax8.set_title('Discoverable: Error Breakdown', fontsize=10, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # Panel 9: Agent belief distributions
    ax9 = fig.add_subplot(gs[2, 2])
    h_final_beliefs = hayekian_result['belief_history'][-1]
    ax9.hist(h_final_beliefs, bins=15, alpha=0.6, color='blue', label='Hayekian')
    ax9.axvline(h_latent, color=color_latent, linestyle='--', linewidth=2, label='Latent')
    ax9.axvline(h_final_price, color=color_market, linestyle='-', linewidth=2, label='Market')
    ax9.set_xlabel('Belief P(Yes)')
    ax9.set_ylabel('Count')
    ax9.set_title('Final Agent Beliefs', fontsize=10, fontweight='bold')
    ax9.legend(fontsize=8)
    ax9.grid(True, alpha=0.3)

    # Panel 10: Summary statistics
    ax10 = fig.add_subplot(gs[2, 3])
    ax10.axis('off')

    summary_text = f"""
HAYEKIAN (Aggregation)
━━━━━━━━━━━━━━━━━━━━
Trades: {len(hayekian_result['trade_log'])}
Error vs Latent: {abs(h_final_price - h_latent):.3f}
Error vs Realized: {abs(h_final_price - h_realized):.3f}

DISCOVERABLE (Discovery)
━━━━━━━━━━━━━━━━━━━━━━━
Trades: {len(discoverable_result['trade_log'])}
Discoveries: {len([d for d in discoverable_result['discovery_log'] if d['success']])}
Info Discovered: {discoverable_result['signals_discovered_history'][-1]*100:.0f}%
Error vs Latent: {abs(d_final_price - d_latent):.3f}
Error vs Realized: {abs(d_final_price - d_realized):.3f}

KEY INSIGHT
━━━━━━━━━━━
"Error vs Realized" mixes two
different targets! A price of
{h_latent:.2f} SHOULD lose when
outcome is {h_realized}.

Better metric: "Error vs Latent"
measures estimation quality.
"""

    ax10.text(0.1, 0.5, summary_text, fontsize=9, family='monospace',
              verticalalignment='center', bbox=dict(boxstyle='round',
              facecolor='wheat', alpha=0.3))

    # Overall title
    fig.suptitle('COMPREHENSIVE SIMULATION DASHBOARD', fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nDashboard saved to: {output_path}")
    plt.close()


def create_mechanism_explanation_chart(output_path):
    """Create a visual explanation of how the simulation works."""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('HOW THE SIMULATION WORKS', fontsize=16, fontweight='bold')

    # Panel 1: Environment generation
    ax1 = axes[0, 0]
    ax1.axis('off')
    ax1.text(0.5, 0.95, 'STEP 1: Environment Generation',
             ha='center', fontsize=12, fontweight='bold')

    mechanism_text = """
HAYEKIAN (Aggregation):
  1. Draw 20 binary signals [0 or 1]
  2. Weighted sum → latent P(Yes)
  3. Sample outcome from latent P
  4. Distribute signals to agents (free)

DISCOVERABLE (Discovery):
  1. Draw 20 binary signals [0 or 1]
  2. Weighted sum → latent P(Yes)
  3. Sample outcome from latent P
  4. Hide signals; agents pay to discover

KEY: Latent probability ≠ Realized outcome!
     Latent P(Yes)=0.6 means outcome is Yes
     60% of the time, No 40% of the time.
"""

    ax1.text(0.05, 0.5, mechanism_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    # Panel 2: Agent beliefs
    ax2 = axes[0, 1]
    ax2.axis('off')
    ax2.text(0.5, 0.95, 'STEP 2: Agent Beliefs',
             ha='center', fontsize=12, fontweight='bold')

    belief_text = """
Each agent observes signals with NOISE:

  True signal: 1
  Precision: 0.85
  → 85% chance agent sees "1"
  → 15% chance agent sees "0"

Agent updates beliefs via Bayes' rule:

  Prior: P(Yes) = 0.5
  Signal points to Yes
  → Posterior: P(Yes) = 0.73

Multiple signals → belief moves further.

PROBLEM: If signals are wrong,
         perfect aggregation makes
         you confidently WRONG!
"""

    ax2.text(0.05, 0.5, belief_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # Panel 3: Market mechanism
    ax3 = axes[1, 0]
    ax3.axis('off')
    ax3.text(0.5, 0.95, 'STEP 3: LMSR Market Mechanism',
             ha='center', fontsize=12, fontweight='bold')

    market_text = """
LMSR provides infinite liquidity:

  Cost = b * log(sum(exp(q_i / b)))

  where b = liquidity parameter
        q_i = shares of outcome i

  Price = ∂Cost/∂q

High b → prices move SLOWLY
Low b → prices move QUICKLY

Current: b=100

Agent trading rule:
  IF belief - price > threshold:
     Buy (belief - price) * wealth * kelly
  ELSE:
     Don't trade

Trade threshold = 0.02 (2%)
Kelly fraction = 0.3 (conservative)
"""

    ax3.text(0.05, 0.5, market_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

    # Panel 4: Metrics explanation
    ax4 = axes[1, 1]
    ax4.axis('off')
    ax4.text(0.5, 0.95, 'STEP 4: What We Measure',
             ha='center', fontsize=12, fontweight='bold')

    metrics_text = """
TWO DIFFERENT ERRORS:

1) ESTIMATION ERROR (what we want)
   |Price - Latent Probability|

   Measures: How well did market
             estimate TRUE probability?

   Good aggregation → small error

2) OUTCOME ERROR (misleading!)
   |Price - Realized Outcome|

   Measures: Did we predict correctly?

   Mixing these confuses calibration
   with single-outcome luck!

BETTER METRICS:
  - Brier score across many runs
  - Calibration curves
  - Price vs Latent scatter plots

BASELINES TO COMPARE:
  - Mean agent belief
  - Best individual agent
  - Pooled posterior (oracle)
"""

    ax4.text(0.05, 0.5, metrics_text, fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='#ffcccc', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Mechanism explanation saved to: {output_path}")
    plt.close()


def main():
    """Run comprehensive visualizations."""

    print("\n" + "="*70)
    print("COMPREHENSIVE VISUALIZATION DASHBOARD")
    print("="*70)
    print("\nThis will create detailed visualizations to explain:")
    print("  1. What agents are doing")
    print("  2. How markets behave")
    print("  3. How well aggregation works")
    print("  4. Why current metrics are misleading")
    print()

    seed = 42

    # Run simulations
    hayekian_result = run_detailed_hayekian_simulation(seed)
    discoverable_result = run_detailed_discoverable_simulation(seed)

    # Create visualizations
    output_dir = Path(__file__).parent

    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)

    dashboard_path = output_dir / "comprehensive_dashboard.png"
    create_comprehensive_dashboard(hayekian_result, discoverable_result, dashboard_path)

    mechanism_path = output_dir / "mechanism_explanation.png"
    create_mechanism_explanation_chart(mechanism_path)

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)

    print("""
1. METRICS ARE MISLEADING
   - Currently measuring "Price vs Realized Outcome"
   - This confuses ESTIMATION quality with single-outcome LUCK
   - A price of 0.60 SHOULD lose 40% of the time!

2. RIGHT METRICS TO USE
   - Price vs Latent Probability (estimation error)
   - Brier score across many runs (calibration)
   - Calibration curves (reliability diagrams)

3. WHY PRICES BARELY MOVE
   - Liquidity parameter b=100 is TOO HIGH
   - Trade threshold 0.02 is CONSERVATIVE
   - Agent beliefs cluster near 0.5 (weak signals)

4. DISCOVERY IS NOT SCARCE
   - Agents discover 100% of information
   - No genuine Grossman-Stiglitz tradeoff
   - Need: budget limits, time limits, diminishing returns

5. MISSING BASELINES
   - No comparison to mean belief
   - No comparison to pooled posterior
   - Can't tell if market is adding value
""")

    print("\n" + "="*70)
    print("VISUALIZATIONS COMPLETE")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  1. {dashboard_path}")
    print(f"  2. {mechanism_path}")
    print()


if __name__ == "__main__":
    main()
