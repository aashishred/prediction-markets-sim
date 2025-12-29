# Prediction Markets Simulation

An agent-based simulation framework for testing theoretical claims about prediction markets, specifically testing whether their value comes from aggregating existing knowledge or incentivizing discovery of new information.

> **For detailed development history, design decisions, and implementation rationale**, see [history.md](history.md)

---

## Core Theoretical Claims

This project tests five key hypotheses about prediction markets:

**H1: Aggregation Alone Has Limited Value**
- Markets that only aggregate existing beliefs face Coasean constraints
- Transaction costs may exceed informational gains when information is already distributed

**H2: Discovery Creates Most Value**
- Markets that incentivize discovery of new information create more value than pure aggregation
- This requires discoverers to profit enough to cover research costs (Grossman-Stiglitz)

**H3: Coasean Constraints Bind**
- Below some threshold of dispersed information, markets fail to beat expert judgment
- The cost of coordination exceeds the accuracy improvement

**H4: Leakiness Determines Market Utility**
- Information "leakiness" ranges from Knightian uncertainty (no info) to full disclosure
- Markets perform best at medium leakiness (enough to discover, not so much it's trivial)

**H5: Subsidies Enable Specialization**
- Above some liquidity threshold, professional forecasters become viable
- Market accuracy jumps discontinuously as specialists enter

### Information Leakiness Spectrum

| Type | Information Status | Market Role |
|------|-------------------|-------------|
| **Knightian** | No information available | Limited value (nothing to aggregate/discover) |
| **Hayekian** | Distributed tacit knowledge | Aggregation valuable |
| **Discoverable** | Information exists but must be found | Discovery incentives valuable |

---

## Architecture

### Core Components

```
prediction_markets/
├── markets/          # LMSR, order book (future)
├── agents/           # Informed, Discoverer, Noise traders
├── environments/     # Hayekian, Discoverable, Unified
├── simulation/       # Runner and metrics
└── analysis/         # Statistics and visualization (future)

experiments/          # H1-H5 experimental tests
tests/               # Unit and integration tests
```

### Key Innovation: Continuous Leakiness Parameter

**UnifiedEnvironment** replaces discrete environment types with continuous spectrum:

- **Leakiness ∈ [0, 1]**: Controls what fraction of total information "leaks" into present
- **Free signals**: Distributed to agents at no cost (Hayekian-style)
- **Discoverable signals**: Require costly discovery
- **Hidden signals**: Never available (Knightian uncertainty)

```python
# Low leakiness (0.2) → Knightian-like
# Medium leakiness (0.5) → Hayekian-like (25% free, 25% discoverable)
# High leakiness (0.9) → Discoverable-like (5% free, 85% discoverable)
```

This enables smooth testing across the information spectrum rather than three discrete regimes.

### Quick Usage Example

```python
from prediction_markets import (
    LMSRMarket, BinaryContract,
    InformedAgent, NoiseTrader,
    UnifiedEnvironment, UnifiedConfig,
    Simulation, SimulationConfig,
)

# Create a market
contract = BinaryContract(name="Will X happen?")
market = LMSRMarket(contract=contract, liquidity=50)

# Create environment with 50% leakiness (Hayekian-like)
config = UnifiedConfig(leakiness=0.5, n_signals=20, random_seed=42)
env = UnifiedEnvironment(config=config)

# Create agents
agents = [
    InformedAgent(agent_id="informed_1", signal_precision=0.8),
    NoiseTrader(agent_id="noise_1", trade_probability=0.3),
]

# Run simulation
sim = Simulation(
    environment=env,
    market=market,
    agents=agents,
    config=SimulationConfig(n_steps=50)
)
result = sim.run()
print(f"Final price: {result.final_prices}")
```

---

## Current Status

### Completed ✅

**Core Infrastructure**
- [x] LMSR market mechanism with calibrated liquidity parameter
- [x] Agent framework (InformedAgent, NoiseTrader, DiscovererAgent)
- [x] Hayekian environment (distributed tacit knowledge)
- [x] Discoverable environment (costly discovery)
- [x] **Unified environment** with continuous leakiness parameter
- [x] Simulation runner with configurable parameters
- [x] General discovery model (uncertain validity + unclear bearing)
- [x] **Clean module exports** with top-level imports (v0.3.0)

**Metrics & Analysis**
- [x] **Fixed metric confusion** (estimation error vs outcome error properly separated)
- [x] **Baseline comparisons** (mean belief, best individual, prior)
- [x] **Calibration metrics** (Brier score, calibration curves, ECE)
- [x] **Liquidity calibration tool** (find optimal LMSR parameter)
- [x] **Comprehensive visualizations** (10-panel dashboard + mechanism explanation)

**Testing & Quality**
- [x] Unit tests for markets, contracts, agents (12 tests)
- [x] Unit tests for metrics (16 tests)
- [x] Unit tests for environments (15 tests)
- [x] **43 tests passing** with pytest

**Experiments**
- [x] Experiment 1: Aggregation in Hayekian environment
- [x] Experiment 2: Discovery vs Aggregation comparison
- [x] Leakiness spectrum test (verifies UnifiedEnvironment)

### In Progress 🔨

- [ ] Complete H4 experiment (market performance across leakiness spectrum)
- [ ] Dynamic real-time visualization system

### Pending 📋

**Infrastructure (GPT-5 Feedback)**
- [ ] Refactor signal generation (θ drawn first, signals as evidence not votes)
- [ ] Continuous noise trader flow (small orders each tick)
- [ ] Time horizons and capital constraints for discoverers
- [ ] Order book implementation (price-time priority matching)
- [ ] Market maker agent (Avellaneda-Stoikov or naive spread)

**Complete Experiment Suite**
- [ ] H1: Aggregation value test
- [ ] H2: Discovery value with ROI metrics
- [ ] H3: Coasean constraints with transaction costs
- [ ] H4: Leakiness spectrum (started)
- [ ] H5: Subsidy phase transitions

**Advanced Metrics & Statistics**
- [ ] Knowledge production metrics (entropy reduction, counterfactual discovery value)
- [ ] Paired simulation designs for causal attribution
- [ ] Bootstrap CIs, permutation tests, pre-registered endpoints
- [ ] Heavy-tail robust statistics (median + MAD)

**Realism Upgrades**
- [ ] True Knightian environment (regime shifts, model uncertainty)
- [ ] Grossman-Stiglitz equilibrium dynamics (tight incentive tradeoffs)

---

## Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# On Windows: venv\Scripts\activate

# Install dependencies
pip install numpy scipy pandas matplotlib plotly statsmodels pytest

# Run tests to verify installation
python -m pytest tests/ -v
```

### Run Visualizations

**Comprehensive Dashboard (RECOMMENDED)**
```bash
python experiments/comprehensive_visualization.py
```
Generates detailed multi-panel visualizations showing price evolution, trading activity, discovery attempts, error decomposition, and agent beliefs.

**Test Unified Environment**
```bash
python experiments/test_unified_environment.py
```
Demonstrates how information structure changes across the leakiness spectrum (0.0 to 1.0).

**Calibrate LMSR Liquidity**
```bash
python experiments/calibrate_liquidity.py
```
Finds optimal liquidity parameter for desired price movement. Currently recommends b=50-75 for 15-25% impact.

### Run Experiments

```bash
# Aggregation test (Hayekian environment)
python experiments/experiment_01_aggregation.py

# Discovery vs Aggregation comparison
python experiments/experiment_02_discovery.py

# Leakiness spectrum (in progress)
python experiments/experiment_leakiness_spectrum.py
```

---

## Key Results So Far

### Experiment 1: Pure Aggregation (Hayekian)
- Market correctly aggregates distributed beliefs
- BUT: aggregated noisy signals = aggregated noise
- **Validates H1**: Aggregation alone has limited value when signal quality is poor

### Experiment 2: Discovery vs Aggregation
- Discovery-enabled markets achieve similar accuracy to pure aggregation
- Higher variance reflects uncertainty from discovery process
- **Supports Grossman-Stiglitz**: Markets efficient enough to justify discovery costs, but not perfectly efficient

### Metrics Fix (Critical)
- Original metrics confused estimation quality with single-outcome luck
- A price of 60% SHOULD lose 40% of the time!
- Now properly measure:
  - **Estimation error**: |Price - Latent Probability|
  - **Calibration**: Brier score across runs
  - **Value-add**: Comparison to mean belief, best individual, prior

### Unified Environment
- Smooth transition from Knightian (leakiness=0.0) to full disclosure (leakiness=1.0)
- Information partitions cleanly into free/discoverable/hidden
- Discovery cost scales inversely with leakiness (high leakiness = cheaper discovery)

---

## Files to Note

| File | Purpose |
|------|---------|
| [readme.md](readme.md) | **This file** - Project overview and current status |
| [history.md](history.md) | Complete development log with design decisions and rationale |
| [GPT-5_Feedback.md](GPT-5_Feedback.md) | External review with improvement suggestions |
| [calibrate_liquidity.py](experiments/calibrate_liquidity.py) | Find optimal LMSR parameter empirically |
| [comprehensive_visualization.py](experiments/comprehensive_visualization.py) | 10-panel dashboard |
| [test_unified_environment.py](experiments/test_unified_environment.py) | Verify leakiness spectrum |
| [unified.py](prediction_markets/environments/unified.py) | Continuous leakiness environment |
| [metrics.py](prediction_markets/simulation/metrics.py) | Fixed estimation vs outcome errors |

---

## Theoretical Background

### Key Sources
- **Hayek (1945)** — "The Use of Knowledge in Society" (distributed tacit knowledge)
- **Hayek (1968)** — "Competition as a Discovery Procedure" (markets discover, not just aggregate)
- **Knight (1921)** — "Risk, Uncertainty, and Profit" (Knightian uncertainty)
- **Grossman & Stiglitz (1980)** — "Impossibility of Informationally Efficient Markets" (discovery requires rents)
- **Coase (1937)** — "The Nature of the Firm" (transaction costs)
- **Hanson** — Logarithmic Market Scoring Rule (LMSR) for prediction markets

### Blog Series
Detailed theoretical development in accompanying blog posts:
- §1: Introduction to prediction markets
- §2: Common objections (liquidity, manipulation, etc.)
- §3: The serious objection (aggregation vs discovery, Coasean constraints)
- §4: Implications (where markets create value, where they fail)

---

## Contributing

This is a research project testing specific theoretical hypotheses. The roadmap is driven by the experimental agenda (H1-H5) and the need for rigorous statistical testing.

**Current priorities** (from GPT-5 feedback):
1. Complete H1-H5 experiment suite with clean designs
2. Implement order book for mechanism comparison
3. Refactor signal generation model (θ first, signals as evidence)
4. Add statistical rigor (bootstrap CIs, permutation tests, paired designs)
5. Create real-time dynamic visualizations

---

## License

MIT License - See LICENSE file for details

---

## Contact

For questions about the theoretical framework or simulation design, see the blog series in `Writing/` or the development history in `history.md`.
