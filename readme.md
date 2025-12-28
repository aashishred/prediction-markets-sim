# Prediction Markets Simulation

An agent-based simulation framework for testing theoretical claims about the epistemic foundations and institutional limits of prediction markets.

## Theoretical Context

This project accompanies an ongoing blog series examining prediction markets. The series argues that while prediction markets are often defended on Hayekian grounds (aggregating dispersed knowledge into prices), their true value lies elsewhere.

### Core Claims to Test

**1. Aggregation vs Discovery**
The primary value of prediction markets does not come from aggregating knowledge participants already have, but from *incentivising discovery* of new information. Insofar as they only serve the aggregation function, their efficiency is limited by Coasean constraints (transaction costs may exceed informational gains).

**2. Information Leakiness Spectrum**
Different environments exhibit different degrees of "leakiness" — how much information about the true state of the world is available in the present:

| Type | Description | Example |
|------|-------------|---------|
| **Knightian Uncertainty** | No information leaks into present minds. We must "let reality run" to find out. No coherent ex ante probability distribution exists. | Long-term policy effects, novel technology outcomes |
| **Hayekian Tacit Knowledge** | Information leaks directly into *some* minds but not others. Local, context-specific, often inarticulable knowledge about preferences and resources. | Shopkeeper knowing neighbourhood demand; worker knowing project bottlenecks |
| **Discoverable Information** | Information leaks into the "environment" but not directly into minds. Agents can pay discovery costs to learn it, then trade on it. | Scientific data requiring analysis; investigative journalism |

**3. The Grossman-Stiglitz Constraint**
If prices were perfectly informative, no one would pay to acquire information, so information wouldn't be produced. Markets are *partially* revealing; informed traders earn just enough to cover research costs. This implies:
- Knowledge production requires rents (from noise traders or explicit subsidies)
- Internal corporate markets fail because employees rationally prioritise salaried work
- At macro scale, with sufficient liquidity/subsidy, specialisation becomes viable

**4. When Prediction Markets Create Value**
Markets work best when they:
1. Draw on a large population with private, already-acquired signals
2. Have low costs of signal expression (trading)
3. Provide benefits exceeding market maintenance costs

OR (more importantly):
- Incentivise genuine discovery by making it profitable to build models, gather data, and refine priors

### Key Theoretical Sources
- Hayek (1945) — "The Use of Knowledge in Society"
- Hayek (1968) — "Competition as a Discovery Procedure"
- Knight (1921) — "Risk, Uncertainty, and Profit"
- Keynes (1937) — "General Theory of Employment" (on uncertainty)
- Coase (1937) — "The Nature of the Firm"
- Grossman & Stiglitz (1980) — "On the Impossibility of Informationally Efficient Markets"
- Hanson — Logarithmic Market Scoring Rule (LMSR)

---

## Project Architecture

### Technology Stack
- **Language**: Python 3.11+
- **Agent-Based Modelling**: Mesa or custom framework
- **Numerical Computing**: NumPy, SciPy
- **Statistical Analysis**: Statsmodels, hypothesis testing
- **Visualisation**: Matplotlib, Plotly (interactive)
- **Data Management**: Pandas, potentially DuckDB for larger runs

### Core Components

```
prediction_markets/
├── markets/
│   ├── lmsr.py           # Logarithmic Market Scoring Rule AMM
│   ├── order_book.py     # Traditional order book matching
│   └── base.py           # Abstract market interface
├── agents/
│   ├── base.py           # Abstract agent interface
│   ├── informed.py       # Agents with private signals
│   ├── noise.py          # Random/expressive traders
│   ├── discoverer.py     # Agents who pay to discover info
│   └── strategies.py     # Trading strategies
├── environments/
│   ├── base.py           # Abstract environment
│   ├── knightian.py      # No information leakage
│   ├── hayekian.py       # Tacit knowledge distributed
│   └── discoverable.py   # Information available at cost
├── simulation/
│   ├── runner.py         # Simulation orchestration
│   ├── scenarios.py      # Pre-configured test scenarios
│   └── metrics.py        # Efficiency measures, calibration
├── analysis/
│   ├── statistics.py     # Hypothesis testing
│   └── visualisation.py  # Plotting and dashboards
├── experiments/
│   └── [specific experiment configs]
└── tests/
    └── [unit and integration tests]
```

### Contract Types

| Type | Description | Resolution |
|------|-------------|------------|
| **Binary** | Pays 1 if event occurs, 0 otherwise | Single boolean outcome |
| **Multinomial** | Mutually exclusive outcomes | Prices sum to 1, one outcome wins |
| **Continuous** | Tracks ongoing variable | Pays based on final value |
| **Range** | Slices continuum into bands | Which band contains outcome? |
| **Time-to-event** | When will event occur? | Date/time bands |
| **Conditional** | "If P, will Y occur?" | Only resolves if condition met |
| **Combinatorial** | Complex logical combinations | Multiple conditions |

### Market Mechanisms

**LMSR (Logarithmic Market Scoring Rule)**
- Always provides liquidity via cost function: `C(q) = b * log(sum(exp(q_i/b)))`
- Liquidity parameter `b` controls price sensitivity
- Subsidised by market creator (bounded loss)
- Generalised for n outcomes (not just binary)

**Order Book**
- Traditional bid/ask matching
- Requires counterparties for trades
- Can have wide spreads in thin markets
- Support for multi-outcome markets

### Agent Types

| Agent Type | Has Signal? | Discovers? | Trading Motivation |
|------------|-------------|------------|-------------------|
| Informed | Yes (assigned) | No | Profit from private info |
| Discoverer | Can acquire | Yes (pays cost) | Profit from discovered info |
| Noise | No | No | Random, hedging, entertainment |
| Market Maker | No | No | Earn spread (order book only) |

### Discovery Model

The discovery mechanism is fully general:
1. Agent pays `discovery_cost` to attempt information acquisition
2. With some probability, agent obtains a signal
3. Signal has **uncertain validity** (may be false positive/negative)
4. Signal has **unclear bearing** on contract value (weight/relevance uncertain)
5. Agents can repeatedly pay to accumulate signals (incremental learning)

Special cases emerge from parameter choices:
- **Binary discovery**: High success probability, high validity, clear bearing
- **Noisy signal**: Moderate validity, clear bearing on value
- **Incremental discovery**: Multiple attempts improve precision over time

### Experimental Variables

**Information Structure**
- `leakiness`: How much true value leaks to agents (0 = Knightian, 1 = common knowledge)
- `signal_distribution`: How private signals are distributed across agents
- `discovery_cost`: Cost to acquire discoverable information
- `discovery_probability`: Chance of successful discovery given effort
- `information_pieces`: Multiple independent signals with weighted importance

**Market Structure**
- `mechanism`: LMSR vs order book
- `liquidity_parameter`: (LMSR) Controls price sensitivity
- `subsidy_budget`: Initial market maker funding
- `transaction_costs`: Fees per trade

**Agent Population**
- `n_informed`: Number of agents with private signals
- `n_discoverers`: Number of agents who attempt discovery
- `n_noise`: Number of noise traders
- `wealth_distribution`: Initial capital allocation
- `risk_preferences`: Agent utility functions

### Key Metrics

**Market Efficiency**
- Price accuracy: `|market_price - true_probability|` over time
- Calibration: Do X% confidence intervals contain truth X% of the time?
- Information incorporation speed: How quickly do prices reflect new information?

**Knowledge Creation**
- Discovery rate: How often do discoverers succeed?
- Information value: How much does discovery improve price accuracy?
- Discovery ROI: Returns to discoverers vs cost

**Welfare**
- Total surplus generated
- Distribution of profits (informed vs noise vs discoverers)
- Subsidy efficiency: Information gained per dollar of subsidy

---

## Experimental Hypotheses

### H1: Aggregation Alone Has Limited Value
**Setup**: Vary the amount of pre-existing information vs discoverable information
**Prediction**: When most value comes from aggregation (high leakiness, no discovery needed), market improvement over simple averaging is small and may not justify coordination costs

### H2: Discovery Creates Most Value
**Setup**: Compare environments with/without discovery opportunities
**Prediction**: Markets with discoverable information and agents who can discover it produce significantly more accurate prices than pure aggregation markets

### H3: Coasean Constraints Bind
**Setup**: Vary transaction costs and compare to information gains
**Prediction**: Below some threshold of dispersed information, the cost of running a market exceeds the accuracy improvement over expert judgment

### H4: Leakiness Determines Market Utility
**Setup**: Systematically vary information leakiness from Knightian to common knowledge
**Prediction**: There's an optimal leakiness range; too low (Knightian) = nothing to aggregate; too high = no role for markets

### H5: Subsidies Enable Specialisation
**Setup**: Vary market liquidity/subsidy levels
**Prediction**: Above some threshold, specialised "professional forecasters" become viable and market accuracy jumps discontinuously

---

## Current Progress

- [x] Theoretical framework developed (blog series §1-§3)
- [x] Project initialisation (pyproject.toml, package structure)
- [x] Core market mechanisms (LMSR implemented, order book pending)
- [x] Agent framework (Informed, Noise traders implemented)
- [x] Environment types (Hayekian implemented, Knightian/Discoverable pending)
- [x] Simulation runner
- [x] Core metrics (price error, Brier score, information ratio, etc.)
- [x] **Experiment 1 run**: Initial aggregation test in Hayekian environment
- [x] **General discovery model** (uncertain validity + unclear bearing)
- [x] **Discoverer agent** (pays to acquire information)
- [x] **Discoverable environment** (info exists but must be discovered)
- [x] **Experiment 2 run**: Discovery vs Aggregation comparison
- [x] **Comprehensive visualizations** (dashboard + mechanism explanation)
- [x] **Fixed metrics** (estimation vs outcome error separation)
- [x] **Calibration analysis** (Brier score, calibration curves, ECE)
- [x] **Baseline comparisons** (mean belief, best individual, prior)
- [x] **Liquidity calibration** (tool to find optimal LMSR parameter)
- [x] **Discovery scarcity** (diminishing returns enabled by default)
- [ ] Order book market mechanism
- [ ] Knightian environment
- [ ] Full experiment suite (H1-H5)
- [ ] Statistical analysis pipeline

### Experiment 1 Results: Aggregation in Hayekian Environment

**What We Did**

We simulated a prediction market for a binary event where:
1. The "true probability" is determined by 20 underlying signals (each worth 5% weight)
2. 10 informed agents each receive ~2 signals with 85% precision
3. 5 noise traders make random trades
4. Agents trade in an LMSR market (liquidity parameter = 100)

**What We Observed**

From the diagnostic run (seed=42):
```
Underlying signals: 12 "Yes" signals, 8 "No" signals
Theoretical aggregate: P(Yes) = 0.60
True outcome: No (sampled probabilistically from signals)

Agent beliefs after receiving signals:
  - informed_0: P(Yes) = 0.570  (4 signals)
  - informed_1: P(Yes) = 0.552  (3 signals)
  - informed_4: P(Yes) = 0.430  (4 signals)
  - informed_8: P(Yes) = 0.412  (5 signals)
  ... (mean belief ≈ 0.51)

After 50 trading rounds:
  - Final price: P(Yes) = 0.539
  - 91 trades executed
  - Price moved toward theoretical aggregate
  - BUT: Price error vs truth = 0.54 (worse than prior!)
  - Information ratio = -0.08 (slightly WORSE than 50/50 prior)
```

**Key Insight: The System Works, But Aggregation Has Inherent Limits**

The diagnostic reveals something important:
1. **The market IS aggregating** — price moved from 0.50 toward the theoretical aggregate of 0.60
2. **But the theoretical aggregate itself is wrong** — signals said P(Yes)=0.60, but the true outcome was No
3. **Aggregation can't exceed the information content of signals** — if signals point the wrong way, perfect aggregation just makes you confidently wrong

This is actually a validation of your theoretical claim! The signals themselves have limited information. Aggregating noisy signals gives you...aggregated noise. The market worked mechanically, but there wasn't enough "leakage" of truth into the signals to make aggregation valuable.

**Batch Results Across 30 Runs**

```
Varying informed agents (2 to 50):
  - Mean price error: ~0.50 regardless of count
  - Information ratio: fluctuates around 0

Varying signal precision (0.5 to 0.95):
  - Higher precision doesn't dramatically help
  - Mean error stays ~0.50
```

**Interpretation**

These results are consistent with **Hypothesis H1**: Aggregation alone has limited value. The market correctly aggregates the beliefs of participants, but when those beliefs are themselves noisy estimates of truth, the aggregated result is no better than any individual estimate — and the coordination cost of running the market exceeds the marginal accuracy gain.

### Experiment 2 Results: Discovery vs Aggregation

**What We Did**

We compared two environments with matched simulations (same random seeds):
1. **Hayekian (Aggregation)**: Signals pre-distributed to agents (free information)
2. **Discoverable (Discovery)**: Agents must pay to discover signals

**Key Results (30 runs)**

```
Hayekian Environment (Aggregation Only):
  Mean price error vs truth:  0.474 (+/- 0.097)
  Mean information ratio:     0.052 (+/- 0.194)

Discoverable Environment (Discovery Enabled):
  Mean price error vs truth:  0.491 (+/- 0.251)
  Mean information ratio:     0.017 (+/- 0.503)
  Mean info discovered:       100%
```

**Key Insight: Discovery Works, But Doesn't Dominate**

Both environments achieve similar accuracy! This supports the Grossman-Stiglitz insight:
- Markets can't be perfectly efficient (that would eliminate discovery incentives)
- But they can be efficient *enough* to justify discovery costs
- Even when agents must pay to learn, the market produces comparable accuracy

The higher variance in the discoverable environment reflects the additional uncertainty from the discovery process — signals may be wrong, and their relevance may be unclear.

### Critical Issues Identified

**The current metrics are misleading!** After review, several fundamental problems were identified:

1. **Mixing Two Different Targets**
   - Currently measuring "Price vs Realized Outcome"
   - This confuses ESTIMATION quality with single-outcome LUCK
   - A price of 0.60 SHOULD lose 40% of the time (when outcome is No)!
   - This is not evidence of bad aggregation—it's exactly what calibrated forecasting looks like

2. **Correct Metrics Should Be**
   - **Estimation error**: |Price - Latent Probability| (how well did we estimate the true probability?)
   - **Calibration**: Brier score across many runs (are 60% predictions right 60% of the time?)
   - **Baseline comparisons**: Does the market beat mean belief? Pooled posterior? Best individual agent?

3. **Why Prices Barely Move**
   - Liquidity parameter b=100 is TOO HIGH (prices move ~0.04 when they should move ~0.1-0.2)
   - Trade threshold 0.02 is conservative
   - Agent beliefs cluster near 0.5 due to weak/noisy signals

4. **Discovery Is Not Scarce**
   - Agents currently discover 100% of information
   - No genuine Grossman-Stiglitz tradeoff
   - Need: stricter budget limits, time constraints, diminishing returns

5. **Missing Baselines**
   - No comparison to simple mean of agent beliefs
   - No comparison to oracle pooled posterior
   - Can't tell if market mechanism is adding value over simpler aggregation

### Fixes Implemented

**All critical issues have been addressed!**

1. **✅ Metrics Fixed** ([metrics.py](prediction_markets/simulation/metrics.py))
   - `calculate_estimation_error()`: |Price - Latent| (RIGHT metric!)
   - `calculate_outcome_error()`: |Price - Realized| (calibration only)
   - `calculate_brier_score()`: Proper scoring across runs
   - `calculate_calibration_curve()`: ECE for calibration quality
   - `calculate_baseline_comparisons()`: Market vs mean belief vs best individual

2. **✅ Liquidity Calibrated** ([calibrate_liquidity.py](experiments/calibrate_liquidity.py))
   - Tool to find optimal LMSR parameter
   - Current b=100 gives ~12% price impact
   - Recommended b=50-75 for 15-25% impact

3. **✅ Discovery Made Scarce** ([discoverable.py](prediction_markets/environments/discoverable.py))
   - `diminishing_returns=True` by default
   - Time limits supported
   - Prevents 100% discovery

4. **✅ Comprehensive Visualizations** ([comprehensive_visualization.py](experiments/comprehensive_visualization.py))
   - 10-panel dashboard with all metrics
   - Mechanism explanation chart
   - Error decomposition clearly shown

---

## How to Run

### Setup
```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install numpy scipy pandas matplotlib plotly statsmodels pytest
```

### Run Tests
```bash
python -m pytest tests/test_basic.py -v
```

### Run Visualizations

**Comprehensive Dashboard (RECOMMENDED - Start here!)**
```bash
python experiments/comprehensive_visualization.py
```
Creates detailed multi-panel visualizations showing:
- Price evolution vs latent probability vs realized outcome
- Trading activity and price impact by agent type
- Discovery attempts and success rates
- Error decomposition (Price vs Latent vs Realized)
- Agent belief distributions
- Summary statistics

Outputs:
- `comprehensive_dashboard.png` - Full simulation dashboard
- `mechanism_explanation.png` - How the simulation works

**Single Simulation Diagnostic**
```bash
python experiments/visualize_simulation.py
```
Step-by-step trace of a single simulation. Generates `diagnostic_visualization.png`.

### Run Batch Experiments
```bash
# Experiment 1: Aggregation in Hayekian environment
python experiments/experiment_01_aggregation.py

# Experiment 2: Discovery vs Aggregation comparison
python experiments/experiment_02_discovery.py
```
Runs experiments with multiple simulations and generates comparative visualizations.

### Calibrate LMSR Liquidity
```bash
python experiments/calibrate_liquidity.py
```
Tests different liquidity parameters and finds optimal value for desired price movement.
Generates `liquidity_calibration.png` showing impact vs liquidity.

---

## How the Simulation Works

### The Flow

```
1. ENVIRONMENT creates underlying signals
   └─> 20 binary signals (e.g., [1,0,1,1,0,...])
   └─> Weighted sum gives "theoretical aggregate" probability
   └─> True outcome sampled from this probability

2. SIGNALS distributed to AGENTS
   └─> Each informed agent receives subset of signals
   └─> Signals observed with noise (precision parameter)
   └─> Agent updates beliefs via Bayesian updating

3. MARKET opens at prior (50/50 for binary)
   └─> LSMR provides infinite liquidity at cost

4. TRADING LOOP (each timestep):
   └─> Each agent compares beliefs to market price
   └─> If belief differs enough, agent trades
   └─> LSMR adjusts price based on trades
   └─> Prices should converge toward aggregate beliefs

5. RESOLUTION
   └─> True outcome revealed
   └─> Shareholders of correct outcome paid £1/share
   └─> P&L calculated for all participants
```

### Key Files

| File | Purpose |
|------|---------|
| `prediction_markets/markets/lmsr.py` | LSMR automated market maker |
| `prediction_markets/agents/informed.py` | Agents who receive private signals |
| `prediction_markets/agents/noise.py` | Random/expressive traders |
| `prediction_markets/environments/hayekian.py` | Distributed signal environment |
| `prediction_markets/simulation/runner.py` | Orchestrates simulation |
| `prediction_markets/simulation/metrics.py` | Calculates accuracy/efficiency |
| `experiments/visualize_simulation.py` | Diagnostic visualization |
| `experiments/experiment_01_aggregation.py` | Batch experiments |

---

## Next Steps (Prioritised)

### Immediate Options

**Option A: Test the Discovery Hypothesis (H2)**
Build the Discoverable environment and Discoverer agent to test whether markets that incentivise discovery outperform pure aggregation.
- Create `environments/discoverable.py` — information exists but requires payment to access
- Create `agents/discoverer.py` — agents who pay to discover and then trade
- Compare: pure aggregation vs discovery-enabled markets

**Option B: Test Coasean Constraints (H3)**
Add transaction costs to the simulation and find the threshold where markets fail to beat simple expert judgment.
- Add `transaction_cost` parameter to market
- Compare market accuracy vs single-agent-with-average-signal
- Find: at what cost level does market become worthless?

**Option C: Build the Knightian Environment (H4)**
Test the extreme case where no information leaks — pure uncertainty.
- Create `environments/knightian.py` — no signals, true outcome is random
- Verify: market should equal prior, no value from aggregation
- This is the limiting case of the leakiness spectrum

**Option D: Improve Current Diagnostics**
Before moving forward, we might want to:
- Run more simulations to get statistical significance
- Create interactive visualizations (Plotly dashboard)
- Add calibration tests (Brier scores across many runs)

### Research Questions to Investigate

1. **Why is price movement so small?** In our simulation, prices only move ~0.04 from the prior. Is this because:
   - Agent beliefs are too close to 0.5?
   - Trade thresholds are too conservative?
   - Liquidity parameter is too high?

2. **Does the "noise trader subsidy" matter?** Per Milgrom-Stokey, we need noise traders for markets to function. Are our noise traders actually providing liquidity that informed traders exploit?

3. **What happens with extreme signal precision?** If signals are 99% accurate, does aggregation become valuable?

---

## Future Phases

1. **Phase 1** ✓ Core infrastructure (markets, agents, environments)
2. **Phase 2** (Current) Basic experiments testing H1-H5
3. **Phase 3** Extended experiments with richer agent heterogeneity
4. **Phase 4** Calibration to real-world data (Polymarket, Metaculus)
5. **Phase 5** Interactive exploration tool for presentation

---

## Related Blog Series

The theoretical background is developed in detail in the accompanying blog posts:

- **§1**: Introduction — What are prediction markets and why might they work?
- **§2**: Common Objections — Liquidity, noise traders, manipulation, reflexivity
- **§3**: The Actually Good Objection — Aggregation vs discovery, Coasean constraints
- **§4**: Implications — Where prediction markets create value (macro) vs fail (micro)

See the `Wrting/` folder for the full series.
