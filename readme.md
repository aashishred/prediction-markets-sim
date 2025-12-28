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
- [ ] Order book market mechanism
- [ ] Discoverer agent
- [ ] Discoverable environment
- [ ] Knightian environment
- [ ] Full experiment suite (H1-H5)
- [ ] Statistical analysis pipeline
- [ ] Visualisation dashboard

### Experiment 1 Results (Preliminary)

Initial results from the aggregation experiment show:
- Markets struggle to beat the prior (information ratio ≈ 0)
- More agents doesn't dramatically improve accuracy
- Signal precision has limited effect on final prices

**Interpretation**: This may support the hypothesis that pure aggregation has limited value. However, further investigation is needed to understand agent trading behaviour and ensure signals are being used effectively.

---

## Future Planned Developments

1. **Phase 1**: Core infrastructure (markets, agents, environments)
2. **Phase 2**: Basic experiments testing H1-H5
3. **Phase 3**: Extended experiments with richer agent heterogeneity
4. **Phase 4**: Calibration to real-world data (Polymarket, Metaculus)
5. **Phase 5**: Interactive exploration tool for presentation

---

## Related Blog Series

The theoretical background is developed in detail in the accompanying blog posts:

- **§1**: Introduction — What are prediction markets and why might they work?
- **§2**: Common Objections — Liquidity, noise traders, manipulation, reflexivity
- **§3**: The Actually Good Objection — Aggregation vs discovery, Coasean constraints
- **§4**: Implications — Where prediction markets create value (macro) vs fail (micro)

See the `Wrting/` folder for the full series.
