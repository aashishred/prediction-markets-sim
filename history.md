# Prediction Markets Simulation - Development History

This file tracks all significant changes, decisions, and rationales throughout the project development.

---

## Phase 1: Initial Implementation (Completed)

### Core Infrastructure
- **Markets**: Implemented LMSR (Logarithmic Market Scoring Rule) with generalized cost function
- **Agents**: Created InformedAgent, NoiseTrader, and DiscovererAgent with Kelly criterion position sizing
- **Environments**: Built Hayekian (distributed knowledge) and Discoverable (costly discovery) environments
- **Simulation**: Developed runner with configurable timesteps, agent ordering, and data collection

### Key Design Decisions
1. **Signal-based Information Model**: Information represented as discrete signals with weights, allowing flexible distribution
2. **Bayesian Belief Updating**: Agents use precision-weighted updates to incorporate signals
3. **Kelly Criterion Trading**: Position sizing based on edge and current wealth to prevent bankruptcy
4. **Subsidized Market Maker**: LMSR provides guaranteed liquidity through subsidy

---

## Phase 2: Metrics and Calibration Fixes (Completed)

### Problem Identified (GPT-5.2 Feedback)
**Critical confusion between estimation error and outcome error**

- **Estimation Error**: |Price - Latent Probability| (measures how well we estimated true probability)
- **Outcome Error**: |Price - Realized Outcome| (measures prediction accuracy on single draw)

These are fundamentally different! A market saying 60% should lose 40% of the time.

### Changes Made

#### 1. Metrics Refactor ([metrics.py](prediction_markets/simulation/metrics.py:23-53))
```python
def calculate_estimation_error(result, at_step=None):
    """
    Calculate ESTIMATION ERROR: |Price - Latent Probability|.

    This is what we care about! Did the market estimate the true
    probability correctly?
    """
    return np.sum(np.abs(prices - result.theoretical_aggregate)) / 2

def calculate_outcome_error(result, at_step=None):
    """
    Calculate OUTCOME ERROR: |Price - Realized Outcome|.

    WARNING: Only use for Brier scores across runs!
    Single-run outcome error is misleading.
    """
    # Returns |price - realized_outcome|
```

**Rationale**: Separating these prevents confusing calibration with luck. We now properly measure:
- Estimation quality (vs latent)
- Calibration (Brier score across runs)
- Baseline comparisons

#### 2. Baseline Comparisons ([metrics.py](prediction_markets/simulation/metrics.py:294-339))
Added comparisons to prove market value-add:
- **Mean belief**: Simple average of all agent beliefs
- **Best individual**: Best single agent
- **Prior**: Uniform distribution

**Rationale**: Markets must beat naive aggregation to be useful. This quantifies value-add.

#### 3. Calibration Analysis ([metrics.py](prediction_markets/simulation/metrics.py:342-402))
Implemented proper calibration metrics:
- **Brier score**: Mean squared error across runs
- **Calibration curves**: Are X% predictions right X% of the time?
- **Expected Calibration Error (ECE)**: Weighted average miscalibration

**Rationale**: Calibration is about long-run frequency matching, not single outcomes.

#### 4. Liquidity Calibration ([calibrate_liquidity.py](experiments/calibrate_liquidity.py))
**Problem**: With b=100, prices barely moved (0.04 change with informed trades)

**Solution**: Created empirical calibration tool testing b ∈ [5, 200]

**Results**:
- b=100: ~12% price impact per typical trade
- b=50-75: ~15-25% price impact (recommended)
- b=20: ~35% price impact (too volatile)

**Decision**: Use b=50 as default for visible price discovery

**Bug Found & Fixed**: Agents weren't trading when edges were exactly equal
- Root cause: `if best_edge > abs(worst_edge)` failed when edges equal
- Fix: Manually set asymmetric beliefs for calibration test

#### 5. Discovery Scarcity ([discoverable.py](prediction_markets/environments/discoverable.py:56))
**Problem**: 100% discovery made discovery look trivial

**Solution**:
- Set `diminishing_returns=True` by default (later discoveries less valuable)
- Added `time_limit` parameter to restrict discovery window

**Rationale**: Real discovery faces diminishing returns and time constraints

#### 6. Comprehensive Visualizations ([comprehensive_visualization.py](experiments/comprehensive_visualization.py))
Created 10-panel dashboard showing:
1. **Price Evolution**: Market price vs latent probability vs realized outcome
2. **Trading Activity**: Volume and price impact by agent type
3. **Discovery Attempts**: Success rates and timing
4. **Error Decomposition**: Estimation error vs outcome error over time
5. **Agent Beliefs**: Distribution of final beliefs
6. **Mechanism Explanation**: 4-panel guide to simulation mechanics

**Rationale**: Needed to understand what's actually happening in simulations

---

## Phase 3: Continuous Leakiness Spectrum (In Progress)

### Motivation (GPT-5 Feedback Point #1)
Current system has three discrete environment types (Knightian, Hayekian, Discoverable).
Better: unified model with continuous leakiness parameter.

### Design: UnifiedEnvironment

**Leakiness Parameter** (0.0 to 1.0):
- Controls what fraction of total information "leaks" into the present
- Of leaked information:
  - `free_fraction = (1 - leakiness)` → distributed to agent minds (Hayekian)
  - `discovery_fraction = leakiness` → requires costly discovery

**Information Flow**:
```
Total Signals (20)
    ↓
leakiness parameter determines how many leak
    ↓
Leaked Signals split into:
    - Free signals (in agent minds, no cost)
    - Discoverable signals (cost to acquire)
    - Hidden signals (never available)
```

**Regime Mapping**:
- **Knightian** (leakiness < 0.3): Little information available, mostly free
- **Hayekian** (0.3 ≤ leakiness < 0.7): Mix of free and discoverable
- **Discoverable** (leakiness ≥ 0.7): Most info requires discovery

**Discovery Cost Scaling**:
```python
cost_multiplier = 2.0 - leakiness  # High leakiness = cheaper discovery
cost = base_cost * cost_multiplier
```

### Implementation ([unified.py](prediction_markets/environments/unified.py))

Key methods:
- `_partition_signals_by_leakiness()`: Splits signals into free/discoverable/hidden
- `generate_signals()`: Distributes only free signals
- `get_discovery_opportunities()`: Returns discoverable signals with costs
- `theoretical_aggregate()`: Oracle aggregation of all leaked signals (not hidden)

**Backward Compatibility**: Old environment classes can wrap UnifiedEnvironment with fixed leakiness values

### Testing ([test_unified_environment.py](experiments/test_unified_environment.py))
Verified across spectrum (0.0 to 1.0):
- At leakiness=0.0: 0% leaked, 100% hidden
- At leakiness=0.5: 50% leaked (25% free, 25% discoverable), 50% hidden
- At leakiness=1.0: 100% leaked (0% free, 100% discoverable), 0% hidden

**Result**: Clean transition from aggregation-dominated to discovery-dominated regimes

### Rationale
1. **Testable Hypothesis**: Can now plot market performance as smooth curve across leakiness
2. **Theoretical Clarity**: "Leakiness" is now a measurable parameter, not just a label
3. **H4 Experiment**: Can test if markets excel at medium leakiness (0.3-0.7)

---

## Next Steps (From GPT-5 Feedback)

### Immediate
1. **Split Documentation**: Separate readme.md (overview) from history.md (this file)
2. **Finish H4 Experiment**: Complete leakiness spectrum experiment
3. **Dynamic Visualization**: Real-time convergence display

### Major Refactors Needed

#### Signal Generation Model (GPT-5 Point #2)
**Problem**: Current model has "signals vote to make θ"
- θ = weighted average of signals
- This makes aggregation conceptually privileged

**Solution**: Invert the model
- θ drawn first from prior (e.g., Beta(2,2))
- Signals generated from θ with controlled noise
- Signals become evidence about θ, not constituents of θ

**Benefits**:
- Cleaner epistemic interpretation
- Discovery and aggregation on equal footing
- Easier to parameterize information content

#### Grossman-Stiglitz Equilibrium (GPT-5 Point #3)
**Problem**: Discovery doesn't create tight incentive tradeoffs yet

**Needed**:
1. Continuous noise trader flow (small orders each tick)
2. Competition among discoverers (price moves reduce others' profits)
3. Time horizons and capital constraints
4. Stochastic, sometimes misleading discovery
5. Gradual information selling (repeated trades)

**Expected Outcome**: "Knife-edge" region where discovery just barely profitable

#### Order Book Implementation (GPT-5 Point #4)
**Must Have**:
- Limit orders (price, quantity)
- Market orders (quantity)
- Price-time priority matching
- Optional market maker agent
- Two modes: Pure CLOB vs CLOB+MM

**Rationale**: Test if discovery story is mechanism-robust or LMSR artifact

#### Clean Experiment Suite (GPT-5 Point #5)
Map each hypothesis to single experiment:
- **H1**: Aggregation value (vary free info, discovery off)
- **H2**: Discovery value (fixed low free info, vary discovery cost/strength)
- **H3**: Coasean constraints (add transaction costs, compare to expert)
- **H4**: Leakiness spectrum (already building)
- **H5**: Subsidy phase transitions (vary liquidity, look for entry thresholds)

#### Knowledge Production Metrics (GPT-5 Point #6)
Add operational measures:
1. **Information gain**: Entropy reduction about θ
2. **Counterfactual discovery value**: Paired runs (discovery on vs off)

#### Statistical Rigor (GPT-5 Point #7)
Implement:
- Bootstrap confidence intervals
- Permutation tests for mechanism comparisons
- Paired designs (same seed, different mechanism)
- Pre-registered primary endpoints
- Heavy-tail robust statistics (median + MAD)

#### True Knightian Uncertainty (GPT-5 Point #8)
Make Knightian environment actually Knightian:
- Regime shifts (data-generating process changes mid-run)
- Model uncertainty (agents have wrong hypothesis class)
- Test if markets overfit stale patterns

---

## Design Rationale Summary

### Why Agent-Based Simulation?
- **Testability**: Can isolate mechanisms (aggregation vs discovery)
- **Causal Attribution**: Paired designs with same seed
- **Mechanism Comparison**: LMSR vs order book vs baselines
- **Emergent Behavior**: Grossman-Stiglitz equilibria arise endogenously

### Why These Metrics?
- **Estimation Error**: Tests hypothesis about probability estimation
- **Brier Score**: Tests calibration across uncertainty
- **Baseline Comparisons**: Proves market value-add isn't trivial
- **Discovery ROI**: Tests whether discovery incentives work

### Why This Information Model?
- **Signals**: Modular, interpretable, parameterizable
- **Leakiness**: Unified framework from Knightian to full disclosure
- **Weights**: Allows importance heterogeneity
- **Precision**: Models noisy observation

### Why LMSR First?
- **Guaranteed Liquidity**: Isolates aggregation/discovery from liquidity provision
- **Measurable Subsidy**: Can vary liquidity parameter cleanly
- **Analytic Tractability**: Cost function has known properties
- **Realistic**: Used by real prediction markets (Metaculus, etc.)

---

## Open Questions

1. **Optimal signal generation model**: Signals as votes vs signals as evidence?
2. **Discovery dynamics**: One-shot vs gradual information selling?
3. **Market maker specification**: Avellaneda-Stoikov vs naive spread?
4. **Knightian operationalization**: Regime shifts vs model misspecification?
5. **Calibration targets**: What Brier score / ECE is "good"?

---

## Lessons Learned

### Metric Confusion is Deadly
Mixing estimation error with outcome error led to weeks of ambiguous results.
**Lesson**: Be pedantically precise about what you're measuring.

### Liquidity Matters More Than Expected
Price impact scales inversely with liquidity. Small changes in `b` dramatically affect behavior.
**Lesson**: Calibrate market parameters empirically, not just theoretically.

### Discovery Needs Real Costs
If discovery is trivial (100% success, no scarcity), it doesn't test the hypothesis.
**Lesson**: Diminishing returns and time limits are essential.

### Visualization is Not Optional
Couldn't understand what was happening without comprehensive visualizations.
**Lesson**: Build visualization tools early, not as an afterthought.

### Equal Edges Break Agent Logic
When `best_edge == abs(worst_edge)`, agents didn't trade due to tie-breaking logic.
**Lesson**: Test edge cases (literally) in agent decision rules.

---

## File Organization

### Core Library (`prediction_markets/`)
- `markets/`: LMSR and (future) order book implementations
- `agents/`: InformedAgent, NoiseTrader, DiscovererAgent
- `environments/`: Hayekian, Discoverable, Unified
- `simulation/`: Runner and metrics

### Experiments (`experiments/`)
- `calibrate_liquidity.py`: Find optimal LMSR parameter
- `comprehensive_visualization.py`: Multi-panel simulation analysis
- `test_unified_environment.py`: Verify leakiness spectrum
- `experiment_leakiness_spectrum.py`: H4 test (in progress)

### Documentation
- `readme.md`: Project overview, hypotheses, current status
- `history.md`: This file - complete development log
- `GPT-5_Feedback.md`: External review and improvement suggestions

---

## Version History

### v0.1 - Initial Implementation
- Basic LMSR market
- Hayekian and Discoverable environments
- InformedAgent and NoiseTrader

### v0.2 - Metrics Fix
- Separated estimation vs outcome error
- Added baseline comparisons
- Implemented calibration metrics
- Fixed liquidity parameter

### v0.3 - Continuous Leakiness (Current)
- Unified environment with leakiness parameter
- Smooth transition across information regimes
- Backward compatible with old environments

### v0.4 - Planned (GPT-5 Feedback)
- Signal generation refactor (θ first, signals as evidence)
- Order book implementation
- Grossman-Stiglitz equilibrium dynamics
- Complete H1-H5 experiment suite
- Statistical rigor upgrades
