"""
Metrics for evaluating prediction market performance.

CRITICAL DISTINCTION:
- ESTIMATION ERROR: |Price - Latent Probability| (how well did we estimate?)
- OUTCOME ERROR: |Price - Realized Outcome| (did we predict correctly?)

These are DIFFERENT targets! A price of 0.60 SHOULD lose 40% of the time.
Mixing them confuses calibration with single-outcome luck.

Correct metrics:
- Estimation error (vs latent probability)
- Brier score across runs (calibration)
- Baseline comparisons (mean belief, pooled posterior)
"""

from typing import Any
import numpy as np

from .runner import SimulationResult


def calculate_estimation_error(
    result: SimulationResult,
    at_step: int | None = None
) -> float:
    """
    Calculate ESTIMATION ERROR: |Price - Latent Probability|.

    This measures how well the market estimated the TRUE probability,
    NOT whether it predicted the realized outcome correctly.

    For a well-calibrated market:
    - Latent P = 0.60 → Price should be ~0.60
    - Outcome can still be 0 (happens 40% of the time!)

    Args:
        result: Simulation result
        at_step: Step to evaluate (default: final)

    Returns:
        Estimation error (lower is better)
    """
    if result.theoretical_aggregate is None:
        return np.nan

    if at_step is None:
        prices = result.final_prices
    else:
        prices = result.price_history[at_step]

    # Error vs LATENT probability (theoretical aggregate)
    return np.sum(np.abs(prices - result.theoretical_aggregate)) / 2


def calculate_outcome_error(
    result: SimulationResult,
    at_step: int | None = None
) -> float:
    """
    Calculate OUTCOME ERROR: |Price - Realized Outcome|.

    WARNING: This is NOT a good metric for single runs!
    A price of 0.60 SHOULD lose when outcome is 0.

    Use this metric ONLY:
    - Aggregated across many runs (Brier score)
    - For calibration analysis

    Args:
        result: Simulation result
        at_step: Step to evaluate (default: final)

    Returns:
        Outcome error (misleading for single runs!)
    """
    if at_step is None:
        prices = result.final_prices
    else:
        prices = result.price_history[at_step]

    # True REALIZED outcome (0 or 1)
    n_outcomes = len(prices)
    true_probs = np.zeros(n_outcomes)

    if isinstance(result.true_value, (int, np.integer)):
        true_probs[result.true_value] = 1.0
    else:
        true_probs[0] = result.true_value

    return np.sum(np.abs(prices - true_probs)) / 2


def calculate_price_error(
    result: SimulationResult,
    at_step: int | None = None
) -> float:
    """
    DEPRECATED: Use calculate_estimation_error() or calculate_outcome_error().

    This was ambiguous - use the explicit versions instead.
    For backward compatibility, this returns outcome error.
    """
    return calculate_outcome_error(result, at_step)


def calculate_price_error_vs_aggregate(result: SimulationResult) -> float:
    """
    Calculate error between market price and theoretical aggregate.

    This measures how well the market aggregates dispersed information,
    comparing to what perfect aggregation would achieve.

    Args:
        result: Simulation result

    Returns:
        Error vs theoretical aggregate (lower = better aggregation)
    """
    if result.theoretical_aggregate is None:
        return np.nan

    return np.sum(np.abs(result.final_prices - result.theoretical_aggregate)) / 2


def calculate_brier_score(
    results: list[SimulationResult],
    at_step: int | None = None
) -> float:
    """
    Calculate Brier score across multiple simulation runs.

    The Brier score measures calibration: if market says 70% for outcome X,
    outcome X should happen ~70% of the time across many runs.

    Brier = mean((probability - outcome)^2)

    Args:
        results: List of simulation results
        at_step: Step to evaluate (default: final)

    Returns:
        Brier score (lower is better, 0 = perfect)
    """
    if not results:
        return np.nan

    errors = []
    for result in results:
        if at_step is None:
            prices = result.final_prices
        else:
            prices = result.price_history[at_step]

        # For binary, Brier score is (p - outcome)^2 for outcome 1
        if len(prices) == 2:
            prob = prices[1]  # Probability of "Yes"
            outcome = float(result.true_value)
            errors.append((prob - outcome) ** 2)
        else:
            # Multinomial Brier: sum of (p_i - 1{i=outcome})^2
            true_idx = int(result.true_value)
            for i, p in enumerate(prices):
                target = 1.0 if i == true_idx else 0.0
                errors.append((p - target) ** 2)

    return np.mean(errors)


def calculate_information_ratio(result: SimulationResult) -> float:
    """
    Calculate how much information the market captures.

    Information ratio = 1 - (market_error / prior_error)

    A ratio of 1 means the market is perfectly informed.
    A ratio of 0 means the market is no better than the prior.
    Negative means the market is worse than the prior.

    Args:
        result: Simulation result

    Returns:
        Information ratio
    """
    market_error = calculate_price_error(result)

    # Prior error (uniform distribution)
    n_outcomes = len(result.final_prices)
    prior = np.ones(n_outcomes) / n_outcomes
    true_probs = np.zeros(n_outcomes)
    if isinstance(result.true_value, (int, np.integer)):
        true_probs[result.true_value] = 1.0
    else:
        true_probs[0] = result.true_value

    prior_error = np.sum(np.abs(prior - true_probs)) / 2

    if prior_error < 1e-10:
        return 1.0 if market_error < 1e-10 else 0.0

    return 1 - (market_error / prior_error)


def calculate_aggregation_efficiency(result: SimulationResult) -> float:
    """
    Calculate how efficiently the market aggregates vs simple average.

    Compares market accuracy to what would be achieved by simply
    averaging all agents' beliefs.

    Efficiency = 1 - (market_error / average_error)

    Args:
        result: Simulation result

    Returns:
        Aggregation efficiency (1 = market equals average, >1 = market beats average)
    """
    if result.theoretical_aggregate is None:
        return np.nan

    market_error = calculate_price_error(result)

    # Error of theoretical aggregate
    n_outcomes = len(result.final_prices)
    true_probs = np.zeros(n_outcomes)
    if isinstance(result.true_value, (int, np.integer)):
        true_probs[result.true_value] = 1.0
    else:
        true_probs[0] = result.true_value

    aggregate_error = np.sum(np.abs(result.theoretical_aggregate - true_probs)) / 2

    if aggregate_error < 1e-10:
        return 1.0 if market_error < 1e-10 else 0.0

    return 1 - (market_error / aggregate_error)


def calculate_convergence_speed(
    result: SimulationResult,
    threshold: float = 0.1
) -> int | None:
    """
    Calculate how many steps until price is within threshold of truth.

    Args:
        result: Simulation result
        threshold: Maximum error to consider "converged"

    Returns:
        Step number when converged, or None if never converged
    """
    for step in range(len(result.price_history)):
        error = calculate_price_error(result, at_step=step)
        if error < threshold:
            return step
    return None


def calculate_welfare_distribution(result: SimulationResult) -> dict[str, Any]:
    """
    Analyze how profits/losses are distributed among participants.

    Returns breakdown of:
    - Total profit/loss for informed vs noise traders
    - Subsidy efficiency (information gained per dollar of subsidy)
    - Gini coefficient of profit distribution

    Args:
        result: Simulation result

    Returns:
        Dictionary with welfare statistics
    """
    pnls = list(result.agent_pnl.values())

    if not pnls:
        return {"total_pnl": 0, "mean_pnl": 0, "std_pnl": 0}

    total_pnl = sum(pnls)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p < 0]

    # Gini coefficient
    pnls_sorted = np.sort(np.abs(pnls))
    n = len(pnls_sorted)
    if n == 0 or np.sum(pnls_sorted) == 0:
        gini = 0
    else:
        cumsum = np.cumsum(pnls_sorted)
        gini = (2 * np.sum((np.arange(1, n + 1) * pnls_sorted)) / (n * np.sum(pnls_sorted))) - (n + 1) / n

    # Subsidy efficiency: information gained per dollar
    if result.total_subsidy_spent > 0:
        info_ratio = calculate_information_ratio(result)
        subsidy_efficiency = info_ratio / result.total_subsidy_spent
    else:
        subsidy_efficiency = np.nan

    return {
        "total_pnl": total_pnl,
        "mean_pnl": np.mean(pnls),
        "std_pnl": np.std(pnls),
        "n_winners": len(winners),
        "n_losers": len(losers),
        "total_winner_profit": sum(winners) if winners else 0,
        "total_loser_loss": sum(losers) if losers else 0,
        "gini_coefficient": gini,
        "subsidy_spent": result.total_subsidy_spent,
        "subsidy_efficiency": subsidy_efficiency,
    }


def calculate_trading_activity(result: SimulationResult) -> dict[str, Any]:
    """
    Analyze trading activity patterns.

    Args:
        result: Simulation result

    Returns:
        Dictionary with trading statistics
    """
    trades = result.trades
    if not trades:
        return {"n_trades": 0}

    volumes = [abs(t.shares) for t in trades]
    costs = [abs(t.cost) for t in trades]

    # Activity by agent
    trades_per_agent = {}
    for aid in result.agent_trades:
        trades_per_agent[aid] = len(result.agent_trades[aid])

    return {
        "n_trades": len(trades),
        "total_volume": sum(volumes),
        "mean_trade_size": np.mean(volumes),
        "total_costs": sum(costs),
        "mean_cost": np.mean(costs),
        "trades_per_agent": trades_per_agent,
        "most_active_agent": max(trades_per_agent, key=trades_per_agent.get) if trades_per_agent else None,
    }


def calculate_baseline_comparisons(result: SimulationResult, agent_beliefs: list[np.ndarray] | None = None) -> dict[str, float]:
    """
    Compare market to baseline aggregation methods.

    Baselines:
    1. Mean belief: Simple average of all agent beliefs
    2. Best individual: Best single agent's belief
    3. Prior: Uniform distribution

    Args:
        result: Simulation result
        agent_beliefs: List of final agent beliefs (if available)

    Returns:
        Dictionary with errors for each baseline
    """
    if result.theoretical_aggregate is None:
        return {}

    latent = result.theoretical_aggregate
    market_price = result.final_prices

    # Market estimation error
    market_error = np.sum(np.abs(market_price - latent)) / 2

    baselines = {
        "market_error": market_error,
    }

    # Prior (uniform)
    n_outcomes = len(latent)
    prior = np.ones(n_outcomes) / n_outcomes
    baselines["prior_error"] = np.sum(np.abs(prior - latent)) / 2

    # Mean belief and best individual (if available)
    if agent_beliefs is not None and len(agent_beliefs) > 0:
        mean_belief = np.mean(agent_beliefs, axis=0)
        baselines["mean_belief_error"] = np.sum(np.abs(mean_belief - latent)) / 2

        # Best individual agent
        individual_errors = [np.sum(np.abs(belief - latent)) / 2 for belief in agent_beliefs]
        baselines["best_individual_error"] = min(individual_errors)
        baselines["worst_individual_error"] = max(individual_errors)
        baselines["median_individual_error"] = np.median(individual_errors)

    return baselines


def calculate_calibration_curve(results: list[SimulationResult], n_bins: int = 10) -> dict[str, Any]:
    """
    Calculate calibration curve: are X% predictions right X% of the time?

    Args:
        results: List of simulation results
        n_bins: Number of probability bins

    Returns:
        Dictionary with bin centers, empirical frequencies, and counts
    """
    if not results:
        return {}

    # Collect (predicted_prob, outcome) pairs
    predictions = []
    outcomes = []

    for result in results:
        if len(result.final_prices) == 2:  # Binary only for now
            prob = result.final_prices[1]  # P(Yes)
            outcome = float(result.true_value)
            predictions.append(prob)
            outcomes.append(outcome)

    if not predictions:
        return {}

    predictions = np.array(predictions)
    outcomes = np.array(outcomes)

    # Bin the predictions
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    empirical_freq = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (predictions >= bins[i]) & (predictions < bins[i + 1])
        if i == n_bins - 1:  # Include right endpoint
            mask = (predictions >= bins[i]) & (predictions <= bins[i + 1])

        counts[i] = np.sum(mask)
        if counts[i] > 0:
            empirical_freq[i] = np.mean(outcomes[mask])
        else:
            empirical_freq[i] = np.nan

    # Expected calibration error (ECE)
    valid_bins = ~np.isnan(empirical_freq)
    if np.any(valid_bins):
        ece = np.sum(counts[valid_bins] * np.abs(bin_centers[valid_bins] - empirical_freq[valid_bins])) / np.sum(counts)
    else:
        ece = np.nan

    return {
        "bin_centers": bin_centers,
        "empirical_frequencies": empirical_freq,
        "counts": counts,
        "expected_calibration_error": ece,
    }


def summarize_batch(results: list[SimulationResult]) -> dict[str, Any]:
    """
    Summarize results across multiple simulation runs.

    NEW: Properly separates estimation vs outcome errors.

    Args:
        results: List of simulation results

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    # ESTIMATION errors (vs latent probability)
    estimation_errors = [calculate_estimation_error(r) for r in results if not np.isnan(calculate_estimation_error(r))]

    # OUTCOME errors (vs realized outcome - only for Brier)
    outcome_errors = [calculate_outcome_error(r) for r in results]
    info_ratios = [calculate_information_ratio(r) for r in results]
    agg_efficiencies = [calculate_aggregation_efficiency(r) for r in results]

    # Filter out NaN values
    agg_efficiencies_clean = [x for x in agg_efficiencies if not np.isnan(x)]

    # Brier score across all runs (proper calibration metric)
    brier = calculate_brier_score(results)

    # Calibration curve
    calibration = calculate_calibration_curve(results)

    return {
        "n_runs": len(results),
        # ESTIMATION metrics (what we care about!)
        "mean_estimation_error": np.mean(estimation_errors) if estimation_errors else np.nan,
        "std_estimation_error": np.std(estimation_errors) if estimation_errors else np.nan,
        # OUTCOME metrics (for calibration only)
        "mean_outcome_error": np.mean(outcome_errors),
        "std_outcome_error": np.std(outcome_errors),
        "brier_score": brier,
        # Other metrics
        "mean_information_ratio": np.mean(info_ratios),
        "std_information_ratio": np.std(info_ratios),
        "mean_aggregation_efficiency": np.mean(agg_efficiencies_clean) if agg_efficiencies_clean else np.nan,
        # Calibration
        "calibration": calibration,
        # Raw data
        "all_estimation_errors": estimation_errors,
        "all_outcome_errors": outcome_errors,
        "all_info_ratios": info_ratios,
    }
