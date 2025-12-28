"""
Metrics for evaluating prediction market performance.

These metrics help test the core hypotheses:
- How accurate are market prices compared to truth?
- How much does aggregation help vs simple averaging?
- How is value distributed among participants?
"""

from typing import Any
import numpy as np

from .runner import SimulationResult


def calculate_price_error(
    result: SimulationResult,
    at_step: int | None = None
) -> float:
    """
    Calculate the absolute error between market price and true probability.

    For binary contracts, this is |price - truth|.
    For multinomial, this is the L1 distance / 2.

    Args:
        result: Simulation result
        at_step: Step to evaluate (default: final)

    Returns:
        Price error (lower is better)
    """
    if at_step is None:
        prices = result.final_prices
    else:
        prices = result.price_history[at_step]

    # True probabilities
    n_outcomes = len(prices)
    true_probs = np.zeros(n_outcomes)

    if isinstance(result.true_value, (int, np.integer)):
        true_probs[result.true_value] = 1.0
    else:
        # For continuous, true_probs[0] = true_value
        true_probs[0] = result.true_value

    # L1 distance divided by 2 (max possible is 1)
    return np.sum(np.abs(prices - true_probs)) / 2


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


def summarize_batch(results: list[SimulationResult]) -> dict[str, Any]:
    """
    Summarize results across multiple simulation runs.

    Args:
        results: List of simulation results

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {}

    price_errors = [calculate_price_error(r) for r in results]
    info_ratios = [calculate_information_ratio(r) for r in results]
    agg_efficiencies = [calculate_aggregation_efficiency(r) for r in results]

    # Filter out NaN values
    agg_efficiencies_clean = [x for x in agg_efficiencies if not np.isnan(x)]

    # Brier score across all runs
    brier = calculate_brier_score(results)

    return {
        "n_runs": len(results),
        "mean_price_error": np.mean(price_errors),
        "std_price_error": np.std(price_errors),
        "mean_information_ratio": np.mean(info_ratios),
        "std_information_ratio": np.std(info_ratios),
        "mean_aggregation_efficiency": np.mean(agg_efficiencies_clean) if agg_efficiencies_clean else np.nan,
        "brier_score": brier,
        "all_price_errors": price_errors,
        "all_info_ratios": info_ratios,
    }
