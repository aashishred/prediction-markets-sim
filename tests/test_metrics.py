"""Tests for simulation metrics."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prediction_markets.simulation.metrics import (
    calculate_estimation_error,
    calculate_outcome_error,
    calculate_brier_score,
    calculate_information_ratio,
    calculate_baseline_comparisons,
    calculate_calibration_curve,
    summarize_batch,
)
from prediction_markets.simulation.runner import SimulationResult, SimulationConfig


def create_mock_result(
    final_prices: np.ndarray,
    true_value: int,
    theoretical_aggregate: np.ndarray | None = None,
    price_history: np.ndarray | None = None,
) -> SimulationResult:
    """Create a mock SimulationResult for testing."""
    n_outcomes = len(final_prices)
    if price_history is None:
        price_history = np.array([final_prices])

    return SimulationResult(
        config=SimulationConfig(),
        n_steps=len(price_history),
        true_value=true_value,
        price_history=price_history,
        quantity_history=np.zeros_like(price_history),
        trades=[],
        final_prices=final_prices,
        agent_final_wealth={},
        agent_pnl={},
        agent_trades={},
        payouts={},
        total_subsidy_spent=0.0,
        theoretical_aggregate=theoretical_aggregate,
        prior=np.ones(n_outcomes) / n_outcomes,
    )


class TestEstimationError:
    """Test estimation error calculation (vs latent probability)."""

    def test_perfect_estimation(self):
        """When price equals latent probability, error should be 0."""
        result = create_mock_result(
            final_prices=np.array([0.3, 0.7]),
            true_value=1,
            theoretical_aggregate=np.array([0.3, 0.7]),
        )
        error = calculate_estimation_error(result)
        assert np.isclose(error, 0.0)

    def test_imperfect_estimation(self):
        """Error should measure distance from latent probability."""
        result = create_mock_result(
            final_prices=np.array([0.5, 0.5]),
            true_value=1,
            theoretical_aggregate=np.array([0.3, 0.7]),
        )
        # |0.5 - 0.3| + |0.5 - 0.7| = 0.2 + 0.2 = 0.4, divided by 2 = 0.2
        error = calculate_estimation_error(result)
        assert np.isclose(error, 0.2)

    def test_missing_aggregate_returns_nan(self):
        """When no theoretical aggregate, should return NaN."""
        result = create_mock_result(
            final_prices=np.array([0.5, 0.5]),
            true_value=1,
            theoretical_aggregate=None,
        )
        error = calculate_estimation_error(result)
        assert np.isnan(error)


class TestOutcomeError:
    """Test outcome error calculation (vs realized outcome)."""

    def test_correct_prediction(self):
        """When price is 1.0 for winning outcome, error should be 0."""
        result = create_mock_result(
            final_prices=np.array([0.0, 1.0]),
            true_value=1,
        )
        error = calculate_outcome_error(result)
        assert np.isclose(error, 0.0)

    def test_wrong_prediction(self):
        """When price is 0.0 for winning outcome, error should be 0.5."""
        result = create_mock_result(
            final_prices=np.array([1.0, 0.0]),
            true_value=1,
        )
        # |1.0 - 0| + |0.0 - 1| = 1 + 1 = 2, divided by 2 = 1.0
        error = calculate_outcome_error(result)
        assert np.isclose(error, 1.0)

    def test_uncertain_prediction(self):
        """Test with uncertain 50/50 prediction."""
        result = create_mock_result(
            final_prices=np.array([0.5, 0.5]),
            true_value=1,
        )
        # |0.5 - 0| + |0.5 - 1| = 0.5 + 0.5 = 1.0, divided by 2 = 0.5
        error = calculate_outcome_error(result)
        assert np.isclose(error, 0.5)


class TestBrierScore:
    """Test Brier score calculation across runs."""

    def test_perfect_calibration(self):
        """Perfect predictions should have Brier score of 0."""
        results = [
            create_mock_result(np.array([0.0, 1.0]), true_value=1),
            create_mock_result(np.array([1.0, 0.0]), true_value=0),
        ]
        brier = calculate_brier_score(results)
        assert np.isclose(brier, 0.0)

    def test_uncertain_predictions(self):
        """50/50 predictions should have Brier score of 0.25."""
        results = [
            create_mock_result(np.array([0.5, 0.5]), true_value=1),
            create_mock_result(np.array([0.5, 0.5]), true_value=0),
        ]
        brier = calculate_brier_score(results)
        # (0.5 - 1)^2 + (0.5 - 0)^2 = 0.25 + 0.25 = 0.5 / 2 = 0.25
        assert np.isclose(brier, 0.25)

    def test_empty_results(self):
        """Empty results should return NaN."""
        brier = calculate_brier_score([])
        assert np.isnan(brier)


class TestInformationRatio:
    """Test information ratio calculation."""

    def test_perfect_prediction(self):
        """Perfect prediction should have info ratio of 1."""
        result = create_mock_result(
            final_prices=np.array([0.0, 1.0]),
            true_value=1,
        )
        ratio = calculate_information_ratio(result)
        assert np.isclose(ratio, 1.0)

    def test_prior_prediction(self):
        """Uniform prediction should have info ratio of 0."""
        result = create_mock_result(
            final_prices=np.array([0.5, 0.5]),
            true_value=1,
        )
        ratio = calculate_information_ratio(result)
        assert np.isclose(ratio, 0.0)


class TestBaselineComparisons:
    """Test baseline comparison calculations."""

    def test_with_theoretical_aggregate(self):
        """Should return market error vs baselines."""
        result = create_mock_result(
            final_prices=np.array([0.4, 0.6]),
            true_value=1,
            theoretical_aggregate=np.array([0.3, 0.7]),
        )
        baselines = calculate_baseline_comparisons(result)

        assert "market_error" in baselines
        assert "prior_error" in baselines
        # Market error: |0.4-0.3| + |0.6-0.7| = 0.2, /2 = 0.1
        assert np.isclose(baselines["market_error"], 0.1)

    def test_without_theoretical_aggregate(self):
        """Should return empty dict when no aggregate."""
        result = create_mock_result(
            final_prices=np.array([0.4, 0.6]),
            true_value=1,
            theoretical_aggregate=None,
        )
        baselines = calculate_baseline_comparisons(result)
        assert baselines == {}


class TestCalibrationCurve:
    """Test calibration curve calculation."""

    def test_empty_results(self):
        """Empty results should return empty dict."""
        curve = calculate_calibration_curve([])
        assert curve == {}

    def test_with_binary_results(self):
        """Should compute calibration bins for binary results."""
        results = [
            create_mock_result(np.array([0.3, 0.7]), true_value=1),
            create_mock_result(np.array([0.3, 0.7]), true_value=1),
            create_mock_result(np.array([0.3, 0.7]), true_value=0),
        ]
        curve = calculate_calibration_curve(results, n_bins=5)

        assert "bin_centers" in curve
        assert "empirical_frequencies" in curve
        assert "counts" in curve
        assert "expected_calibration_error" in curve


class TestSummarizeBatch:
    """Test batch summary statistics."""

    def test_empty_batch(self):
        """Empty batch should return empty dict."""
        summary = summarize_batch([])
        assert summary == {}

    def test_with_results(self):
        """Should compute summary statistics."""
        results = [
            create_mock_result(
                np.array([0.4, 0.6]),
                true_value=1,
                theoretical_aggregate=np.array([0.3, 0.7]),
            ),
            create_mock_result(
                np.array([0.35, 0.65]),
                true_value=0,
                theoretical_aggregate=np.array([0.4, 0.6]),
            ),
        ]
        summary = summarize_batch(results)

        assert "n_runs" in summary
        assert summary["n_runs"] == 2
        assert "mean_estimation_error" in summary
        assert "brier_score" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
