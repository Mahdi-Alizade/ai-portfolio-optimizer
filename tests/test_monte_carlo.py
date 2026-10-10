import numpy as np
import pandas as pd
import pytest
from src.quant.monte_carlo import MonteCarloSimulator


@pytest.fixture
def mock_sim_data():
    dates = pd.date_range(start="2025-01-01", periods=252, freq="B")
    np.random.seed(99)
    ret_1 = np.random.normal(0.0006, 0.012, 252)
    ret_2 = np.random.normal(0.0004, 0.010, 252)
    return pd.DataFrame({"STK1": ret_1, "STK2": ret_2}, index=dates)


def test_monte_carlo_percentile_monotonicity(mock_sim_data):
    weights = {"STK1": 0.5, "STK2": 0.5}
    simulator = MonteCarloSimulator(daily_returns=mock_sim_data, weights=weights, initial_investment=10000.0)

    res = simulator.run_simulation(time_horizon_years=1, num_simulations=500, random_seed=42)
    p = res["percentiles"]

    # percentiles must be strictly monotonic
    assert p["p5_worst_case"] <= p["p25_conservative"]
    assert p["p25_conservative"] <= p["p50_median"]
    assert p["p50_median"] <= p["p75_optimistic"]
    assert p["p75_optimistic"] <= p["p95_best_case"]


def test_monte_carlo_probability_range(mock_sim_data):
    weights = {"STK1": 0.6, "STK2": 0.4}
    simulator = MonteCarloSimulator(daily_returns=mock_sim_data, weights=weights, initial_investment=10000.0)

    res = simulator.run_simulation(time_horizon_years=1, num_simulations=200)

    prob = res["probability_of_profit"]
    assert 0.0 <= prob <= 100.0
    assert res["mean_terminal_wealth"] > 0.0