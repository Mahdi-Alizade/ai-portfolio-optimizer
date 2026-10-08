import numpy as np
import pandas as pd
import pytest
from src.quant.backtest_engine import PortfolioBacktestEngine


@pytest.fixture
def mock_backtest_data():
    dates = pd.date_range(start="2025-01-01", periods=252, freq="B")
    
    # constant positive growth for asset 1 (+0.1% daily)
    ret_pos = np.full(252, 0.001)
    # volatile asset 2
    np.random.seed(77)
    ret_vol = np.random.normal(0.0005, 0.015, 252)

    return pd.DataFrame({"STABLE": ret_pos, "VOLATILE": ret_vol}, index=dates)


def test_cumulative_return_and_cagr_positive(mock_backtest_data):
    # allocate 100% to positive asset
    weights = {"STABLE": 1.0, "VOLATILE": 0.0}
    engine = PortfolioBacktestEngine(daily_returns=mock_backtest_data, weights=weights, risk_free_rate=0.03)

    metrics = engine.generate_performance_metrics()
    assert metrics["cumulative_return"] > 0.0
    assert metrics["cagr_annualized_return"] > 0.0
    assert metrics["max_drawdown"] == 0.0  # purely increasing asset has 0 drawdown


def test_drawdown_is_always_non_positive(mock_backtest_data):
    weights = {"STABLE": 0.20, "VOLATILE": 0.80}
    engine = PortfolioBacktestEngine(daily_returns=mock_backtest_data, weights=weights, risk_free_rate=0.03)

    _, max_dd = engine.calculate_drawdown_series()
    # drawdown must be 0.0 or negative by financial definition
    assert max_dd <= 0.0


def test_sortino_ratio_calculation(mock_backtest_data):
    weights = {"STABLE": 0.50, "VOLATILE": 0.50}
    engine = PortfolioBacktestEngine(daily_returns=mock_backtest_data, weights=weights, risk_free_rate=0.02)

    sortino = engine.calculate_sortino_ratio()
    assert isinstance(sortino, float)