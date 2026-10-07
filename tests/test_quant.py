import numpy as np
import pandas as pd
import pytest
from src.quant.mpt_optimizer import ModernPortfolioOptimizer
from src.quant.black_litterman import BlackLittermanOptimizer


@pytest.fixture
def mock_returns():
    # simulate 252 trading days for 3 assets with known seed
    np.random.seed(42)
    dates = pd.date_range(start="2025-01-01", periods=252, freq="B")
    
    # asset A: moderate return, moderate risk
    ret_a = np.random.normal(0.0008, 0.012, 252)
    # asset B: high return, high risk
    ret_b = np.random.normal(0.0012, 0.020, 252)
    # asset C: lower risk defensive
    ret_c = np.random.normal(0.0004, 0.008, 252)

    df = pd.DataFrame(
        {
            "ASSET_A": ret_a,
            "ASSET_B": ret_b,
            "ASSET_C": ret_c
        },
        index=dates
    )
    return df


def test_mpt_max_sharpe_weights_sum_to_one(mock_returns):
    optimizer = ModernPortfolioOptimizer(daily_returns=mock_returns, risk_free_rate=0.03)
    result = optimizer.optimize_max_sharpe()

    assert result["success"] is True
    weights = result["weights"]
    total_weight = sum(weights.values())

    # tolerance check for float precision
    assert pytest.approx(total_weight, abs=1e-3) == 1.0

    # check no short selling
    for asset, w in weights.items():
        assert w >= 0.0
        assert w <= 1.0


def test_mpt_min_volatility_reduces_risk(mock_returns):
    optimizer = ModernPortfolioOptimizer(daily_returns=mock_returns, risk_free_rate=0.03)
    min_vol_res = optimizer.optimize_min_volatility()
    
    # equal weights volatility as benchmark
    equal_w = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    _, benchmark_vol, _ = optimizer.portfolio_performance(equal_w)

    optimized_vol = min_vol_res["volatility"]
    assert optimized_vol <= benchmark_vol


def test_black_litterman_with_bullish_view(mock_returns):
    bl = BlackLittermanOptimizer(daily_returns=mock_returns, risk_free_rate=0.03)

    # strong bullish view on ASSET_B
    views = {"ASSET_B": 0.25}
    confidence = {"ASSET_B": 0.85}

    result = bl.optimize_with_views(views_dict=views, confidence_dict=confidence)

    assert result["success"] is True
    weights = result["weights"]
    total_weight = sum(weights.values())

    assert pytest.approx(total_weight, abs=1e-3) == 1.0
    assert "posterior_returns" in result
    assert result["posterior_returns"]["ASSET_B"] is not None