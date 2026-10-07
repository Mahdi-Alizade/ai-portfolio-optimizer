import numpy as np
import pandas as pd
import pytest
from src.quant.risk_engine import PortfolioRiskEngine


@pytest.fixture
def mock_asset_returns():
    np.random.seed(101)
    dates = pd.date_range(start="2025-01-01", periods=252, freq="B")
    
    # generate return series with negative skew/fat tails
    ret_1 = np.random.normal(0.0005, 0.015, 252)
    ret_2 = np.random.normal(0.0007, 0.022, 252)
    
    return pd.DataFrame({"SYM_1": ret_1, "SYM_2": ret_2}, index=dates)


def test_var_and_cvar_mathematical_consistency(mock_asset_returns):
    weights = {"SYM_1": 0.60, "SYM_2": 0.40}
    engine = PortfolioRiskEngine(daily_returns=mock_asset_returns, weights=weights, confidence_level=0.95)
    
    hist_var = engine.calculate_historical_var()
    cvar = engine.calculate_cvar()

    assert hist_var >= 0.0
    assert cvar >= 0.0
    # Expected Shortfall (CVaR) must always be greater than or equal to VaR by definition
    assert cvar >= hist_var


def test_concentration_limit_violation_detection(mock_asset_returns):
    # allocate 80% to SYM_1
    heavy_weights = {"SYM_1": 0.80, "SYM_2": 0.20}
    engine = PortfolioRiskEngine(daily_returns=mock_asset_returns, weights=heavy_weights)

    audit_result = engine.evaluate_concentration_limits(max_allowed_weight=0.50)
    assert audit_result["has_violation"] is True
    assert "SYM_1" in audit_result["violating_assets"]
    assert audit_result["violating_assets"]["SYM_1"] == 0.80


def test_stress_test_crisis_drawdown_generation(mock_asset_returns):
    weights = {"SYM_1": 0.50, "SYM_2": 0.50}
    engine = PortfolioRiskEngine(daily_returns=mock_asset_returns, weights=weights)

    stress_report = engine.run_crisis_stress_test()
    drawdowns = stress_report["simulated_scenario_drawdowns"]

    assert "2008_Global_Financial_Crisis" in drawdowns
    assert "2020_COVID_Liquidity_Shock" in drawdowns
    # drawdowns must represent negative values (loss)
    assert drawdowns["2008_Global_Financial_Crisis"] < 0.0