from typing import Dict, List, Optional
import numpy as np
import pandas as pd


class PortfolioRiskEngine:
    def __init__(
        self,
        daily_returns: pd.DataFrame,
        weights: Dict[str, float],
        confidence_level: float = 0.95
    ):
        self.returns = daily_returns
        self.weights_dict = weights
        self.confidence_level = confidence_level
        self.assets = list(daily_returns.columns)

        # align weight array with dataframe column order
        weight_list = []
        for col in self.assets:
            w = self.weights_dict.get(col, 0.0)
            weight_list.append(w)
        self.weights = np.array(weight_list)

        # compute time-series of total portfolio daily returns
        self.portfolio_daily_returns = self.returns.dot(self.weights)

    def calculate_historical_var(self) -> float:
        # historical quantile method
        percentile = (1.0 - self.confidence_level) * 100.0
        var_value = np.percentile(self.portfolio_daily_returns, percentile)
        # return positive percentage representation of loss
        loss_var = -float(var_value)
        return round(max(0.0, loss_var), 4)

    def calculate_parametric_var(self) -> float:
        # parametric gaussian method based on normal distribution quantile
        mean_ret = float(self.portfolio_daily_returns.mean())
        std_ret = float(self.portfolio_daily_returns.std())

        # z-scores for common confidence levels
        if abs(self.confidence_level - 0.99) < 0.005:
            z_score = 2.326
        elif abs(self.confidence_level - 0.90) < 0.005:
            z_score = 1.282
        else:
            # default 95%
            z_score = 1.645

        loss_var = -(mean_ret - z_score * std_ret)
        return round(max(0.0, loss_var), 4)

    def calculate_cvar(self) -> float:
        # conditional VaR / Expected Shortfall
        percentile = (1.0 - self.confidence_level) * 100.0
        cutoff_threshold = np.percentile(self.portfolio_daily_returns, percentile)
        
        tail_losses = self.portfolio_daily_returns[self.portfolio_daily_returns <= cutoff_threshold]
        if len(tail_losses) > 0:
            expected_shortfall = -float(tail_losses.mean())
        else:
            expected_shortfall = -float(cutoff_threshold)

        return round(max(0.0, expected_shortfall), 4)

    def evaluate_concentration_limits(self, max_allowed_weight: float = 0.35) -> Dict[str, object]:
        violating_assets = {}
        for asset, weight_value in self.weights_dict.items():
            if weight_value > max_allowed_weight:
                violating_assets[asset] = round(weight_value, 4)

        has_violation = len(violating_assets) > 0
        status_message = "Concentration risk within limits"
        if has_violation:
            status_message = f"Assets exceed max weight threshold of {max_allowed_weight * 100}%"

        result = {
            "max_allowed_weight": max_allowed_weight,
            "has_violation": has_violation,
            "violating_assets": violating_assets,
            "message": status_message
        }
        return result

    def run_crisis_stress_test(self) -> Dict[str, object]:
        # stylized macro crisis drawdown shocks based on broad equity factor sensitivities
        # scenario definition: approximate peak-to-trough market equity index drop
        scenarios = {
            "2008_Global_Financial_Crisis": -0.50,
            "2020_COVID_Liquidity_Shock": -0.34,
            "2022_Inflation_Rate_Hike_Cycle": -0.22,
            "Mild_Correction_Scenario": -0.10
        }

        # calculate portfolio beta proxy relative to average market variance
        annual_vol = float(self.portfolio_daily_returns.std() * np.sqrt(252))
        market_benchmark_vol = 0.18  # baseline S&P long-run volatility benchmark
        
        if market_benchmark_vol > 0:
            portfolio_beta_proxy = annual_vol / market_benchmark_vol
        else:
            portfolio_beta_proxy = 1.0

        simulated_results = {}
        for scenario_name, market_drop in scenarios.items():
            # estimated portfolio impact scaled by volatility factor
            estimated_impact = market_drop * portfolio_beta_proxy
            # cap maximum theoretical loss at 100%
            constrained_impact = max(-1.0, estimated_impact)
            simulated_results[scenario_name] = round(constrained_impact, 4)

        return {
            "portfolio_beta_proxy": round(portfolio_beta_proxy, 4),
            "simulated_scenario_drawdowns": simulated_results
        }

    def generate_comprehensive_risk_report(self, max_allowed_weight: float = 0.35) -> Dict[str, object]:
        hist_var = self.calculate_historical_var()
        param_var = self.calculate_parametric_var()
        cvar = self.calculate_cvar()
        concentration = self.evaluate_concentration_limits(max_allowed_weight)
        stress_tests = self.run_crisis_stress_test()

        report = {
            "confidence_level": self.confidence_level,
            "daily_value_at_risk": {
                "historical_var_1d": hist_var,
                "parametric_var_1d": param_var
            },
            "daily_conditional_var_1d": cvar,
            "concentration_audit": concentration,
            "stress_testing": stress_tests
        }
        return report