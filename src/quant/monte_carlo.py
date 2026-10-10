from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class MonteCarloSimulator:
    def __init__(
        self,
        daily_returns: pd.DataFrame,
        weights: Dict[str, float],
        initial_investment: float = 10000.0,
        trading_days: int = 252
    ):
        self.returns = daily_returns
        self.assets = list(daily_returns.columns)
        self.weights_dict = weights
        self.initial_investment = initial_investment
        self.trading_days = trading_days

        # vector of weights ordered by columns
        w_list = [self.weights_dict.get(asset, 0.0) for asset in self.assets]
        self.weights = np.array(w_list)

        # annual and daily parameters
        self.mean_daily_returns = self.returns.mean().values
        self.cov_matrix = self.returns.cov().values

    def run_simulation(
        self,
        time_horizon_years: int = 1,
        num_simulations: int = 1000,
        random_seed: int = 42
    ) -> Dict[str, object]:
        np.random.seed(random_seed)

        num_days = int(time_horizon_years * self.trading_days)
        num_assets = len(self.assets)

        # cholesky decomposition for correlated asset shocks
        # add minimal jitter to diagonal to avoid non-positive definite edge cases
        cov_stable = self.cov_matrix + np.eye(num_assets) * 1e-8
        cholesky_l = np.linalg.cholesky(cov_stable)

        # asset daily variance
        asset_variances = np.diag(cov_stable)
        drift = self.mean_daily_returns - 0.5 * asset_variances

        # simulate random shocks: shape = (num_simulations, num_days, num_assets)
        uncorrelated_shocks = np.random.normal(0, 1, size=(num_simulations, num_days, num_assets))

        # correlated shocks via L: shock * L.T
        correlated_shocks = np.einsum('ijk,lk->ijl', uncorrelated_shocks, cholesky_l)

        # calculate daily log returns: drift + shock
        daily_log_returns = drift + correlated_shocks

        # calculate simple returns: exp(log_ret) - 1
        daily_simple_returns = np.exp(daily_log_returns) - 1.0

        # portfolio return per day per simulation: dot product with weights
        # shape: (num_simulations, num_days)
        portfolio_daily_returns = np.dot(daily_simple_returns, self.weights)

        # wealth trajectory
        growth_factors = 1.0 + portfolio_daily_returns
        
        # prepend starting wealth of 1.0
        ones = np.ones((num_simulations, 1))
        all_factors = np.hstack([ones, growth_factors])
        wealth_trajectories = self.initial_investment * np.cumprod(all_factors, axis=1)

        final_portfolio_values = wealth_trajectories[:, -1]

        # calculate percentile outcomes
        percentiles = {
            "p5_worst_case": round(float(np.percentile(final_portfolio_values, 5)), 2),
            "p25_conservative": round(float(np.percentile(final_portfolio_values, 25)), 2),
            "p50_median": round(float(np.percentile(final_portfolio_values, 50)), 2),
            "p75_optimistic": round(float(np.percentile(final_portfolio_values, 75)), 2),
            "p95_best_case": round(float(np.percentile(final_portfolio_values, 95)), 2),
        }

        # probability of preserving initial capital
        capital_preserved_count = np.sum(final_portfolio_values >= self.initial_investment)
        prob_preservation = float(capital_preserved_count / num_simulations)

        # expected average outcome
        mean_final_value = float(np.mean(final_portfolio_values))
        expected_gain_pct = (mean_final_value - self.initial_investment) / self.initial_investment

        results = {
            "initial_investment": self.initial_investment,
            "horizon_years": time_horizon_years,
            "simulations_count": num_simulations,
            "percentiles": percentiles,
            "mean_terminal_wealth": round(mean_final_value, 2),
            "expected_gain_percentage": round(expected_gain_pct * 100, 2),
            "probability_of_profit": round(prob_preservation * 100, 2)
        }

        return results