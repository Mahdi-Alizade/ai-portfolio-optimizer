from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class ModernPortfolioOptimizer:
    def __init__(self, daily_returns: pd.DataFrame, risk_free_rate: float = 0.04):
        self.returns = daily_returns
        self.assets = list(daily_returns.columns)
        self.num_assets = len(self.assets)
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252

        # calculate annualized expected returns and covariance matrix
        mean_daily = self.returns.mean()
        self.expected_returns = mean_daily * self.trading_days

        daily_cov = self.returns.cov()
        self.cov_matrix = daily_cov * self.trading_days

    def portfolio_performance(self, weights: np.ndarray) -> Tuple[float, float, float]:
        # calculate annualized return
        ret = np.dot(weights, self.expected_returns)

        # calculate annualized volatility (variance then square root)
        variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        volatility = np.sqrt(variance)

        # calculate annualized sharpe ratio
        excess_return = ret - self.risk_free_rate
        if volatility > 0:
            sharpe_ratio = excess_return / volatility
        else:
            sharpe_ratio = 0.0

        return ret, volatility, sharpe_ratio

    def _negative_sharpe(self, weights: np.ndarray) -> float:
        ret, vol, sharpe = self.portfolio_performance(weights)
        # we minimize the negative to maximize actual sharpe
        return -sharpe

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        ret, vol, sharpe = self.portfolio_performance(weights)
        return vol

    def optimize_max_sharpe(self) -> Dict[str, object]:
        # initial guess: equal weight allocation across assets
        equal_weight = 1.0 / self.num_assets
        initial_guess = np.array([equal_weight] * self.num_assets)

        # constraint: sum of weights equals 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        # bounds: no short-selling (weights must be between 0 and 1)
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))

        result = minimize(
            fun=self._negative_sharpe,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(optimal_weights)

        # format allocation dictionary with rounded percentages
        weights_dict = {}
        for index in range(self.num_assets):
            symbol = self.assets[index]
            weight_val = float(optimal_weights[index])
            weights_dict[symbol] = round(weight_val, 4)

        output_data = {
            "strategy": "Maximum Sharpe Ratio",
            "expected_return": round(float(ret), 4),
            "volatility": round(float(vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "weights": weights_dict,
            "success": bool(result.success)
        }

        return output_data

    def optimize_min_volatility(self) -> Dict[str, object]:
        # initial guess: equal weight allocation
        equal_weight = 1.0 / self.num_assets
        initial_guess = np.array([equal_weight] * self.num_assets)

        # constraints and bounds
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))

        result = minimize(
            fun=self._portfolio_volatility,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        min_vol_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(min_vol_weights)

        weights_dict = {}
        for index in range(self.num_assets):
            symbol = self.assets[index]
            weight_val = float(min_vol_weights[index])
            weights_dict[symbol] = round(weight_val, 4)

        output_data = {
            "strategy": "Minimum Volatility",
            "expected_return": round(float(ret), 4),
            "volatility": round(float(vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "weights": weights_dict,
            "success": bool(result.success)
        }

        return output_data