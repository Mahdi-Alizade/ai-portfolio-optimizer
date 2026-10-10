from typing import Dict, List, Optional, Tuple
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

    def optimize_max_sharpe(
        self,
        current_weights: Optional[Dict[str, float]] = None,
        turnover_penalty: float = 0.0
    ) -> Dict[str, object]:
        # initial guess: equal weight allocation across assets
        equal_weight = 1.0 / self.num_assets
        initial_guess = np.array([equal_weight] * self.num_assets)

        # parse current weights vector if rebalancing friction is modeled
        target_current_w = None
        if current_weights is not None:
            c_list = [current_weights.get(asset, 0.0) for asset in self.assets]
            target_current_w = np.array(c_list)

        def objective(w: np.ndarray) -> float:
            _, _, sharpe = self.portfolio_performance(w)
            penalty = 0.0
            if target_current_w is not None and turnover_penalty > 0.0:
                # L1 norm turnover cost
                turnover = np.sum(np.abs(w - target_current_w))
                penalty = turnover_penalty * turnover
            return -(sharpe - penalty)

        # constraint: sum of weights equals 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        # bounds: no short-selling (weights must be between 0 and 1)
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))

        result = minimize(
            fun=objective,
            x0=initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        ret, vol, sharpe = self.portfolio_performance(optimal_weights)

        # calculate executed turnover if current weights were supplied
        turnover_executed = 0.0
        if target_current_w is not None:
            turnover_executed = float(np.sum(np.abs(optimal_weights - target_current_w)) / 2.0)

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
            "turnover_rate": round(turnover_executed, 4),
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

        def vol_objective(w: np.ndarray) -> float:
            _, vol, _ = self.portfolio_performance(w)
            return vol

        result = minimize(
            fun=vol_objective,
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