from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize


class BlackLittermanOptimizer:
    def __init__(
        self,
        daily_returns: pd.DataFrame,
        market_caps: Optional[Dict[str, float]] = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.04
    ):
        self.returns = daily_returns
        self.assets = list(daily_returns.columns)
        self.num_assets = len(self.assets)
        self.delta = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252

        # calculate annualized covariance matrix
        daily_cov = self.returns.cov()
        self.cov_matrix = daily_cov.values * self.trading_days

        # determine benchmark market weights
        if market_caps is not None and len(market_caps) == self.num_assets:
            total_cap = sum(market_caps.values())
            weights_list = []
            for asset in self.assets:
                cap = market_caps.get(asset, 0.0)
                weights_list.append(cap / total_cap)
            self.market_weights = np.array(weights_list)
        else:
            # fallback: equal weighting as market proxy
            equal_val = 1.0 / self.num_assets
            self.market_weights = np.array([equal_val] * self.num_assets)

        # calculate implied equilibrium excess returns (Pi)
        self.implied_returns = self.delta * np.dot(self.cov_matrix, self.market_weights)

    def calculate_posterior_returns(
        self,
        p_matrix: np.ndarray,
        q_vector: np.ndarray,
        omega_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        num_views = len(q_vector)

        # if omega is not provided, use default heuristic: tau * P * Sigma * P.T
        if omega_matrix is None:
            middle_term = np.dot(self.cov_matrix, p_matrix.T)
            omega_matrix = self.tau * np.dot(p_matrix, middle_term)
            # make sure diagonal has variance to avoid singularity
            np.fill_diagonal(omega_matrix, np.diag(omega_matrix) + 1e-6)

        sigma = self.cov_matrix
        tau_sigma = self.tau * sigma

        inv_tau_sigma = np.linalg.inv(tau_sigma)
        inv_omega = np.linalg.inv(omega_matrix)

        # first factor: [(tau * Sigma)^-1 + P^T * Omega^-1 * P]^-1
        p_trans = p_matrix.T
        p_trans_inv_omega = np.dot(p_trans, inv_omega)
        middle_bracket = inv_tau_sigma + np.dot(p_trans_inv_omega, p_matrix)
        first_factor = np.linalg.inv(middle_bracket)

        # second factor: (tau * Sigma)^-1 * Pi + P^T * Omega^-1 * Q
        term1 = np.dot(inv_tau_sigma, self.implied_returns)
        term2 = np.dot(p_trans_inv_omega, q_vector)
        second_factor = term1 + term2

        # posterior expected return vector
        posterior_returns = np.dot(first_factor, second_factor)
        return posterior_returns

    def optimize_with_views(
        self,
        views_dict: Dict[str, float],
        confidence_dict: Optional[Dict[str, float]] = None
    ) -> Dict[str, object]:
        # transform views_dict to P matrix and Q vector
        view_assets = []
        q_list = []

        for symbol, return_val in views_dict.items():
            if symbol in self.assets:
                view_assets.append(symbol)
                q_list.append(return_val)

        if len(view_assets) == 0:
            # if no valid views, return equilibrium weights
            allocation_dict = {}
            for i, asset in enumerate(self.assets):
                allocation_dict[asset] = round(float(self.market_weights[i]), 4)
            return {
                "strategy": "Black-Litterman (Equilibrium Only)",
                "weights": allocation_dict,
                "posterior_returns": {asset: round(float(self.implied_returns[i]), 4) for i, asset in enumerate(self.assets)}
            }

        num_views = len(view_assets)
        p_matrix = np.zeros((num_views, self.num_assets))
        q_vector = np.array(q_list)

        for row_idx, symbol in enumerate(view_assets):
            col_idx = self.assets.index(symbol)
            p_matrix[row_idx, col_idx] = 1.0

        # build diagonal omega using confidence scale if provided
        omega_matrix = None
        if confidence_dict is not None:
            omega_matrix = np.zeros((num_views, num_views))
            for i, symbol in enumerate(view_assets):
                # confidence between 0.01 and 1.0
                conf = confidence_dict.get(symbol, 0.5)
                conf = max(0.01, min(conf, 1.0))
                # higher confidence means smaller variance in view error
                variance_scale = (1.0 - conf) / conf
                asset_idx = self.assets.index(symbol)
                asset_variance = self.cov_matrix[asset_idx, asset_idx]
                omega_matrix[i, i] = self.tau * asset_variance * variance_scale + 1e-6

        posterior_expected_returns = self.calculate_posterior_returns(p_matrix, q_vector, omega_matrix)

        # find constrained optimal weights (long-only, sum to 1) using posterior returns
        def objective(w):
            port_ret = np.dot(w, posterior_expected_returns)
            port_var = np.dot(w.T, np.dot(self.cov_matrix, w))
            # maximize utility: return - (delta / 2) * variance
            utility = port_ret - (self.delta / 2.0) * port_var
            return -utility

        initial_weights = np.array([1.0 / self.num_assets] * self.num_assets)
        bounds = tuple((0.0, 1.0) for _ in range(self.num_assets))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})

        opt_res = minimize(
            fun=objective,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimized_weights = opt_res.x

        weights_out = {}
        posterior_returns_out = {}
        for idx in range(self.num_assets):
            asset_symbol = self.assets[idx]
            weights_out[asset_symbol] = round(float(optimized_weights[idx]), 4)
            posterior_returns_out[asset_symbol] = round(float(posterior_expected_returns[idx]), 4)

        result_payload = {
            "strategy": "Black-Litterman (AI-Informed Views)",
            "weights": weights_out,
            "posterior_returns": posterior_returns_out,
            "prior_equilibrium_returns": {
                asset: round(float(self.implied_returns[i]), 4)
                for i, asset in enumerate(self.assets)
            },
            "success": bool(opt_res.success)
        }

        return result_payload