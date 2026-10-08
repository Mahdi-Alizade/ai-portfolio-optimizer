from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


class PortfolioBacktestEngine:
    def __init__(
        self,
        daily_returns: pd.DataFrame,
        weights: Dict[str, float],
        risk_free_rate: float = 0.04
    ):
        self.returns = daily_returns
        self.weights_dict = weights
        self.risk_free_rate = risk_free_rate
        self.trading_days = 252
        self.assets = list(daily_returns.columns)

        # align weight array with columns
        weight_list = []
        for col in self.assets:
            w = self.weights_dict.get(col, 0.0)
            weight_list.append(w)
        self.weights = np.array(weight_list)

        # calculate historical portfolio daily return series
        self.portfolio_series = self.returns.dot(self.weights)

    def calculate_cumulative_returns(self) -> pd.Series:
        # compute (1 + r1) * (1 + r2) ... - 1
        growth_factors = 1.0 + self.portfolio_series
        cumulative_growth = growth_factors.cumprod()
        cumulative_returns = cumulative_growth - 1.0
        return cumulative_returns

    def calculate_drawdown_series(self) -> Tuple[pd.Series, float]:
        growth_factors = 1.0 + self.portfolio_series
        wealth_index = growth_factors.cumprod()
        previous_peaks = wealth_index.cummax()
        
        # drawdown relative to peak
        drawdown_series = (wealth_index - previous_peaks) / previous_peaks
        max_drawdown = float(drawdown_series.min())

        return drawdown_series, max_drawdown

    def calculate_sortino_ratio(self) -> float:
        mean_daily = float(self.portfolio_series.mean())
        annualized_return = mean_daily * self.trading_days

        # downside deviation: consider only negative returns relative to 0 or target
        negative_returns = self.portfolio_series[self.portfolio_series < 0.0]

        if len(negative_returns) > 1:
            downside_std = float(negative_returns.std() * np.sqrt(self.trading_days))
        else:
            downside_std = 0.0

        excess_return = annualized_return - self.risk_free_rate

        if downside_std > 0:
            sortino = excess_return / downside_std
        else:
            sortino = 0.0

        return round(float(sortino), 4)

    def generate_performance_metrics(self) -> Dict[str, object]:
        total_days = len(self.portfolio_series)
        if total_days == 0:
            return {}

        # total cumulative return
        growth_factors = 1.0 + self.portfolio_series
        final_wealth = float(growth_factors.prod())
        cumulative_return = final_wealth - 1.0

        # annualized return via geometric compounding
        years = total_days / self.trading_days
        if years > 0:
            cagr = (final_wealth ** (1.0 / years)) - 1.0
        else:
            cagr = cumulative_return

        # annualized volatility
        ann_volatility = float(self.portfolio_series.std() * np.sqrt(self.trading_days))

        # sharpe ratio
        excess_ret = cagr - self.risk_free_rate
        if ann_volatility > 0:
            sharpe = excess_ret / ann_volatility
        else:
            sharpe = 0.0

        # sortino ratio
        sortino = self.calculate_sortino_ratio()

        # max drawdown and calmar ratio
        _, max_drawdown = self.calculate_drawdown_series()
        abs_mdd = abs(max_drawdown)

        if abs_mdd > 0:
            calmar_ratio = cagr / abs_mdd
        else:
            calmar_ratio = 0.0

        metrics_payload = {
            "total_trading_days": total_days,
            "cumulative_return": round(float(cumulative_return), 4),
            "cagr_annualized_return": round(float(cagr), 4),
            "annualized_volatility": round(float(ann_volatility), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "sortino_ratio": round(float(sortino), 4),
            "max_drawdown": round(float(max_drawdown), 4),
            "calmar_ratio": round(float(calmar_ratio), 4)
        }

        return metrics_payload