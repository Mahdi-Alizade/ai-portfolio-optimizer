from typing import Dict, List, Optional
import pandas as pd

from src.core.config import settings
from src.data.market_data import MarketDataLoader
from src.data.news_loader import FinancialNewsLoader
from src.quant.mpt_optimizer import ModernPortfolioOptimizer
from src.quant.black_litterman import BlackLittermanOptimizer
from src.quant.risk_engine import PortfolioRiskEngine
from src.quant.backtest_engine import PortfolioBacktestEngine
from src.quant.monte_carlo import MonteCarloSimulator
from src.ai.sentiment_extractor import SentimentViewExtractor


class PortfolioOptimizationService:
    def __init__(self, risk_free_rate: Optional[float] = None):
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        else:
            self.risk_free_rate = settings.risk_free_rate
        self.sentiment_extractor = SentimentViewExtractor()
        self.news_loader = FinancialNewsLoader(max_articles_per_ticker=3)

    def run_full_optimization(
        self,
        tickers: List[str],
        news_context: Optional[str] = None,
        auto_fetch_news: bool = True,
        lookback_years: int = 2,
        max_asset_allocation: float = 0.40,
        run_monte_carlo: bool = True
    ) -> Dict[str, object]:
        # step 1: fetch market historical prices and calculate returns
        normalized_tickers = [t.upper().strip() for t in tickers]
        loader = MarketDataLoader(tickers=normalized_tickers, lookback_years=lookback_years)
        daily_returns = loader.calculate_daily_returns()
        summary_stats = loader.get_summary_statistics()

        # benchmark: equal weight portfolio
        equal_weight_val = round(1.0 / len(normalized_tickers), 4)
        equal_weights = {t: equal_weight_val for t in normalized_tickers}
        benchmark_bt = PortfolioBacktestEngine(
            daily_returns=daily_returns,
            weights=equal_weights,
            risk_free_rate=self.risk_free_rate
        )
        benchmark_performance = benchmark_bt.generate_performance_metrics()

        # step 2: run baseline Modern Portfolio Theory optimization
        mpt = ModernPortfolioOptimizer(
            daily_returns=daily_returns,
            risk_free_rate=self.risk_free_rate
        )
        max_sharpe_result = mpt.optimize_max_sharpe()
        min_vol_result = mpt.optimize_min_volatility()

        # risk audit and backtest
        risk_engine_sharpe = PortfolioRiskEngine(daily_returns=daily_returns, weights=max_sharpe_result["weights"])
        max_sharpe_result["risk_audit"] = risk_engine_sharpe.generate_comprehensive_risk_report(max_allowed_weight=max_asset_allocation)
        
        bt_sharpe = PortfolioBacktestEngine(daily_returns=daily_returns, weights=max_sharpe_result["weights"], risk_free_rate=self.risk_free_rate)
        max_sharpe_result["backtest_performance"] = bt_sharpe.generate_performance_metrics()

        risk_engine_min_vol = PortfolioRiskEngine(daily_returns=daily_returns, weights=min_vol_result["weights"])
        min_vol_result["risk_audit"] = risk_engine_min_vol.generate_comprehensive_risk_report(max_allowed_weight=max_asset_allocation)
        
        bt_min_vol = PortfolioBacktestEngine(daily_returns=daily_returns, weights=min_vol_result["weights"], risk_free_rate=self.risk_free_rate)
        min_vol_result["backtest_performance"] = bt_min_vol.generate_performance_metrics()

        # monte carlo for max sharpe
        if run_monte_carlo:
            mc_sharpe = MonteCarloSimulator(daily_returns=daily_returns, weights=max_sharpe_result["weights"])
            max_sharpe_result["forward_monte_carlo"] = mc_sharpe.run_simulation(time_horizon_years=1, num_simulations=1000)

        # step 3: resolve financial news context
        active_news_context = ""
        if news_context and len(news_context.strip()) > 0:
            active_news_context = news_context.strip()
        elif auto_fetch_news:
            active_news_context = self.news_loader.build_news_context(normalized_tickers)

        # step 4: extract AI investor views and run Black-Litterman
        ai_views_list = []
        bl_result = None

        if active_news_context and len(active_news_context.strip()) > 0:
            views_dict, conf_dict, parsed_views = self.sentiment_extractor.extract_views(
                tickers=normalized_tickers,
                news_context=active_news_context
            )
            ai_views_list = [view.model_dump() for view in parsed_views]

            bl_optimizer = BlackLittermanOptimizer(daily_returns=daily_returns, risk_free_rate=self.risk_free_rate)
            bl_result = bl_optimizer.optimize_with_views(views_dict=views_dict, confidence_dict=conf_dict)

            risk_engine_bl = PortfolioRiskEngine(daily_returns=daily_returns, weights=bl_result["weights"])
            bl_result["risk_audit"] = risk_engine_bl.generate_comprehensive_risk_report(max_allowed_weight=max_asset_allocation)
            
            bt_bl = PortfolioBacktestEngine(daily_returns=daily_returns, weights=bl_result["weights"], risk_free_rate=self.risk_free_rate)
            bl_result["backtest_performance"] = bt_bl.generate_performance_metrics()

            if run_monte_carlo:
                mc_bl = MonteCarloSimulator(daily_returns=daily_returns, weights=bl_result["weights"])
                bl_result["forward_monte_carlo"] = mc_bl.run_simulation(time_horizon_years=1, num_simulations=1000)

        # step 5: build response
        formatted_summary = {}
        for ticker in normalized_tickers:
            if ticker in summary_stats.index:
                formatted_summary[ticker] = {
                    "annual_return": round(float(summary_stats.loc[ticker, "Annual Return"]), 4),
                    "annual_volatility": round(float(summary_stats.loc[ticker, "Annual Volatility"]), 4)
                }

        response = {
            "tickers": normalized_tickers,
            "lookback_years": lookback_years,
            "risk_free_rate": self.risk_free_rate,
            "historical_metrics": formatted_summary,
            "news_context_used": active_news_context if len(active_news_context) < 300 else active_news_context[:300] + "...",
            "equal_weight_benchmark": {
                "weights": equal_weights,
                "performance": benchmark_performance
            },
            "baseline_mpt": {
                "max_sharpe": max_sharpe_result,
                "min_volatility": min_vol_result
            },
            "ai_views": ai_views_list,
            "black_litterman": bl_result
        }

        return response