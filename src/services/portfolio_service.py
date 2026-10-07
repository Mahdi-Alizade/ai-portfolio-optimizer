from typing import Dict, List, Optional
import pandas as pd

from src.core.config import settings
from src.data.market_data import MarketDataLoader
from src.quant.mpt_optimizer import ModernPortfolioOptimizer
from src.quant.black_litterman import BlackLittermanOptimizer
from src.quant.risk_engine import PortfolioRiskEngine
from src.ai.sentiment_extractor import SentimentViewExtractor


class PortfolioOptimizationService:
    def __init__(self, risk_free_rate: Optional[float] = None):
        if risk_free_rate is not None:
            self.risk_free_rate = risk_free_rate
        else:
            self.risk_free_rate = settings.risk_free_rate
        self.sentiment_extractor = SentimentViewExtractor()

    def run_full_optimization(
        self,
        tickers: List[str],
        news_context: Optional[str] = None,
        lookback_years: int = 2,
        max_asset_allocation: float = 0.40
    ) -> Dict[str, object]:
        # step 1: fetch market historical prices and calculate returns
        normalized_tickers = [t.upper().strip() for t in tickers]
        loader = MarketDataLoader(tickers=normalized_tickers, lookback_years=lookback_years)
        daily_returns = loader.calculate_daily_returns()
        summary_stats = loader.get_summary_statistics()

        # step 2: run baseline Modern Portfolio Theory optimization
        mpt = ModernPortfolioOptimizer(
            daily_returns=daily_returns,
            risk_free_rate=self.risk_free_rate
        )
        max_sharpe_result = mpt.optimize_max_sharpe()
        min_vol_result = mpt.optimize_min_volatility()

        # generate risk metrics for baseline MPT portfolios
        risk_engine_sharpe = PortfolioRiskEngine(
            daily_returns=daily_returns,
            weights=max_sharpe_result["weights"]
        )
        max_sharpe_result["risk_audit"] = risk_engine_sharpe.generate_comprehensive_risk_report(
            max_allowed_weight=max_asset_allocation
        )

        risk_engine_min_vol = PortfolioRiskEngine(
            daily_returns=daily_returns,
            weights=min_vol_result["weights"]
        )
        min_vol_result["risk_audit"] = risk_engine_min_vol.generate_comprehensive_risk_report(
            max_allowed_weight=max_asset_allocation
        )

        # step 3: extract AI investor views if news context is provided
        ai_views_list = []
        bl_result = None

        if news_context and len(news_context.strip()) > 0:
            views_dict, conf_dict, parsed_views = self.sentiment_extractor.extract_views(
                tickers=normalized_tickers,
                news_context=news_context
            )
            ai_views_list = [view.model_dump() for view in parsed_views]

            # step 4: run Black-Litterman using AI-derived views
            bl_optimizer = BlackLittermanOptimizer(
                daily_returns=daily_returns,
                risk_free_rate=self.risk_free_rate
            )
            bl_result = bl_optimizer.optimize_with_views(
                views_dict=views_dict,
                confidence_dict=conf_dict
            )

            # audit risk metrics for Black-Litterman allocations
            risk_engine_bl = PortfolioRiskEngine(
                daily_returns=daily_returns,
                weights=bl_result["weights"]
            )
            bl_result["risk_audit"] = risk_engine_bl.generate_comprehensive_risk_report(
                max_allowed_weight=max_asset_allocation
            )

        # step 5: build consolidated payload
        formatted_summary = {}
        for ticker in normalized_tickers:
            if ticker in summary_stats.index:
                ann_ret = float(summary_stats.loc[ticker, "Annual Return"])
                ann_vol = float(summary_stats.loc[ticker, "Annual Volatility"])
                formatted_summary[ticker] = {
                    "annual_return": round(ann_ret, 4),
                    "annual_volatility": round(ann_vol, 4)
                }

        response = {
            "tickers": normalized_tickers,
            "lookback_years": lookback_years,
            "risk_free_rate": self.risk_free_rate,
            "historical_metrics": formatted_summary,
            "baseline_mpt": {
                "max_sharpe": max_sharpe_result,
                "min_volatility": min_vol_result
            },
            "ai_views": ai_views_list,
            "black_litterman": bl_result
        }

        return response