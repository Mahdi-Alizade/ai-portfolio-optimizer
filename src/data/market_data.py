from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import yfinance as yf

from src.data.cache_manager import SQLitePriceCache


class MarketDataLoader:
    def __init__(self, tickers: List[str], lookback_years: int = 2, use_cache: bool = True):
        self.tickers = [t.upper().strip() for t in tickers]
        self.lookback_years = lookback_years
        self.use_cache = use_cache
        self.cache = SQLitePriceCache() if use_cache else None
        self.price_history = None

    def _calculate_date_range(self):
        end_date = datetime.today()
        total_days = self.lookback_years * 365
        start_date = end_date - timedelta(days=total_days)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        return start_date_str, end_date_str

    def fetch_price_data(self) -> pd.DataFrame:
        start_date_str, end_date_str = self._calculate_date_range()

        # step 1: check if cached data covers all requested tickers
        if self.use_cache and self.cache is not None:
            cached_df = self.cache.get_cached_prices(
                tickers=self.tickers,
                start_date=start_date_str,
                end_date=end_date_str
            )
            # verify all tickers are present and data is non-empty
            if not cached_df.empty:
                missing_tickers = [t for t in self.tickers if t not in cached_df.columns]
                # require reasonable history length (at least ~80% expected trading days)
                expected_min_days = int(self.lookback_years * 252 * 0.70)
                if len(missing_tickers) == 0 and len(cached_df) >= expected_min_days:
                    self.price_history = cached_df[self.tickers].dropna()
                    return self.price_history

        # step 2: download fresh data from network provider
        downloaded = yf.download(
            tickers=self.tickers,
            start=start_date_str,
            end=end_date_str,
            progress=False,
            auto_adjust=False
        )

        if "Adj Close" in downloaded.columns:
            adjusted_prices = downloaded["Adj Close"]
        else:
            adjusted_prices = downloaded["Close"]

        # handle single ticker Series to DataFrame conversion
        if isinstance(adjusted_prices, pd.Series):
            ticker_name = self.tickers[0]
            adjusted_prices = adjusted_prices.to_frame(name=ticker_name)

        clean_prices = adjusted_prices.dropna()
        self.price_history = clean_prices

        # step 3: persist newly fetched data into local cache
        if self.use_cache and self.cache is not None and not clean_prices.empty:
            self.cache.save_prices(clean_prices)

        return clean_prices

    def calculate_daily_returns(self) -> pd.DataFrame:
        if self.price_history is None:
            self.fetch_price_data()

        daily_returns = self.price_history.pct_change()
        daily_returns = daily_returns.dropna()

        return daily_returns

    def get_summary_statistics(self) -> pd.DataFrame:
        daily_returns = self.calculate_daily_returns()
        trading_days = 252

        mean_returns = daily_returns.mean()
        annual_returns = mean_returns * trading_days

        daily_std = daily_returns.std()
        annual_volatility = daily_std * (trading_days ** 0.5)

        stats_dict = {
            "Annual Return": annual_returns,
            "Annual Volatility": annual_volatility
        }

        stats_df = pd.DataFrame(stats_dict)
        return stats_df