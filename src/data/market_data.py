from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf


class MarketDataLoader:
    def __init__(self, tickers: list[str], lookback_years: int = 2):
        self.tickers = tickers
        self.lookback_years = lookback_years
        self.raw_data = None
        self.price_history = None

    def _calculate_date_range(self):
        end_date = datetime.today()
        total_days = self.lookback_years * 365
        start_date = end_date - timedelta(days=total_days)

        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        return start_date_str, end_date_str

    def fetch_price_data(self) -> pd.DataFrame:
        start_date, end_date = self._calculate_date_range()

        # download historical price series
        downloaded = yf.download(
            tickers=self.tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False
        )

        self.raw_data = downloaded

        # extract adjusted close prices
        if "Adj Close" in downloaded.columns:
            adjusted_prices = downloaded["Adj Close"]
        else:
            adjusted_prices = downloaded["Close"]

        # handle single ticker dataframe format edge case
        if isinstance(adjusted_prices, pd.Series):
            ticker_name = self.tickers[0]
            adjusted_prices = adjusted_prices.to_frame(name=ticker_name)

        # drop missing values to align dates
        clean_prices = adjusted_prices.dropna()
        self.price_history = clean_prices

        return clean_prices

    def calculate_daily_returns(self) -> pd.DataFrame:
        if self.price_history is None:
            self.fetch_price_data()

        # calculate percentage change day over day
        daily_returns = self.price_history.pct_change()
        daily_returns = daily_returns.dropna()

        return daily_returns

    def get_summary_statistics(self) -> pd.DataFrame:
        daily_returns = self.calculate_daily_returns()

        # annualizing factor for trading days in a year
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