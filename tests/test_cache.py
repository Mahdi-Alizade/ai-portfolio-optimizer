import os
import pandas as pd
import pytest
from src.data.cache_manager import SQLitePriceCache


@pytest.fixture
def temp_cache():
    # test database inside data directory with dedicated test name
    cache = SQLitePriceCache(db_filename="test_market_cache.db")
    yield cache

    # cleanup after test completes
    if os.path.exists(cache.db_path):
        try:
            os.remove(cache.db_path)
        except OSError:
            pass


def test_sqlite_cache_save_and_retrieve(temp_cache):
    dates = pd.date_range(start="2025-01-01", periods=5, freq="D")
    sample_df = pd.DataFrame(
        {
            "AAPL": [150.0, 152.0, 151.5, 153.0, 155.0],
            "MSFT": [300.0, 302.5, 301.0, 304.0, 305.5]
        },
        index=dates
    )

    # save sample dataframe to cache
    temp_cache.save_prices(sample_df)

    # retrieve cached data
    retrieved_df = temp_cache.get_cached_prices(
        tickers=["AAPL", "MSFT"],
        start_date="2025-01-01",
        end_date="2025-01-05"
    )

    assert not retrieved_df.empty
    assert "AAPL" in retrieved_df.columns
    assert "MSFT" in retrieved_df.columns
    assert len(retrieved_df) == 5
    assert float(retrieved_df.loc["2025-01-01", "AAPL"].iloc[0] if isinstance(retrieved_df.loc["2025-01-01", "AAPL"], pd.Series) else retrieved_df.loc["2025-01-01", "AAPL"]) == 150.0


def test_sqlite_cache_returns_empty_on_missing_ticker(temp_cache):
    result = temp_cache.get_cached_prices(
        tickers=["UNKNOWN_TICKER"],
        start_date="2025-01-01",
        end_date="2025-01-05"
    )
    assert result.empty