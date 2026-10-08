import pytest
from unittest.mock import patch, MagicMock
from src.data.news_loader import FinancialNewsLoader


def test_build_news_context_formatting():
    loader = FinancialNewsLoader(max_articles_per_ticker=2)

    fake_news_item_1 = {
        "content": {
            "title": "Quarterly Earnings Surge",
            "summary": "Revenue increased by 15 percent year-over-year.",
            "provider": {"displayName": "Financial Times"}
        }
    }
    fake_news_item_2 = {
        "content": {
            "title": "New Product Launch",
            "summary": "Enterprise cloud platform gains adoption.",
            "provider": {"displayName": "Bloomberg"}
        }
    }

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.news = [fake_news_item_1, fake_news_item_2]

    with patch("yfinance.Ticker", return_value=mock_ticker_instance):
        context = loader.build_news_context(tickers=["AAPL"])

        assert "=== News for AAPL ===" in context
        assert "Quarterly Earnings Surge" in context
        assert "Financial Times" in context
        assert "New Product Launch" in context


def test_news_loader_handles_empty_feed():
    loader = FinancialNewsLoader(max_articles_per_ticker=2)

    mock_ticker_instance = MagicMock()
    mock_ticker_instance.news = []

    with patch("yfinance.Ticker", return_value=mock_ticker_instance):
        articles = loader.fetch_news_for_ticker("TEST")
        assert articles == []