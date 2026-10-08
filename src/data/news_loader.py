from typing import Dict, List, Optional
import yfinance as yf


class FinancialNewsLoader:
    def __init__(self, max_articles_per_ticker: int = 3):
        self.max_articles_per_ticker = max_articles_per_ticker

    def fetch_news_for_ticker(self, ticker: str) -> List[Dict[str, str]]:
        clean_ticker = ticker.upper().strip()
        articles_list = []

        try:
            ticker_obj = yf.Ticker(clean_ticker)
            raw_news = ticker_obj.news

            if not raw_news:
                return []

            for item in raw_news[:self.max_articles_per_ticker]:
                # yfinance news payloads vary: extract title, publisher, and summary
                # newer yfinance versions wrap article attributes in a 'content' dict
                content = item.get("content", item)
                
                title = content.get("title", "")
                summary = content.get("summary", "")
                provider = content.get("provider", {})
                publisher = provider.get("displayName", "") if isinstance(provider, dict) else str(provider)

                # sanitize strings
                clean_title = title.strip().replace("\n", " ")
                clean_summary = summary.strip().replace("\n", " ")

                if clean_title:
                    article_entry = {
                        "ticker": clean_ticker,
                        "title": clean_title,
                        "publisher": publisher,
                        "summary": clean_summary
                    }
                    articles_list.append(article_entry)

        except Exception as exc:
            # graceful fallback on network or ticker parsing errors
            articles_list.append({
                "ticker": clean_ticker,
                "title": f"Market update for {clean_ticker}",
                "publisher": "System Fallback",
                "summary": f"Could not retrieve live news feed due to: {str(exc)}"
            })

        return articles_list

    def fetch_news_for_portfolio(self, tickers: List[str]) -> Dict[str, List[Dict[str, str]]]:
        portfolio_news = {}
        for ticker in tickers:
            symbol = ticker.upper().strip()
            articles = self.fetch_news_for_ticker(symbol)
            portfolio_news[symbol] = articles
        return portfolio_news

    def build_news_context(self, tickers: List[str]) -> str:
        news_data = self.fetch_news_for_portfolio(tickers)
        context_paragraphs = []

        for ticker, articles in news_data.items():
            if not articles:
                continue

            ticker_header = f"=== News for {ticker} ==="
            article_lines = [ticker_header]

            for idx, art in enumerate(articles, start=1):
                headline = art.get("title", "")
                summary = art.get("summary", "")
                publisher = art.get("publisher", "")

                line = f"Article {idx}: {headline}"
                if publisher:
                    line += f" (Source: {publisher})"
                if summary:
                    line += f"\nSummary: {summary}"

                article_lines.append(line)

            paragraph = "\n".join(article_lines)
            context_paragraphs.append(paragraph)

        consolidated_context = "\n\n".join(context_paragraphs)
        return consolidated_context