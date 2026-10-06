import json
from typing import Dict, List, Tuple
from openai import OpenAI
from src.core.config import settings
from src.ai.schemas import PortfolioViewsResponse, SingleAssetView


class SentimentViewExtractor:
    def __init__(self, api_key: str = None):
        self.api_key = api_key if api_key else settings.openai_api_key
        if not self.api_key:
            # fallback or mock mode if api key is missing
            self.client = None
        else:
            self.client = OpenAI(api_key=self.api_key)

    def _build_prompt(self, tickers: List[str], news_context: str) -> str:
        tickers_str = ", ".join(tickers)
        
        prompt_lines = [
            f"You are an institutional financial analyst evaluating the following assets: {tickers_str}.",
            "Based on the provided market news and financial context, formulate specific investment views.",
            "Rules for your assessment:",
            "1. Quantify the expected annualized excess return (expected_excess_return) typically between -0.30 (-30%) and +0.30 (+30%).",
            "2. Assign a realistic confidence level between 0.10 (low certainty) and 0.90 (high conviction).",
            "3. Provide a concise, factual explanation in the reasoning field.",
            "",
            "Financial Context / News Articles:",
            news_context
        ]

        full_prompt = "\n".join(prompt_lines)
        return full_prompt

    def extract_views(self, tickers: List[str], news_context: str) -> Tuple[Dict[str, float], Dict[str, float], List[SingleAssetView]]:
        # if no client or key, return conservative neutral views
        if self.client is None:
            neutral_views = {}
            neutral_conf = {}
            for t in tickers:
                neutral_views[t] = 0.05
                neutral_conf[t] = 0.50
            return neutral_views, neutral_conf, []

        user_content = self._build_prompt(tickers, news_context)

        system_instruction = (
            "You are a quantitative research assistant. Extract objective, structured investor views "
            "strictly adhering to the requested schema. Do not output conversational text."
        )

        response = self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format=PortfolioViewsResponse,
            temperature=0.2
        )

        parsed_data = response.choices[0].message.parsed
        
        views_dict = {}
        confidence_dict = {}

        for view in parsed_data.views:
            ticker_upper = view.ticker.upper().strip()
            if ticker_upper in tickers:
                views_dict[ticker_upper] = float(view.expected_excess_return)
                confidence_dict[ticker_upper] = float(view.confidence)

        return views_dict, confidence_dict, parsed_data.views