from typing import List, Optional
from pydantic import BaseModel, Field


class SingleAssetView(BaseModel):
    ticker: str = Field(description="The ticker symbol of the asset (e.g. AAPL, MSFT)")
    expected_excess_return: float = Field(
        description="The estimated annualized excess return based on qualitative insights, between -0.50 and +0.50"
    )
    confidence: float = Field(
        description="Confidence level in the assessment from 0.05 to 0.95",
        ge=0.05,
        le=0.95
    )
    sentiment: str = Field(description="Sentiment classification: Bullish, Bearish, or Neutral")
    reasoning: str = Field(description="Brief explanation of the thesis supporting this view")


class PortfolioViewsResponse(BaseModel):
    views: List[SingleAssetView] = Field(description="List of extracted investor views for the analyzed assets")
    market_overview: Optional[str] = Field(description="Summary of macro market conditions observed in the input text")