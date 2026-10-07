from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator


class PortfolioOptimizationRequest(BaseModel):
    tickers: List[str] = Field(
        ...,
        description="List of stock ticker symbols (minimum 2 symbols required)",
        examples=[["AAPL", "MSFT", "GOOGL", "AMZN"]]
    )
    news_context: Optional[str] = Field(
        None,
        description="Optional qualitative market news or earnings summary to extract AI views",
        examples=["Nvidia reported strong data center revenue, while Apple faces smartphone market saturation."]
    )
    lookback_years: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Historical data lookback window in years"
    )
    risk_free_rate: Optional[float] = Field(
        default=0.04,
        ge=0.0,
        le=0.20,
        description="Annual risk-free benchmark rate"
    )
    max_asset_allocation: Optional[float] = Field(
        default=0.40,
        ge=0.10,
        le=1.00,
        description="Maximum allowed portfolio allocation threshold for a single asset (for concentration risk auditing)"
    )

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: List[str]) -> List[str]:
        cleaned_list = []
        for item in v:
            clean_item = item.strip().upper()
            if len(clean_item) > 0 and clean_item not in cleaned_list:
                cleaned_list.append(clean_item)

        if len(cleaned_list) < 2:
            raise ValueError("At least 2 distinct valid ticker symbols are required for portfolio optimization")
        return cleaned_list


class OptimizationResponseModel(BaseModel):
    status: str = "success"
    data: Dict[str, Any]