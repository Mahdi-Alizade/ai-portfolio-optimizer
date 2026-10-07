import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# load environment variables from .env file if it exists
load_dotenv()


class Settings(BaseSettings):
    # API settings
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    market_data_api_key: str = os.getenv("MARKET_DATA_API_KEY", "")

    # Application settings
    app_name: str = "AI Portfolio Optimizer"
    debug: bool = True
    log_level: str = "INFO"

    # Default financial parameters
    risk_free_rate: float = 0.04
    default_lookback_years: int = 2

    # Pydantic v2 configuration
    model_config = SettingsConfigDict(case_sensitive=False)


settings = Settings()