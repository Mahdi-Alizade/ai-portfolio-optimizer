# AI-Powered Quantitative Portfolio Optimizer

A deterministic institutional-grade portfolio allocation, risk management, and Monte Carlo simulation engine integrated with automated live news ingestion and structured LLM investor view synthesis.

---

## Architecture Flow

[Live Financial News / RSS Feeds] ─────────┐
▼
[Structured AI View Extractor]
(Pydantic Schema Validation)
│
▼
[Investor Views Matrix (P, Q, Ω)]
│
[Historical Prices (SQLite Cache)]        │
│                          │
├──────────────────────────┼──────────────► [Black-Litterman Engine]
│                          │                (Posterior Expected Returns)
│                          │                               │
├─► [MPT SLSQP Optimizer] ─┴───────────────────────────────┤
│   (Turnover & Friction Penalty Regularized)              │
│                                                          ▼
├──────────────────────────────────────────────► [Risk & Audit Engine]
│                                                (VaR, CVaR, Concentration)
│                                                          │
├──────────────────────────────────────────────► [Backtesting Engine]
│                                                (CAGR, Sortino, MDD)
│                                                          │
└──────────────────────────────────────────────► [Monte Carlo Simulator]
(Correlated Geometric Brownian)
│
▼
[FastAPI & CLI Delivery]


---

## Core Capabilities

1. **Deterministic Quant Engines (`src/quant/`)**:
   - **Modern Portfolio Theory (MPT)**: SLSQP solver with optional L1 transaction cost and turnover penalty ($\lambda_{turnover} \sum \vert{}w_i - w_i^{current}\vert{}$).
   - **Black-Litterman Master Formula**: Mathematical blending of equilibrium implied returns ($\Pi = \delta \Sigma w_{mkt}$) with qualitative AI views and diagonal error variance ($\Omega$).
   - **Monte Carlo Simulator (`src/quant/monte_carlo.py`)**: Multivariate Geometric Brownian Motion utilizing Cholesky decomposition of the covariance matrix for correlated asset shocks (1,000+ paths generating P5, P50, and P95 distribution percentiles).
   - **Risk & Macro Stress-Testing (`src/quant/risk_engine.py`)**: 1-day 95% Historical & Parametric VaR, Conditional VaR (Expected Shortfall), concentration threshold audits, and stylized stress testing (2008 GFC, 2020 COVID shock, 2022 rate hike cycles).
   - **Attribution & Backtesting (`src/quant/backtest_engine.py`)**: Historical compounding, CAGR, Sortino Ratio, Maximum Drawdown (MDD), and Calmar Ratio benchmarking.

2. **Automated Live Ingestion & Local Caching (`src/data/`)**:
   - **SQLite Price Cache (`src/data/cache_manager.py`)**: Persistent local storage preventing API rate limits and network latency.
   - **Financial News Loader (`src/data/news_loader.py`)**: Automated live news feed aggregation and sanitation per ticker symbol.

3. **Structured AI Integration (`src/ai/`)**:
   - Uses OpenAI structured outputs (`beta.chat.completions.parse`) with strict Pydantic models to guarantee deterministic extraction of excess return estimates and bounded confidence levels.

---

## Running the Application

### 1. Interactive CLI Runner
```powershell
python cli.py --tickers AAPL MSFT NVDA GOOGL --lookback 2 --max-weight 0.35
2. FastAPI REST Server
PowerShell
uvicorn main:app --reload --port 8000
Swagger UI: http://127.0.0.1:8000/docs

3. Automated Test Suite (19 passing unit tests)
PowerShell
pytest -v tests/