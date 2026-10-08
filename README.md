# AI-Powered Quantitative Portfolio Optimizer

A high-performance quantitative portfolio allocation and risk audit engine integrated with structured AI sentiment extraction. The system combines classical Modern Portfolio Theory (MPT) with the Black-Litterman model, SQLite-backed market data caching, parametric/historical risk audits (VaR, CVaR, stress testing), and walk-forward performance attribution metrics.

---

## Architectural Workflow

[Market Financial Context / News]
│
▼
[Structured AI View Parser]
(OpenAI Pydantic Enforcement)
│
▼
[Investor Views Vector & Uncertainty] ──┐
(Q, P, Ω)                  │
▼
[Market Data Ingestion] ──────► [Black-Litterman Engine] ──► [Optimized Allocation]
(SQLite Cached Pricing)      (Posterior Expected Returns)            │
│                                                        │
├──────────────► [Modern Portfolio Theory] ──────────────┤
│                  (SLSQP Sharp / Min Vol)                │
│                                                        ▼
├──────────────────────────────────────────────► [Risk & Audit Engine]
│                                                (VaR, CVaR, Shocks)
│                                                        │
└──────────────────────────────────────────────► [Backtesting Engine]
(Sortino, Calmar, MDD)
│
▼
[FastAPI REST Delivery]


---

## Core Capabilities

### 1. Mathematical Quant Engines (`src/quant/`)
- **Modern Portfolio Theory (MPT)**: Sequential Least Squares Programming (`SLSQP`) solving for:
  - Maximum Sharpe Allocation ($\max \frac{\mathbf{w}^T \mathbf{\mu} - r_f}{\sigma_p}$)
  - Minimum Volatility Allocation ($\min \mathbf{w}^T \mathbf{\Sigma} \mathbf{w}$)
  - Bounds: Long-only ($0 \le w_i \le 1$), Full Investment ($\sum w_i = 1$).
- **Black-Litterman Model**:
  - Computes market equilibrium excess returns $\mathbf{\Pi} = \delta \mathbf{\Sigma} \mathbf{w}_{mkt}$.
  - Blends subjective investor views with the market prior via:
    $$E[R] = \left[(\tau \mathbf{\Sigma})^{-1} + \mathbf{P}^T \mathbf{\Omega}^{-1} \mathbf{P}\right]^{-1} \left[(\tau \mathbf{\Sigma})^{-1} \mathbf{\Pi} + \mathbf{P}^T \mathbf{\Omega}^{-1} \mathbf{Q}\right]$$
- **Portfolio Risk Engine (`src/quant/risk_engine.py`)**:
  - **1-Day Historical & Parametric Value at Risk (VaR)** at 95% confidence.
  - **Conditional Value at Risk (CVaR / Expected Shortfall)** evaluating tail risk.
  - **Historical Crisis Stress-Testing**: Simulating drawdowns under the 2008 GFC, 2020 COVID shock, and 2022 rate hike cycles.
  - **Concentration Risk Audit**: Flags individual asset overweight violations.
- **Backtesting & Attribution Engine (`src/quant/backtest_engine.py`)**:
  - Cumulative Wealth Compounding, CAGR, Sortino Ratio, Maximum Drawdown (MDD), and Calmar Ratio benchmarking.

### 2. High-Performance Data Layer (`src/data/`)
- **SQLite Price Cache (`src/data/cache_manager.py`)**: Automatic local caching of historical trading sessions to prevent API rate-limiting, reduce network latency, and guarantee deterministic query playback.

### 3. Structured LLM Layer (`src/ai/`)
- Uses deterministic OpenAI structured outputs (`beta.chat.completions.parse`) with strict Pydantic schemas (`SingleAssetView`, `PortfolioViewsResponse`) to translate unstructured news into bounded quantitative vectors ($Q$) and diagonal confidence variances ($\Omega$).

---

## Directory Structure

```text
ai-portfolio-optimizer/
├── src/
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── schemas.py                 # Pydantic schemas for view generation
│   │   └── sentiment_extractor.py     # Structured LLM view extractor
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes_schemas.py          # API validation models
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                  # Pydantic v2 application settings
│   ├── data/
│   │   ├── __init__.py
│   │   ├── cache_manager.py           # SQLite price cache storage
│   │   └── market_data.py             # Historical pricing & return calculation
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── backtest_engine.py         # Sortino, Calmar, MDD & performance metrics
│   │   ├── black_litterman.py         # Black-Litterman matrix solver
│   │   ├── mpt_optimizer.py           # SLSQP Sharpe & variance minimizer
│   │   └── risk_engine.py             # Historical VaR, CVaR & macro stress tests
│   └── services/
│       ├── __init__.py
│       └── portfolio_service.py       # Orchestration pipeline
├── tests/
│   ├── __init__.py
│   ├── test_api.py                    # API route and validation tests
│   ├── test_backtest.py               # Cumulative return and drawdown tests
│   ├── test_cache.py                  # SQLite cache save and retrieval tests
│   ├── test_quant.py                  # Matrix and constraint unit tests
│   └── test_risk.py                   # VaR, CVaR and stress testing tests
├── .env.example
├── .gitignore
├── main.py                            # FastAPI entry point
├── requirements.txt
└── README.md
Quickstart
1. Environment Setup
PowerShell
python -m venv venv ; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
2. Configuration
PowerShell
Copy-Item .env.example .env
Provide your OpenAI API key in .env:

Plaintext
OPENAI_API_KEY=your_openai_api_key_here
DEBUG=True
3. Run Test Suite
PowerShell
pytest -v tests/
4. Start the Application
PowerShell
uvicorn main:app --reload --port 8000
Interactive API docs: http://127.0.0.1:8000/docs