# AI-Powered Quantitative Portfolio Optimizer

A deterministic quantitative portfolio allocation engine integrated with structured AI sentiment extraction. The system combines classical Modern Portfolio Theory (MPT) with the Black-Litterman model, using Large Language Models (LLMs) strictly as structured view generators rather than unconstrained black-box predictors.

---

## Architecture Overview

Financial systems require deterministic rigor. LLMs should not perform floating-point matrix operations or directly guess portfolio weights. This engine enforces a strict separation of concerns:

[Market News / Transcripts]│▼[Structured LLM Parser (Pydantic)] ──────► [Investor Views Matrix (P, Q, Ω)]│[Historical Price Feeds (Yahoo Fin)]                   ││                                         ▼▼                              [Black-Litterman Engine][Covariance & Return Matrices] ──────────► [Posterior Expected Returns]│                                         │▼                                         ▼[Markowitz MPT Engine (SLSQP)]             [Constrained Optimization]│                                         │└───────────────┬─────────────────────────┘▼[FastAPI REST API / Validated Output]
### Key Modules:
1. **Data Ingestion (`src/data`)**: Automated retrieval and cleansing of adjusted close price series with annualization constants (252 trading days).
2. **Deterministic Quant Engine (`src/quant`)**:
   - **Modern Portfolio Theory (MPT)**: Sequential Least Squares Programming (`SLSQP`) to locate Maximum Sharpe and Minimum Volatility frontiers under long-only bounds ($\sum w_i = 1, 0 \le w_i \le 1$).
   - **Black-Litterman Model**: Merges market equilibrium prior returns ($\Pi = \delta \Sigma w_{mkt}$) with subjective investor views and uncertainty diagonals ($\Omega$) to calculate posterior expected returns.
3. **Structured AI Layer (`src/ai`)**: Converts financial text into validated Pydantic schemas (`SingleAssetView`, `PortfolioViewsResponse`) via deterministic OpenAI structured parsing (`beta.chat.completions.parse`).
4. **Service Orchestrator (`src/services`)**: Pipelines historical data loading, baseline Markowitz calculation, LLM view extraction, and Black-Litterman optimization into an unified response payload.
5. **API Layer (`src/api`, `main.py`)**: Asynchronous REST endpoints with Pydantic request validation and CORS middleware.

---

## Mathematical Formulation

### 1. Modern Portfolio Theory (Markowitz)
- **Portfolio Return**: $E(R_p) = \mathbf{w}^T \mathbf{\mu}$
- **Portfolio Volatility**: $\sigma_p = \sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}$
- **Sharpe Ratio Maximization**:
  $$\max_{\mathbf{w}} \frac{\mathbf{w}^T \mathbf{\mu} - r_f}{\sqrt{\mathbf{w}^T \mathbf{\Sigma} \mathbf{w}}} \quad \text{s.t.} \quad \sum_{i=1}^N w_i = 1, \quad 0 \le w_i \le 1$$

### 2. Black-Litterman Master Formula
Posterior expected return vector $E[R]$:
$$E[R] = \left[(\tau \mathbf{\Sigma})^{-1} + \mathbf{P}^T \mathbf{\Omega}^{-1} \mathbf{P}\right]^{-1} \left[(\tau \mathbf{\Sigma})^{-1} \mathbf{\Pi} + \mathbf{P}^T \mathbf{\Omega}^{-1} \mathbf{Q}\right]$$
Where:
- $\mathbf{\Pi}$: Implied equilibrium excess return vector ($\delta \mathbf{\Sigma} \mathbf{w}_{mkt}$)
- $\mathbf{P}$: Link matrix identifying assets involved in specific views
- $\mathbf{Q}$: Vector of expected returns per view (quantified by AI)
- $\mathbf{\Omega}$: Diagonal covariance matrix representing uncertainty in views
- $\tau$: Scalar weighting relative confidence in prior distribution

---

## Directory Structure

```text
ai-portfolio-optimizer/
├── src/
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── schemas.py                 # Pydantic structured view schemas
│   │   └── sentiment_extractor.py     # OpenAI structured parser integration
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes_schemas.py          # API request and response validators
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                  # Environment loader & system settings
│   ├── data/
│   │   ├── __init__.py
│   │   └── market_data.py             # Historical pricing & return statistics
│   ├── quant/
│   │   ├── __init__.py
│   │   ├── black_litterman.py         # Black-Litterman posterior engine
│   │   └── mpt_optimizer.py           # SLSQP Sharpe & variance minimizer
│   └── services/
│       ├── __init__.py
│       └── portfolio_service.py       # Pipeline orchestration layer
├── tests/
│   ├── __init__.py
│   ├── test_api.py                    # FastAPI route and validation tests
│   └── test_quant.py                  # Matrix and constraint unit tests
├── .env.example                       # Environment configuration template
├── .gitignore                         # Git exclusion rules
├── main.py                            # FastAPI entry point
├── requirements.txt                   # Production and testing dependencies
└── README.md                          # Technical documentation
Installation & Setup1. Virtual Environment ActivationIn Windows PowerShell:PowerShellpython -m venv venv ; .\venv\Scripts\Activate.ps1
2. Dependency InstallationPowerShellpip install -r requirements.txt
3. Environment ConfigurationCopy the template and supply your OpenAI API key:PowerShellCopy-Item .env.example .env
Edit .env:PlaintextOPENAI_API_KEY=your_openai_api_key_here
DEBUG=True
Running the ApplicationStart the FastAPI application using Uvicorn:PowerShelluvicorn main:app --reload --port 8000
Interactive Swagger Documentation: http://127.0.0.1:8000/docsHealth Check: http://127.0.0.1:8000/healthRunning TestsExecute the automated test suite with coverage verification:PowerShellpytest -v tests/
Test validations cover:Weight normalization ($\sum w_i = 1.0$) and non-negativity constraint compliance.Variance reduction properties under the minimum volatility objective.Regularization of posterior returns in the Black-Litterman engine without matrix singularity.Input validation boundaries on API endpoints (e.g., rejecting portfolios with $< 2$ assets).