from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import settings
from src.api.routes_schemas import PortfolioOptimizationRequest, OptimizationResponseModel
from src.services.portfolio_service import PortfolioOptimizationService

app = FastAPI(
    title=settings.app_name,
    version="1.2.0",
    description="Deterministic Quantitative Portfolio Optimizer with Automated Live News Ingestion and Risk Auditing"
)

# allow CORS for frontend integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    health_payload = {
        "status": "healthy",
        "service": settings.app_name,
        "debug_mode": settings.debug
    }
    return health_payload


@app.post("/api/v1/optimize", response_model=OptimizationResponseModel, status_code=status.HTTP_200_OK)
def optimize_portfolio(payload: PortfolioOptimizationRequest):
    try:
        service = PortfolioOptimizationService(risk_free_rate=payload.risk_free_rate)
        
        result = service.run_full_optimization(
            tickers=payload.tickers,
            news_context=payload.news_context,
            auto_fetch_news=payload.auto_fetch_news if payload.auto_fetch_news is not None else True,
            lookback_years=payload.lookback_years,
            max_asset_allocation=payload.max_asset_allocation
        )

        return OptimizationResponseModel(
            status="success",
            data=result
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization pipeline failed: {str(exc)}" if 'exc' in locals() else f"Optimization pipeline failed: {str(ex)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)