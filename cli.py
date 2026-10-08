import argparse
import sys
from src.services.portfolio_service import PortfolioOptimizationService


def print_section_header(title: str) -> None:
    border = "=" * 70
    print(f"\n{border}")
    print(f" {title.upper()}")
    print(f"{border}\n")


def print_table_row(col1: str, col2: str, width1: int = 35, width2: int = 30) -> None:
    print(f"  {col1:<{width1}} | {col2:<{width2}}")


def run_cli():
    parser = argparse.ArgumentParser(
        description="Run quantitative portfolio optimization with risk audits and backtesting directly from CLI."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "MSFT", "NVDA", "GOOGL"],
        help="List of stock tickers (minimum 2 tickers). Example: --tickers AAPL MSFT NVDA"
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=2,
        help="Historical lookback period in years (default: 2)"
    )
    parser.add_argument(
        "--risk-free",
        type=float,
        default=0.04,
        help="Annual risk-free benchmark rate (default: 0.04)"
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.35,
        help="Maximum allowed single-asset concentration threshold (default: 0.35)"
    )
    parser.add_argument(
        "--news",
        type=str,
        default=(
            "Nvidia demonstrated strong enterprise data center growth and pricing power. "
            "Apple faced moderate saturation in mature smartphone replacement cycles. "
            "Microsoft cloud infrastructure consumption continued to accelerate."
        ),
        help="Financial news context to extract qualitative views for Black-Litterman."
    )

    args = parser.parse_args()

    cleaned_tickers = []
    for t in args.tickers:
        item = t.strip().upper()
        if len(item) > 0 and item not in cleaned_tickers:
            cleaned_tickers.append(item)

    if len(cleaned_tickers) < 2:
        print("[ERROR] At least 2 distinct ticker symbols are required.")
        sys.exit(1)

    print_section_header("Starting Portfolio Optimization Pipeline")
    print(f"  Tickers: {', '.join(cleaned_tickers)}")
    print(f"  Lookback Window: {args.lookback} years")
    print(f"  Risk-Free Benchmark: {args.risk_free * 100:.2f}%")
    print(f"  Max Single Asset Limit: {args.max_weight * 100:.2f}%\n")

    service = PortfolioOptimizationService(risk_free_rate=args.risk_free)

    try:
        results = service.run_full_optimization(
            tickers=cleaned_tickers,
            news_context=args.news,
            lookback_years=args.lookback,
            max_asset_allocation=args.max_weight
        )
    except Exception as exc:
        print(f"[ERROR] Optimization pipeline execution failed: {str(exc)}")
        sys.exit(1)

    # 1. Historical Metrics Summary
    print_section_header("Historical Annualized Statistics")
    print_table_row("Asset Ticker", "Annual Return | Volatility")
    print("  " + "-" * 66)
    for ticker, stats in results["historical_metrics"].items():
        ret_pct = f"{stats['annual_return'] * 100:.2f}%"
        vol_pct = f"{stats['annual_volatility'] * 100:.2f}%"
        print_table_row(ticker, f"{ret_pct:<13} | {vol_pct}")

    # 2. Modern Portfolio Theory Allocations
    print_section_header("Baseline Modern Portfolio Theory (MPT)")

    strategies = [
        ("Max Sharpe Allocation", results["baseline_mpt"]["max_sharpe"]),
        ("Minimum Volatility Allocation", results["baseline_mpt"]["min_volatility"])
    ]

    for strat_name, strat_data in strategies:
        print(f"\n>> {strat_name}:")
        print(f"   Expected Return: {strat_data['expected_return'] * 100:.2f}% | Volatility: {strat_data['volatility'] * 100:.2f}% | Sharpe: {strat_data['sharpe_ratio']:.2f}")
        
        weight_str = ", ".join([f"{sym}: {wt * 100:.1f}%" for sym, wt in strat_data["weights"].items() if wt > 0.001])
        print(f"   Allocations: {weight_str}")

        # Risk Audit
        risk = strat_data.get("risk_audit", {})
        var_1d = risk.get("daily_value_at_risk", {}).get("historical_var_1d", 0.0)
        cvar_1d = risk.get("daily_conditional_var_1d", 0.0)
        conc = risk.get("concentration_audit", {})
        print(f"   Risk Audit -> 1D 95% VaR: {var_1d * 100:.2f}% | 1D CVaR: {cvar_1d * 100:.2f}% | Limit Exceeded: {conc.get('has_violation')}")

        # Backtest Performance
        bt = strat_data.get("backtest_performance", {})
        print(f"   Backtest   -> Cumulative: {bt.get('cumulative_return', 0.0) * 100:.2f}% | Max Drawdown: {bt.get('max_drawdown', 0.0) * 100:.2f}% | Sortino: {bt.get('sortino_ratio', 0.0):.2f}")

    # 3. AI-Informed Black-Litterman Allocation
    bl_data = results.get("black_litterman")
    if bl_data:
        print_section_header("AI-Informed Black-Litterman Allocation")
        print(">> Extracted AI Views:")
        for view in results.get("ai_views", []):
            print(f"   - {view['ticker']}: Return View={view['expected_excess_return'] * 100:+.2f}%, Conf={view['confidence'] * 100:.0f}%, Sentiment={view['sentiment']}")
            print(f"     Reasoning: {view['reasoning']}")

        print(f"\n>> Black-Litterman Posterior Weights:")
        bl_weight_str = ", ".join([f"{sym}: {wt * 100:.1f}%" for sym, wt in bl_data["weights"].items() if wt > 0.001])
        print(f"   Allocations: {bl_weight_str}")

        bl_risk = bl_data.get("risk_audit", {})
        bl_var = bl_risk.get("daily_value_at_risk", {}).get("historical_var_1d", 0.0)
        bl_cvar = bl_risk.get("daily_conditional_var_1d", 0.0)
        print(f"   Risk Audit -> 1D 95% VaR: {bl_var * 100:.2f}% | 1D CVaR: {bl_cvar * 100:.2f}%")

        bl_bt = bl_data.get("backtest_performance", {})
        print(f"   Backtest   -> Cumulative: {bl_bt.get('cumulative_return', 0.0) * 100:.2f}% | Max Drawdown: {bl_bt.get('max_drawdown', 0.0) * 100:.2f}% | Sortino: {bl_bt.get('sortino_ratio', 0.0):.2f}")

    # 4. Stress Test Scenario Drawdowns
    print_section_header("Macro Stress Testing (Simulated Crisis Drawdowns)")
    sample_risk = results["baseline_mpt"]["max_sharpe"].get("risk_audit", {}).get("stress_testing", {})
    drawdowns = sample_risk.get("simulated_scenario_drawdowns", {})
    for crisis_name, drop in drawdowns.items():
        clean_name = crisis_name.replace("_", " ")
        print_table_row(clean_name, f"{drop * 100:.2f}%")

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    run_cli()