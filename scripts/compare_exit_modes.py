#!/usr/bin/env python3
"""Compare ORB strategy with target exits vs EOD-only exits."""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.data.fetcher import IBKRDataFetcher
from src.backtest.engine import ORBBacktester
from src.utils.metrics import calculate_metrics, print_report


def fetch_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch data in chunks."""
    fetcher = IBKRDataFetcher()
    if not fetcher.connect():
        print("Failed to connect to IBKR")
        sys.exit(1)

    all_data = []
    current_start = start_date

    print("Fetching historical data...")
    while current_start < end_date:
        chunk_end = min(current_start + timedelta(days=30), end_date)
        duration_days = (chunk_end - current_start).days + 1

        df = fetcher.fetch_historical_bars(
            symbol=symbol,
            duration=f"{duration_days} D",
            bar_size="5 mins",
            end_date=chunk_end,
            use_rth=True,
        )

        if not df.empty:
            all_data.append(df)

        current_start = chunk_end + timedelta(days=1)

    fetcher.disconnect()

    if not all_data:
        return pd.DataFrame()

    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df


def run_backtest_mode(df: pd.DataFrame, symbol: str, initial_capital: float, use_eod_exit: bool) -> dict:
    """Run backtest and return metrics."""
    backtester = ORBBacktester(initial_capital=initial_capital, use_eod_exit=use_eod_exit)
    result = backtester.run(df, symbol)

    if not result.trades:
        return None

    trades_data = []
    for t in result.trades:
        trades_data.append({
            "date": t.date,
            "pnl": t.pnl,
            "pnl_r": t.pnl_r,
            "direction": t.direction.value,
            "exit_reason": t.exit_reason,
            "commission": t.commission,
        })
    trades_df = pd.DataFrame(trades_data)

    metrics = calculate_metrics(trades_df, result.equity_curve, initial_capital)
    metrics["trades"] = result.trades
    return metrics


def main():
    print("=" * 70)
    print("ORB STRATEGY COMPARISON: Target Exits vs EOD-Only Exits")
    print("=" * 70)

    symbol = "TSLA"
    start_date = datetime(2025, 12, 1)
    end_date = datetime(2026, 1, 25)
    initial_capital = 25000.0

    print(f"\nSymbol: {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Capital: ${initial_capital:,.2f}")

    # Fetch data once
    df = fetch_data(symbol, start_date, end_date)
    if df.empty:
        print("No data fetched!")
        return

    print(f"Loaded {len(df)} bars")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Run both backtests
    print("\n" + "=" * 70)
    print("BACKTEST 1: Standard Mode (10R Target)")
    print("=" * 70)
    metrics_target = run_backtest_mode(df, symbol, initial_capital, use_eod_exit=False)

    print("\n" + "=" * 70)
    print("BACKTEST 2: EOD-Only Mode (No Target, Exit at EOD or Stop)")
    print("=" * 70)
    metrics_eod = run_backtest_mode(df, symbol, initial_capital, use_eod_exit=True)

    if not metrics_target or not metrics_eod:
        print("Error running backtests")
        return

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    comparison = [
        ("Total Trades", metrics_target["total_trades"], metrics_eod["total_trades"]),
        ("Winning Trades", metrics_target["winning_trades"], metrics_eod["winning_trades"]),
        ("Win Rate", f"{metrics_target['win_rate']:.1%}", f"{metrics_eod['win_rate']:.1%}"),
        ("", "", ""),
        ("Total P&L", f"${metrics_target['total_pnl']:,.2f}", f"${metrics_eod['total_pnl']:,.2f}"),
        ("Total Return", f"{metrics_target['total_return']:.2%}", f"{metrics_eod['total_return']:.2%}"),
        ("Final Capital", f"${metrics_target['final_capital']:,.2f}", f"${metrics_eod['final_capital']:,.2f}"),
        ("", "", ""),
        ("Avg P&L/Trade", f"${metrics_target['avg_pnl']:,.2f}", f"${metrics_eod['avg_pnl']:,.2f}"),
        ("Avg Win", f"${metrics_target['avg_win']:,.2f}", f"${metrics_eod['avg_win']:,.2f}"),
        ("Avg Loss", f"${metrics_target['avg_loss']:,.2f}", f"${metrics_eod['avg_loss']:,.2f}"),
        ("Profit Factor", f"{metrics_target['profit_factor']:.2f}", f"{metrics_eod['profit_factor']:.2f}"),
        ("", "", ""),
        ("Avg R", f"{metrics_target['avg_r']:.2f}", f"{metrics_eod['avg_r']:.2f}"),
        ("Best Trade (R)", f"{metrics_target['max_r']:.2f}", f"{metrics_eod['max_r']:.2f}"),
        ("Worst Trade (R)", f"{metrics_target['min_r']:.2f}", f"{metrics_eod['min_r']:.2f}"),
        ("", "", ""),
        ("Sharpe Ratio", f"{metrics_target['sharpe_ratio']:.2f}", f"{metrics_eod['sharpe_ratio']:.2f}"),
        ("Max Drawdown", f"{metrics_target['max_drawdown']:.2%}", f"{metrics_eod['max_drawdown']:.2%}"),
        ("", "", ""),
        ("Stop Exits", metrics_target["stop_exits"], metrics_eod["stop_exits"]),
        ("Target Exits", metrics_target["target_exits"], metrics_eod["target_exits"]),
        ("EOD Exits", metrics_target["eod_exits"], metrics_eod["eod_exits"]),
    ]

    print(f"\n{'Metric':<20} {'10R Target':>18} {'EOD-Only':>18} {'Difference':>15}")
    print("-" * 75)

    for metric, val1, val2 in comparison:
        if metric == "":
            print()
            continue

        # Calculate difference for numeric values
        diff = ""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            d = val2 - val1
            if isinstance(val1, int):
                diff = f"{d:+d}"
            else:
                diff = f"{d:+.2f}"
        elif isinstance(val1, str) and val1.startswith("$"):
            try:
                v1 = float(val1.replace("$", "").replace(",", ""))
                v2 = float(val2.replace("$", "").replace(",", ""))
                diff = f"${v2 - v1:+,.2f}"
            except:
                pass
        elif isinstance(val1, str) and val1.endswith("%"):
            try:
                v1 = float(val1.replace("%", ""))
                v2 = float(val2.replace("%", ""))
                diff = f"{v2 - v1:+.1f}%"
            except:
                pass

        print(f"{metric:<20} {str(val1):>18} {str(val2):>18} {diff:>15}")

    # Print conclusion
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    pnl_diff = metrics_eod["total_pnl"] - metrics_target["total_pnl"]
    wr_diff = metrics_eod["win_rate"] - metrics_target["win_rate"]

    if pnl_diff > 0:
        print(f"\nEOD-Only mode outperformed by ${pnl_diff:,.2f}")
    else:
        print(f"\n10R Target mode outperformed by ${-pnl_diff:,.2f}")

    print(f"Win rate difference: {wr_diff:+.1%}")

    print("\nKey insights:")
    if metrics_target["target_exits"] == 0:
        print("- No trades hit the 10R target (too ambitious for this timeframe)")
    if metrics_eod["win_rate"] > metrics_target["win_rate"]:
        print("- EOD-Only has higher win rate (winners aren't stopped out chasing target)")
    if metrics_eod["avg_pnl"] > metrics_target["avg_pnl"]:
        print("- EOD-Only has higher average P&L per trade")
    if metrics_eod["max_drawdown"] < metrics_target["max_drawdown"]:
        print("- EOD-Only has lower max drawdown")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
