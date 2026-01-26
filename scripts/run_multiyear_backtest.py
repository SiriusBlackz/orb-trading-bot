#!/usr/bin/env python3
"""Run multi-year backtest by fetching data in chunks."""

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


def main():
    print("=" * 60)
    print("ORB TRADING BOT - MULTI-YEAR BACKTEST")
    print("=" * 60)

    symbol = "TSLA"
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2026, 1, 25)
    initial_capital = 25000.0

    print(f"Symbol: {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Capital: ${initial_capital:,.2f}")
    print("=" * 60)

    # Fetch data in yearly chunks
    fetcher = IBKRDataFetcher()
    if not fetcher.connect():
        print("Failed to connect to IBKR")
        sys.exit(1)

    all_data = []
    current_start = start_date

    print("\nFetching historical data in chunks...")
    while current_start < end_date:
        chunk_end = min(current_start + timedelta(days=364), end_date)
        duration_days = (chunk_end - current_start).days + 1

        print(f"  Fetching {current_start.date()} to {chunk_end.date()} ({duration_days} days)...")

        df = fetcher.fetch_historical_bars(
            symbol=symbol,
            duration=f"{duration_days} D",
            bar_size="5 mins",
            end_date=chunk_end,
            use_rth=True,
        )

        if not df.empty:
            all_data.append(df)
            print(f"    Got {len(df)} bars")

        current_start = chunk_end + timedelta(days=1)

    fetcher.disconnect()

    if not all_data:
        print("No data fetched!")
        sys.exit(1)

    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"\nTotal bars: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Run backtest
    print("\nRunning backtest...")
    backtester = ORBBacktester(initial_capital=initial_capital)
    result = backtester.run(df, symbol)

    # Build trades DataFrame for metrics
    if result.trades:
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

        # Calculate and print detailed metrics
        metrics = calculate_metrics(trades_df, result.equity_curve, initial_capital)
        print_report(metrics, f"ORB Strategy Results - {symbol} (3 Years)")

        # Print trade details
        print("\nTrade Details (showing first 20 and last 20):")
        print("-" * 100)

        trades = result.trades
        if len(trades) > 40:
            show_trades = list(enumerate(trades[:20])) + [(-1, None)] + list(enumerate(trades[-20:], len(trades) - 20))
        else:
            show_trades = list(enumerate(trades))

        for i, t in show_trades:
            if t is None:
                print(f"... ({len(trades) - 40} more trades) ...")
                continue
            pnl_sign = "+" if t.pnl >= 0 else ""
            print(
                f"{t.date.date()} | {t.direction.value:5} | "
                f"Entry: ${t.entry_price:>7.2f} | Exit: ${t.exit_price:>7.2f} | "
                f"Shares: {t.shares:>4} | PnL: {pnl_sign}${t.pnl:>9.2f} | "
                f"{t.pnl_r:>6.2f}R | {t.exit_reason}"
            )
        print("-" * 100)

        # Summary stats
        print("\nAdditional Statistics:")
        print("-" * 60)

        # By year
        trades_df["year"] = pd.to_datetime(trades_df["date"]).dt.year
        print("\nPerformance by Year:")
        for year in sorted(trades_df["year"].unique()):
            year_trades = trades_df[trades_df["year"] == year]
            year_pnl = year_trades["pnl"].sum()
            year_wins = (year_trades["pnl"] > 0).sum()
            year_total = len(year_trades)
            print(f"  {year}: {year_total:3} trades, Win Rate: {year_wins/year_total:.1%}, P&L: ${year_pnl:>10,.2f}")

        # By direction
        print("\nPerformance by Direction:")
        for direction in ["LONG", "SHORT"]:
            dir_trades = trades_df[trades_df["direction"] == direction]
            if len(dir_trades) > 0:
                dir_pnl = dir_trades["pnl"].sum()
                dir_wins = (dir_trades["pnl"] > 0).sum()
                dir_total = len(dir_trades)
                print(f"  {direction:5}: {dir_total:3} trades, Win Rate: {dir_wins/dir_total:.1%}, P&L: ${dir_pnl:>10,.2f}")

        # By exit reason
        print("\nPerformance by Exit Reason:")
        for reason in ["target", "stop", "eod"]:
            reason_trades = trades_df[trades_df["exit_reason"] == reason]
            if len(reason_trades) > 0:
                reason_pnl = reason_trades["pnl"].sum()
                reason_total = len(reason_trades)
                print(f"  {reason:6}: {reason_total:3} trades, P&L: ${reason_pnl:>10,.2f}")

    else:
        print("No trades were generated during the backtest period")


if __name__ == "__main__":
    main()
