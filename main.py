#!/usr/bin/env python3
"""ORB Trading Bot - Opening Range Breakout Strategy.

Entry point for backtesting, paper trading, and live trading.
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure loguru
logger.remove()  # Remove default handler

# Console handler - colorful output
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
    level="INFO",
    colorize=True,
)

# File handler - detailed logs
logger.add(
    LOGS_DIR / "orb.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ORB Trading Bot - Opening Range Breakout Strategy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run backtest for last 30 days
  python main.py --mode backtest --symbol TSLA --start 2026-01-01 --end 2026-01-25

  # Run paper trading
  python main.py --mode paper --symbol TSLA

  # Run live trading (requires confirmation)
  python main.py --mode live --symbol TSLA --capital 50000
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "paper", "live"],
        required=True,
        help="Trading mode: backtest, paper, or live",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        default=os.getenv("SYMBOL", "TSLA"),
        help=f"Stock symbol to trade (default: {os.getenv('SYMBOL', 'TSLA')})",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date for backtest (YYYY-MM-DD). Default: 30 days ago",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date for backtest (YYYY-MM-DD). Default: today",
    )

    parser.add_argument(
        "--capital",
        type=float,
        default=float(os.getenv("STARTING_CAPITAL", 25000)),
        help=f"Starting capital (default: ${os.getenv('STARTING_CAPITAL', 25000)})",
    )

    return parser.parse_args()


def run_backtest(symbol: str, start_date: datetime, end_date: datetime, capital: float) -> None:
    """Run backtest mode."""
    from src.backtest.engine import ORBBacktester, run_backtest as execute_backtest
    from src.data.fetcher import fetch_and_cache_data
    from src.utils.metrics import calculate_metrics, print_report

    logger.info("=" * 60)
    logger.info("ORB TRADING BOT - BACKTEST MODE")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")
    logger.info(f"Capital: ${capital:,.2f}")
    logger.info("=" * 60)

    # Fetch data
    logger.info("Fetching historical data...")
    df = fetch_and_cache_data(symbol, start_date, end_date)

    if df.empty:
        logger.error("No data fetched. Check IBKR connection and date range.")
        return

    logger.info(f"Loaded {len(df)} bars")

    # Run backtest
    logger.info("Running backtest...")
    backtester = ORBBacktester(initial_capital=capital)
    result = backtester.run(df, symbol)

    # Build trades DataFrame for metrics
    if result.trades:
        import pandas as pd

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
        metrics = calculate_metrics(trades_df, result.equity_curve, capital)
        print_report(metrics, f"ORB Backtest Results - {symbol}")

        # Print individual trades
        print("\nTrade Details:")
        print("-" * 80)
        for t in result.trades:
            pnl_color = "+" if t.pnl >= 0 else ""
            print(
                f"{t.date.date()} | {t.direction.value:5} | "
                f"Entry: ${t.entry_price:>7.2f} | Exit: ${t.exit_price:>7.2f} | "
                f"Shares: {t.shares:>4} | PnL: {pnl_color}${t.pnl:>8.2f} | "
                f"{t.pnl_r:>6.2f}R | {t.exit_reason}"
            )
        print("-" * 80)
    else:
        logger.warning("No trades were generated during the backtest period")


def run_paper_trading(symbol: str, capital: float) -> None:
    """Run paper trading mode."""
    from src.trading.executor import PaperTrader

    logger.info("=" * 60)
    logger.info("ORB TRADING BOT - PAPER TRADING MODE")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Capital: ${capital:,.2f}")
    logger.info("=" * 60)
    logger.info("Starting paper trader... Press Ctrl+C to stop")

    # Override capital in environment
    os.environ["STARTING_CAPITAL"] = str(capital)

    trader = PaperTrader(symbol=symbol, paper_mode=True)
    trader.start()


def run_live_trading(symbol: str, capital: float) -> None:
    """Run live trading mode with confirmation."""
    from src.trading.executor import PaperTrader

    logger.warning("=" * 60)
    logger.warning("ORB TRADING BOT - LIVE TRADING MODE")
    logger.warning("=" * 60)
    logger.warning(f"Symbol: {symbol}")
    logger.warning(f"Capital: ${capital:,.2f}")
    logger.warning("=" * 60)
    logger.warning("")
    logger.warning("WARNING: You are about to start LIVE TRADING with REAL MONEY!")
    logger.warning("This will place actual orders in your IBKR account.")
    logger.warning("")

    confirmation = input("Type 'YES' to confirm and start live trading: ")

    if confirmation.strip() != "YES":
        logger.info("Live trading cancelled.")
        return

    logger.info("Live trading confirmed. Starting...")
    logger.info("Press Ctrl+C to stop")

    # Override capital in environment
    os.environ["STARTING_CAPITAL"] = str(capital)

    trader = PaperTrader(symbol=symbol, paper_mode=False)
    trader.start()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    logger.info(f"ORB Trading Bot starting in {args.mode.upper()} mode")

    if args.mode == "backtest":
        # Parse dates
        if args.end:
            end_date = datetime.strptime(args.end, "%Y-%m-%d")
        else:
            end_date = datetime.now()

        if args.start:
            start_date = datetime.strptime(args.start, "%Y-%m-%d")
        else:
            start_date = end_date - timedelta(days=30)

        run_backtest(args.symbol, start_date, end_date, args.capital)

    elif args.mode == "paper":
        run_paper_trading(args.symbol, args.capital)

    elif args.mode == "live":
        run_live_trading(args.symbol, args.capital)


if __name__ == "__main__":
    main()
