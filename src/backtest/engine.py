"""Backtesting engine for ORB strategy."""

import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.strategy.orb import Direction, ORBSignal, ORBSignalGenerator

load_dotenv()


def get_symbols_from_env() -> list[str]:
    """Parse SYMBOLS from environment variable.

    Returns:
        List of symbols from SYMBOLS env var (comma-separated) or single SYMBOL.
    """
    symbols_str = os.getenv("SYMBOLS", "")
    if symbols_str:
        return [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

    # Fallback to single SYMBOL for backwards compatibility
    single = os.getenv("SYMBOL", "TSLA")
    return [single.upper()]

# Constants
COMMISSION_PER_SHARE = 0.0005  # Round trip commission per share
MARKET_CLOSE = time(16, 0)


@dataclass
class TradeResult:
    """Result of a single trade."""

    date: datetime
    symbol: str
    direction: Direction
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    pnl_r: float  # P&L in R-multiples
    exit_reason: str  # 'stop', 'target', 'eod'
    commission: float
    entry_time: datetime
    exit_time: datetime


@dataclass
class BacktestResult:
    """Complete backtest results."""

    trades: list[TradeResult] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_pnl_r: float = 0.0
    total_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0


@dataclass
class MultiSymbolBacktestResult:
    """Results from multi-symbol backtest."""

    symbol_results: dict[str, BacktestResult] = field(default_factory=dict)
    combined_trades: list[TradeResult] = field(default_factory=list)
    combined_equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    initial_capital: float = 0.0
    final_capital: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0


class ORBBacktester:
    """Backtests the Opening Range Breakout strategy."""

    def __init__(
        self,
        initial_capital: float = None,
        risk_per_trade: float = None,
        max_leverage: float = None,
        use_eod_exit: bool = False,
        profit_target_r: float = None,
    ):
        self.initial_capital = initial_capital or float(os.getenv("STARTING_CAPITAL", 25000))
        self.risk_per_trade = risk_per_trade or float(os.getenv("RISK_PER_TRADE", 0.01))
        self.max_leverage = max_leverage or float(os.getenv("MAX_LEVERAGE", 4))
        self.use_eod_exit = use_eod_exit
        self.profit_target_r = profit_target_r or float(os.getenv("PROFIT_TARGET_R", 10))

    def calculate_position_size(self, account_value: float, entry_price: float, risk_per_share: float) -> int:
        """Calculate position size based on risk and leverage constraints."""
        if risk_per_share <= 0 or entry_price <= 0:
            return 0

        risk_based_shares = account_value * self.risk_per_trade / risk_per_share
        leverage_based_shares = self.max_leverage * account_value / entry_price
        shares = int(min(risk_based_shares, leverage_based_shares))

        return max(shares, 0)

    def simulate_trade(self, signal: ORBSignal, day_data: pd.DataFrame, account_value: float) -> TradeResult | None:
        """Simulate a single trade through the trading day."""
        shares = self.calculate_position_size(account_value, signal.entry_price, signal.risk_per_share)

        if shares == 0:
            logger.warning(f"Position size is 0 for {signal.symbol} on {signal.date.date()}")
            return None

        day_data = day_data.copy()
        day_data["date"] = pd.to_datetime(day_data["date"])
        day_data["time"] = day_data["date"].dt.time
        day_data = day_data.sort_values("date")

        entry_bar_time = time(9, 35)
        entry_bar = day_data[day_data["time"] == entry_bar_time]

        if entry_bar.empty:
            logger.warning(f"Entry bar not found for {signal.date.date()}")
            return None

        entry_time = entry_bar.iloc[0]["date"]
        bars_after_entry = day_data[day_data["date"] > entry_time]

        exit_price = None
        exit_time = None
        exit_reason = None

        for _, bar in bars_after_entry.iterrows():
            bar_high = bar["high"]
            bar_low = bar["low"]
            bar_close = bar["close"]
            bar_time = bar["time"]
            bar_datetime = bar["date"]

            if signal.direction == Direction.LONG:
                if bar_low <= signal.stop_price:
                    exit_price = signal.stop_price
                    exit_time = bar_datetime
                    exit_reason = "stop"
                    break
                elif signal.target_price is not None and bar_high >= signal.target_price:
                    exit_price = signal.target_price
                    exit_time = bar_datetime
                    exit_reason = "target"
                    break
            else:
                if bar_high >= signal.stop_price:
                    exit_price = signal.stop_price
                    exit_time = bar_datetime
                    exit_reason = "stop"
                    break
                elif signal.target_price is not None and bar_low <= signal.target_price:
                    exit_price = signal.target_price
                    exit_time = bar_datetime
                    exit_reason = "target"
                    break

            if bar_time >= MARKET_CLOSE or bar_datetime == bars_after_entry.iloc[-1]["date"]:
                exit_price = bar_close
                exit_time = bar_datetime
                exit_reason = "eod"
                break

        if exit_price is None:
            last_bar = day_data.iloc[-1]
            exit_price = last_bar["close"]
            exit_time = last_bar["date"]
            exit_reason = "eod"

        commission = shares * COMMISSION_PER_SHARE * 2

        if signal.direction == Direction.LONG:
            gross_pnl = (exit_price - signal.entry_price) * shares
        else:
            gross_pnl = (signal.entry_price - exit_price) * shares

        pnl = gross_pnl - commission
        pnl_r = gross_pnl / (signal.risk_per_share * shares) if signal.risk_per_share > 0 else 0.0

        return TradeResult(
            date=signal.date,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_price=signal.entry_price,
            exit_price=exit_price,
            shares=shares,
            pnl=pnl,
            pnl_r=pnl_r,
            exit_reason=exit_reason,
            commission=commission,
            entry_time=entry_time,
            exit_time=exit_time,
        )

    def run(self, df: pd.DataFrame, symbol: str) -> BacktestResult:
        """Run backtest on historical data."""
        exit_mode = "EOD-only" if self.use_eod_exit else "Target/Stop"
        logger.info(f"Starting backtest for {symbol} with ${self.initial_capital:,.2f} initial capital ({exit_mode})")

        signal_generator = ORBSignalGenerator(
            symbol=symbol,
            use_eod_exit=self.use_eod_exit,
            profit_target_r=self.profit_target_r,
        )
        signals = signal_generator.generate_signals(df)

        if not signals:
            logger.warning("No signals generated")
            return BacktestResult(initial_capital=self.initial_capital, final_capital=self.initial_capital)

        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["trade_date"] = df["date"].dt.date

        account_value = self.initial_capital
        trades: list[TradeResult] = []
        equity_data: list[dict] = [{"date": df["trade_date"].min(), "equity": account_value}]

        for signal in signals:
            trade_date = signal.date.date()
            day_data = df[df["trade_date"] == trade_date]

            if day_data.empty:
                continue

            result = self.simulate_trade(signal, day_data, account_value)

            if result:
                account_value += result.pnl
                trades.append(result)
                equity_data.append({"date": trade_date, "equity": account_value})

                logger.debug(
                    f"{result.date.date()}: {result.direction.value} {result.shares} shares, "
                    f"PnL=${result.pnl:.2f} ({result.pnl_r:.2f}R), Exit={result.exit_reason}"
                )

        equity_curve = pd.DataFrame(equity_data)
        result = self._calculate_metrics(trades, equity_curve, self.initial_capital, account_value)

        return result

    def _calculate_metrics(
        self,
        trades: list[TradeResult],
        equity_curve: pd.DataFrame,
        initial_capital: float,
        final_capital: float,
    ) -> BacktestResult:
        """Calculate backtest performance metrics."""
        if not trades:
            return BacktestResult(initial_capital=initial_capital, final_capital=final_capital)

        total_trades = len(trades)
        long_trades = sum(1 for t in trades if t.direction == Direction.LONG)
        short_trades = sum(1 for t in trades if t.direction == Direction.SHORT)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        pnl_r_values = [t.pnl_r for t in trades]
        avg_pnl_r = np.mean(pnl_r_values) if pnl_r_values else 0.0

        total_return = (final_capital - initial_capital) / initial_capital

        if len(equity_curve) > 1:
            days = (equity_curve["date"].max() - equity_curve["date"].min()).days
            if days > 0:
                annualized_return = ((1 + total_return) ** (252 / days)) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0

        equity_curve = equity_curve.copy()
        equity_curve["returns"] = equity_curve["equity"].pct_change()
        daily_returns = equity_curve["returns"].dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["peak"]) / equity_curve["peak"]
        max_drawdown = abs(equity_curve["drawdown"].min())

        return BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            initial_capital=initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            avg_pnl_r=avg_pnl_r,
            total_trades=total_trades,
            long_trades=long_trades,
            short_trades=short_trades,
        )


def _find_cached_alpaca_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame | None:
    """Check for cached Alpaca data and filter to requested date range."""
    data_dir = Path(__file__).parent.parent.parent / "data"
    if not data_dir.exists():
        return None

    alpaca_files = list(data_dir.glob(f"{symbol}_*_alpaca.csv"))
    if not alpaca_files:
        logger.debug(f"No Alpaca cache files found for {symbol}")
        return None

    for cache_file in sorted(alpaca_files, reverse=True):
        logger.info(f"Checking Alpaca cache: {cache_file.name}")
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"])
            df = df[(df["date"].dt.date >= start_date.date()) & (df["date"].dt.date <= end_date.date())]

            if not df.empty:
                logger.success(f"Loaded {len(df)} bars from Alpaca cache ({cache_file.name})")
                return df
        except Exception as e:
            logger.warning(f"Failed to load {cache_file.name}: {e}")
            continue

    return None


def run_backtest(symbol: str, start_date: datetime, end_date: datetime) -> BacktestResult:
    """Fetch data and run backtest, printing results."""
    logger.info(f"Running backtest for {symbol} from {start_date.date()} to {end_date.date()}")

    df = _find_cached_alpaca_data(symbol, start_date, end_date)

    if df is None or df.empty:
        from src.data.fetcher import fetch_and_cache_data
        logger.info("No cached Alpaca data found, fetching from IBKR...")
        df = fetch_and_cache_data(symbol, start_date, end_date)

    if df.empty:
        logger.error("No data fetched, cannot run backtest")
        return BacktestResult()

    backtester = ORBBacktester()
    result = backtester.run(df, symbol)

    print("\n" + "=" * 60)
    print(f"BACKTEST RESULTS: {symbol}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 60)
    print(f"Initial Capital:    ${result.initial_capital:>12,.2f}")
    print(f"Final Capital:      ${result.final_capital:>12,.2f}")
    print(f"Total Return:       {result.total_return:>12.2%}")
    print(f"Annualized Return:  {result.annualized_return:>12.2%}")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:>12.2f}")
    print(f"Max Drawdown:       {result.max_drawdown:>12.2%}")
    print("-" * 60)
    print(f"Total Trades:       {result.total_trades:>12}")
    print(f"Long Trades:        {result.long_trades:>12}")
    print(f"Short Trades:       {result.short_trades:>12}")
    print(f"Win Rate:           {result.win_rate:>12.2%}")
    print(f"Avg P&L (R):        {result.avg_pnl_r:>12.2f}")
    print("=" * 60)

    return result


def run_multi_symbol_backtest(
    symbols: list[str],
    start_date: datetime,
    end_date: datetime,
    parallel: bool = True,
) -> MultiSymbolBacktestResult:
    """Run backtest for multiple symbols and produce comparison report."""
    logger.info(f"Running multi-symbol backtest for {symbols}")
    logger.info(f"Period: {start_date.date()} to {end_date.date()}")

    symbol_results: dict[str, BacktestResult] = {}

    if parallel:
        with ThreadPoolExecutor(max_workers=min(len(symbols), 4)) as executor:
            futures = {
                executor.submit(_run_single_backtest, sym, start_date, end_date): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    result = future.result()
                    symbol_results[symbol] = result
                except Exception as e:
                    logger.error(f"Backtest failed for {symbol}: {e}")
                    symbol_results[symbol] = BacktestResult()
    else:
        for symbol in symbols:
            try:
                result = _run_single_backtest(symbol, start_date, end_date)
                symbol_results[symbol] = result
            except Exception as e:
                logger.error(f"Backtest failed for {symbol}: {e}")
                symbol_results[symbol] = BacktestResult()

    combined = _combine_symbol_results(symbol_results)
    _print_comparison_report(symbol_results, combined, start_date, end_date)

    return combined


def _run_single_backtest(symbol: str, start_date: datetime, end_date: datetime) -> BacktestResult:
    """Run backtest for a single symbol without printing."""
    logger.info(f"Running backtest for {symbol}...")

    df = _find_cached_alpaca_data(symbol, start_date, end_date)

    if df is None or df.empty:
        from src.data.fetcher import fetch_and_cache_data
        logger.info(f"No cached Alpaca data for {symbol}, fetching from IBKR...")
        df = fetch_and_cache_data(symbol, start_date, end_date)

    if df.empty:
        logger.error(f"No data for {symbol}, skipping")
        return BacktestResult()

    backtester = ORBBacktester()
    result = backtester.run(df, symbol)

    logger.info(f"Completed {symbol}: {result.total_trades} trades, {result.total_return:.2%} return")

    return result


def _combine_symbol_results(symbol_results: dict[str, BacktestResult]) -> MultiSymbolBacktestResult:
    """Combine results from multiple symbols into a combined result."""
    all_trades = []
    for symbol, result in symbol_results.items():
        all_trades.extend(result.trades)

    all_trades.sort(key=lambda t: t.date)

    initial_capital = float(os.getenv("STARTING_CAPITAL", 25000))
    account_value = initial_capital
    equity_data = []

    if all_trades:
        current_date = all_trades[0].date.date()
        equity_data.append({"date": current_date, "equity": account_value})

        for trade in all_trades:
            account_value += trade.pnl
            trade_date = trade.date.date()
            if trade_date != current_date:
                current_date = trade_date
            equity_data.append({"date": current_date, "equity": account_value})

    equity_curve = pd.DataFrame(equity_data) if equity_data else pd.DataFrame()

    final_capital = account_value
    total_return = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0

    annualized_return = 0.0
    if len(equity_curve) > 1:
        days = (equity_curve["date"].max() - equity_curve["date"].min()).days
        if days > 0:
            annualized_return = ((1 + total_return) ** (252 / days)) - 1

    sharpe_ratio = 0.0
    if len(equity_curve) > 1:
        equity_curve = equity_curve.copy()
        equity_curve["returns"] = equity_curve["equity"].pct_change()
        daily_returns = equity_curve["returns"].dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)

    max_drawdown = 0.0
    if len(equity_curve) > 0:
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["peak"]) / equity_curve["peak"]
        max_drawdown = abs(equity_curve["drawdown"].min())

    return MultiSymbolBacktestResult(
        symbol_results=symbol_results,
        combined_trades=all_trades,
        combined_equity_curve=equity_curve,
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
    )


def _print_comparison_report(
    symbol_results: dict[str, BacktestResult],
    combined: MultiSymbolBacktestResult,
    start_date: datetime,
    end_date: datetime,
) -> None:
    """Print a comparison report for all symbols."""
    print("\n" + "=" * 100)
    print("MULTI-SYMBOL BACKTEST COMPARISON REPORT")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print("=" * 100)

    print(f"\n{'Symbol':<8} {'Trades':>8} {'Win Rate':>10} {'Total Ret':>12} {'Ann. Ret':>12} "
          f"{'Sharpe':>8} {'Max DD':>10} {'Avg R':>8} {'Final $':>14}")
    print("-" * 100)

    sorted_symbols = sorted(
        symbol_results.items(),
        key=lambda x: x[1].total_return,
        reverse=True,
    )

    for symbol, result in sorted_symbols:
        print(
            f"{symbol:<8} "
            f"{result.total_trades:>8} "
            f"{result.win_rate:>9.1%} "
            f"{result.total_return:>11.2%} "
            f"{result.annualized_return:>11.2%} "
            f"{result.sharpe_ratio:>8.2f} "
            f"{result.max_drawdown:>9.2%} "
            f"{result.avg_pnl_r:>8.2f} "
            f"${result.final_capital:>12,.2f}"
        )

    print("-" * 100)

    total_trades = sum(r.total_trades for r in symbol_results.values())
    winning_trades = sum(
        sum(1 for t in r.trades if t.pnl > 0)
        for r in symbol_results.values()
    )
    combined_win_rate = winning_trades / total_trades if total_trades > 0 else 0

    print(
        f"{'COMBINED':<8} "
        f"{total_trades:>8} "
        f"{combined_win_rate:>9.1%} "
        f"{combined.total_return:>11.2%} "
        f"{combined.annualized_return:>11.2%} "
        f"{combined.sharpe_ratio:>8.2f} "
        f"{combined.max_drawdown:>9.2%} "
        f"{'-':>8} "
        f"${combined.final_capital:>12,.2f}"
    )

    print("=" * 100)

    print("\nPERFORMANCE RANKING (by Total Return):")
    print("-" * 40)
    for i, (symbol, result) in enumerate(sorted_symbols, 1):
        medal = "#1" if i == 1 else "#2" if i == 2 else "#3" if i == 3 else f"#{i}"
        print(f"  {medal} {symbol}: {result.total_return:+.2%} ({result.total_trades} trades)")

    print("\nBEST PERFORMERS:")
    print("-" * 40)

    if symbol_results:
        best_return = max(symbol_results.items(), key=lambda x: x[1].total_return)
        best_sharpe = max(symbol_results.items(), key=lambda x: x[1].sharpe_ratio)
        best_winrate = max(symbol_results.items(), key=lambda x: x[1].win_rate)
        lowest_dd = min(symbol_results.items(), key=lambda x: x[1].max_drawdown)

        print(f"  Best Return:     {best_return[0]} ({best_return[1].total_return:+.2%})")
        print(f"  Best Sharpe:     {best_sharpe[0]} ({best_sharpe[1].sharpe_ratio:.2f})")
        print(f"  Best Win Rate:   {best_winrate[0]} ({best_winrate[1].win_rate:.1%})")
        print(f"  Lowest Drawdown: {lowest_dd[0]} ({lowest_dd[1].max_drawdown:.2%})")

    print("=" * 100)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ORB Strategy Backtester")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides SYMBOLS env var)",
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run single-symbol backtest (first symbol only)",
    )
    args = parser.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = get_symbols_from_env()

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        start_date = datetime(2022, 1, 1)

    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    else:
        end_date = datetime.now()

    if args.single or len(symbols) == 1:
        run_backtest(symbols[0], start_date, end_date)
    else:
        run_multi_symbol_backtest(symbols, start_date, end_date)
