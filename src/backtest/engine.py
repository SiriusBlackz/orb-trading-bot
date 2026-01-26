"""Backtesting engine for ORB strategy."""

import os
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger

from src.strategy.orb import Direction, ORBSignal, ORBSignalGenerator

load_dotenv()

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
        """Calculate position size based on risk and leverage constraints.

        Formula: shares = int(min(account * risk_pct / risk_per_share, max_leverage * account / price))

        Args:
            account_value: Current account value.
            entry_price: Entry price per share.
            risk_per_share: Dollar risk per share (entry - stop).

        Returns:
            Number of shares to trade.
        """
        if risk_per_share <= 0 or entry_price <= 0:
            return 0

        # Risk-based position size
        risk_based_shares = account_value * self.risk_per_trade / risk_per_share

        # Leverage-based position size
        leverage_based_shares = self.max_leverage * account_value / entry_price

        shares = int(min(risk_based_shares, leverage_based_shares))

        return max(shares, 0)

    def simulate_trade(self, signal: ORBSignal, day_data: pd.DataFrame, account_value: float) -> TradeResult | None:
        """Simulate a single trade through the trading day.

        Walks through bars after entry checking for stop, target, or end-of-day exit.

        Args:
            signal: ORB signal with entry/stop/target prices.
            day_data: Intraday data for the trading day.
            account_value: Current account value for position sizing.

        Returns:
            TradeResult if trade was executed, None otherwise.
        """
        shares = self.calculate_position_size(account_value, signal.entry_price, signal.risk_per_share)

        if shares == 0:
            logger.warning(f"Position size is 0 for {signal.symbol} on {signal.date.date()}")
            return None

        day_data = day_data.copy()
        day_data["date"] = pd.to_datetime(day_data["date"])
        day_data["time"] = day_data["date"].dt.time
        day_data = day_data.sort_values("date")

        # Find entry bar (9:35) and get bars after entry
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

        # Walk through each bar after entry
        for _, bar in bars_after_entry.iterrows():
            bar_high = bar["high"]
            bar_low = bar["low"]
            bar_close = bar["close"]
            bar_time = bar["time"]
            bar_datetime = bar["date"]

            if signal.direction == Direction.LONG:
                # Check stop first (assumes stop hit before target on same bar)
                if bar_low <= signal.stop_price:
                    exit_price = signal.stop_price
                    exit_time = bar_datetime
                    exit_reason = "stop"
                    break
                # Check target (skip if use_eod_exit)
                elif signal.target_price is not None and bar_high >= signal.target_price:
                    exit_price = signal.target_price
                    exit_time = bar_datetime
                    exit_reason = "target"
                    break
            else:  # SHORT
                # Check stop first
                if bar_high >= signal.stop_price:
                    exit_price = signal.stop_price
                    exit_time = bar_datetime
                    exit_reason = "stop"
                    break
                # Check target (skip if use_eod_exit)
                elif signal.target_price is not None and bar_low <= signal.target_price:
                    exit_price = signal.target_price
                    exit_time = bar_datetime
                    exit_reason = "target"
                    break

            # Check for end of day
            if bar_time >= MARKET_CLOSE or bar_datetime == bars_after_entry.iloc[-1]["date"]:
                exit_price = bar_close
                exit_time = bar_datetime
                exit_reason = "eod"
                break

        if exit_price is None:
            # Fallback to last bar close
            last_bar = day_data.iloc[-1]
            exit_price = last_bar["close"]
            exit_time = last_bar["date"]
            exit_reason = "eod"

        # Calculate P&L
        commission = shares * COMMISSION_PER_SHARE * 2  # Round trip

        if signal.direction == Direction.LONG:
            gross_pnl = (exit_price - signal.entry_price) * shares
        else:  # SHORT
            gross_pnl = (signal.entry_price - exit_price) * shares

        pnl = gross_pnl - commission

        # Calculate R-multiple
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
        """Run backtest on historical data.

        Generates signals and simulates all trades, tracking equity curve.

        Args:
            df: Historical intraday data.
            symbol: Stock symbol.

        Returns:
            BacktestResult with all metrics.
        """
        exit_mode = "EOD-only" if self.use_eod_exit else "Target/Stop"
        logger.info(f"Starting backtest for {symbol} with ${self.initial_capital:,.2f} initial capital ({exit_mode})")

        # Generate signals
        signal_generator = ORBSignalGenerator(
            symbol=symbol,
            use_eod_exit=self.use_eod_exit,
            profit_target_r=self.profit_target_r,
        )
        signals = signal_generator.generate_signals(df)

        if not signals:
            logger.warning("No signals generated")
            return BacktestResult(initial_capital=self.initial_capital, final_capital=self.initial_capital)

        # Prepare data
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["trade_date"] = df["date"].dt.date

        # Track equity
        account_value = self.initial_capital
        trades: list[TradeResult] = []
        equity_data: list[dict] = [{"date": df["trade_date"].min(), "equity": account_value}]

        # Simulate each signal
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

        # Build equity curve
        equity_curve = pd.DataFrame(equity_data)

        # Calculate metrics
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
            return BacktestResult(
                initial_capital=initial_capital,
                final_capital=final_capital,
            )

        # Basic metrics
        total_trades = len(trades)
        long_trades = sum(1 for t in trades if t.direction == Direction.LONG)
        short_trades = sum(1 for t in trades if t.direction == Direction.SHORT)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # P&L metrics
        pnl_r_values = [t.pnl_r for t in trades]
        avg_pnl_r = np.mean(pnl_r_values) if pnl_r_values else 0.0

        # Return metrics
        total_return = (final_capital - initial_capital) / initial_capital

        # Annualized return (assuming 252 trading days)
        if len(equity_curve) > 1:
            days = (equity_curve["date"].max() - equity_curve["date"].min()).days
            if days > 0:
                annualized_return = ((1 + total_return) ** (252 / days)) - 1
            else:
                annualized_return = 0.0
        else:
            annualized_return = 0.0

        # Sharpe ratio (daily returns)
        equity_curve = equity_curve.copy()
        equity_curve["returns"] = equity_curve["equity"].pct_change()
        daily_returns = equity_curve["returns"].dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
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
    """Check for cached Alpaca data and filter to requested date range.

    Args:
        symbol: Stock symbol.
        start_date: Requested start date.
        end_date: Requested end date.

    Returns:
        DataFrame if cached data found with data in range, None otherwise.
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    if not data_dir.exists():
        return None

    # Look for Alpaca cache files for this symbol
    alpaca_files = list(data_dir.glob(f"{symbol}_*_alpaca.csv"))
    if not alpaca_files:
        logger.debug(f"No Alpaca cache files found for {symbol}")
        return None

    # Sort by filename (newest date range first)
    for cache_file in sorted(alpaca_files, reverse=True):
        logger.info(f"Checking Alpaca cache: {cache_file.name}")
        try:
            df = pd.read_csv(cache_file, parse_dates=["date"])

            # Filter to requested date range
            df = df[(df["date"].dt.date >= start_date.date()) & (df["date"].dt.date <= end_date.date())]

            if not df.empty:
                logger.success(f"Loaded {len(df)} bars from Alpaca cache ({cache_file.name})")
                return df
        except Exception as e:
            logger.warning(f"Failed to load {cache_file.name}: {e}")
            continue

    return None


def run_backtest(symbol: str, start_date: datetime, end_date: datetime) -> BacktestResult:
    """Fetch data and run backtest, printing results.

    Checks for cached Alpaca data first, then falls back to IBKR.

    Args:
        symbol: Stock symbol to backtest.
        start_date: Start date for backtest.
        end_date: End date for backtest.

    Returns:
        BacktestResult with all metrics.
    """
    logger.info(f"Running backtest for {symbol} from {start_date.date()} to {end_date.date()}")

    # First, check for cached Alpaca data
    df = _find_cached_alpaca_data(symbol, start_date, end_date)

    # Fall back to IBKR if no Alpaca cache found
    if df is None or df.empty:
        from src.data.fetcher import fetch_and_cache_data
        logger.info("No cached Alpaca data found, fetching from IBKR...")
        df = fetch_and_cache_data(symbol, start_date, end_date)

    if df.empty:
        logger.error("No data fetched, cannot run backtest")
        return BacktestResult()

    # Run backtest
    backtester = ORBBacktester()
    result = backtester.run(df, symbol)

    # Print results
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


if __name__ == "__main__":
    # Test with synthetic data
    logger.info("Testing ORBBacktester with synthetic data")

    # Create synthetic 5-minute data for multiple days
    def create_day_data(base_date: datetime, scenario: str) -> list[dict]:
        """Create synthetic intraday data for testing."""
        bars = []
        # Market hours: 9:30 AM to 4:00 PM (78 five-minute bars)
        current_time = base_date.replace(hour=9, minute=30, second=0, microsecond=0)
        end_time = base_date.replace(hour=16, minute=0, second=0, microsecond=0)

        if scenario == "long_win":
            # Bullish first candle, price goes to target
            price = 250.0
            bars.append({"date": current_time, "open": 250.0, "high": 252.0, "low": 249.0, "close": 251.5, "volume": 100000})
            current_time += timedelta(minutes=5)
            bars.append({"date": current_time, "open": 251.5, "high": 252.5, "low": 251.0, "close": 252.0, "volume": 80000})
            current_time += timedelta(minutes=5)
            # Entry at 252.0, stop at 249.0, risk = 3.0, target = 252 + 30 = 282
            price = 252.0
            while current_time <= end_time:
                price += 2.0  # Trending up
                bars.append({
                    "date": current_time,
                    "open": price - 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 50000,
                })
                current_time += timedelta(minutes=5)
                if price >= 282:
                    break
            # Fill rest of day
            while current_time <= end_time:
                bars.append({
                    "date": current_time,
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 30000,
                })
                current_time += timedelta(minutes=5)

        elif scenario == "long_stop":
            # Bullish first candle, price drops to stop
            bars.append({"date": current_time, "open": 250.0, "high": 252.0, "low": 249.0, "close": 251.5, "volume": 100000})
            current_time += timedelta(minutes=5)
            bars.append({"date": current_time, "open": 251.5, "high": 252.5, "low": 251.0, "close": 252.0, "volume": 80000})
            current_time += timedelta(minutes=5)
            # Entry at 252.0, stop at 249.0
            price = 252.0
            while current_time <= end_time:
                price -= 1.5  # Trending down
                bars.append({
                    "date": current_time,
                    "open": price + 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 50000,
                })
                current_time += timedelta(minutes=5)
                if price <= 249:
                    break
            # Fill rest of day
            while current_time <= end_time:
                bars.append({
                    "date": current_time,
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 30000,
                })
                current_time += timedelta(minutes=5)

        elif scenario == "short_win":
            # Bearish first candle, price drops to target
            bars.append({"date": current_time, "open": 260.0, "high": 261.0, "low": 257.0, "close": 258.0, "volume": 100000})
            current_time += timedelta(minutes=5)
            bars.append({"date": current_time, "open": 258.0, "high": 258.5, "low": 257.0, "close": 257.5, "volume": 80000})
            current_time += timedelta(minutes=5)
            # Entry at 257.5, stop at 261.0, risk = 3.5, target = 257.5 - 35 = 222.5
            price = 257.5
            while current_time <= end_time:
                price -= 2.5  # Trending down
                bars.append({
                    "date": current_time,
                    "open": price + 0.5,
                    "high": price + 1.0,
                    "low": price - 1.0,
                    "close": price,
                    "volume": 50000,
                })
                current_time += timedelta(minutes=5)
                if price <= 222.5:
                    break
            # Fill rest of day
            while current_time <= end_time:
                bars.append({
                    "date": current_time,
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                    "volume": 30000,
                })
                current_time += timedelta(minutes=5)

        elif scenario == "eod_exit":
            # Bullish candle, but price chops around until EOD
            bars.append({"date": current_time, "open": 250.0, "high": 252.0, "low": 249.0, "close": 251.5, "volume": 100000})
            current_time += timedelta(minutes=5)
            bars.append({"date": current_time, "open": 251.5, "high": 252.5, "low": 251.0, "close": 252.0, "volume": 80000})
            current_time += timedelta(minutes=5)
            # Entry at 252.0, stop at 249.0, target at 282.0
            # Price oscillates between stop and target
            price = 252.0
            direction = 1
            while current_time <= end_time:
                price += direction * 0.5
                if price > 255:
                    direction = -1
                elif price < 250:
                    direction = 1
                bars.append({
                    "date": current_time,
                    "open": price - direction * 0.2,
                    "high": price + 0.3,
                    "low": price - 0.3,
                    "close": price,
                    "volume": 40000,
                })
                current_time += timedelta(minutes=5)

        return bars

    # Create multi-day synthetic data
    all_bars = []
    scenarios = [
        (datetime(2024, 1, 15), "long_win"),
        (datetime(2024, 1, 16), "long_stop"),
        (datetime(2024, 1, 17), "short_win"),
        (datetime(2024, 1, 18), "eod_exit"),
    ]

    for base_date, scenario in scenarios:
        all_bars.extend(create_day_data(base_date, scenario))

    df = pd.DataFrame(all_bars)
    logger.info(f"Created synthetic data with {len(df)} bars across {len(scenarios)} days")

    # Run backtest
    backtester = ORBBacktester(initial_capital=25000)
    result = backtester.run(df, symbol="TSLA")

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS (Synthetic Data)")
    print("=" * 60)
    print(f"Initial Capital:    ${result.initial_capital:>12,.2f}")
    print(f"Final Capital:      ${result.final_capital:>12,.2f}")
    print(f"Total Return:       {result.total_return:>12.2%}")
    print(f"Sharpe Ratio:       {result.sharpe_ratio:>12.2f}")
    print(f"Max Drawdown:       {result.max_drawdown:>12.2%}")
    print("-" * 60)
    print(f"Total Trades:       {result.total_trades:>12}")
    print(f"Long Trades:        {result.long_trades:>12}")
    print(f"Short Trades:       {result.short_trades:>12}")
    print(f"Win Rate:           {result.win_rate:>12.2%}")
    print(f"Avg P&L (R):        {result.avg_pnl_r:>12.2f}")
    print("=" * 60)

    # Print individual trades
    print("\nTrade Details:")
    print("-" * 60)
    for trade in result.trades:
        print(
            f"{trade.date.date()} | {trade.direction.value:5} | "
            f"Entry: ${trade.entry_price:.2f} | Exit: ${trade.exit_price:.2f} | "
            f"Shares: {trade.shares:4} | PnL: ${trade.pnl:>8.2f} | "
            f"{trade.pnl_r:>5.2f}R | {trade.exit_reason}"
        )

    # Verify results
    print("\n" + "=" * 60)
    print("Verification:")
    print("=" * 60)
    assert result.total_trades == 4, f"Expected 4 trades, got {result.total_trades}"
    assert result.long_trades == 3, f"Expected 3 long trades, got {result.long_trades}"
    assert result.short_trades == 1, f"Expected 1 short trade, got {result.short_trades}"

    # Check exit reasons
    exit_reasons = [t.exit_reason for t in result.trades]
    assert "target" in exit_reasons, "Expected at least one target exit"
    assert "stop" in exit_reasons, "Expected at least one stop exit"
    assert "eod" in exit_reasons, "Expected at least one EOD exit"

    logger.success("All tests passed!")
