"""Profit target optimizer for ORB strategy."""

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from loguru import logger

from src.backtest.engine import ORBBacktester, TradeResult
from src.strategy.orb import Direction, ORBSignal, ORBSignalGenerator


@dataclass
class OptimizationResult:
    """Result of a single optimization run with a specific target R."""

    target_r: float | None  # None means EOD-only exit
    total_return: float
    sharpe_ratio: float
    win_rate: float
    avg_r: float
    profit_factor: float
    max_drawdown: float
    total_trades: int

    @property
    def target_label(self) -> str:
        """Human-readable target label."""
        return "EOD" if self.target_r is None else f"{self.target_r}R"


class ORBOptimizer:
    """Optimizes ORB strategy profit targets across multiple R:R ratios."""

    def __init__(
        self,
        initial_capital: float = 25000.0,
        risk_per_trade: float = 0.01,
        max_leverage: float = 4.0,
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.max_leverage = max_leverage
        self._results: list[OptimizationResult] = []

    def _calculate_profit_factor(self, trades: list[TradeResult]) -> float:
        """Calculate profit factor: gross profits / gross losses."""
        gross_profits = sum(t.pnl for t in trades if t.pnl > 0)
        gross_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))

        if gross_losses == 0:
            return float("inf") if gross_profits > 0 else 0.0

        return gross_profits / gross_losses

    def _generate_signals_with_target(
        self, df: pd.DataFrame, symbol: str, target_r: float | None
    ) -> list[ORBSignal]:
        """Generate ORB signals with a specific target R multiple.

        Args:
            df: Historical intraday data.
            symbol: Stock symbol.
            target_r: Target R multiple (e.g., 2 for 2:1 R:R), or None for EOD exit.

        Returns:
            List of ORBSignal objects with the specified target.
        """
        # For EOD-only exit
        if target_r is None:
            generator = ORBSignalGenerator(symbol=symbol, use_eod_exit=True)
            return generator.generate_signals(df)

        # For specific target R, we need to modify the REWARD_RISK_RATIO
        generator = ORBSignalGenerator(symbol=symbol, use_eod_exit=False)
        generator.REWARD_RISK_RATIO = target_r
        return generator.generate_signals(df)

    def _run_single_optimization(
        self, df: pd.DataFrame, symbol: str, target_r: float | None
    ) -> OptimizationResult:
        """Run backtest for a single target R value.

        Args:
            df: Historical intraday data.
            symbol: Stock symbol.
            target_r: Target R multiple or None for EOD exit.

        Returns:
            OptimizationResult with all metrics.
        """
        # Create backtester
        use_eod = target_r is None
        backtester = ORBBacktester(
            initial_capital=self.initial_capital,
            risk_per_trade=self.risk_per_trade,
            max_leverage=self.max_leverage,
            use_eod_exit=use_eod,
        )

        # Generate signals with specific target
        signals = self._generate_signals_with_target(df, symbol, target_r)

        if not signals:
            return OptimizationResult(
                target_r=target_r,
                total_return=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                avg_r=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                total_trades=0,
            )

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

            result = backtester.simulate_trade(signal, day_data, account_value)

            if result:
                account_value += result.pnl
                trades.append(result)
                equity_data.append({"date": trade_date, "equity": account_value})

        if not trades:
            return OptimizationResult(
                target_r=target_r,
                total_return=0.0,
                sharpe_ratio=0.0,
                win_rate=0.0,
                avg_r=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                total_trades=0,
            )

        # Calculate metrics
        equity_curve = pd.DataFrame(equity_data)
        total_return = (account_value - self.initial_capital) / self.initial_capital

        # Win rate
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        win_rate = winning_trades / len(trades)

        # Average R
        avg_r = sum(t.pnl_r for t in trades) / len(trades)

        # Profit factor
        profit_factor = self._calculate_profit_factor(trades)

        # Sharpe ratio
        equity_curve["returns"] = equity_curve["equity"].pct_change()
        daily_returns = equity_curve["returns"].dropna()
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252**0.5)
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        equity_curve["peak"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] - equity_curve["peak"]) / equity_curve["peak"]
        max_drawdown = abs(equity_curve["drawdown"].min())

        return OptimizationResult(
            target_r=target_r,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            avg_r=avg_r,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            total_trades=len(trades),
        )

    def optimize_profit_target(
        self,
        df: pd.DataFrame,
        symbol: str,
        targets: list[float | None] | None = None,
    ) -> list[OptimizationResult]:
        """Test multiple profit target levels and return sorted results.

        Args:
            df: Historical intraday data.
            symbol: Stock symbol.
            targets: List of target R values to test. None means EOD-only exit.
                    Defaults to [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None].

        Returns:
            List of OptimizationResult sorted by Sharpe ratio (best first).
        """
        if targets is None:
            targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]

        logger.info(f"Running optimization for {symbol} with {len(targets)} target levels")

        results = []
        for target in targets:
            target_label = "EOD" if target is None else f"{target}R"
            logger.info(f"Testing target: {target_label}")

            result = self._run_single_optimization(df, symbol, target)
            results.append(result)

        # Sort by Sharpe ratio (descending)
        results.sort(key=lambda r: r.sharpe_ratio, reverse=True)
        self._results = results

        logger.success(f"Optimization complete. Best target: {results[0].target_label}")

        return results

    def find_optimal(self) -> OptimizationResult | None:
        """Return the target with the best Sharpe ratio.

        Returns:
            OptimizationResult with highest Sharpe ratio, or None if no results.
        """
        if not self._results:
            logger.warning("No optimization results available. Run optimize_profit_target first.")
            return None

        return self._results[0]

    def print_optimization_report(self, results: list[OptimizationResult] | None = None) -> None:
        """Display a comparison table of all optimization results.

        Args:
            results: List of OptimizationResult to display. Uses stored results if None.
        """
        if results is None:
            results = self._results

        if not results:
            print("No optimization results to display.")
            return

        # Header
        print("\n" + "=" * 100)
        print("ORB STRATEGY PROFIT TARGET OPTIMIZATION REPORT")
        print("=" * 100)

        # Column headers
        print(
            f"{'Target':>8} | {'Return':>10} | {'Sharpe':>8} | {'Win Rate':>10} | "
            f"{'Avg R':>8} | {'PF':>8} | {'Max DD':>10} | {'Trades':>8}"
        )
        print("-" * 100)

        # Data rows
        for r in results:
            pf_str = f"{r.profit_factor:.2f}" if r.profit_factor != float("inf") else "inf"
            print(
                f"{r.target_label:>8} | {r.total_return:>9.2%} | {r.sharpe_ratio:>8.2f} | "
                f"{r.win_rate:>9.2%} | {r.avg_r:>8.2f} | {pf_str:>8} | "
                f"{r.max_drawdown:>9.2%} | {r.total_trades:>8}"
            )

        print("=" * 100)

        # Best performer summary
        best = results[0]
        print(f"\nBEST PERFORMER: {best.target_label}")
        print(f"  Sharpe Ratio:  {best.sharpe_ratio:.2f}")
        print(f"  Total Return:  {best.total_return:.2%}")
        print(f"  Win Rate:      {best.win_rate:.2%}")
        print(f"  Profit Factor: {best.profit_factor:.2f}" if best.profit_factor != float("inf") else "  Profit Factor: inf")
        print(f"  Max Drawdown:  {best.max_drawdown:.2%}")
        print("=" * 100 + "\n")


def run_optimization(symbol: str, start_date: datetime, end_date: datetime) -> list[OptimizationResult]:
    """Fetch data and run profit target optimization.

    Args:
        symbol: Stock symbol to optimize.
        start_date: Start date for backtest data.
        end_date: End date for backtest data.

    Returns:
        List of OptimizationResult sorted by Sharpe ratio.
    """
    from src.data.fetcher import fetch_and_cache_data

    logger.info(f"Running optimization for {symbol} from {start_date.date()} to {end_date.date()}")

    df = fetch_and_cache_data(symbol, start_date, end_date)

    if df.empty:
        logger.error("No data fetched, cannot run optimization")
        return []

    optimizer = ORBOptimizer()
    results = optimizer.optimize_profit_target(df, symbol)
    optimizer.print_optimization_report()

    return results


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Try to load cached TSLA data
    DATA_DIR = Path(__file__).parent.parent.parent / "data"
    cached_files = list(DATA_DIR.glob("TSLA_*.csv"))

    if cached_files:
        # Use the most recent cached file
        cache_file = sorted(cached_files)[-1]
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["date"])
        logger.success(f"Loaded {len(df)} bars from cache")

        # Run optimization
        optimizer = ORBOptimizer()
        targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]

        results = optimizer.optimize_profit_target(df, "TSLA", targets)
        optimizer.print_optimization_report()

        # Show optimal
        optimal = optimizer.find_optimal()
        if optimal:
            print(f"Optimal profit target: {optimal.target_label} (Sharpe: {optimal.sharpe_ratio:.2f})")
    else:
        logger.error("No cached TSLA data found in data/ folder.")
        logger.info("Run the data fetcher first to cache some data:")
        logger.info("  python -m src.data.fetcher")
        sys.exit(1)
