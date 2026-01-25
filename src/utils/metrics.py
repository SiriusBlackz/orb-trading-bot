"""Performance metrics and reporting utilities."""

import numpy as np
import pandas as pd
from loguru import logger


def calculate_metrics(trades_df: pd.DataFrame, equity_df: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate comprehensive performance metrics.

    Args:
        trades_df: DataFrame with trade results. Expected columns:
            - pnl: Profit/loss per trade
            - pnl_r: R-multiple per trade
            - direction: 'LONG' or 'SHORT'
            - exit_reason: 'stop', 'target', or 'eod'
            - commission: Commission per trade
        equity_df: DataFrame with equity curve. Expected columns:
            - date: Date
            - equity: Account equity
        initial_capital: Starting capital.

    Returns:
        Dictionary with all performance metrics.
    """
    metrics = {}

    # Handle empty trades
    if trades_df.empty:
        return _empty_metrics(initial_capital)

    # ===== Trade Counts =====
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df["pnl"] > 0])
    losing_trades = len(trades_df[trades_df["pnl"] < 0])
    breakeven_trades = len(trades_df[trades_df["pnl"] == 0])

    metrics["total_trades"] = total_trades
    metrics["winning_trades"] = winning_trades
    metrics["losing_trades"] = losing_trades
    metrics["win_rate"] = winning_trades / total_trades if total_trades > 0 else 0.0

    # ===== Direction Breakdown =====
    metrics["long_trades"] = len(trades_df[trades_df["direction"] == "LONG"])
    metrics["short_trades"] = len(trades_df[trades_df["direction"] == "SHORT"])

    # ===== P&L Metrics =====
    total_pnl = trades_df["pnl"].sum()
    metrics["total_pnl"] = total_pnl
    metrics["avg_pnl"] = trades_df["pnl"].mean()

    winners = trades_df[trades_df["pnl"] > 0]["pnl"]
    losers = trades_df[trades_df["pnl"] < 0]["pnl"]

    metrics["avg_win"] = winners.mean() if len(winners) > 0 else 0.0
    metrics["avg_loss"] = losers.mean() if len(losers) > 0 else 0.0

    # ===== R-Multiple Metrics =====
    metrics["avg_r"] = trades_df["pnl_r"].mean()
    metrics["max_r"] = trades_df["pnl_r"].max()
    metrics["min_r"] = trades_df["pnl_r"].min()

    # ===== Profit Factor =====
    gross_profit = winners.sum() if len(winners) > 0 else 0.0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 0.0
    metrics["profit_factor"] = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # ===== Capital & Returns =====
    final_capital = initial_capital + total_pnl
    metrics["initial_capital"] = initial_capital
    metrics["final_capital"] = final_capital
    metrics["total_return"] = (final_capital - initial_capital) / initial_capital

    # Annualized return
    if len(equity_df) > 1:
        equity_df = equity_df.copy()
        equity_df["date"] = pd.to_datetime(equity_df["date"])
        days = (equity_df["date"].max() - equity_df["date"].min()).days
        if days > 0:
            metrics["annualized_return"] = ((1 + metrics["total_return"]) ** (252 / days)) - 1
        else:
            metrics["annualized_return"] = 0.0
    else:
        metrics["annualized_return"] = 0.0

    # ===== Risk Metrics =====
    # Sharpe ratio
    equity_df = equity_df.copy()
    equity_df["returns"] = equity_df["equity"].pct_change()
    daily_returns = equity_df["returns"].dropna()

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        metrics["sharpe_ratio"] = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        metrics["sharpe_ratio"] = 0.0

    # Max drawdown
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = (equity_df["equity"] - equity_df["peak"]) / equity_df["peak"]
    metrics["max_drawdown"] = abs(equity_df["drawdown"].min())

    # Calmar ratio
    if metrics["max_drawdown"] > 0:
        metrics["calmar_ratio"] = metrics["annualized_return"] / metrics["max_drawdown"]
    else:
        metrics["calmar_ratio"] = float("inf") if metrics["annualized_return"] > 0 else 0.0

    # ===== Exit Reasons =====
    metrics["stop_exits"] = len(trades_df[trades_df["exit_reason"] == "stop"])
    metrics["target_exits"] = len(trades_df[trades_df["exit_reason"] == "target"])
    metrics["eod_exits"] = len(trades_df[trades_df["exit_reason"] == "eod"])

    # ===== Consecutive Wins/Losses =====
    metrics["max_consecutive_wins"] = _max_consecutive(trades_df["pnl"] > 0)
    metrics["max_consecutive_losses"] = _max_consecutive(trades_df["pnl"] < 0)

    # ===== Costs =====
    metrics["total_commission"] = trades_df["commission"].sum()

    return metrics


def _max_consecutive(series: pd.Series) -> int:
    """Calculate maximum consecutive True values in a boolean series."""
    if series.empty:
        return 0

    max_streak = 0
    current_streak = 0

    for val in series:
        if val:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def _empty_metrics(initial_capital: float) -> dict:
    """Return empty metrics dict when no trades."""
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "long_trades": 0,
        "short_trades": 0,
        "total_pnl": 0.0,
        "avg_pnl": 0.0,
        "avg_win": 0.0,
        "avg_loss": 0.0,
        "avg_r": 0.0,
        "max_r": 0.0,
        "min_r": 0.0,
        "profit_factor": 0.0,
        "initial_capital": initial_capital,
        "final_capital": initial_capital,
        "total_return": 0.0,
        "annualized_return": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown": 0.0,
        "calmar_ratio": 0.0,
        "stop_exits": 0,
        "target_exits": 0,
        "eod_exits": 0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "total_commission": 0.0,
    }


def print_report(metrics: dict, title: str = "Performance Report") -> None:
    """Print a formatted performance report.

    Args:
        metrics: Dictionary of metrics from calculate_metrics().
        title: Report title.
    """
    width = 60

    print("\n" + "=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

    # ===== Capital Section =====
    print("\n--- Capital ---")
    print(f"Initial Capital:      ${metrics['initial_capital']:>14,.2f}")
    print(f"Final Capital:        ${metrics['final_capital']:>14,.2f}")
    print(f"Net P&L:              ${metrics['total_pnl']:>14,.2f}")

    # ===== Returns Section =====
    print("\n--- Returns ---")
    print(f"Total Return:         {metrics['total_return']:>14.2%}")
    print(f"Annualized Return:    {metrics['annualized_return']:>14.2%}")

    # ===== Risk Metrics Section =====
    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:>14.2f}")
    print(f"Max Drawdown:         {metrics['max_drawdown']:>14.2%}")
    calmar = metrics['calmar_ratio']
    if calmar == float("inf"):
        print(f"Calmar Ratio:         {'inf':>14}")
    else:
        print(f"Calmar Ratio:         {calmar:>14.2f}")

    # ===== Trade Statistics Section =====
    print("\n--- Trade Statistics ---")
    print(f"Total Trades:         {metrics['total_trades']:>14}")
    print(f"Winning Trades:       {metrics['winning_trades']:>14}")
    print(f"Losing Trades:        {metrics['losing_trades']:>14}")
    print(f"Win Rate:             {metrics['win_rate']:>14.2%}")
    print(f"Long Trades:          {metrics['long_trades']:>14}")
    print(f"Short Trades:         {metrics['short_trades']:>14}")

    # ===== P&L Per Trade Section =====
    print("\n--- P&L Per Trade ---")
    print(f"Average P&L:          ${metrics['avg_pnl']:>14,.2f}")
    print(f"Average Win:          ${metrics['avg_win']:>14,.2f}")
    print(f"Average Loss:         ${metrics['avg_loss']:>14,.2f}")
    pf = metrics['profit_factor']
    if pf == float("inf"):
        print(f"Profit Factor:        {'inf':>14}")
    else:
        print(f"Profit Factor:        {pf:>14.2f}")

    # ===== R-Multiples Section =====
    print("\n--- R-Multiples ---")
    print(f"Average R:            {metrics['avg_r']:>14.2f}")
    print(f"Best Trade (R):       {metrics['max_r']:>14.2f}")
    print(f"Worst Trade (R):      {metrics['min_r']:>14.2f}")

    # ===== Exit Reasons Section =====
    print("\n--- Exit Reasons ---")
    print(f"Stop Exits:           {metrics['stop_exits']:>14}")
    print(f"Target Exits:         {metrics['target_exits']:>14}")
    print(f"EOD Exits:            {metrics['eod_exits']:>14}")

    # ===== Streaks Section =====
    print("\n--- Streaks ---")
    print(f"Max Consecutive Wins: {metrics['max_consecutive_wins']:>14}")
    print(f"Max Consecutive Loss: {metrics['max_consecutive_losses']:>14}")

    # ===== Costs Section =====
    print("\n--- Costs ---")
    print(f"Total Commission:     ${metrics['total_commission']:>14,.2f}")

    print("\n" + "=" * width)


if __name__ == "__main__":
    # Test with sample data
    logger.info("Testing metrics calculation and reporting")

    # Create sample trades DataFrame
    trades_data = [
        {"date": "2024-01-15", "pnl": 750.00, "pnl_r": 10.0, "direction": "LONG", "exit_reason": "target", "commission": 0.50},
        {"date": "2024-01-16", "pnl": -250.00, "pnl_r": -1.0, "direction": "LONG", "exit_reason": "stop", "commission": 0.50},
        {"date": "2024-01-17", "pnl": 875.00, "pnl_r": 10.0, "direction": "SHORT", "exit_reason": "target", "commission": 0.50},
        {"date": "2024-01-18", "pnl": 125.00, "pnl_r": 1.5, "direction": "LONG", "exit_reason": "eod", "commission": 0.50},
        {"date": "2024-01-19", "pnl": -300.00, "pnl_r": -1.0, "direction": "SHORT", "exit_reason": "stop", "commission": 0.50},
        {"date": "2024-01-22", "pnl": 500.00, "pnl_r": 6.0, "direction": "LONG", "exit_reason": "eod", "commission": 0.50},
        {"date": "2024-01-23", "pnl": -275.00, "pnl_r": -1.0, "direction": "LONG", "exit_reason": "stop", "commission": 0.50},
        {"date": "2024-01-24", "pnl": -280.00, "pnl_r": -1.0, "direction": "SHORT", "exit_reason": "stop", "commission": 0.50},
        {"date": "2024-01-25", "pnl": 600.00, "pnl_r": 8.0, "direction": "LONG", "exit_reason": "target", "commission": 0.50},
        {"date": "2024-01-26", "pnl": 450.00, "pnl_r": 5.5, "direction": "LONG", "exit_reason": "eod", "commission": 0.50},
    ]
    trades_df = pd.DataFrame(trades_data)
    trades_df["date"] = pd.to_datetime(trades_df["date"])

    # Create sample equity curve
    initial_capital = 25000.0
    equity_values = [initial_capital]
    for pnl in trades_df["pnl"]:
        equity_values.append(equity_values[-1] + pnl)

    equity_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-14"] + trades_df["date"].dt.strftime("%Y-%m-%d").tolist()),
        "equity": equity_values,
    })

    # Calculate metrics
    metrics = calculate_metrics(trades_df, equity_df, initial_capital)

    # Print report
    print_report(metrics, "ORB Strategy Backtest Results")

    # Verify key metrics
    print("\n--- Verification ---")
    expected_total_trades = 10
    expected_winning = 6
    expected_losing = 4
    expected_total_pnl = sum(t["pnl"] for t in trades_data)

    assert metrics["total_trades"] == expected_total_trades, f"Expected {expected_total_trades} trades"
    assert metrics["winning_trades"] == expected_winning, f"Expected {expected_winning} winning trades"
    assert metrics["losing_trades"] == expected_losing, f"Expected {expected_losing} losing trades"
    assert abs(metrics["total_pnl"] - expected_total_pnl) < 0.01, f"Expected total PnL of {expected_total_pnl}"
    assert metrics["stop_exits"] == 4, "Expected 4 stop exits"
    assert metrics["target_exits"] == 3, "Expected 3 target exits"
    assert metrics["eod_exits"] == 3, "Expected 3 EOD exits"
    assert metrics["max_consecutive_wins"] == 2, "Expected max 2 consecutive wins"
    assert metrics["max_consecutive_losses"] == 2, "Expected max 2 consecutive losses"

    logger.success("All tests passed!")

    # Test empty trades case
    print("\n--- Testing Empty Trades ---")
    empty_metrics = calculate_metrics(pd.DataFrame(), pd.DataFrame({"date": [], "equity": []}), 25000.0)
    print_report(empty_metrics, "Empty Backtest")
    assert empty_metrics["total_trades"] == 0
    logger.success("Empty trades test passed!")
