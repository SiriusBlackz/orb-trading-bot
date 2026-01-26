"""System validation tests for ORB Trading Bot.

Run with: pytest tests/test_setup.py -v
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestImports:
    """Test that all modules can be imported successfully."""

    def test_import_data_fetcher(self):
        """Import historical data fetcher."""
        from src.data.fetcher import IBKRDataFetcher, fetch_and_cache_data
        print("✓ src.data.fetcher imported successfully")
        assert IBKRDataFetcher is not None
        assert fetch_and_cache_data is not None

    def test_import_data_realtime(self):
        """Import real-time data manager."""
        from src.data.realtime import RealtimeDataManager
        print("✓ src.data.realtime imported successfully")
        assert RealtimeDataManager is not None

    def test_import_strategy_orb(self):
        """Import ORB strategy."""
        from src.strategy.orb import Direction, ORBSignal, ORBSignalGenerator
        print("✓ src.strategy.orb imported successfully")
        assert Direction is not None
        assert ORBSignal is not None
        assert ORBSignalGenerator is not None

    def test_import_backtest_engine(self):
        """Import backtesting engine."""
        from src.backtest.engine import BacktestResult, ORBBacktester, TradeResult
        print("✓ src.backtest.engine imported successfully")
        assert BacktestResult is not None
        assert ORBBacktester is not None
        assert TradeResult is not None

    def test_import_trading_monitor(self):
        """Import trading monitor."""
        from src.trading.monitor import ORBMonitor, TradeStatus
        print("✓ src.trading.monitor imported successfully")
        assert ORBMonitor is not None
        assert TradeStatus is not None

    def test_import_trading_executor(self):
        """Import order executor."""
        from src.trading.executor import OrderExecutor, PaperTrader
        print("✓ src.trading.executor imported successfully")
        assert OrderExecutor is not None
        assert PaperTrader is not None

    def test_import_utils_metrics(self):
        """Import metrics utilities."""
        from src.utils.metrics import calculate_metrics, print_report
        print("✓ src.utils.metrics imported successfully")
        assert calculate_metrics is not None
        assert print_report is not None


class TestEnvironment:
    """Test environment configuration."""

    def test_env_file_exists(self):
        """Check .env file exists."""
        env_path = PROJECT_ROOT / ".env"
        assert env_path.exists(), ".env file not found. Copy .env.example to .env"
        print("✓ .env file exists")

    def test_required_settings(self):
        """Verify required settings are present."""
        from dotenv import load_dotenv
        load_dotenv()

        required_settings = [
            "IBKR_HOST",
            "IBKR_PORT",
            "IBKR_CLIENT_ID",
            "SYMBOL",
            "STARTING_CAPITAL",
            "RISK_PER_TRADE",
            "MAX_LEVERAGE",
        ]

        missing = []
        for setting in required_settings:
            value = os.getenv(setting)
            if value is None:
                missing.append(setting)
            else:
                print(f"✓ {setting} = {value}")

        assert not missing, f"Missing required settings: {missing}"

    def test_settings_valid_values(self):
        """Verify settings have valid values."""
        from dotenv import load_dotenv
        load_dotenv()

        # Check port is numeric
        port = os.getenv("IBKR_PORT")
        assert port.isdigit(), f"IBKR_PORT must be numeric, got: {port}"
        print(f"✓ IBKR_PORT is valid: {port}")

        # Check capital is positive
        capital = float(os.getenv("STARTING_CAPITAL", 0))
        assert capital > 0, f"STARTING_CAPITAL must be positive, got: {capital}"
        print(f"✓ STARTING_CAPITAL is valid: ${capital:,.2f}")

        # Check risk is between 0 and 1
        risk = float(os.getenv("RISK_PER_TRADE", 0))
        assert 0 < risk <= 1, f"RISK_PER_TRADE must be between 0 and 1, got: {risk}"
        print(f"✓ RISK_PER_TRADE is valid: {risk:.2%}")


class TestTWSConnection:
    """Test TWS/Gateway connection (requires TWS running)."""

    @pytest.fixture
    def fetcher(self):
        """Create a data fetcher instance."""
        from src.data.fetcher import IBKRDataFetcher
        return IBKRDataFetcher()

    def test_connect_to_tws(self, fetcher):
        """Test connecting to TWS/Gateway."""
        connected = fetcher.connect()
        assert connected, "Failed to connect to TWS/Gateway. Is it running?"
        print("✓ Connected to TWS/Gateway successfully")
        fetcher.disconnect()
        print("✓ Disconnected cleanly")


class TestHistoricalData:
    """Test historical data fetching (requires TWS running)."""

    @pytest.fixture
    def fetcher(self):
        """Create and connect a data fetcher."""
        from src.data.fetcher import IBKRDataFetcher
        f = IBKRDataFetcher()
        if not f.connect():
            pytest.skip("TWS not available")
        yield f
        f.disconnect()

    def test_fetch_one_day_data(self, fetcher):
        """Fetch 1 day of TSLA historical data."""
        import pandas as pd

        symbol = os.getenv("SYMBOL", "TSLA")
        df = fetcher.fetch_historical_bars(
            symbol=symbol,
            duration="1 D",
            bar_size="5 mins",
            use_rth=True,
        )

        assert not df.empty, f"No data returned for {symbol}"
        assert len(df) > 0, "DataFrame is empty"

        # Check required columns
        required_cols = ["date", "open", "high", "low", "close", "volume"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

        print(f"✓ Fetched {len(df)} bars for {symbol}")
        print(f"✓ Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"✓ Columns: {list(df.columns)}")


class TestSignalGeneration:
    """Test ORB signal generation."""

    @pytest.fixture
    def sample_data(self):
        """Create sample intraday data."""
        import pandas as pd
        from datetime import datetime

        # Create bullish day data
        data = [
            {"date": datetime(2024, 1, 15, 9, 30), "open": 250.0, "high": 252.0, "low": 249.0, "close": 251.5, "volume": 100000},
            {"date": datetime(2024, 1, 15, 9, 35), "open": 251.5, "high": 252.5, "low": 251.0, "close": 252.0, "volume": 80000},
            {"date": datetime(2024, 1, 15, 9, 40), "open": 252.0, "high": 253.0, "low": 251.5, "close": 252.5, "volume": 60000},
        ]
        return pd.DataFrame(data)

    def test_generate_signal(self, sample_data):
        """Generate ORB signal from sample data."""
        from src.strategy.orb import Direction, ORBSignalGenerator

        generator = ORBSignalGenerator(symbol="TSLA")
        signal = generator.generate_signal(sample_data)

        assert signal is not None, "No signal generated"
        assert signal.direction == Direction.LONG, f"Expected LONG, got {signal.direction}"
        assert signal.entry_price == 252.0, f"Wrong entry price: {signal.entry_price}"
        assert signal.stop_price == 249.0, f"Wrong stop price: {signal.stop_price}"

        print(f"✓ Signal generated: {signal.direction.value}")
        print(f"✓ Entry: ${signal.entry_price:.2f}")
        print(f"✓ Stop: ${signal.stop_price:.2f}")
        print(f"✓ Target: ${signal.target_price:.2f}")
        print(f"✓ Risk/Share: ${signal.risk_per_share:.2f}")

    def test_generate_signals_multiple_days(self, sample_data):
        """Generate signals from multiple days."""
        import pandas as pd
        from datetime import datetime
        from src.strategy.orb import ORBSignalGenerator

        # Add another day
        day2 = [
            {"date": datetime(2024, 1, 16, 9, 30), "open": 255.0, "high": 256.0, "low": 253.0, "close": 253.5, "volume": 100000},
            {"date": datetime(2024, 1, 16, 9, 35), "open": 253.5, "high": 254.0, "low": 252.5, "close": 253.0, "volume": 80000},
        ]
        multi_day = pd.concat([sample_data, pd.DataFrame(day2)], ignore_index=True)

        generator = ORBSignalGenerator(symbol="TSLA")
        signals = generator.generate_signals(multi_day)

        assert len(signals) == 2, f"Expected 2 signals, got {len(signals)}"
        print(f"✓ Generated {len(signals)} signals from 2 days")


class TestBacktest:
    """Test backtesting engine."""

    @pytest.fixture
    def sample_multi_day_data(self):
        """Create multi-day sample data for backtesting."""
        import pandas as pd
        from datetime import datetime, timedelta

        bars = []
        base_date = datetime(2024, 1, 15, 9, 30)

        # Day 1: Bullish, price goes up
        price = 250.0
        for i in range(78):  # Full trading day
            bars.append({
                "date": base_date + timedelta(minutes=i * 5),
                "open": price,
                "high": price + 1,
                "low": price - 0.5,
                "close": price + 0.5,
                "volume": 50000,
            })
            price += 0.3

        # Day 2: Bearish, price goes down
        base_date = datetime(2024, 1, 16, 9, 30)
        price = 260.0
        for i in range(78):
            bars.append({
                "date": base_date + timedelta(minutes=i * 5),
                "open": price,
                "high": price + 0.5,
                "low": price - 1,
                "close": price - 0.5,
                "volume": 50000,
            })
            price -= 0.3

        return pd.DataFrame(bars)

    def test_run_backtest(self, sample_multi_day_data):
        """Run a mini backtest on sample data."""
        from src.backtest.engine import ORBBacktester

        backtester = ORBBacktester(initial_capital=25000)
        result = backtester.run(sample_multi_day_data, symbol="TSLA")

        assert result is not None, "Backtest returned None"
        assert result.total_trades > 0, "No trades executed"
        assert result.initial_capital == 25000, "Wrong initial capital"

        print(f"✓ Backtest completed")
        print(f"✓ Total trades: {result.total_trades}")
        print(f"✓ Final capital: ${result.final_capital:,.2f}")
        print(f"✓ Total return: {result.total_return:.2%}")
        print(f"✓ Win rate: {result.win_rate:.2%}")


class TestMetrics:
    """Test metrics calculation."""

    @pytest.fixture
    def sample_trades_and_equity(self):
        """Create sample trades and equity data."""
        import pandas as pd

        trades_df = pd.DataFrame([
            {"date": "2024-01-15", "pnl": 500.0, "pnl_r": 2.0, "direction": "LONG", "exit_reason": "target", "commission": 0.50},
            {"date": "2024-01-16", "pnl": -250.0, "pnl_r": -1.0, "direction": "SHORT", "exit_reason": "stop", "commission": 0.50},
            {"date": "2024-01-17", "pnl": 300.0, "pnl_r": 1.5, "direction": "LONG", "exit_reason": "eod", "commission": 0.50},
        ])

        equity_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-14", "2024-01-15", "2024-01-16", "2024-01-17"]),
            "equity": [25000, 25500, 25250, 25550],
        })

        return trades_df, equity_df

    def test_calculate_metrics(self, sample_trades_and_equity):
        """Calculate metrics from sample data."""
        from src.utils.metrics import calculate_metrics

        trades_df, equity_df = sample_trades_and_equity
        metrics = calculate_metrics(trades_df, equity_df, initial_capital=25000)

        assert metrics is not None, "Metrics returned None"
        assert metrics["total_trades"] == 3, f"Wrong trade count: {metrics['total_trades']}"
        assert metrics["winning_trades"] == 2, f"Wrong winning count: {metrics['winning_trades']}"
        assert metrics["losing_trades"] == 1, f"Wrong losing count: {metrics['losing_trades']}"

        print(f"✓ Metrics calculated successfully")
        print(f"✓ Total trades: {metrics['total_trades']}")
        print(f"✓ Win rate: {metrics['win_rate']:.2%}")
        print(f"✓ Total P&L: ${metrics['total_pnl']:,.2f}")
        print(f"✓ Avg R: {metrics['avg_r']:.2f}")
        print(f"✓ Profit factor: {metrics['profit_factor']:.2f}")

    def test_print_report(self, sample_trades_and_equity, capsys):
        """Test report printing."""
        from src.utils.metrics import calculate_metrics, print_report

        trades_df, equity_df = sample_trades_and_equity
        metrics = calculate_metrics(trades_df, equity_df, initial_capital=25000)
        print_report(metrics, "Test Report")

        captured = capsys.readouterr()
        assert "Test Report" in captured.out
        assert "Capital" in captured.out
        assert "Win Rate" in captured.out
        print("✓ Report printed successfully")


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
