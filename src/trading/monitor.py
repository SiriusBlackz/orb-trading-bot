"""Live ORB signal monitoring for real-time trading."""

import os
from datetime import datetime, time, timedelta
from enum import Enum

import pandas as pd
import pytz
from dotenv import load_dotenv
from loguru import logger

from src.data.realtime import RealtimeDataManager
from src.strategy.orb import Direction, ORBSignal, ORBSignalGenerator

load_dotenv()

# Eastern Time zone
ET = pytz.timezone("America/New_York")


class TradeStatus(Enum):
    """Current trade status."""

    WAITING_FOR_OPEN = "waiting_for_open"
    WAITING_FOR_SIGNAL = "waiting_for_signal"
    SIGNAL_GENERATED = "signal_generated"
    IN_TRADE = "in_trade"
    TRADE_CLOSED = "trade_closed"
    NO_SIGNAL = "no_signal"


class ORBMonitor:
    """Monitors real-time data for ORB trading signals."""

    MARKET_OPEN = time(9, 30)
    FIRST_CANDLE_END = time(9, 35)
    MARKET_CLOSE = time(16, 0)

    def __init__(self, symbol: str = None, client_id_offset: int = 0):
        """Initialize the ORB monitor.

        Args:
            symbol: Stock ticker symbol.
            client_id_offset: Offset to add to base client ID for unique connections.
        """
        self.symbol = symbol or os.getenv("SYMBOL", "TSLA")
        self.data_manager = RealtimeDataManager(client_id_offset=client_id_offset)
        self.signal_generator = ORBSignalGenerator(symbol=self.symbol)

        # State variables
        self._status = TradeStatus.WAITING_FOR_OPEN
        self._todays_date: datetime.date | None = None
        self._first_candle: pd.Series | None = None
        self._second_candle: pd.Series | None = None
        self._todays_signal: ORBSignal | None = None
        self._bars_today: list[pd.Series] = []
        self._entry_price: float | None = None
        self._exit_price: float | None = None
        self._exit_reason: str | None = None

    def _now_et(self) -> datetime:
        """Get current time in Eastern Time."""
        return datetime.now(ET)

    def _is_market_open(self) -> bool:
        """Check if current time is within market hours (9:30-16:00 ET).

        Returns:
            True if market is open, False otherwise.
        """
        now = self._now_et()
        current_time = now.time()
        # Also check if it's a weekday
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False
        return self.MARKET_OPEN <= current_time < self.MARKET_CLOSE

    def _is_first_candle_complete(self) -> bool:
        """Check if the first candle (9:30-9:35) is complete.

        Returns:
            True if time is 9:35 ET or later, False otherwise.
        """
        current_time = self._now_et().time()
        return current_time >= self.FIRST_CANDLE_END

    def _reset_daily_state(self) -> None:
        """Reset state for a new trading day."""
        self._todays_date = self._now_et().date()
        self._first_candle = None
        self._second_candle = None
        self._todays_signal = None
        self._bars_today = []
        self._entry_price = None
        self._exit_price = None
        self._exit_reason = None
        self._status = TradeStatus.WAITING_FOR_OPEN
        logger.info(f"Reset state for new trading day: {self._todays_date}")

    def _on_bar(self, bar: pd.Series) -> None:
        """Callback for new bar data.

        Stores bars, generates signal at 9:35, and monitors for stop/target.

        Args:
            bar: New bar data as pd.Series.
        """
        now = self._now_et()
        today = now.date()

        # Check if it's a new day
        if self._todays_date != today:
            self._reset_daily_state()

        # Parse bar time
        bar_datetime = pd.to_datetime(bar["date"])
        if bar_datetime.tzinfo is None:
            bar_datetime = ET.localize(bar_datetime)
        else:
            bar_datetime = bar_datetime.astimezone(ET)

        bar_time = bar_datetime.time()
        bar_date = bar_datetime.date()

        # Only process bars from today
        if bar_date != today:
            return

        # Store bar
        self._bars_today.append(bar)

        logger.debug(f"Bar received: {bar_time} O={bar['open']:.2f} H={bar['high']:.2f} "
                    f"L={bar['low']:.2f} C={bar['close']:.2f}")

        # State machine
        if self._status == TradeStatus.WAITING_FOR_OPEN:
            if self._is_market_open():
                self._status = TradeStatus.WAITING_FOR_SIGNAL
                logger.info("Market is open, waiting for first candle to complete...")

        if self._status == TradeStatus.WAITING_FOR_SIGNAL:
            # Capture first candle (9:30 bar)
            if bar_time == self.MARKET_OPEN or (
                self._first_candle is None and
                self.MARKET_OPEN <= bar_time < self.FIRST_CANDLE_END
            ):
                self._first_candle = bar.copy()
                logger.info(f"First candle (9:30): O={bar['open']:.2f} H={bar['high']:.2f} "
                           f"L={bar['low']:.2f} C={bar['close']:.2f}")

            # Capture second candle and generate signal (9:35 bar)
            if bar_time >= self.FIRST_CANDLE_END and self._second_candle is None:
                self._second_candle = bar.copy()
                logger.info(f"Second candle (9:35): O={bar['open']:.2f} H={bar['high']:.2f} "
                           f"L={bar['low']:.2f} C={bar['close']:.2f}")
                self._generate_signal()

        elif self._status == TradeStatus.SIGNAL_GENERATED or self._status == TradeStatus.IN_TRADE:
            self._check_exit_conditions(bar)

    def _generate_signal(self) -> None:
        """Generate ORB signal from collected candles."""
        if self._first_candle is None or self._second_candle is None:
            logger.warning("Cannot generate signal: missing candles")
            self._status = TradeStatus.NO_SIGNAL
            return

        # Build day data DataFrame for signal generator
        day_data = pd.DataFrame([
            self._first_candle.to_dict(),
            self._second_candle.to_dict(),
        ])

        signal = self.signal_generator.generate_signal(day_data)

        if signal is None:
            logger.warning("No signal generated (doji or invalid)")
            self._status = TradeStatus.NO_SIGNAL
            return

        self._todays_signal = signal
        self._entry_price = signal.entry_price
        self._status = TradeStatus.SIGNAL_GENERATED

        # Log signal details
        logger.success("=" * 60)
        logger.success(f"ORB SIGNAL GENERATED")
        logger.success("=" * 60)
        logger.success(f"Symbol:      {signal.symbol}")
        logger.success(f"Direction:   {signal.direction.value}")
        logger.success(f"Entry:       ${signal.entry_price:.2f}")
        logger.success(f"Stop:        ${signal.stop_price:.2f}")
        logger.success(f"Target:      ${signal.target_price:.2f}")
        logger.success(f"Risk/Share:  ${signal.risk_per_share:.2f}")
        logger.success("=" * 60)

    def _check_exit_conditions(self, bar: pd.Series) -> None:
        """Check if stop or target has been hit.

        Args:
            bar: Current bar data.
        """
        if self._todays_signal is None:
            return

        signal = self._todays_signal
        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]
        bar_time = pd.to_datetime(bar["date"])

        if signal.direction == Direction.LONG:
            if bar_low <= signal.stop_price:
                self._exit_price = signal.stop_price
                self._exit_reason = "stop"
                self._status = TradeStatus.TRADE_CLOSED
                logger.warning(f"STOP HIT: Exited LONG at ${self._exit_price:.2f}")
            elif bar_high >= signal.target_price:
                self._exit_price = signal.target_price
                self._exit_reason = "target"
                self._status = TradeStatus.TRADE_CLOSED
                logger.success(f"TARGET HIT: Exited LONG at ${self._exit_price:.2f}")
        else:  # SHORT
            if bar_high >= signal.stop_price:
                self._exit_price = signal.stop_price
                self._exit_reason = "stop"
                self._status = TradeStatus.TRADE_CLOSED
                logger.warning(f"STOP HIT: Exited SHORT at ${self._exit_price:.2f}")
            elif bar_low <= signal.target_price:
                self._exit_price = signal.target_price
                self._exit_reason = "target"
                self._status = TradeStatus.TRADE_CLOSED
                logger.success(f"TARGET HIT: Exited SHORT at ${self._exit_price:.2f}")

        # Check for end of day
        bar_time_only = bar_time.time() if hasattr(bar_time, 'time') else bar_time
        if isinstance(bar_time_only, datetime):
            bar_time_only = bar_time_only.time()

        if bar_time_only >= self.MARKET_CLOSE and self._status != TradeStatus.TRADE_CLOSED:
            self._exit_price = bar_close
            self._exit_reason = "eod"
            self._status = TradeStatus.TRADE_CLOSED
            logger.info(f"EOD EXIT: Closed at ${self._exit_price:.2f}")

        if self._status == TradeStatus.TRADE_CLOSED:
            self._log_trade_result()

    def _log_trade_result(self) -> None:
        """Log the final trade result."""
        if self._todays_signal is None or self._exit_price is None:
            return

        signal = self._todays_signal

        if signal.direction == Direction.LONG:
            pnl_per_share = self._exit_price - signal.entry_price
        else:
            pnl_per_share = signal.entry_price - self._exit_price

        r_multiple = pnl_per_share / signal.risk_per_share

        logger.info("=" * 60)
        logger.info("TRADE RESULT")
        logger.info("=" * 60)
        logger.info(f"Direction:   {signal.direction.value}")
        logger.info(f"Entry:       ${signal.entry_price:.2f}")
        logger.info(f"Exit:        ${self._exit_price:.2f}")
        logger.info(f"P&L/Share:   ${pnl_per_share:.2f}")
        logger.info(f"R-Multiple:  {r_multiple:.2f}R")
        logger.info(f"Exit Reason: {self._exit_reason}")
        logger.info("=" * 60)

    def start(self) -> bool:
        """Start monitoring for ORB signals.

        Returns:
            True if started successfully, False otherwise.
        """
        logger.info(f"Starting ORB monitor for {self.symbol}")

        if not self.data_manager.connect():
            logger.error("Failed to connect to IBKR")
            return False

        if not self.data_manager.subscribe_bars(self.symbol, "5 mins", self._on_bar):
            logger.error(f"Failed to subscribe to {self.symbol}")
            self.data_manager.disconnect()
            return False

        self._todays_date = self._now_et().date()

        if self._is_market_open():
            self._status = TradeStatus.WAITING_FOR_SIGNAL
            logger.info("Market is open, monitoring for signals...")
        else:
            logger.info("Market is closed, waiting for market open...")

        return True

    def stop(self) -> None:
        """Stop monitoring and disconnect."""
        logger.info("Stopping ORB monitor")
        self.data_manager.disconnect()

    def get_todays_signal(self) -> ORBSignal | None:
        """Get today's ORB signal if generated.

        Returns:
            ORBSignal if generated today, None otherwise.
        """
        return self._todays_signal

    def get_status(self) -> TradeStatus:
        """Get current monitor status.

        Returns:
            Current TradeStatus.
        """
        return self._status

    def run(self) -> None:
        """Run the monitor until interrupted."""
        if not self.start():
            return

        try:
            self.data_manager.run_forever()
        except KeyboardInterrupt:
            logger.info("Monitor interrupted")
        finally:
            self.stop()


if __name__ == "__main__":
    import signal
    import sys

    logger.info("Starting ORB Monitor test")

    symbol = os.getenv("SYMBOL", "TSLA")
    monitor = ORBMonitor(symbol=symbol)

    def shutdown(signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutting down...")
        monitor.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Show current market status
    now_et = datetime.now(ET)
    logger.info(f"Current time (ET): {now_et.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"Market open: {monitor._is_market_open()}")
    logger.info(f"First candle complete: {monitor._is_first_candle_complete()}")

    # Run the monitor
    logger.info(f"Monitoring {symbol} for ORB signals... Press Ctrl+C to stop")
    monitor.run()
