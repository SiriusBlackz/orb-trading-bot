"""Real-time market data manager for live trading."""

import os
import threading
import time as time_module
from collections import defaultdict
from datetime import datetime, time
from typing import Callable

import pandas as pd
from dotenv import load_dotenv
from ib_insync import IB, Stock, util
from loguru import logger

load_dotenv()

# Reconnection constants
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 5


class RealtimeDataManager:
    """Manages real-time market data subscriptions from IBKR."""

    def __init__(self, client_id_offset: int = 0):
        """Initialize the data manager.

        Args:
            client_id_offset: Offset to add to base client ID for unique connections.
        """
        self.ib = IB()
        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_PORT", 7497))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", 1)) + client_id_offset

        self._connected = False
        self._subscriptions: dict[str, any] = {}  # symbol -> bars object
        self._subscription_config: dict[str, str] = {}  # symbol -> bar_size (for resubscription)
        self._callbacks: dict[str, list[Callable]] = defaultdict(list)
        self._current_bars: dict[str, pd.Series] = {}
        self._bar_events: dict[str, threading.Event] = defaultdict(threading.Event)
        self._target_times: dict[str, datetime] = {}

        # Reconnection state
        self._reconnect_attempts = 0
        self._reconnecting = False
        self._disconnect_callback: Callable[[], None] | None = None
        self._reconnect_callback: Callable[[], None] | None = None

        # Register error handlers
        self.ib.errorEvent += self._on_error
        self.ib.disconnectedEvent += self._on_disconnected

    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway.

        Returns:
            True if connection successful, False otherwise.
        """
        if self._connected:
            logger.warning("Already connected to IBKR")
            return True

        try:
            logger.info(f"Connecting to IBKR at {self.host}:{self.port} (client_id={self.client_id})")
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            self._reconnect_attempts = 0  # Reset on successful connect
            logger.success("Connected to IBKR successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from IBKR and clean up subscriptions."""
        if not self._connected:
            logger.warning("Not connected to IBKR")
            return

        # Cancel all subscriptions
        for symbol, bars in self._subscriptions.items():
            try:
                if bars is not None:
                    self.ib.cancelHistoricalData(bars)
                    logger.debug(f"Cancelled subscription for {symbol}")
            except Exception as e:
                logger.warning(f"Error cancelling subscription for {symbol}: {e}")

        self._subscriptions.clear()
        self._callbacks.clear()
        self._current_bars.clear()
        self._bar_events.clear()
        self._target_times.clear()

        self.ib.disconnect()
        self._connected = False
        logger.info("Disconnected from IBKR")

    def set_disconnect_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called on disconnect."""
        self._disconnect_callback = callback

    def set_reconnect_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called after successful reconnect."""
        self._reconnect_callback = callback

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract) -> None:
        """Handle IBKR error events.

        Error codes:
            1100: Connectivity lost
            1101: Connectivity restored (data lost)
            1102: Connectivity restored (data maintained)
            2110: Connectivity between TWS and server is broken
        """
        if errorCode in (1100, 2110):
            logger.error(
                f"IBKR_DISCONNECT | error_code={errorCode} message={errorString}"
            )
            self._connected = False
            if not self._reconnecting:
                self._handle_disconnect()
        elif errorCode == 1101:
            logger.warning(
                f"IBKR_RECONNECT | error_code={errorCode} message={errorString} data_lost=True"
            )
            self._connected = True
            self._reconnecting = False
            self._reconnect_attempts = 0
            # Data was lost - need to resubscribe
            self._resubscribe_all()
            if self._reconnect_callback:
                self._reconnect_callback()
        elif errorCode == 1102:
            logger.info(
                f"IBKR_RECONNECT | error_code={errorCode} message={errorString} data_maintained=True"
            )
            self._connected = True
            self._reconnecting = False
            self._reconnect_attempts = 0
            if self._reconnect_callback:
                self._reconnect_callback()
        elif errorCode >= 2000:
            # Warning codes - log but don't treat as disconnect
            logger.warning(f"IBKR_WARNING | code={errorCode} message={errorString}")

    def _on_disconnected(self) -> None:
        """Handle IB disconnected event."""
        logger.warning("IBKR_DISCONNECTED | Connection to IBKR lost")
        self._connected = False
        if not self._reconnecting:
            self._handle_disconnect()

    def _handle_disconnect(self) -> None:
        """Handle disconnect and initiate reconnection."""
        if self._reconnecting:
            return

        self._reconnecting = True

        # Notify via callback
        if self._disconnect_callback:
            try:
                self._disconnect_callback()
            except Exception as e:
                logger.error(f"DISCONNECT_CALLBACK_ERROR | error={e}")

        # Start reconnection in a separate thread to not block
        reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        """Attempt to reconnect with retries."""
        while self._reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            logger.info(
                f"RECONNECT_ATTEMPT | attempt={self._reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} "
                f"waiting={RECONNECT_DELAY}s"
            )

            time_module.sleep(RECONNECT_DELAY)

            try:
                # Create new IB instance
                self.ib = IB()
                self.ib.errorEvent += self._on_error
                self.ib.disconnectedEvent += self._on_disconnected

                logger.info(f"RECONNECT | Attempting connection to {self.host}:{self.port}")
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self._connected = True
                self._reconnecting = False
                self._reconnect_attempts = 0

                logger.success(
                    f"RECONNECT_SUCCESS | Connected after {self._reconnect_attempts} attempts"
                )

                # Resubscribe to all data feeds
                self._resubscribe_all()

                # Notify via callback
                if self._reconnect_callback:
                    try:
                        self._reconnect_callback()
                    except Exception as e:
                        logger.error(f"RECONNECT_CALLBACK_ERROR | error={e}")

                return

            except Exception as e:
                logger.error(
                    f"RECONNECT_FAILED | attempt={self._reconnect_attempts} error={e}"
                )

        # Max attempts reached
        logger.error(
            f"RECONNECT_EXHAUSTED | max_attempts={MAX_RECONNECT_ATTEMPTS} giving_up=True"
        )
        self._reconnecting = False

    def _resubscribe_all(self) -> None:
        """Resubscribe to all previously subscribed data feeds."""
        if not self._subscription_config:
            logger.info("RESUBSCRIBE | No subscriptions to restore")
            return

        logger.info(f"RESUBSCRIBE | Restoring {len(self._subscription_config)} subscriptions")

        # Store old config and clear subscriptions
        old_config = self._subscription_config.copy()
        old_callbacks = {sym: cbs.copy() for sym, cbs in self._callbacks.items()}
        self._subscriptions.clear()

        for symbol, bar_size in old_config.items():
            try:
                # Resubscribe without callbacks first
                contract = Stock(symbol, "SMART", "USD")
                self.ib.qualifyContracts(contract)

                bars = self.ib.reqHistoricalData(
                    contract,
                    endDateTime="",
                    durationStr="1 D",
                    barSizeSetting=bar_size,
                    whatToShow="TRADES",
                    useRTH=True,
                    formatDate=1,
                    keepUpToDate=True,
                )

                bars.updateEvent += self._on_bar_update
                self._subscriptions[symbol] = bars

                # Restore callbacks
                if symbol in old_callbacks:
                    self._callbacks[symbol] = old_callbacks[symbol]

                # Update current bar
                if bars:
                    latest = bars[-1]
                    self._current_bars[symbol] = pd.Series({
                        "date": latest.date,
                        "open": latest.open,
                        "high": latest.high,
                        "low": latest.low,
                        "close": latest.close,
                        "volume": latest.volume,
                        "average": getattr(latest, "average", None),
                        "barCount": getattr(latest, "barCount", None),
                    })

                logger.success(f"RESUBSCRIBE | Restored {symbol} {bar_size}")

            except Exception as e:
                logger.error(f"RESUBSCRIBE_ERROR | symbol={symbol} error={e}")

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self._connected and self.ib.isConnected()

    def _on_bar_update(self, bars, has_new_bar: bool) -> None:
        """Handle incoming bar updates.

        Args:
            bars: BarDataList from ib_insync.
            has_new_bar: True if a new bar was added.
        """
        if not bars:
            return

        symbol = bars.contract.symbol
        latest_bar = bars[-1]

        # Convert to Series for consistent interface
        bar_data = pd.Series({
            "date": latest_bar.date,
            "open": latest_bar.open,
            "high": latest_bar.high,
            "low": latest_bar.low,
            "close": latest_bar.close,
            "volume": latest_bar.volume,
            "average": getattr(latest_bar, "average", None),
            "barCount": getattr(latest_bar, "barCount", None),
        })

        self._current_bars[symbol] = bar_data

        if has_new_bar:
            logger.debug(f"New bar for {symbol}: {latest_bar.date} O={latest_bar.open:.2f} "
                        f"H={latest_bar.high:.2f} L={latest_bar.low:.2f} C={latest_bar.close:.2f}")

            # Call registered callbacks
            for callback in self._callbacks.get(symbol, []):
                try:
                    callback(bar_data)
                except Exception as e:
                    logger.error(f"Error in callback for {symbol}: {e}")

            # Check if this bar matches a target time
            target_time = self._target_times.get(symbol)
            if target_time is not None:
                bar_time = pd.to_datetime(latest_bar.date)
                if isinstance(target_time, time):
                    if bar_time.time() >= target_time:
                        self._bar_events[symbol].set()
                elif bar_time >= target_time:
                    self._bar_events[symbol].set()

    def subscribe_bars(
        self,
        symbol: str,
        bar_size: str = "5 mins",
        callback: Callable[[pd.Series], None] | None = None,
    ) -> bool:
        """Subscribe to real-time bars for a symbol.

        Uses reqHistoricalData with keepUpToDate=True for streaming bars.

        Args:
            symbol: Stock ticker symbol.
            bar_size: Bar size (e.g., '5 mins', '1 min', '15 mins').
            callback: Function to call with each new bar (receives pd.Series).

        Returns:
            True if subscription successful, False otherwise.
        """
        if not self._connected:
            logger.error("Not connected to IBKR. Call connect() first.")
            return False

        if symbol in self._subscriptions:
            logger.warning(f"Already subscribed to {symbol}")
            if callback:
                self._callbacks[symbol].append(callback)
            return True

        contract = Stock(symbol, "SMART", "USD")

        try:
            self.ib.qualifyContracts(contract)
        except Exception as e:
            logger.error(f"Failed to qualify contract for {symbol}: {e}")
            return False

        try:
            # Request historical data with keepUpToDate for streaming
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",  # Empty for current time
                durationStr="1 D",
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
                keepUpToDate=True,  # Enable streaming updates
            )

            # Register update handler
            bars.updateEvent += self._on_bar_update

            self._subscriptions[symbol] = bars
            self._subscription_config[symbol] = bar_size  # Store for resubscription

            if callback:
                self._callbacks[symbol].append(callback)

            # Initialize current bar
            if bars:
                latest = bars[-1]
                self._current_bars[symbol] = pd.Series({
                    "date": latest.date,
                    "open": latest.open,
                    "high": latest.high,
                    "low": latest.low,
                    "close": latest.close,
                    "volume": latest.volume,
                    "average": getattr(latest, "average", None),
                    "barCount": getattr(latest, "barCount", None),
                })

            logger.success(f"Subscribed to {bar_size} bars for {symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to subscribe to {symbol}: {e}")
            return False

    def get_current_bar(self, symbol: str) -> pd.Series | None:
        """Get the latest bar for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Series with bar data, or None if not available.
        """
        return self._current_bars.get(symbol)

    def wait_for_bar(self, symbol: str, target_time: datetime | time, timeout: float = 300) -> pd.Series | None:
        """Block until a bar with the target timestamp arrives.

        Args:
            symbol: Stock ticker symbol.
            target_time: Target bar time (datetime or time object).
            timeout: Maximum seconds to wait.

        Returns:
            The bar that matched, or None if timeout.
        """
        if symbol not in self._subscriptions:
            logger.error(f"Not subscribed to {symbol}")
            return None

        # Check if current bar already matches
        current = self._current_bars.get(symbol)
        if current is not None:
            bar_time = pd.to_datetime(current["date"])
            if isinstance(target_time, time):
                if bar_time.time() >= target_time:
                    return current
            elif bar_time >= target_time:
                return current

        # Set target and wait
        self._target_times[symbol] = target_time
        self._bar_events[symbol].clear()

        logger.info(f"Waiting for {symbol} bar at {target_time}...")

        # Wait with periodic ib.sleep to process events
        waited = 0.0
        interval = 0.5
        while waited < timeout:
            self.ib.sleep(interval)
            if self._bar_events[symbol].is_set():
                self._target_times.pop(symbol, None)
                return self._current_bars.get(symbol)
            waited += interval

        logger.warning(f"Timeout waiting for {symbol} bar at {target_time}")
        self._target_times.pop(symbol, None)
        return None

    def run_forever(self) -> None:
        """Run the event loop indefinitely. Use Ctrl+C to stop."""
        if not self._connected:
            logger.error("Not connected to IBKR")
            return

        logger.info("Running event loop (Ctrl+C to stop)...")
        try:
            self.ib.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.disconnect()


if __name__ == "__main__":
    import signal
    import sys

    # Test: Subscribe to TSLA and print each new bar
    logger.info("Testing RealtimeDataManager - subscribing to TSLA 5-min bars")

    manager = RealtimeDataManager()

    def on_new_bar(bar: pd.Series) -> None:
        """Callback for new bars."""
        print(f"\n{'='*60}")
        print(f"NEW BAR: {bar['date']}")
        print(f"  Open:   ${bar['open']:.2f}")
        print(f"  High:   ${bar['high']:.2f}")
        print(f"  Low:    ${bar['low']:.2f}")
        print(f"  Close:  ${bar['close']:.2f}")
        print(f"  Volume: {bar['volume']:,.0f}")
        print(f"{'='*60}")

    def shutdown(signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutting down...")
        manager.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if not manager.connect():
        logger.error("Failed to connect to IBKR")
        sys.exit(1)

    symbol = os.getenv("SYMBOL", "TSLA")

    if not manager.subscribe_bars(symbol, "5 mins", on_new_bar):
        logger.error(f"Failed to subscribe to {symbol}")
        manager.disconnect()
        sys.exit(1)

    # Show current bar
    current = manager.get_current_bar(symbol)
    if current is not None:
        logger.info(f"Current bar for {symbol}:")
        print(f"  Date:   {current['date']}")
        print(f"  Open:   ${current['open']:.2f}")
        print(f"  High:   ${current['high']:.2f}")
        print(f"  Low:    ${current['low']:.2f}")
        print(f"  Close:  ${current['close']:.2f}")
        print(f"  Volume: {current['volume']:,.0f}")

    logger.info(f"Listening for {symbol} bars... Press Ctrl+C to stop")

    # Run event loop
    manager.run_forever()
