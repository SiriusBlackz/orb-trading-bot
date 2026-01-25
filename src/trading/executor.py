"""Order execution for ORB trading strategy."""

import os
import sys
from datetime import datetime, time
from pathlib import Path

import pytz
from dotenv import load_dotenv
from ib_insync import IB, LimitOrder, MarketOrder, Stock, StopOrder
from loguru import logger

from src.strategy.orb import Direction, ORBSignal
from src.trading.monitor import ORBMonitor, TradeStatus

load_dotenv()

# Eastern Time zone
ET = pytz.timezone("America/New_York")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Set up file logging for trades
LOGS_DIR.mkdir(exist_ok=True)
logger.add(
    LOGS_DIR / "trades.log",
    rotation="1 day",
    retention="30 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    filter=lambda record: "trade" in record["extra"] or "order" in record["extra"],
)

# Create bound loggers
trade_logger = logger.bind(trade=True)
order_logger = logger.bind(order=True)


class OrderExecutor:
    """Executes orders through IBKR TWS."""

    def __init__(self, paper_mode: bool = True):
        """Initialize order executor.

        Args:
            paper_mode: If True, connect to paper trading port (7497).
                       If False, connect to live trading port (7496).
        """
        self.paper_mode = paper_mode
        self.ib = IB()

        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        # Paper trading uses 7497, live uses 7496
        default_port = 7497 if paper_mode else 7496
        self.port = int(os.getenv("IBKR_PORT", default_port))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", 1)) + 10  # Different client ID from data

        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", 0.01))
        self.max_leverage = float(os.getenv("MAX_LEVERAGE", 4))

        self._connected = False
        self._active_orders: dict[str, list[int]] = {}  # symbol -> order IDs

    def connect(self) -> bool:
        """Connect to IBKR TWS/Gateway.

        Returns:
            True if connection successful, False otherwise.
        """
        if self._connected:
            logger.warning("Already connected to IBKR")
            return True

        mode = "PAPER" if self.paper_mode else "LIVE"
        try:
            logger.info(f"Connecting to IBKR ({mode}) at {self.host}:{self.port}")
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            logger.success(f"Connected to IBKR ({mode}) successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")

    def _create_bracket_order(
        self,
        action: str,
        quantity: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
    ) -> tuple[MarketOrder, StopOrder, LimitOrder]:
        """Create a bracket order with entry, stop loss, and take profit.

        Args:
            action: 'BUY' or 'SELL'.
            quantity: Number of shares.
            entry_price: Entry price (used for reference, order is market).
            stop_price: Stop loss price.
            target_price: Take profit price.

        Returns:
            Tuple of (parent_order, stop_order, profit_order).
        """
        # Parent order - Market order for immediate entry
        parent = MarketOrder(action, quantity)
        parent.orderId = self.ib.client.getReqId()
        parent.transmit = False  # Don't transmit until children are attached

        # Determine child order actions
        exit_action = "SELL" if action == "BUY" else "BUY"

        # Stop loss order
        stop = StopOrder(exit_action, quantity, stop_price)
        stop.orderId = self.ib.client.getReqId()
        stop.parentId = parent.orderId
        stop.transmit = False

        # Take profit order
        profit = LimitOrder(exit_action, quantity, target_price)
        profit.orderId = self.ib.client.getReqId()
        profit.parentId = parent.orderId
        profit.transmit = True  # Transmit all orders when this one is placed

        return parent, stop, profit

    def execute_signal(self, signal: ORBSignal, account_value: float) -> list[int] | None:
        """Execute an ORB signal by placing a bracket order.

        Args:
            signal: ORBSignal with entry/stop/target prices.
            account_value: Current account value for position sizing.

        Returns:
            List of order IDs if successful, None otherwise.
        """
        if not self._connected:
            logger.error("Not connected to IBKR. Call connect() first.")
            return None

        # Calculate position size
        quantity = self._calculate_position_size(
            account_value, signal.entry_price, signal.risk_per_share
        )

        if quantity <= 0:
            logger.warning(f"Position size is 0, cannot execute signal")
            return None

        # Determine action
        action = "BUY" if signal.direction == Direction.LONG else "SELL"

        # Create contract
        contract = Stock(signal.symbol, "SMART", "USD")
        try:
            self.ib.qualifyContracts(contract)
        except Exception as e:
            logger.error(f"Failed to qualify contract for {signal.symbol}: {e}")
            return None

        # Create bracket order
        parent, stop, profit = self._create_bracket_order(
            action=action,
            quantity=quantity,
            entry_price=signal.entry_price,
            stop_price=signal.stop_price,
            target_price=signal.target_price,
        )

        # Place orders
        try:
            order_logger.info(f"Placing {action} bracket order for {quantity} {signal.symbol}")
            order_logger.info(f"  Entry: MARKET (target ~${signal.entry_price:.2f})")
            order_logger.info(f"  Stop: ${signal.stop_price:.2f}")
            order_logger.info(f"  Target: ${signal.target_price:.2f}")

            parent_trade = self.ib.placeOrder(contract, parent)
            stop_trade = self.ib.placeOrder(contract, stop)
            profit_trade = self.ib.placeOrder(contract, profit)

            order_ids = [parent.orderId, stop.orderId, profit.orderId]
            self._active_orders[signal.symbol] = order_ids

            order_logger.success(f"Bracket order placed. Order IDs: {order_ids}")

            # Set up fill callback
            parent_trade.filledEvent += lambda trade: self._on_fill(trade, "ENTRY")
            stop_trade.filledEvent += lambda trade: self._on_fill(trade, "STOP")
            profit_trade.filledEvent += lambda trade: self._on_fill(trade, "TARGET")

            return order_ids

        except Exception as e:
            logger.error(f"Failed to place bracket order: {e}")
            return None

    def _calculate_position_size(
        self, account_value: float, entry_price: float, risk_per_share: float
    ) -> int:
        """Calculate position size based on risk and leverage constraints.

        Formula: shares = int(min(account * risk_pct / risk_per_share, max_leverage * account / price))
        """
        if risk_per_share <= 0 or entry_price <= 0:
            return 0

        risk_based_shares = account_value * self.risk_per_trade / risk_per_share
        leverage_based_shares = self.max_leverage * account_value / entry_price

        shares = int(min(risk_based_shares, leverage_based_shares))
        return max(shares, 0)

    def _on_fill(self, trade, fill_type: str) -> None:
        """Handle order fill events."""
        fill = trade.fills[-1] if trade.fills else None
        if fill:
            trade_logger.info(
                f"ORDER FILLED ({fill_type}): {trade.contract.symbol} "
                f"{fill.execution.side} {fill.execution.shares} @ ${fill.execution.price:.2f}"
            )

    def cancel_orders(self, order_ids: list[int]) -> bool:
        """Cancel open orders by ID.

        Args:
            order_ids: List of order IDs to cancel.

        Returns:
            True if cancellation requests sent, False otherwise.
        """
        if not self._connected:
            logger.error("Not connected to IBKR")
            return False

        try:
            for order_id in order_ids:
                # Find the order in open orders
                for trade in self.ib.openTrades():
                    if trade.order.orderId == order_id:
                        self.ib.cancelOrder(trade.order)
                        order_logger.info(f"Cancelled order {order_id}")
                        break

            return True
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Close any open position in a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            True if position closed or no position, False on error.
        """
        if not self._connected:
            logger.error("Not connected to IBKR")
            return False

        position = self.get_position(symbol)

        if position == 0:
            logger.info(f"No open position in {symbol}")
            return True

        # Cancel any active orders first
        if symbol in self._active_orders:
            self.cancel_orders(self._active_orders[symbol])
            del self._active_orders[symbol]

        # Create market order to flatten
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)

        action = "SELL" if position > 0 else "BUY"
        quantity = abs(position)

        order = MarketOrder(action, quantity)

        try:
            order_logger.info(f"Closing position: {action} {quantity} {symbol}")
            trade = self.ib.placeOrder(contract, order)
            trade.filledEvent += lambda t: self._on_fill(t, "CLOSE")
            return True
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False

    def get_position(self, symbol: str) -> int:
        """Get current position size for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Position size (positive for long, negative for short, 0 for flat).
        """
        if not self._connected:
            return 0

        for position in self.ib.positions():
            if position.contract.symbol == symbol:
                return int(position.position)

        return 0

    def get_account_value(self) -> float:
        """Get current account balance.

        Returns:
            Net liquidation value of the account.
        """
        if not self._connected:
            return float(os.getenv("STARTING_CAPITAL", 25000))

        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)

        # Fallback to starting capital
        return float(os.getenv("STARTING_CAPITAL", 25000))


class PaperTrader:
    """Combines ORBMonitor and OrderExecutor for automated paper trading."""

    MARKET_CLOSE = time(16, 0)
    EOD_CLOSE_TIME = time(15, 55)  # Close positions 5 min before market close

    def __init__(self, symbol: str = None, paper_mode: bool = True):
        """Initialize paper trader.

        Args:
            symbol: Stock ticker symbol.
            paper_mode: If True, use paper trading.
        """
        self.symbol = symbol or os.getenv("SYMBOL", "TSLA")
        self.paper_mode = paper_mode

        self.monitor = ORBMonitor(symbol=self.symbol)
        self.executor = OrderExecutor(paper_mode=paper_mode)

        self._order_ids: list[int] | None = None
        self._signal_executed = False
        self._position_closed = False

    def _now_et(self) -> datetime:
        """Get current time in Eastern Time."""
        return datetime.now(ET)

    def _on_signal_generated(self) -> None:
        """Called when ORB signal is generated at 9:35."""
        signal = self.monitor.get_todays_signal()
        if signal is None or self._signal_executed:
            return

        trade_logger.info("=" * 60)
        trade_logger.info("ORB SIGNAL DETECTED - EXECUTING TRADE")
        trade_logger.info("=" * 60)
        trade_logger.info(f"Symbol: {signal.symbol}")
        trade_logger.info(f"Direction: {signal.direction.value}")
        trade_logger.info(f"Entry: ${signal.entry_price:.2f}")
        trade_logger.info(f"Stop: ${signal.stop_price:.2f}")
        trade_logger.info(f"Target: ${signal.target_price:.2f}")

        # Get account value and execute
        account_value = self.executor.get_account_value()
        trade_logger.info(f"Account Value: ${account_value:,.2f}")

        self._order_ids = self.executor.execute_signal(signal, account_value)

        if self._order_ids:
            self._signal_executed = True
            trade_logger.success(f"Trade executed successfully")
        else:
            trade_logger.error("Failed to execute trade")

    def _check_eod_close(self) -> None:
        """Check if it's time to close positions for EOD."""
        if self._position_closed:
            return

        current_time = self._now_et().time()

        if current_time >= self.EOD_CLOSE_TIME:
            position = self.executor.get_position(self.symbol)
            if position != 0:
                trade_logger.info("EOD approaching - closing position")
                self.executor.close_position(self.symbol)
                self._position_closed = True

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        import time as time_module

        while True:
            # Let ib_insync process events
            self.executor.ib.sleep(0.1)
            self.monitor.data_manager.ib.sleep(0.1)

            # Check if signal was generated
            status = self.monitor.get_status()
            if status == TradeStatus.SIGNAL_GENERATED and not self._signal_executed:
                self._on_signal_generated()

            # Check for EOD close
            if self._signal_executed:
                self._check_eod_close()

            # Check if trade is closed
            if status == TradeStatus.TRADE_CLOSED:
                trade_logger.info("Trade closed by stop/target")
                break

            # Check if market is closed
            if not self.monitor._is_market_open() and self._now_et().time() > self.MARKET_CLOSE:
                trade_logger.info("Market closed for the day")
                break

            time_module.sleep(0.5)

    def start(self) -> None:
        """Start the paper trader."""
        mode = "PAPER" if self.paper_mode else "LIVE"
        trade_logger.info(f"Starting PaperTrader ({mode}) for {self.symbol}")

        # Connect executor
        if not self.executor.connect():
            logger.error("Failed to connect executor to IBKR")
            return

        # Start monitor
        if not self.monitor.start():
            logger.error("Failed to start monitor")
            self.executor.disconnect()
            return

        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            trade_logger.info("Paper trader interrupted")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the paper trader and clean up."""
        trade_logger.info("Stopping PaperTrader")

        # Close any open position
        position = self.executor.get_position(self.symbol)
        if position != 0:
            trade_logger.info(f"Closing remaining position: {position} shares")
            self.executor.close_position(self.symbol)

        self.monitor.stop()
        self.executor.disconnect()

        trade_logger.info("PaperTrader stopped")


if __name__ == "__main__":
    import signal

    logger.info("Starting Order Executor / Paper Trader test")

    symbol = os.getenv("SYMBOL", "TSLA")

    # Test OrderExecutor connection and account info
    logger.info("=" * 60)
    logger.info("Testing OrderExecutor")
    logger.info("=" * 60)

    executor = OrderExecutor(paper_mode=True)

    if executor.connect():
        account_value = executor.get_account_value()
        logger.info(f"Account Value: ${account_value:,.2f}")

        position = executor.get_position(symbol)
        logger.info(f"Current {symbol} Position: {position}")

        executor.disconnect()
    else:
        logger.error("Could not connect to IBKR for executor test")

    # Test PaperTrader (comment out for quick test)
    logger.info("=" * 60)
    logger.info("Testing PaperTrader")
    logger.info("=" * 60)
    logger.info(f"Starting paper trader for {symbol}")
    logger.info("Press Ctrl+C to stop")

    trader = PaperTrader(symbol=symbol, paper_mode=True)

    def shutdown(signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutting down...")
        trader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    trader.start()
