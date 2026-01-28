"""Order execution for ORB trading strategy."""

import json
import os
import sys
import threading
import time as time_module
from datetime import datetime, time
from pathlib import Path
from typing import Any, Callable

import pytz
from dotenv import load_dotenv
from ib_insync import IB, LimitOrder, MarketOrder, Stock, StopOrder
from loguru import logger

from src.strategy.orb import Direction, ORBSignal
from src.trading.monitor import ORBMonitor, TradeStatus

load_dotenv()

# Reconnection constants
RECONNECT_DELAY = 5  # seconds
MAX_RECONNECT_ATTEMPTS = 5

# Eastern Time zone
ET = pytz.timezone("America/New_York")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
ACTIVE_TRADES_FILE = LOGS_DIR / "active_trades.json"  # Stores multiple trades by symbol


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

# Set up file logging for trades - comprehensive logging without filtering
LOGS_DIR.mkdir(exist_ok=True)
logger.add(
    LOGS_DIR / "trades.log",
    rotation="10 MB",
    retention="90 days",
    level="DEBUG",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    filter=lambda record: record["extra"].get("file_log", False),
    backtrace=True,
    diagnose=True,
)

# Create a file logger that logs to both console and file
file_logger = logger.bind(file_log=True)


class OrderExecutor:
    """Executes orders through IBKR TWS."""

    def __init__(self, paper_mode: bool = True, client_id_offset: int = 0):
        """Initialize order executor.

        Args:
            paper_mode: If True, connect to paper trading port (7497).
                       If False, connect to live trading port (7496).
            client_id_offset: Offset to add to base client ID for unique connections.
        """
        self.paper_mode = paper_mode
        self.ib = IB()

        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        # Paper trading uses 7497, live uses 7496
        default_port = 7497 if paper_mode else 7496
        self.port = int(os.getenv("IBKR_PORT", default_port))
        # Add 10 to separate from data connections, plus symbol-specific offset
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", 1)) + 10 + client_id_offset

        self.risk_per_trade = float(os.getenv("RISK_PER_TRADE", 0.01))
        self.max_leverage = float(os.getenv("MAX_LEVERAGE", 4))

        self._connected = False
        self._active_orders: dict[str, list[int]] = {}  # symbol -> order IDs

        # Reconnection state
        self._reconnect_attempts = 0
        self._reconnecting = False
        self._disconnect_callback: Callable[[], None] | None = None
        self._reconnect_callback: Callable[[bool], None] | None = None  # bool = position_ok

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

        mode = "PAPER" if self.paper_mode else "LIVE"
        try:
            logger.info(f"Connecting to IBKR ({mode}) at {self.host}:{self.port}")
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self._connected = True
            self._reconnect_attempts = 0  # Reset on successful connect
            file_logger.success(
                f"IBKR_CONNECT | mode={mode} host={self.host} port={self.port} "
                f"client_id={self.client_id}"
            )
            return True
        except Exception as e:
            file_logger.error(f"IBKR_CONNECT_ERROR | error={e} type={type(e).__name__}")
            return False

    def disconnect(self) -> None:
        """Disconnect from IBKR."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            file_logger.info("IBKR_DISCONNECT | status=disconnected")

    def set_disconnect_callback(self, callback: Callable[[], None]) -> None:
        """Set callback to be called on disconnect."""
        self._disconnect_callback = callback

    def set_reconnect_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback to be called after successful reconnect.

        Args:
            callback: Function that takes a bool (True if position is intact).
        """
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
            file_logger.error(
                f"EXECUTOR_DISCONNECT | error_code={errorCode} message={errorString}"
            )
            self._connected = False
            if not self._reconnecting:
                self._handle_disconnect()
        elif errorCode == 1101:
            file_logger.warning(
                f"EXECUTOR_RECONNECT | error_code={errorCode} message={errorString} data_lost=True"
            )
            self._connected = True
            self._reconnecting = False
            self._reconnect_attempts = 0
            self._verify_and_notify()
        elif errorCode == 1102:
            file_logger.info(
                f"EXECUTOR_RECONNECT | error_code={errorCode} message={errorString} data_maintained=True"
            )
            self._connected = True
            self._reconnecting = False
            self._reconnect_attempts = 0
            self._verify_and_notify()
        elif errorCode >= 2000:
            # Warning codes - log but don't treat as disconnect
            file_logger.warning(f"EXECUTOR_WARNING | code={errorCode} message={errorString}")

    def _on_disconnected(self) -> None:
        """Handle IB disconnected event."""
        file_logger.warning("EXECUTOR_DISCONNECTED | Connection to IBKR lost")
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
                file_logger.error(f"DISCONNECT_CALLBACK_ERROR | error={e}")

        # Start reconnection in a separate thread
        reconnect_thread = threading.Thread(target=self._reconnect_loop, daemon=True)
        reconnect_thread.start()

    def _reconnect_loop(self) -> None:
        """Attempt to reconnect with retries."""
        while self._reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
            self._reconnect_attempts += 1
            file_logger.info(
                f"EXECUTOR_RECONNECT_ATTEMPT | attempt={self._reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS} "
                f"waiting={RECONNECT_DELAY}s"
            )

            time_module.sleep(RECONNECT_DELAY)

            try:
                # Create new IB instance
                self.ib = IB()
                self.ib.errorEvent += self._on_error
                self.ib.disconnectedEvent += self._on_disconnected

                mode = "PAPER" if self.paper_mode else "LIVE"
                file_logger.info(
                    f"EXECUTOR_RECONNECT | Attempting connection to {self.host}:{self.port} ({mode})"
                )
                self.ib.connect(self.host, self.port, clientId=self.client_id)
                self._connected = True
                self._reconnecting = False
                self._reconnect_attempts = 0

                file_logger.success(
                    f"EXECUTOR_RECONNECT_SUCCESS | Connected after {self._reconnect_attempts} attempts"
                )

                # Verify position and notify
                self._verify_and_notify()
                return

            except Exception as e:
                file_logger.error(
                    f"EXECUTOR_RECONNECT_FAILED | attempt={self._reconnect_attempts} error={e}"
                )

        # Max attempts reached
        file_logger.error(
            f"EXECUTOR_RECONNECT_EXHAUSTED | max_attempts={MAX_RECONNECT_ATTEMPTS} giving_up=True"
        )
        self._reconnecting = False

    def _verify_and_notify(self) -> None:
        """Verify position status after reconnect and notify via callback."""
        position_ok = True

        # Check if we have any tracked positions
        for symbol in list(self._active_orders.keys()):
            position = self.get_position(symbol)
            open_orders = self.get_open_orders_for_symbol(symbol)

            file_logger.info(
                f"POSITION_VERIFY | symbol={symbol} position={position} "
                f"open_orders={len(open_orders)}"
            )

            # If we have a position but no orders, something is wrong
            if position != 0 and len(open_orders) == 0:
                file_logger.warning(
                    f"POSITION_UNPROTECTED | symbol={symbol} position={position} "
                    f"no_stop_target=True"
                )
                position_ok = False

        if self._reconnect_callback:
            try:
                self._reconnect_callback(position_ok)
            except Exception as e:
                file_logger.error(f"RECONNECT_CALLBACK_ERROR | error={e}")

    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self._connected and self.ib.isConnected()

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
            file_logger.error(f"ORDER_ERROR | symbol={signal.symbol} reason=not_connected")
            return None

        # Calculate position size
        quantity = self._calculate_position_size(
            account_value, signal.entry_price, signal.risk_per_share
        )

        if quantity <= 0:
            file_logger.warning(
                f"ORDER_SKIP | symbol={signal.symbol} reason=position_size_zero "
                f"account_value={account_value} entry_price={signal.entry_price} "
                f"risk_per_share={signal.risk_per_share}"
            )
            return None

        # Determine action
        action = "BUY" if signal.direction == Direction.LONG else "SELL"

        # Create contract
        contract = Stock(signal.symbol, "SMART", "USD")
        try:
            self.ib.qualifyContracts(contract)
        except Exception as e:
            file_logger.error(
                f"CONTRACT_ERROR | symbol={signal.symbol} error={e} type={type(e).__name__}"
            )
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
            file_logger.info(
                f"ORDER_SUBMIT | symbol={signal.symbol} action={action} quantity={quantity} "
                f"type=bracket entry_price=${signal.entry_price:.2f} "
                f"stop_price=${signal.stop_price:.2f} target_price=${signal.target_price:.2f}"
            )

            parent_trade = self.ib.placeOrder(contract, parent)
            stop_trade = self.ib.placeOrder(contract, stop)
            profit_trade = self.ib.placeOrder(contract, profit)

            order_ids = [parent.orderId, stop.orderId, profit.orderId]
            self._active_orders[signal.symbol] = order_ids

            file_logger.success(
                f"ORDER_ACCEPTED | symbol={signal.symbol} order_ids={order_ids} "
                f"parent_id={parent.orderId} stop_id={stop.orderId} target_id={profit.orderId}"
            )

            # Set up fill callback
            parent_trade.filledEvent += lambda trade: self._on_fill(trade, "ENTRY", signal.symbol)
            stop_trade.filledEvent += lambda trade: self._on_fill(trade, "STOP", signal.symbol)
            profit_trade.filledEvent += lambda trade: self._on_fill(trade, "TARGET", signal.symbol)

            return order_ids

        except Exception as e:
            file_logger.error(f"ORDER_ERROR | symbol={signal.symbol} error={e} type={type(e).__name__}")
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

    def _on_fill(self, trade, fill_type: str, symbol: str = None) -> None:
        """Handle order fill events."""
        fill = trade.fills[-1] if trade.fills else None
        if fill:
            sym = symbol or trade.contract.symbol
            file_logger.info(
                f"FILL | symbol={sym} type={fill_type} side={fill.execution.side} "
                f"shares={fill.execution.shares} price=${fill.execution.price:.2f} "
                f"order_id={trade.order.orderId} exec_id={fill.execution.execId} "
                f"time={fill.execution.time}"
            )

    def cancel_orders(self, order_ids: list[int]) -> bool:
        """Cancel open orders by ID.

        Args:
            order_ids: List of order IDs to cancel.

        Returns:
            True if cancellation requests sent, False otherwise.
        """
        if not self._connected:
            file_logger.error("ORDER_CANCEL_ERROR | reason=not_connected")
            return False

        try:
            for order_id in order_ids:
                # Find the order in open orders
                for trade in self.ib.openTrades():
                    if trade.order.orderId == order_id:
                        self.ib.cancelOrder(trade.order)
                        file_logger.info(
                            f"ORDER_CANCEL | order_id={order_id} "
                            f"symbol={trade.contract.symbol}"
                        )
                        break

            return True
        except Exception as e:
            file_logger.error(f"ORDER_CANCEL_ERROR | order_ids={order_ids} error={e}")
            return False

    def close_position(self, symbol: str) -> bool:
        """Close any open position in a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            True if position closed or no position, False on error.
        """
        if not self._connected:
            file_logger.error(f"CLOSE_POSITION_ERROR | symbol={symbol} reason=not_connected")
            return False

        position = self.get_position(symbol)

        if position == 0:
            file_logger.info(f"CLOSE_POSITION | symbol={symbol} position=0 action=none")
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
            file_logger.info(
                f"CLOSE_POSITION | symbol={symbol} action={action} "
                f"quantity={quantity} type=market"
            )
            trade = self.ib.placeOrder(contract, order)
            trade.filledEvent += lambda t: self._on_fill(t, "CLOSE", symbol)
            return True
        except Exception as e:
            file_logger.error(f"CLOSE_POSITION_ERROR | symbol={symbol} error={e}")
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

    def get_open_orders_for_symbol(self, symbol: str) -> list[dict]:
        """Get all open orders for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            List of order info dicts with order_id, order_type, action, quantity, price.
        """
        if not self._connected:
            return []

        orders = []
        for trade in self.ib.openTrades():
            if trade.contract.symbol == symbol:
                order = trade.order
                order_info = {
                    "order_id": order.orderId,
                    "order_type": order.orderType,
                    "action": order.action,
                    "quantity": int(order.totalQuantity),
                }
                # Add price based on order type
                if order.orderType == "STP":
                    order_info["price"] = order.auxPrice
                elif order.orderType == "LMT":
                    order_info["price"] = order.lmtPrice
                orders.append(order_info)

        return orders

    def place_exit_orders(
        self,
        symbol: str,
        position: int,
        stop_price: float,
        target_price: float,
    ) -> list[int] | None:
        """Place stop and target orders for an existing position.

        Args:
            symbol: Stock ticker symbol.
            position: Current position size (positive=long, negative=short).
            stop_price: Stop loss price.
            target_price: Take profit price.

        Returns:
            List of order IDs [stop_id, target_id] if successful, None otherwise.
        """
        if not self._connected:
            file_logger.error(f"EXIT_ORDER_ERROR | symbol={symbol} reason=not_connected")
            return None

        if position == 0:
            file_logger.warning(f"EXIT_ORDER_SKIP | symbol={symbol} reason=no_position")
            return None

        # Create contract
        contract = Stock(symbol, "SMART", "USD")
        try:
            self.ib.qualifyContracts(contract)
        except Exception as e:
            file_logger.error(f"CONTRACT_ERROR | symbol={symbol} error={e}")
            return None

        # Determine exit action based on position direction
        quantity = abs(position)
        exit_action = "SELL" if position > 0 else "BUY"

        # Create stop loss order
        stop = StopOrder(exit_action, quantity, stop_price)
        stop.orderId = self.ib.client.getReqId()
        stop.transmit = True

        # Create take profit order
        profit = LimitOrder(exit_action, quantity, target_price)
        profit.orderId = self.ib.client.getReqId()
        profit.transmit = True

        try:
            file_logger.info(
                f"EXIT_ORDER_SUBMIT | symbol={symbol} action={exit_action} "
                f"quantity={quantity} stop_price=${stop_price:.2f} "
                f"target_price=${target_price:.2f}"
            )

            stop_trade = self.ib.placeOrder(contract, stop)
            profit_trade = self.ib.placeOrder(contract, profit)

            order_ids = [stop.orderId, profit.orderId]
            self._active_orders[symbol] = order_ids

            file_logger.success(
                f"EXIT_ORDER_ACCEPTED | symbol={symbol} "
                f"stop_id={stop.orderId} target_id={profit.orderId}"
            )

            # Set up fill callbacks
            stop_trade.filledEvent += lambda trade: self._on_fill(trade, "STOP", symbol)
            profit_trade.filledEvent += lambda trade: self._on_fill(trade, "TARGET", symbol)

            return order_ids

        except Exception as e:
            file_logger.error(f"EXIT_ORDER_ERROR | symbol={symbol} error={e}")
            return None

    def get_current_price(self, symbol: str) -> float | None:
        """Get current market price for a symbol.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Current price or None if unavailable.
        """
        if not self._connected:
            return None

        contract = Stock(symbol, "SMART", "USD")
        try:
            self.ib.qualifyContracts(contract)
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(1)  # Wait for data
            price = ticker.marketPrice()
            self.ib.cancelMktData(contract)
            if price and price > 0:
                return float(price)
        except Exception as e:
            file_logger.error(f"PRICE_ERROR | symbol={symbol} error={e}")

        return None


class PaperTrader:
    """Combines ORBMonitor and OrderExecutor for automated paper trading."""

    MARKET_CLOSE = time(16, 0)
    EOD_CLOSE_TIME = time(15, 55)  # Close positions 5 min before market close

    def __init__(self, symbol: str = None, paper_mode: bool = True, client_id_offset: int = 0):
        """Initialize paper trader.

        Args:
            symbol: Stock ticker symbol.
            paper_mode: If True, use paper trading.
            client_id_offset: Offset to add to base client IDs for unique connections.
                             Each PaperTrader uses 2 client IDs (data + executor).
        """
        self.symbol = symbol or os.getenv("SYMBOL", "TSLA")
        self.paper_mode = paper_mode

        # Each PaperTrader uses 2 client IDs: one for data, one for executor
        # Data manager uses offset, executor uses offset (plus its internal +10)
        self.monitor = ORBMonitor(symbol=self.symbol, client_id_offset=client_id_offset)
        self.executor = OrderExecutor(paper_mode=paper_mode, client_id_offset=client_id_offset)

        self._order_ids: list[int] | None = None
        self._signal_executed = False
        self._position_closed = False
        self._paused_for_reconnect = False

    def _now_et(self) -> datetime:
        """Get current time in Eastern Time."""
        return datetime.now(ET)

    def _save_trade_state(
        self,
        entry_price: float,
        stop_price: float,
        target_price: float,
        shares: int,
        direction: str,
        order_ids: list[int] | None = None,
    ) -> None:
        """Save active trade state to file for recovery.

        Args:
            entry_price: Entry price of the trade.
            stop_price: Stop loss price.
            target_price: Take profit price.
            shares: Number of shares.
            direction: 'LONG' or 'SHORT'.
            order_ids: List of active order IDs.
        """
        trade_data = {
            "symbol": self.symbol,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "shares": shares,
            "direction": direction,
            "order_ids": order_ids or [],
            "opened_at": self._now_et().isoformat(),
            "updated_at": self._now_et().isoformat(),
        }
        try:
            # Load existing trades
            all_trades = {}
            if ACTIVE_TRADES_FILE.exists():
                with open(ACTIVE_TRADES_FILE, "r") as f:
                    all_trades = json.load(f)

            # Update this symbol's trade
            all_trades[self.symbol] = trade_data

            with open(ACTIVE_TRADES_FILE, "w") as f:
                json.dump(all_trades, f, indent=2)
            file_logger.info(
                f"STATE_SAVE | symbol={self.symbol} entry=${entry_price:.2f} "
                f"stop=${stop_price:.2f} target=${target_price:.2f} "
                f"shares={shares} direction={direction}"
            )
        except Exception as e:
            file_logger.error(f"STATE_SAVE_ERROR | symbol={self.symbol} error={e}")

    def _load_trade_state(self) -> dict[str, Any] | None:
        """Load trade state from file for this symbol.

        Returns:
            Trade state dict or None if no state exists for this symbol.
        """
        if not ACTIVE_TRADES_FILE.exists():
            return None

        try:
            with open(ACTIVE_TRADES_FILE, "r") as f:
                all_trades = json.load(f)

            # Get this symbol's trade
            state = all_trades.get(self.symbol)
            if not state:
                return None

            file_logger.info(
                f"STATE_LOAD | symbol={state['symbol']} entry=${state['entry_price']:.2f} "
                f"stop=${state['stop_price']:.2f} target=${state['target_price']:.2f} "
                f"shares={state['shares']} direction={state['direction']} "
                f"opened_at={state['opened_at']}"
            )
            return state

        except Exception as e:
            file_logger.error(f"STATE_LOAD_ERROR | symbol={self.symbol} error={e}")
            return None

    def _clear_trade_state(self) -> None:
        """Remove this symbol's trade from the state file."""
        if not ACTIVE_TRADES_FILE.exists():
            return

        try:
            with open(ACTIVE_TRADES_FILE, "r") as f:
                all_trades = json.load(f)

            if self.symbol in all_trades:
                del all_trades[self.symbol]
                file_logger.info(f"STATE_CLEAR | symbol={self.symbol} removed from state file")

                # Write back or delete file if empty
                if all_trades:
                    with open(ACTIVE_TRADES_FILE, "w") as f:
                        json.dump(all_trades, f, indent=2)
                else:
                    ACTIVE_TRADES_FILE.unlink()
                    file_logger.info("STATE_CLEAR | state file deleted (no active trades)")

        except Exception as e:
            file_logger.error(f"STATE_CLEAR_ERROR | symbol={self.symbol} error={e}")

    def check_existing_position(self) -> dict[str, Any] | None:
        """Check for existing open position in IBKR.

        Returns:
            Position info dict with position, avg_cost, current_price, pnl
            or None if no position.
        """
        position = self.executor.get_position(self.symbol)

        if position == 0:
            return None

        # Get current price for P&L calculation
        current_price = self.executor.get_current_price(self.symbol)

        # Try to get average cost from IBKR positions
        avg_cost = None
        if self.executor._connected:
            for pos in self.executor.ib.positions():
                if pos.contract.symbol == self.symbol:
                    avg_cost = float(pos.avgCost)
                    break

        # Calculate P&L if we have the data
        unrealized_pnl = None
        if current_price and avg_cost:
            if position > 0:  # Long
                unrealized_pnl = (current_price - avg_cost) * position
            else:  # Short
                unrealized_pnl = (avg_cost - current_price) * abs(position)

        position_info = {
            "position": position,
            "direction": "LONG" if position > 0 else "SHORT",
            "shares": abs(position),
            "avg_cost": avg_cost,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
        }

        file_logger.info(
            f"EXISTING_POSITION | symbol={self.symbol} position={position} "
            f"direction={position_info['direction']} shares={position_info['shares']} "
            f"avg_cost=${avg_cost:.2f if avg_cost else 0:.2f} "
            f"current_price=${current_price:.2f if current_price else 0:.2f} "
            f"unrealized_pnl=${unrealized_pnl:.2f if unrealized_pnl else 0:.2f}"
        )

        return position_info

    def _recover_position(self) -> bool:
        """Attempt to recover an existing position on startup.

        Returns:
            True if position was recovered and managed, False otherwise.
        """
        # Check for existing position
        position_info = self.check_existing_position()
        if not position_info:
            file_logger.info(f"RECOVERY | symbol={self.symbol} no existing position")
            return False

        file_logger.info("=" * 60)
        file_logger.info(f"RECOVERY | Existing position detected for {self.symbol}")
        file_logger.info("=" * 60)

        # Load saved trade state
        trade_state = self._load_trade_state()

        if not trade_state:
            file_logger.warning(
                f"RECOVERY_WARNING | symbol={self.symbol} "
                f"position exists but no trade state file found"
            )
            # We have a position but no state - can't safely manage it
            return False

        # Verify position matches state
        if position_info["shares"] != trade_state["shares"]:
            file_logger.warning(
                f"RECOVERY_MISMATCH | state_shares={trade_state['shares']} "
                f"actual_shares={position_info['shares']}"
            )

        # Check if stop/target orders are still active
        open_orders = self.executor.get_open_orders_for_symbol(self.symbol)
        has_stop = any(o["order_type"] == "STP" for o in open_orders)
        has_target = any(o["order_type"] == "LMT" for o in open_orders)

        file_logger.info(
            f"RECOVERY_ORDERS | symbol={self.symbol} "
            f"open_orders={len(open_orders)} has_stop={has_stop} has_target={has_target}"
        )

        # If missing stop or target, place new exit orders
        if not has_stop or not has_target:
            file_logger.info(
                f"RECOVERY_REORDER | symbol={self.symbol} "
                f"placing new exit orders stop=${trade_state['stop_price']:.2f} "
                f"target=${trade_state['target_price']:.2f}"
            )

            # Cancel any remaining orders first
            if open_orders:
                order_ids = [o["order_id"] for o in open_orders]
                self.executor.cancel_orders(order_ids)
                self.executor.ib.sleep(0.5)  # Wait for cancellation

            # Place new stop and target orders
            order_ids = self.executor.place_exit_orders(
                symbol=self.symbol,
                position=position_info["position"],
                stop_price=trade_state["stop_price"],
                target_price=trade_state["target_price"],
            )

            if order_ids:
                # Update state with new order IDs
                self._save_trade_state(
                    entry_price=trade_state["entry_price"],
                    stop_price=trade_state["stop_price"],
                    target_price=trade_state["target_price"],
                    shares=position_info["shares"],
                    direction=trade_state["direction"],
                    order_ids=order_ids,
                )
                self._order_ids = order_ids
            else:
                file_logger.error(f"RECOVERY_ERROR | failed to place exit orders")
                return False
        else:
            # Orders are active, just restore state
            self._order_ids = [o["order_id"] for o in open_orders]
            file_logger.info(
                f"RECOVERY_SUCCESS | symbol={self.symbol} "
                f"existing orders active order_ids={self._order_ids}"
            )

        # Mark that we have an active position
        self._signal_executed = True

        file_logger.success(
            f"RECOVERY_COMPLETE | symbol={self.symbol} "
            f"entry=${trade_state['entry_price']:.2f} "
            f"stop=${trade_state['stop_price']:.2f} "
            f"target=${trade_state['target_price']:.2f} "
            f"current_pnl=${position_info['unrealized_pnl']:.2f if position_info['unrealized_pnl'] else 0:.2f}"
        )

        return True

    def _on_executor_disconnect(self) -> None:
        """Handle executor disconnect event."""
        file_logger.warning(
            f"DISCONNECT_HANDLER | symbol={self.symbol} pausing_operations=True"
        )
        self._paused_for_reconnect = True

    def _on_executor_reconnect(self, position_ok: bool) -> None:
        """Handle executor reconnect event.

        Args:
            position_ok: True if position and orders are intact.
        """
        file_logger.info(
            f"RECONNECT_HANDLER | symbol={self.symbol} position_ok={position_ok}"
        )

        if not position_ok and self._signal_executed:
            # Position exists but orders are missing - need to restore protection
            trade_state = self._load_trade_state()
            if trade_state:
                position = self.executor.get_position(self.symbol)
                if position != 0:
                    file_logger.info(
                        f"RECONNECT_RESTORE | symbol={self.symbol} "
                        f"restoring stop/target orders"
                    )
                    order_ids = self.executor.place_exit_orders(
                        symbol=self.symbol,
                        position=position,
                        stop_price=trade_state["stop_price"],
                        target_price=trade_state["target_price"],
                    )
                    if order_ids:
                        self._order_ids = order_ids
                        self._save_trade_state(
                            entry_price=trade_state["entry_price"],
                            stop_price=trade_state["stop_price"],
                            target_price=trade_state["target_price"],
                            shares=abs(position),
                            direction=trade_state["direction"],
                            order_ids=order_ids,
                        )
                        file_logger.success(
                            f"RECONNECT_RESTORE_SUCCESS | symbol={self.symbol} "
                            f"order_ids={order_ids}"
                        )
                    else:
                        file_logger.error(
                            f"RECONNECT_RESTORE_FAILED | symbol={self.symbol}"
                        )

        self._paused_for_reconnect = False
        file_logger.info(f"RECONNECT_HANDLER | symbol={self.symbol} resuming_operations=True")

    def _on_data_disconnect(self) -> None:
        """Handle data manager disconnect event."""
        file_logger.warning(
            f"DATA_DISCONNECT_HANDLER | symbol={self.symbol}"
        )

    def _on_data_reconnect(self) -> None:
        """Handle data manager reconnect event."""
        file_logger.info(
            f"DATA_RECONNECT_HANDLER | symbol={self.symbol} data_feeds_restored=True"
        )

    def _on_signal_generated(self) -> None:
        """Called when ORB signal is generated at 9:35."""
        signal = self.monitor.get_todays_signal()
        if signal is None or self._signal_executed:
            return

        file_logger.info("=" * 60)
        file_logger.info("SIGNAL | ORB signal generated")
        file_logger.info("=" * 60)
        file_logger.info(
            f"SIGNAL | symbol={signal.symbol} direction={signal.direction.value} "
            f"entry=${signal.entry_price:.2f} stop=${signal.stop_price:.2f} "
            f"target=${signal.target_price:.2f} risk_per_share=${signal.risk_per_share:.2f}"
        )

        # Get account value and execute
        account_value = self.executor.get_account_value()
        file_logger.info(f"ACCOUNT | value=${account_value:,.2f}")

        self._order_ids = self.executor.execute_signal(signal, account_value)

        if self._order_ids:
            self._signal_executed = True
            file_logger.success(
                f"ORDER_PLACED | symbol={signal.symbol} direction={signal.direction.value} "
                f"order_ids={self._order_ids} entry=${signal.entry_price:.2f} "
                f"stop=${signal.stop_price:.2f} target=${signal.target_price:.2f}"
            )
            # Save trade state for recovery
            # Calculate shares from position size formula
            quantity = self.executor._calculate_position_size(
                account_value, signal.entry_price, signal.risk_per_share
            )
            self._save_trade_state(
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                shares=quantity,
                direction=signal.direction.value,
                order_ids=self._order_ids,
            )
        else:
            file_logger.error(f"ORDER_FAILED | symbol={signal.symbol} reason=execute_signal_returned_none")

    def _check_eod_close(self) -> None:
        """Check if it's time to close positions for EOD."""
        if self._position_closed:
            return

        current_time = self._now_et().time()

        if current_time >= self.EOD_CLOSE_TIME:
            position = self.executor.get_position(self.symbol)
            if position != 0:
                file_logger.info(
                    f"EOD_EXIT | symbol={self.symbol} position={position} "
                    f"time={current_time} reason=end_of_day_close"
                )
                self.executor.close_position(self.symbol)
                self._position_closed = True
                self._clear_trade_state()

    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        last_position = 0

        while True:
            try:
                # Let ib_insync process events
                if self.executor.is_connected():
                    self.executor.ib.sleep(0.1)
                if self.monitor.data_manager.is_connected():
                    self.monitor.data_manager.ib.sleep(0.1)

                # Skip operations if paused for reconnect
                if self._paused_for_reconnect:
                    time_module.sleep(0.5)
                    continue

                # Check if signal was generated
                status = self.monitor.get_status()
                if status == TradeStatus.SIGNAL_GENERATED and not self._signal_executed:
                    self._on_signal_generated()

                # Monitor position changes
                current_position = self.executor.get_position(self.symbol)
                if current_position != last_position:
                    file_logger.info(
                        f"POSITION_CHANGE | symbol={self.symbol} "
                        f"old_position={last_position} new_position={current_position}"
                    )
                    # If position went to zero, trade is closed
                    if current_position == 0 and last_position != 0:
                        self._clear_trade_state()
                    last_position = current_position

                # Check for EOD close
                if self._signal_executed:
                    self._check_eod_close()

                # Check if trade is closed
                if status == TradeStatus.TRADE_CLOSED:
                    file_logger.info(
                        f"EXIT | symbol={self.symbol} reason=stop_or_target_hit "
                        f"final_position={self.executor.get_position(self.symbol)}"
                    )
                    self._clear_trade_state()
                    break

                # Check if market is closed
                if not self.monitor._is_market_open() and self._now_et().time() > self.MARKET_CLOSE:
                    file_logger.info(f"MARKET_CLOSED | symbol={self.symbol} time={self._now_et().time()}")
                    break

                time_module.sleep(0.5)

            except Exception as e:
                file_logger.error(f"ERROR | symbol={self.symbol} error={e} type={type(e).__name__}")
                raise

    def start(self) -> None:
        """Start the paper trader."""
        mode = "PAPER" if self.paper_mode else "LIVE"
        file_logger.info("=" * 60)
        file_logger.info(
            f"BOT_START | symbol={self.symbol} mode={mode} "
            f"time={self._now_et().isoformat()} "
            f"risk_per_trade={self.executor.risk_per_trade} "
            f"max_leverage={self.executor.max_leverage}"
        )
        file_logger.info("=" * 60)

        # Connect executor
        if not self.executor.connect():
            file_logger.error(f"ERROR | reason=executor_connection_failed symbol={self.symbol}")
            return

        # Set up reconnection callbacks
        self.executor.set_disconnect_callback(self._on_executor_disconnect)
        self.executor.set_reconnect_callback(self._on_executor_reconnect)

        file_logger.info(f"CONNECTION | executor connected to IBKR host={self.executor.host} port={self.executor.port}")

        # Check for existing position and attempt recovery
        if self._recover_position():
            file_logger.info(f"RECOVERY | Resuming management of existing position")
        else:
            file_logger.info(f"RECOVERY | No existing position to recover")

        # Start monitor
        if not self.monitor.start():
            file_logger.error(f"ERROR | reason=monitor_start_failed symbol={self.symbol}")
            self.executor.disconnect()
            return

        # Set up data manager reconnection callbacks
        self.monitor.data_manager.set_disconnect_callback(self._on_data_disconnect)
        self.monitor.data_manager.set_reconnect_callback(self._on_data_reconnect)

        file_logger.info(f"MONITOR | started for {self.symbol}")

        close_on_exit = True
        try:
            self._monitor_loop()
        except KeyboardInterrupt:
            file_logger.info(f"BOT_INTERRUPT | symbol={self.symbol} reason=keyboard_interrupt")
            # Don't close position on keyboard interrupt - allow recovery on restart
            close_on_exit = False
        except Exception as e:
            file_logger.error(f"ERROR | symbol={self.symbol} error={e} type={type(e).__name__}")
            raise
        finally:
            self.stop(close_position=close_on_exit)

    def stop(self, close_position: bool = True) -> None:
        """Stop the paper trader and clean up.

        Args:
            close_position: If True, close any open position. If False, keep
                           position open for recovery on restart.
        """
        file_logger.info(f"BOT_STOPPING | symbol={self.symbol} time={self._now_et().isoformat()}")

        # Close any open position if requested
        position = self.executor.get_position(self.symbol)
        if position != 0 and close_position:
            file_logger.info(
                f"POSITION_CLOSE | symbol={self.symbol} position={position} "
                f"reason=bot_shutdown"
            )
            self.executor.close_position(self.symbol)
            self._clear_trade_state()
        elif position != 0:
            file_logger.info(
                f"POSITION_KEEP | symbol={self.symbol} position={position} "
                f"reason=restart_recovery_enabled"
            )

        self.monitor.stop()
        self.executor.disconnect()

        file_logger.info("=" * 60)
        file_logger.info(
            f"BOT_STOP | symbol={self.symbol} time={self._now_et().isoformat()} "
            f"signal_executed={self._signal_executed} position_closed={self._position_closed}"
        )
        file_logger.info("=" * 60)


class MultiSymbolPaperTrader:
    """Manages multiple PaperTrader instances for multi-symbol trading.

    Each symbol runs in its own thread with independent 1% risk allocation.
    """

    def __init__(self, symbols: list[str] = None, paper_mode: bool = True):
        """Initialize multi-symbol paper trader.

        Args:
            symbols: List of stock ticker symbols.
            paper_mode: If True, use paper trading.
        """
        self.symbols = symbols or get_symbols_from_env()
        self.paper_mode = paper_mode

        self._traders: dict[str, PaperTrader] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        file_logger.info(
            f"MULTI_INIT | symbols={self.symbols} count={len(self.symbols)} "
            f"mode={'PAPER' if paper_mode else 'LIVE'}"
        )

    def _run_symbol_trader(self, symbol: str, client_id_offset: int) -> None:
        """Run a single symbol trader in a thread.

        Args:
            symbol: Stock ticker symbol.
            client_id_offset: Unique client ID offset for this symbol's connections.
        """
        import asyncio

        # Create and set a new event loop for this thread (required by ib_insync)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            trader = PaperTrader(
                symbol=symbol,
                paper_mode=self.paper_mode,
                client_id_offset=client_id_offset,
            )
            with self._lock:
                self._traders[symbol] = trader

            file_logger.info(f"THREAD_START | symbol={symbol} client_id_offset={client_id_offset}")
            trader.start()

        except Exception as e:
            file_logger.error(f"THREAD_ERROR | symbol={symbol} error={e} type={type(e).__name__}")
        finally:
            file_logger.info(f"THREAD_END | symbol={symbol}")
            loop.close()

    def start(self) -> None:
        """Start all symbol traders in separate threads."""
        mode = "PAPER" if self.paper_mode else "LIVE"
        file_logger.info("=" * 60)
        file_logger.info(
            f"MULTI_START | symbols={self.symbols} mode={mode} "
            f"time={datetime.now(ET).isoformat()}"
        )
        file_logger.info("=" * 60)

        # Start a thread for each symbol with unique client ID offsets
        # Each symbol needs 2 client IDs (data + executor), so offset by 2 per symbol
        for i, symbol in enumerate(self.symbols):
            client_id_offset = i * 2  # 0, 2, 4, 6, ... for each symbol
            thread = threading.Thread(
                target=self._run_symbol_trader,
                args=(symbol, client_id_offset),
                name=f"Trader-{symbol}",
                daemon=True,
            )
            self._threads[symbol] = thread
            thread.start()
            file_logger.info(f"THREAD_LAUNCHED | symbol={symbol} client_id_offset={client_id_offset}")
            # Small delay to stagger connections
            time_module.sleep(1)

        file_logger.info(f"MULTI_RUNNING | all {len(self.symbols)} threads started")

        # Wait for all threads or stop event
        try:
            while not self._stop_event.is_set():
                # Check if any threads are still alive
                alive_threads = [s for s, t in self._threads.items() if t.is_alive()]
                if not alive_threads:
                    file_logger.info("MULTI_COMPLETE | all symbol threads finished")
                    break
                time_module.sleep(1)
        except KeyboardInterrupt:
            file_logger.info("MULTI_INTERRUPT | keyboard interrupt received")
            self.stop(close_positions=False)

    def stop(self, close_positions: bool = True) -> None:
        """Stop all symbol traders.

        Args:
            close_positions: If True, close all open positions.
        """
        file_logger.info(f"MULTI_STOPPING | close_positions={close_positions}")
        self._stop_event.set()

        with self._lock:
            for symbol, trader in self._traders.items():
                try:
                    file_logger.info(f"STOPPING | symbol={symbol}")
                    trader.stop(close_position=close_positions)
                except Exception as e:
                    file_logger.error(f"STOP_ERROR | symbol={symbol} error={e}")

        # Wait for threads to finish
        for symbol, thread in self._threads.items():
            if thread.is_alive():
                thread.join(timeout=5)
                if thread.is_alive():
                    file_logger.warning(f"THREAD_TIMEOUT | symbol={symbol} did not stop gracefully")

        file_logger.info("=" * 60)
        file_logger.info(
            f"MULTI_STOP | symbols={self.symbols} time={datetime.now(ET).isoformat()}"
        )
        file_logger.info("=" * 60)

    def get_status(self) -> dict[str, dict]:
        """Get status of all symbol traders.

        Returns:
            Dict mapping symbol to status info.
        """
        status = {}
        with self._lock:
            for symbol, trader in self._traders.items():
                status[symbol] = {
                    "thread_alive": self._threads[symbol].is_alive(),
                    "signal_executed": trader._signal_executed,
                    "position_closed": trader._position_closed,
                    "paused": trader._paused_for_reconnect,
                }
        return status


if __name__ == "__main__":
    import argparse
    import signal

    parser = argparse.ArgumentParser(description="ORB Paper Trader")
    parser.add_argument(
        "--symbols",
        type=str,
        help="Comma-separated list of symbols (overrides SYMBOLS env var)",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        default=True,
        help="Use paper trading mode (default)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live trading mode",
    )
    args = parser.parse_args()

    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = get_symbols_from_env()

    paper_mode = not args.live

    mode_str = "PAPER" if paper_mode else "LIVE"
    logger.info(f"Starting Multi-Symbol Paper Trader ({mode_str})")
    logger.info(f"Symbols: {symbols}")
    logger.info("Press Ctrl+C to stop")

    trader = MultiSymbolPaperTrader(symbols=symbols, paper_mode=paper_mode)

    def shutdown(signum, frame):
        """Handle shutdown signal."""
        logger.info("Shutting down (keeping positions for recovery)...")
        trader.stop(close_positions=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    trader.start()
