"""Opening Range Breakout (ORB) strategy implementation."""

from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum

import pandas as pd
from loguru import logger


class Direction(Enum):
    """Trade direction."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


@dataclass
class ORBSignal:
    """Opening Range Breakout signal."""

    date: datetime
    symbol: str
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    risk_per_share: float
    first_candle_open: float
    first_candle_high: float
    first_candle_low: float
    first_candle_close: float


class ORBSignalGenerator:
    """Generates Opening Range Breakout signals from market data."""

    MARKET_OPEN = time(9, 30)
    SECOND_BAR = time(9, 35)
    REWARD_RISK_RATIO = 10

    def __init__(self, symbol: str = "TSLA"):
        self.symbol = symbol

    def _get_first_candle(self, day_data: pd.DataFrame) -> pd.Series | None:
        """Get the 9:30 AM (market open) candle.

        Args:
            day_data: DataFrame with intraday data for a single day.

        Returns:
            Series with OHLCV data for the first candle, or None if not found.
        """
        day_data = day_data.copy()
        day_data["time"] = pd.to_datetime(day_data["date"]).dt.time

        first_candle = day_data[day_data["time"] == self.MARKET_OPEN]

        if first_candle.empty:
            return None

        return first_candle.iloc[0]

    def _get_second_candle(self, day_data: pd.DataFrame) -> pd.Series | None:
        """Get the 9:35 AM candle.

        Args:
            day_data: DataFrame with intraday data for a single day.

        Returns:
            Series with OHLCV data for the second candle, or None if not found.
        """
        day_data = day_data.copy()
        day_data["time"] = pd.to_datetime(day_data["date"]).dt.time

        second_candle = day_data[day_data["time"] == self.SECOND_BAR]

        if second_candle.empty:
            return None

        return second_candle.iloc[0]

    def _determine_direction(self, candle: pd.Series) -> Direction:
        """Determine trade direction based on first candle.

        Args:
            candle: Series with OHLCV data.

        Returns:
            LONG if close > open, SHORT if close < open, NONE if doji.
        """
        open_price = candle["open"]
        close_price = candle["close"]

        if close_price > open_price:
            return Direction.LONG
        elif close_price < open_price:
            return Direction.SHORT
        else:
            return Direction.NONE

    def generate_signal(self, day_data: pd.DataFrame) -> ORBSignal | None:
        """Generate an ORB signal for a single trading day.

        Entry is at the close of the second candle (9:35 bar).
        Stop loss is first candle low (long) or high (short).
        Target is entry +/- 10 * risk_per_share.

        Args:
            day_data: DataFrame with intraday data for a single day.

        Returns:
            ORBSignal if conditions are met, None otherwise.
        """
        first_candle = self._get_first_candle(day_data)
        if first_candle is None:
            logger.warning("First candle (9:30) not found")
            return None

        second_candle = self._get_second_candle(day_data)
        if second_candle is None:
            logger.warning("Second candle (9:35) not found")
            return None

        direction = self._determine_direction(first_candle)

        if direction == Direction.NONE:
            logger.debug("Doji candle detected, no signal generated")
            return None

        # Entry at second candle close
        entry_price = second_candle["close"]
        trade_date = pd.to_datetime(first_candle["date"])

        # Stop and target based on direction
        if direction == Direction.LONG:
            stop_price = first_candle["low"]
            risk_per_share = entry_price - stop_price
            target_price = entry_price + (self.REWARD_RISK_RATIO * risk_per_share)
        else:  # SHORT
            stop_price = first_candle["high"]
            risk_per_share = stop_price - entry_price
            target_price = entry_price - (self.REWARD_RISK_RATIO * risk_per_share)

        # Validate risk is positive
        if risk_per_share <= 0:
            logger.warning(f"Invalid risk calculation: {risk_per_share}")
            return None

        signal = ORBSignal(
            date=trade_date,
            symbol=self.symbol,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            risk_per_share=risk_per_share,
            first_candle_open=first_candle["open"],
            first_candle_high=first_candle["high"],
            first_candle_low=first_candle["low"],
            first_candle_close=first_candle["close"],
        )

        logger.info(
            f"Signal: {direction.value} {self.symbol} @ {entry_price:.2f}, "
            f"stop={stop_price:.2f}, target={target_price:.2f}, risk={risk_per_share:.2f}"
        )

        return signal

    def generate_signals(self, df: pd.DataFrame) -> list[ORBSignal]:
        """Generate ORB signals for all trading days in the data.

        Args:
            df: DataFrame with intraday data spanning multiple days.

        Returns:
            List of ORBSignal objects for each valid trading day.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["trade_date"] = df["date"].dt.date

        signals = []
        unique_dates = df["trade_date"].unique()

        logger.info(f"Processing {len(unique_dates)} trading days")

        for trade_date in unique_dates:
            day_data = df[df["trade_date"] == trade_date]
            signal = self.generate_signal(day_data)

            if signal is not None:
                signals.append(signal)

        logger.success(f"Generated {len(signals)} signals from {len(unique_dates)} days")
        return signals


if __name__ == "__main__":
    # Test with synthetic data
    logger.info("Testing ORB Signal Generator with synthetic data")

    # Create synthetic 5-minute data for 3 days
    dates = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []

    # Day 1: Bullish first candle (LONG signal)
    day1_base = datetime(2024, 1, 15, 9, 30)
    day1_candles = [
        # 9:30 - Bullish candle (close > open)
        {"o": 250.00, "h": 252.50, "l": 249.00, "c": 252.00},
        # 9:35 - Entry candle
        {"o": 252.00, "h": 253.00, "l": 251.50, "c": 252.50},
        # 9:40
        {"o": 252.50, "h": 254.00, "l": 252.00, "c": 253.50},
    ]
    for i, candle in enumerate(day1_candles):
        dates.append(day1_base.replace(minute=30 + i * 5))
        opens.append(candle["o"])
        highs.append(candle["h"])
        lows.append(candle["l"])
        closes.append(candle["c"])
        volumes.append(100000 + i * 10000)

    # Day 2: Bearish first candle (SHORT signal)
    day2_base = datetime(2024, 1, 16, 9, 30)
    day2_candles = [
        # 9:30 - Bearish candle (close < open)
        {"o": 255.00, "h": 256.00, "l": 253.00, "c": 253.50},
        # 9:35 - Entry candle
        {"o": 253.50, "h": 254.00, "l": 252.50, "c": 253.00},
        # 9:40
        {"o": 253.00, "h": 253.50, "l": 251.00, "c": 251.50},
    ]
    for i, candle in enumerate(day2_candles):
        dates.append(day2_base.replace(minute=30 + i * 5))
        opens.append(candle["o"])
        highs.append(candle["h"])
        lows.append(candle["l"])
        closes.append(candle["c"])
        volumes.append(100000 + i * 10000)

    # Day 3: Doji first candle (NO signal)
    day3_base = datetime(2024, 1, 17, 9, 30)
    day3_candles = [
        # 9:30 - Doji candle (close == open)
        {"o": 260.00, "h": 261.00, "l": 259.00, "c": 260.00},
        # 9:35 - Entry candle
        {"o": 260.00, "h": 261.50, "l": 259.50, "c": 261.00},
        # 9:40
        {"o": 261.00, "h": 262.00, "l": 260.50, "c": 261.50},
    ]
    for i, candle in enumerate(day3_candles):
        dates.append(day3_base.replace(minute=30 + i * 5))
        opens.append(candle["o"])
        highs.append(candle["h"])
        lows.append(candle["l"])
        closes.append(candle["c"])
        volumes.append(100000 + i * 10000)

    # Create DataFrame
    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volumes,
    })

    logger.info(f"\nSynthetic data:\n{df.to_string()}")

    # Generate signals
    generator = ORBSignalGenerator(symbol="TSLA")
    signals = generator.generate_signals(df)

    # Display results
    logger.info(f"\n{'='*60}")
    logger.info(f"Generated {len(signals)} signals:")
    logger.info(f"{'='*60}")

    for signal in signals:
        logger.info(f"\nDate: {signal.date.date()}")
        logger.info(f"Direction: {signal.direction.value}")
        logger.info(f"Entry: ${signal.entry_price:.2f}")
        logger.info(f"Stop: ${signal.stop_price:.2f}")
        logger.info(f"Target: ${signal.target_price:.2f}")
        logger.info(f"Risk/Share: ${signal.risk_per_share:.2f}")
        logger.info(f"First Candle: O={signal.first_candle_open:.2f}, "
                   f"H={signal.first_candle_high:.2f}, "
                   f"L={signal.first_candle_low:.2f}, "
                   f"C={signal.first_candle_close:.2f}")

    # Verify expected results
    logger.info(f"\n{'='*60}")
    logger.info("Verification:")
    logger.info(f"{'='*60}")
    assert len(signals) == 2, f"Expected 2 signals, got {len(signals)}"
    assert signals[0].direction == Direction.LONG, "Day 1 should be LONG"
    assert signals[1].direction == Direction.SHORT, "Day 2 should be SHORT"
    logger.success("All tests passed!")
