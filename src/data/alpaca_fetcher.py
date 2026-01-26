"""Alpaca data fetcher for historical market data."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class AlpacaDataFetcher:
    """Fetches historical market data from Alpaca."""

    def __init__(self):
        """Initialize with API credentials from environment."""
        self.api_key = os.getenv("ALPACA_API_KEY")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        if not self.api_key or not self.secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment. "
                "Get your keys at https://alpaca.markets"
            )

        self._client = None

    @property
    def client(self):
        """Lazy-load the Alpaca stock historical data client."""
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            self._client = StockHistoricalDataClient(self.api_key, self.secret_key)
        return self._client

    def fetch_historical_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        bar_size: str = "5Min",
    ) -> pd.DataFrame:
        """Fetch historical bar data from Alpaca.

        Args:
            symbol: Stock ticker symbol (e.g., 'TSLA').
            start_date: Start date for historical data.
            end_date: End date for historical data.
            bar_size: Bar size (default '5Min' for 5-minute bars).

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, symbol.
        """
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        # Map bar size string to TimeFrame
        timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "30Min": TimeFrame(30, TimeFrameUnit.Minute),
            "1Hour": TimeFrame.Hour,
            "1Day": TimeFrame.Day,
        }

        timeframe = timeframe_map.get(bar_size)
        if timeframe is None:
            raise ValueError(f"Unsupported bar size: {bar_size}. Use one of {list(timeframe_map.keys())}")

        logger.info(f"Fetching {bar_size} bars for {symbol} from {start_date.date()} to {end_date.date()}")

        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            feed=DataFeed.IEX,  # IEX is free, SIP requires paid subscription
        )

        try:
            bars = self.client.get_stock_bars(request_params)

            if not bars or symbol not in bars.data or not bars.data[symbol]:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            records = []
            for bar in bars.data[symbol]:
                records.append({
                    "date": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": int(bar.volume),
                    "symbol": symbol,
                })

            df = pd.DataFrame(records)

            # Convert timestamp to datetime and localize to US/Eastern for market hours
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert("America/New_York").dt.tz_localize(None)

            # Filter to regular trading hours (9:30 AM - 4:00 PM ET)
            df["time"] = df["date"].dt.time
            from datetime import time as dt_time
            market_open = dt_time(9, 30)
            market_close = dt_time(16, 0)
            df = df[(df["time"] >= market_open) & (df["time"] < market_close)]
            df = df.drop(columns=["time"])

            df = df.sort_values("date").reset_index(drop=True)

            logger.success(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()


def _get_cache_filename(symbol: str, start_date: datetime, end_date: datetime) -> Path:
    """Generate cache filename for the given parameters."""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    return DATA_DIR / f"{symbol}_{start_str}_{end_str}_5min_alpaca.csv"


def fetch_and_cache_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Fetch historical data from Alpaca with CSV caching.

    Checks for cached data in data/ folder before fetching from Alpaca.
    Saves fetched data to CSV for future use.

    Args:
        symbol: Stock ticker symbol.
        start_date: Start date for data range.
        end_date: End date for data range.

    Returns:
        DataFrame with historical bar data.
    """
    DATA_DIR.mkdir(exist_ok=True)

    cache_file = _get_cache_filename(symbol, start_date, end_date)

    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["date"])
        logger.success(f"Loaded {len(df)} bars from cache")
        return df

    fetcher = AlpacaDataFetcher()
    df = fetcher.fetch_historical_bars(symbol, start_date, end_date)

    if not df.empty:
        df.to_csv(cache_file, index=False)
        logger.success(f"Cached {len(df)} bars to {cache_file}")

    return df


if __name__ == "__main__":
    # Test: Fetch 1 month of TSLA 5-minute data
    logger.info("Starting Alpaca data fetcher test")

    symbol = os.getenv("SYMBOL", "TSLA")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)

    logger.info(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")

    try:
        df = fetch_and_cache_data(symbol, start_date, end_date)

        if not df.empty:
            logger.info(f"\nData shape: {df.shape}")
            logger.info(f"\nColumns: {df.columns.tolist()}")
            logger.info(f"\nFirst 5 rows:\n{df.head()}")
            logger.info(f"\nLast 5 rows:\n{df.tail()}")
            logger.info(f"\nDate range: {df['date'].min()} to {df['date'].max()}")

            # Count trading days
            trading_days = df["date"].dt.date.nunique()
            logger.info(f"\nTrading days: {trading_days}")
            logger.info(f"Bars per day (avg): {len(df) / trading_days:.1f}")
        else:
            logger.error("No data fetched.")

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Make sure ALPACA_API_KEY and ALPACA_SECRET_KEY are set in your .env file")
