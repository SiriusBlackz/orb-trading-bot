"""IBKR data fetcher for historical market data."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ib_insync import IB, Stock, util
from loguru import logger

# Load environment variables
load_dotenv()

# Project root and data directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class IBKRDataFetcher:
    """Fetches historical market data from Interactive Brokers TWS."""

    def __init__(self):
        self.ib = IB()
        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_PORT", 7497))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", 1))
        self._connected = False

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
            logger.success("Connected to IBKR successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IBKR: {e}")
            return False

    def disconnect(self) -> None:
        """Disconnect from IBKR TWS/Gateway."""
        if self._connected:
            self.ib.disconnect()
            self._connected = False
            logger.info("Disconnected from IBKR")
        else:
            logger.warning("Not connected to IBKR")

    def fetch_historical_bars(
        self,
        symbol: str,
        duration: str = "5 D",
        bar_size: str = "5 mins",
        end_date: datetime | None = None,
        use_rth: bool = True,
    ) -> pd.DataFrame:
        """Fetch historical bar data from IBKR.

        Args:
            symbol: Stock ticker symbol (e.g., 'TSLA').
            duration: Duration string (e.g., '5 D', '1 M', '1 Y').
            bar_size: Bar size string (e.g., '5 mins', '1 hour', '1 day').
            end_date: End date for historical data. Defaults to now.
            use_rth: If True, only return Regular Trading Hours data.

        Returns:
            DataFrame with columns: date, open, high, low, close, volume, average, barCount.
        """
        if not self._connected:
            logger.error("Not connected to IBKR. Call connect() first.")
            return pd.DataFrame()

        contract = Stock(symbol, "SMART", "USD")

        # Qualify the contract to get full details
        try:
            self.ib.qualifyContracts(contract)
        except Exception as e:
            logger.error(f"Failed to qualify contract for {symbol}: {e}")
            return pd.DataFrame()

        end_datetime = end_date or datetime.now()
        end_str = end_datetime.strftime("%Y%m%d %H:%M:%S")

        logger.info(f"Fetching {duration} of {bar_size} bars for {symbol} ending {end_str}")

        try:
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime=end_str,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=use_rth,
                formatDate=1,
            )

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            df = util.df(bars)
            logger.success(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            return pd.DataFrame()


def _get_cache_filename(symbol: str, start_date: datetime, end_date: datetime) -> Path:
    """Generate cache filename for the given parameters."""
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    return DATA_DIR / f"{symbol}_{start_str}_{end_str}_5min.csv"


def fetch_and_cache_data(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame:
    """Fetch historical data with CSV caching.

    Checks for cached data in data/ folder before fetching from IBKR.
    Saves fetched data to CSV for future use.

    Args:
        symbol: Stock ticker symbol.
        start_date: Start date for data range.
        end_date: End date for data range.

    Returns:
        DataFrame with historical bar data.
    """
    # Ensure data directory exists
    DATA_DIR.mkdir(exist_ok=True)

    cache_file = _get_cache_filename(symbol, start_date, end_date)

    # Check for cached data
    if cache_file.exists():
        logger.info(f"Loading cached data from {cache_file}")
        df = pd.read_csv(cache_file, parse_dates=["date"])
        logger.success(f"Loaded {len(df)} bars from cache")
        return df

    # Calculate duration in days
    duration_days = (end_date - start_date).days + 1
    duration_str = f"{duration_days} D"

    # Fetch from IBKR
    fetcher = IBKRDataFetcher()
    if not fetcher.connect():
        return pd.DataFrame()

    try:
        df = fetcher.fetch_historical_bars(
            symbol=symbol,
            duration=duration_str,
            bar_size="5 mins",
            end_date=end_date,
            use_rth=True,
        )

        if not df.empty:
            # Filter to requested date range
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"].dt.date >= start_date.date()) & (df["date"].dt.date <= end_date.date())]

            # Save to cache
            df.to_csv(cache_file, index=False)
            logger.success(f"Cached {len(df)} bars to {cache_file}")

        return df

    finally:
        fetcher.disconnect()


if __name__ == "__main__":
    # Test: Fetch 5 days of TSLA 5-minute data
    logger.info("Starting IBKR data fetcher test")

    symbol = os.getenv("SYMBOL", "TSLA")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)

    logger.info(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}")

    df = fetch_and_cache_data(symbol, start_date, end_date)

    if not df.empty:
        logger.info(f"\nData shape: {df.shape}")
        logger.info(f"\nColumns: {df.columns.tolist()}")
        logger.info(f"\nFirst 5 rows:\n{df.head()}")
        logger.info(f"\nLast 5 rows:\n{df.tail()}")
        logger.info(f"\nDate range: {df['date'].min()} to {df['date'].max()}")
    else:
        logger.error("No data fetched. Make sure TWS/Gateway is running and connected.")
