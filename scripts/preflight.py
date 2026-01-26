#!/usr/bin/env python3
"""Preflight check script for ORB Trading Bot.

Performs quick sanity checks to confirm TWS is ready for trading:
- Connection test
- Account value check
- Market data subscription
- Market hours check

Usage: python scripts/preflight.py
"""

import os
import sys
from datetime import datetime, time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytz
from dotenv import load_dotenv
from ib_insync import IB, Stock

load_dotenv()

# Eastern Time
ET = pytz.timezone("America/New_York")

# Market hours
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


class PreflightCheck:
    """Performs preflight checks for trading readiness."""

    def __init__(self):
        self.ib = IB()
        self.host = os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = int(os.getenv("IBKR_PORT", 7497))
        self.client_id = int(os.getenv("IBKR_CLIENT_ID", 1)) + 100  # Use different client ID
        self.symbol = os.getenv("SYMBOL", "TSLA")
        self.checks_passed = 0
        self.checks_failed = 0

    def print_header(self):
        """Print header."""
        print("")
        print("=" * 60)
        print("ORB Trading Bot - Preflight Check")
        print("=" * 60)
        print(f"Time: {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        print(f"Host: {self.host}:{self.port}")
        print(f"Symbol: {self.symbol}")
        print("=" * 60)
        print("")

    def check_pass(self, name: str, details: str = ""):
        """Record a passed check."""
        self.checks_passed += 1
        detail_str = f" - {details}" if details else ""
        print(f"✓ PASS: {name}{detail_str}")

    def check_fail(self, name: str, details: str = ""):
        """Record a failed check."""
        self.checks_failed += 1
        detail_str = f" - {details}" if details else ""
        print(f"✗ FAIL: {name}{detail_str}")

    def check_connection(self) -> bool:
        """Check TWS/Gateway connection."""
        print("\n[1/5] Checking TWS Connection...")

        try:
            self.ib.connect(self.host, self.port, clientId=self.client_id, timeout=10)
            self.check_pass("TWS Connection", "Connected successfully")
            return True
        except Exception as e:
            self.check_fail("TWS Connection", str(e))
            return False

    def check_account_value(self) -> bool:
        """Check account value is accessible."""
        print("\n[2/5] Checking Account Value...")

        try:
            account_values = self.ib.accountValues()
            net_liq = None

            for av in account_values:
                if av.tag == "NetLiquidation" and av.currency == "USD":
                    net_liq = float(av.value)
                    break

            if net_liq is not None:
                self.check_pass("Account Value", f"${net_liq:,.2f}")
                return True
            else:
                self.check_fail("Account Value", "Could not retrieve NetLiquidation")
                return False

        except Exception as e:
            self.check_fail("Account Value", str(e))
            return False

    def check_market_data(self) -> bool:
        """Check market data subscription."""
        print(f"\n[3/5] Checking Market Data for {self.symbol}...")

        try:
            contract = Stock(self.symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            # Request market data
            ticker = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(2)  # Wait for data

            if ticker.last is not None and ticker.last > 0:
                self.check_pass("Market Data", f"Last: ${ticker.last:.2f}")
                self.ib.cancelMktData(contract)
                return True
            elif ticker.close is not None and ticker.close > 0:
                self.check_pass("Market Data", f"Close: ${ticker.close:.2f} (market may be closed)")
                self.ib.cancelMktData(contract)
                return True
            else:
                self.check_fail("Market Data", "No price data received")
                self.ib.cancelMktData(contract)
                return False

        except Exception as e:
            self.check_fail("Market Data", str(e))
            return False

    def check_historical_data(self) -> bool:
        """Check historical data access."""
        print(f"\n[4/5] Checking Historical Data for {self.symbol}...")

        try:
            contract = Stock(self.symbol, "SMART", "USD")
            self.ib.qualifyContracts(contract)

            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="1 D",
                barSizeSetting="5 mins",
                whatToShow="TRADES",
                useRTH=True,
                formatDate=1,
            )

            if bars and len(bars) > 0:
                self.check_pass("Historical Data", f"{len(bars)} bars retrieved")
                return True
            else:
                self.check_fail("Historical Data", "No bars returned")
                return False

        except Exception as e:
            self.check_fail("Historical Data", str(e))
            return False

    def check_market_hours(self) -> bool:
        """Check current time vs market hours."""
        print("\n[5/5] Checking Market Hours...")

        now = datetime.now(ET)
        current_time = now.time()
        is_weekday = now.weekday() < 5

        if not is_weekday:
            self.check_pass("Market Hours", f"Weekend - Market closed ({now.strftime('%A')})")
            return True

        if MARKET_OPEN <= current_time < MARKET_CLOSE:
            self.check_pass("Market Hours", f"Market is OPEN ({now.strftime('%H:%M')} ET)")
            return True
        elif current_time < MARKET_OPEN:
            minutes_until = (datetime.combine(now.date(), MARKET_OPEN) -
                           datetime.combine(now.date(), current_time)).seconds // 60
            self.check_pass("Market Hours", f"Pre-market - Opens in {minutes_until} minutes")
            return True
        else:
            self.check_pass("Market Hours", f"After-hours - Market closed at 16:00 ET")
            return True

    def print_summary(self):
        """Print summary of checks."""
        print("")
        print("=" * 60)
        print("PREFLIGHT SUMMARY")
        print("=" * 60)
        print(f"Passed: {self.checks_passed}")
        print(f"Failed: {self.checks_failed}")
        print("")

        if self.checks_failed == 0:
            print("✓ ALL CHECKS PASSED - System is ready for trading!")
            print("")
            print("Start trading with:")
            print(f"  python main.py --mode paper --symbol {self.symbol}")
            print("")
        else:
            print("✗ SOME CHECKS FAILED - Please resolve issues before trading")
            print("")
            print("Common issues:")
            print("  - TWS/Gateway not running")
            print("  - API not enabled in TWS settings")
            print("  - Wrong port (paper: 7497, live: 7496, gateway: 4001)")
            print("  - 2FA not completed")
            print("")

        print("=" * 60)

    def run(self) -> bool:
        """Run all preflight checks."""
        self.print_header()

        # Run checks
        connected = self.check_connection()

        if connected:
            self.check_account_value()
            self.check_market_data()
            self.check_historical_data()
            self.ib.disconnect()

        self.check_market_hours()

        self.print_summary()

        return self.checks_failed == 0


def main():
    """Main entry point."""
    checker = PreflightCheck()
    success = checker.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
