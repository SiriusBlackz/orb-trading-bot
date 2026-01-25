# ORB Trading Bot

An Opening Range Breakout (ORB) day trading bot for Interactive Brokers (IBKR).

## Strategy Overview

The Opening Range Breakout strategy:
1. Waits for market open (9:30 AM ET)
2. Captures the first 5-minute candle (9:30-9:35)
3. Determines direction: LONG if close > open, SHORT if close < open
4. Enters at the close of the second candle (9:35)
5. Sets stop loss at first candle low (long) or high (short)
6. Sets target at 10x risk (10R reward)
7. Exits at stop, target, or end of day

## Features

- **Backtesting**: Test strategy on historical data with detailed metrics
- **Paper Trading**: Practice with simulated orders
- **Live Trading**: Execute real trades (requires confirmation)
- **Real-time Monitoring**: Watch for signals and track positions
- **Bracket Orders**: Automatic stop loss and take profit

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/orb-trading-bot.git
cd orb-trading-bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

## Configuration

Edit `.env` with your IBKR settings:

```env
# IBKR Connection Settings
IBKR_HOST=127.0.0.1
IBKR_PORT=7497          # 7497 for TWS Paper, 7496 for TWS Live, 4001/4002 for Gateway
IBKR_CLIENT_ID=1

# Trading Parameters
SYMBOL=TSLA
STARTING_CAPITAL=25000
RISK_PER_TRADE=0.01     # 1% risk per trade
MAX_LEVERAGE=4          # Maximum 4x leverage
```

## Usage

### Backtest Mode

Test the strategy on historical data:

```bash
# Backtest last 30 days
python main.py --mode backtest --symbol TSLA

# Backtest specific date range
python main.py --mode backtest --symbol TSLA --start 2024-01-01 --end 2024-01-31

# Backtest with custom capital
python main.py --mode backtest --symbol AAPL --capital 50000
```

### Paper Trading Mode

Practice with simulated orders (requires TWS/Gateway running):

```bash
python main.py --mode paper --symbol TSLA
```

### Live Trading Mode

Execute real trades (requires confirmation):

```bash
python main.py --mode live --symbol TSLA --capital 25000
```

You will be prompted to type `YES` to confirm live trading.

## Project Structure

```
orb-trading-bot/
├── main.py                 # CLI entry point
├── src/
│   ├── data/
│   │   ├── fetcher.py      # Historical data fetching
│   │   └── realtime.py     # Real-time data streaming
│   ├── strategy/
│   │   └── orb.py          # ORB signal generation
│   ├── backtest/
│   │   └── engine.py       # Backtesting engine
│   ├── trading/
│   │   ├── monitor.py      # Live signal monitoring
│   │   └── executor.py     # Order execution
│   └── utils/
│       └── metrics.py      # Performance metrics
├── data/                   # Cached market data (gitignored)
├── logs/                   # Trade and application logs
├── requirements.txt
├── .env.example
└── README.md
```

## IBKR Setup

1. **Download TWS or IB Gateway** from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/tws.php)

2. **Enable API Access**:
   - In TWS: File → Global Configuration → API → Settings
   - Check "Enable ActiveX and Socket Clients"
   - Uncheck "Read-Only API"
   - Set Socket Port (7497 for paper, 7496 for live)

3. **Configure Trusted IPs** (if running remotely):
   - Add your server's IP to "Trusted IPs"

## Cloud Deployment (Railway)

### Important: IB Gateway Requirement

The ORB Trading Bot requires a connection to Interactive Brokers TWS or IB Gateway. **IB Gateway cannot run on Railway** (or most cloud platforms) because:

1. IB Gateway requires a GUI or virtual display
2. IBKR requires 2FA authentication on startup
3. The connection must remain persistent

### Recommended Architecture

```
┌─────────────────┐         ┌─────────────────┐
│   Railway App   │ ──────► │   VPS with      │
│   (ORB Bot)     │         │   IB Gateway    │
└─────────────────┘         └─────────────────┘
        │                           │
        │                           ▼
        │                   ┌─────────────────┐
        └──────────────────►│   IBKR Servers  │
                            └─────────────────┘
```

**Option 1: Dedicated VPS for IB Gateway**
- Run IB Gateway on a VPS (DigitalOcean, AWS, etc.)
- Use a VNC or virtual display (Xvfb)
- Set `IBKR_HOST` to the VPS IP address

**Option 2: Home Server**
- Run IB Gateway on your home computer
- Use a static IP or dynamic DNS
- Configure port forwarding for port 4001/4002
- Set `IBKR_HOST` to your public IP

### Deploying to Railway

1. **Set up IB Gateway** on a separate server (see above)

2. **Deploy to Railway**:
   ```bash
   # Install Railway CLI
   npm install -g @railway/cli

   # Login to Railway
   railway login

   # Initialize project
   railway init

   # Deploy
   railway up
   ```

3. **Configure Environment Variables** in Railway dashboard:
   ```
   IBKR_HOST=<your-ib-gateway-ip>
   IBKR_PORT=4001
   IBKR_CLIENT_ID=1
   SYMBOL=TSLA
   STARTING_CAPITAL=25000
   RISK_PER_TRADE=0.01
   MAX_LEVERAGE=4
   ```

### Docker Deployment

```bash
# Build image
docker build -t orb-trading-bot .

# Run paper trading
docker run -d \
  --name orb-bot \
  -e IBKR_HOST=192.168.1.100 \
  -e IBKR_PORT=4001 \
  -e SYMBOL=TSLA \
  orb-trading-bot

# Run backtest
docker run --rm \
  -e IBKR_HOST=192.168.1.100 \
  orb-trading-bot \
  python main.py --mode backtest --symbol TSLA
```

## Logs

- `logs/orb.log` - Application logs (rotated daily)
- `logs/trades.log` - Trade execution logs

## Risk Warning

Trading involves substantial risk of loss. This software is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly with paper trading before risking real capital.

## License

MIT License
