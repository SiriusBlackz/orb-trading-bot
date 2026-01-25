# DigitalOcean Deployment Guide

Deploy the ORB Trading Bot with IB Gateway on a DigitalOcean droplet.

## Architecture

```
┌─────────────────────────────────────────────────┐
│           DigitalOcean Droplet                  │
│                                                 │
│  ┌─────────────┐     ┌─────────────────────┐   │
│  │  Xvfb       │     │    ORB Trading Bot  │   │
│  │  (Virtual   │     │    (Python)         │   │
│  │   Display)  │     │                     │   │
│  └──────┬──────┘     └──────────┬──────────┘   │
│         │                       │              │
│         ▼                       │              │
│  ┌─────────────┐               │              │
│  │ IB Gateway  │◄──────────────┘              │
│  │ (via IBC)   │    localhost:4001            │
│  └──────┬──────┘                              │
│         │                                      │
│  ┌──────┴──────┐                              │
│  │   x11vnc    │◄─── VNC :5901 (for 2FA)     │
│  └─────────────┘                              │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  IBKR Servers   │
└─────────────────┘
```

## Prerequisites

- DigitalOcean account
- Interactive Brokers account with:
  - API access enabled
  - Paper trading account (recommended for testing)
  - Mobile app for 2FA authentication
- VNC client (RealVNC, TigerVNC, etc.)

## Step 1: Create Droplet

1. Log in to [DigitalOcean](https://cloud.digitalocean.com/)

2. Create a new Droplet:
   - **Image**: Ubuntu 22.04 LTS
   - **Plan**: Basic, Regular Intel
   - **Size**: 2 GB RAM / 1 CPU ($12/month) minimum
   - **Region**: Choose closest to IBKR servers (New York recommended)
   - **Authentication**: SSH keys (recommended) or password
   - **Hostname**: `orb-trading-bot`

3. Note your droplet's IP address

## Step 2: Initial Server Setup

SSH into your droplet:

```bash
ssh root@YOUR_DROPLET_IP
```

Download and run the setup script:

```bash
# Download setup script
wget https://raw.githubusercontent.com/SiriusBlackz/orb-trading-bot/main/deploy/digitalocean/setup.sh

# Make executable and run
chmod +x setup.sh
./setup.sh
```

This will install:
- Python 3.11
- Xvfb (virtual display)
- x11vnc (VNC server)
- Openbox (window manager)
- IB Gateway
- IBC (IB Controller)
- ORB Trading Bot

## Step 3: Configure IBC (IB Gateway Controller)

Edit the IBC configuration with your IBKR credentials:

```bash
nano /home/trader/ibc/config.ini
```

Update these lines:
```ini
IbLoginId=YOUR_IBKR_USERNAME
IbPassword=YOUR_IBKR_PASSWORD
TradingMode=paper    # or 'live' for real trading
```

Save and exit (Ctrl+X, Y, Enter).

## Step 4: Configure Bot Environment

Edit the bot's environment file:

```bash
nano /home/trader/orb-trading-bot/.env
```

Configure:
```env
# IBKR Connection (localhost since Gateway runs on same server)
IBKR_HOST=127.0.0.1
IBKR_PORT=4001
IBKR_CLIENT_ID=1

# Trading Parameters
SYMBOL=TSLA
STARTING_CAPITAL=25000
RISK_PER_TRADE=0.01
MAX_LEVERAGE=4
```

## Step 5: Start IB Gateway

Switch to the trader user and start the gateway:

```bash
su - trader
./start-gateway.sh
```

The script will:
1. Start Xvfb virtual display
2. Start Openbox window manager
3. Start VNC server on port 5901
4. Launch IB Gateway via IBC

## Step 6: Complete 2FA Authentication

1. Open your VNC client

2. Connect to:
   ```
   YOUR_DROPLET_IP:5901
   Password: trader123
   ```

3. You should see IB Gateway's login window

4. Complete 2FA using your IBKR mobile app or security device

5. Once authenticated, Gateway will show "Connected" status

6. **Keep VNC open** until you verify the connection works

## Step 7: Start the Trading Bot

In a new SSH session (keep gateway running):

```bash
# Start bot service
sudo systemctl start orb-bot

# Enable auto-start on boot
sudo systemctl enable orb-bot

# Check status
sudo systemctl status orb-bot

# View logs
tail -f /home/trader/orb-trading-bot/logs/orb.log
```

## Step 8: Verify Everything Works

Check the bot is connected:

```bash
# View bot logs
tail -f /home/trader/orb-trading-bot/logs/orb.log

# View trade logs
tail -f /home/trader/orb-trading-bot/logs/trades.log

# Check service status
sudo systemctl status orb-bot
```

You should see:
- "Connected to IBKR successfully"
- "Subscribed to 5 mins bars for TSLA"
- During market hours: signal generation and trade execution

## Weekly Re-Authentication

IBKR requires re-authentication approximately every week. When the connection drops:

1. **Connect via VNC** to `YOUR_DROPLET_IP:5901`

2. **Restart IB Gateway**:
   ```bash
   su - trader
   pkill -f ibgateway
   ./start-gateway.sh
   ```

3. **Complete 2FA** in the VNC window

4. **Restart the bot**:
   ```bash
   sudo systemctl restart orb-bot
   ```

### Automating Re-Authentication (Advanced)

For fully automated operation, consider:
- [IB Gateway Docker](https://github.com/waqqasansari/ib-gateway-docker) with IBKR Flex authentication
- Using IBKR's "Trust this device" option (less secure)
- Setting up automated 2FA via TOTP (requires additional setup)

## Firewall Configuration

Secure your droplet:

```bash
# Install UFW
sudo apt install ufw

# Allow SSH
sudo ufw allow 22/tcp

# Allow VNC (restrict to your IP for security)
sudo ufw allow from YOUR_HOME_IP to any port 5901

# Enable firewall
sudo ufw enable
```

**Security Note**: VNC port 5901 should only be accessible from your IP address.

## Monitoring Commands

```bash
# Check bot status
sudo systemctl status orb-bot

# View real-time logs
tail -f /home/trader/orb-trading-bot/logs/orb.log

# View trade logs
tail -f /home/trader/orb-trading-bot/logs/trades.log

# View system logs
journalctl -u orb-bot -f

# Check IB Gateway process
ps aux | grep ibgateway

# Check connections
netstat -tlnp | grep 4001
```

## Troubleshooting

### Bot can't connect to IB Gateway

1. Check Gateway is running:
   ```bash
   ps aux | grep ibgateway
   ```

2. Check Gateway port:
   ```bash
   netstat -tlnp | grep 4001
   ```

3. VNC into the server and check Gateway status

### Gateway keeps disconnecting

1. Check IBKR credentials in `/home/trader/ibc/config.ini`
2. Ensure 2FA was completed
3. Check IBC logs: `/home/trader/orb-trading-bot/logs/`

### No trades being executed

1. Verify market hours (9:30 AM - 4:00 PM ET)
2. Check symbol is valid and has options
3. View logs for signal generation:
   ```bash
   grep -i signal /home/trader/orb-trading-bot/logs/orb.log
   ```

### Restart everything

```bash
# Stop bot
sudo systemctl stop orb-bot

# Kill gateway
pkill -f ibgateway
pkill -f Xvfb

# Start gateway
su - trader -c "./start-gateway.sh" &

# After 2FA via VNC
sudo systemctl start orb-bot
```

## Cost Estimate

| Resource | Monthly Cost |
|----------|-------------|
| DigitalOcean Droplet (2GB) | $12 |
| **Total** | **$12/month** |

## Files Reference

| File | Description |
|------|-------------|
| `/home/trader/start-gateway.sh` | Starts Xvfb + VNC + IB Gateway |
| `/home/trader/start-bot.sh` | Starts the trading bot |
| `/home/trader/ibc/config.ini` | IBC/Gateway configuration |
| `/home/trader/orb-trading-bot/.env` | Bot configuration |
| `/etc/systemd/system/orb-bot.service` | Systemd service |
| `/home/trader/orb-trading-bot/logs/` | Application logs |
| `/var/log/orb-bot/` | Service logs |
