#!/bin/bash
# Start ORB Trading Bot in paper trading mode
# Run as trader user

set -e

INSTALL_DIR="/home/trader"
BOT_DIR="${INSTALL_DIR}/orb-trading-bot"
LOG_DIR="${BOT_DIR}/logs"
VENV_DIR="${BOT_DIR}/venv"

echo "=============================================="
echo "Starting ORB Trading Bot"
echo "=============================================="

cd ${BOT_DIR}

# Activate virtual environment
source ${VENV_DIR}/bin/activate

# Pull latest code (optional - comment out for stable deployment)
# git pull origin main

# Check if IB Gateway is responding
echo "[1/3] Checking IB Gateway connection..."
IBKR_HOST=$(grep IBKR_HOST .env | cut -d'=' -f2)
IBKR_PORT=$(grep IBKR_PORT .env | cut -d'=' -f2)

IBKR_HOST=${IBKR_HOST:-127.0.0.1}
IBKR_PORT=${IBKR_PORT:-4001}

MAX_RETRIES=30
RETRY_COUNT=0

while ! nc -z ${IBKR_HOST} ${IBKR_PORT} 2>/dev/null; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ ${RETRY_COUNT} -ge ${MAX_RETRIES} ]; then
        echo "ERROR: Cannot connect to IB Gateway at ${IBKR_HOST}:${IBKR_PORT}"
        echo "Make sure IB Gateway is running and authenticated."
        exit 1
    fi
    echo "Waiting for IB Gateway... (${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 10
done

echo "[2/3] IB Gateway is ready!"

# Load symbol from env
SYMBOL=$(grep SYMBOL .env | cut -d'=' -f2)
SYMBOL=${SYMBOL:-TSLA}

echo "[3/3] Starting paper trading for ${SYMBOL}..."
echo ""
echo "=============================================="
echo "Bot is running - Ctrl+C to stop"
echo "=============================================="
echo ""

# Run the bot
exec python main.py --mode paper --symbol ${SYMBOL} 2>&1 | tee -a ${LOG_DIR}/bot.log
