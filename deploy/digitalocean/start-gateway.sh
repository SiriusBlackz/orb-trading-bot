#!/bin/bash
# Start IB Gateway with Xvfb virtual display
# Run as trader user

set -e

DISPLAY_NUM=1
INSTALL_DIR="/home/trader"
IBC_PATH="/opt/ibc"
GATEWAY_PATH="/opt/ibgateway"
IBC_CONFIG="${INSTALL_DIR}/ibc/config.ini"
LOG_DIR="${INSTALL_DIR}/orb-trading-bot/logs"

echo "=============================================="
echo "Starting IB Gateway"
echo "=============================================="

# Kill any existing Xvfb or Gateway processes
echo "[1/4] Cleaning up existing processes..."
pkill -f "Xvfb :${DISPLAY_NUM}" 2>/dev/null || true
pkill -f "ibgateway" 2>/dev/null || true
pkill -f "x11vnc" 2>/dev/null || true
sleep 2

# Start Xvfb virtual display
echo "[2/4] Starting Xvfb on display :${DISPLAY_NUM}..."
Xvfb :${DISPLAY_NUM} -screen 0 1920x1080x24 &
export DISPLAY=:${DISPLAY_NUM}
sleep 2

# Start lightweight window manager
echo "[3/4] Starting Openbox window manager..."
openbox &
sleep 1

# Start VNC server for remote access
echo "[*] Starting VNC server on port 5901..."
x11vnc -display :${DISPLAY_NUM} \
    -rfbport 5901 \
    -rfbauth ${INSTALL_DIR}/.vnc/passwd \
    -forever \
    -shared \
    -bg \
    -o ${LOG_DIR}/x11vnc.log

# Start IB Gateway via IBC
echo "[4/4] Starting IB Gateway via IBC..."
echo "Using config: ${IBC_CONFIG}"

# Find the gateway start script
if [ -f "${IBC_PATH}/gatewaystart.sh" ]; then
    IBC_SCRIPT="${IBC_PATH}/gatewaystart.sh"
elif [ -f "${IBC_PATH}/scripts/ibcstart.sh" ]; then
    IBC_SCRIPT="${IBC_PATH}/scripts/ibcstart.sh"
else
    IBC_SCRIPT="${IBC_PATH}/ibcstart.sh"
fi

# Start IB Gateway
${IBC_SCRIPT} "${GATEWAY_PATH}" "${IBC_CONFIG}" &

echo ""
echo "=============================================="
echo "IB Gateway Starting..."
echo "=============================================="
echo ""
echo "VNC Access:"
echo "  Host: $(hostname -I | awk '{print $1}'):5901"
echo "  Password: trader123"
echo ""
echo "Connect via VNC to complete 2FA authentication."
echo ""
echo "Gateway API will be available on port 4001"
echo "after authentication is complete."
echo ""
echo "Logs: ${LOG_DIR}/"
echo "=============================================="

# Keep script running and show gateway output
wait
