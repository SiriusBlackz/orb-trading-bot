#!/bin/bash
# DigitalOcean Setup Script for ORB Trading Bot
# Run as root on a fresh Ubuntu 22.04 droplet

set -e

echo "=============================================="
echo "ORB Trading Bot - DigitalOcean Setup"
echo "=============================================="

# Configuration
TRADER_USER="trader"
INSTALL_DIR="/home/${TRADER_USER}"
IBC_VERSION="3.18.0"
GATEWAY_VERSION="10.19"  # Check IBKR website for latest

# Update system
echo "[1/10] Updating system packages..."
apt-get update && apt-get upgrade -y

# Install Python 3.11
echo "[2/10] Installing Python 3.11..."
apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Install display and VNC packages
echo "[3/10] Installing Xvfb, VNC, and window manager..."
apt-get install -y \
    xvfb \
    x11vnc \
    openbox \
    xterm \
    unzip \
    wget \
    curl \
    git \
    openjdk-11-jre \
    libgtk-3-0 \
    libxslt1.1 \
    libxxf86vm1 \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2

# Create trader user
echo "[4/10] Creating trader user..."
if ! id "${TRADER_USER}" &>/dev/null; then
    useradd -m -s /bin/bash ${TRADER_USER}
    echo "${TRADER_USER}:trader123" | chpasswd
    usermod -aG sudo ${TRADER_USER}
fi

# Download and install IB Gateway
echo "[5/10] Downloading IB Gateway..."
cd /tmp
wget -q "https://download2.interactivebrokers.com/installers/ibgateway/stable-standalone/ibgateway-stable-standalone-linux-x64.sh" \
    -O ibgateway-installer.sh
chmod +x ibgateway-installer.sh

echo "[6/10] Installing IB Gateway..."
# Run installer in unattended mode
./ibgateway-installer.sh -q -dir /opt/ibgateway

# Download and install IBC
echo "[7/10] Installing IBC (IB Controller)..."
cd /tmp
wget -q "https://github.com/IbcAlpha/IBC/releases/download/${IBC_VERSION}/IBCLinux-${IBC_VERSION}.zip" \
    -O ibc.zip
unzip -q -o ibc.zip -d /opt/ibc
chmod +x /opt/ibc/*.sh
chmod +x /opt/ibc/*/*.sh 2>/dev/null || true

# Create IBC config directory
mkdir -p ${INSTALL_DIR}/ibc
cat > ${INSTALL_DIR}/ibc/config.ini << 'IBCCONFIG'
# IBC Configuration for ORB Trading Bot
# See https://github.com/IbcAlpha/IBC/blob/master/userguide.md

# Login credentials (set these!)
IbLoginId=
IbPassword=

# Paper trading (change to 'live' for real trading)
TradingMode=paper

# API Settings
AcceptIncomingConnectionAction=accept
AcceptNonBrokerageAccountWarning=yes
AllowBlindTrading=yes
DismissPasswordExpiryWarning=yes
DismissNSEComplianceNotice=yes
ExistingSessionDetectedAction=primary
FIX=no
MinimizeMainWindow=yes
ReadOnlyLogin=no
StoreSettingsOnServer=no

# Auto-restart settings
AutoRestart=yes
AutoRestartTime=01:00

# Connection settings
OverrideTwsApiPort=4001
CommandServerPort=7462
IBCCONFIG

# Clone ORB Trading Bot repository
echo "[8/10] Cloning ORB Trading Bot..."
cd ${INSTALL_DIR}
if [ -d "orb-trading-bot" ]; then
    cd orb-trading-bot && git pull
else
    git clone https://github.com/SiriusBlackz/orb-trading-bot.git
fi
cd orb-trading-bot

# Create virtual environment and install dependencies
echo "[9/10] Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file
cp .env.example .env

# Set ownership
chown -R ${TRADER_USER}:${TRADER_USER} ${INSTALL_DIR}

# Create directories
mkdir -p ${INSTALL_DIR}/orb-trading-bot/logs
mkdir -p ${INSTALL_DIR}/orb-trading-bot/data
mkdir -p /var/log/orb-bot

# Copy startup scripts
echo "[10/10] Installing startup scripts..."
cp deploy/digitalocean/start-gateway.sh ${INSTALL_DIR}/
cp deploy/digitalocean/start-bot.sh ${INSTALL_DIR}/
chmod +x ${INSTALL_DIR}/start-gateway.sh
chmod +x ${INSTALL_DIR}/start-bot.sh
chown ${TRADER_USER}:${TRADER_USER} ${INSTALL_DIR}/start-*.sh

# Install systemd service
cp deploy/digitalocean/orb-bot.service /etc/systemd/system/
systemctl daemon-reload

# Set up VNC password
echo "[*] Setting up VNC password..."
mkdir -p ${INSTALL_DIR}/.vnc
x11vnc -storepasswd trader123 ${INSTALL_DIR}/.vnc/passwd
chown -R ${TRADER_USER}:${TRADER_USER} ${INSTALL_DIR}/.vnc

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit IBC config with your IBKR credentials:"
echo "   nano ${INSTALL_DIR}/ibc/config.ini"
echo ""
echo "2. Edit bot environment variables:"
echo "   nano ${INSTALL_DIR}/orb-trading-bot/.env"
echo "   Set IBKR_HOST=127.0.0.1 and IBKR_PORT=4001"
echo ""
echo "3. Start IB Gateway (as trader user):"
echo "   su - trader"
echo "   ./start-gateway.sh"
echo ""
echo "4. Connect via VNC to complete 2FA:"
echo "   VNC to your_server_ip:5901"
echo "   Password: trader123"
echo ""
echo "5. After 2FA, start the bot service:"
echo "   sudo systemctl start orb-bot"
echo "   sudo systemctl enable orb-bot"
echo ""
echo "=============================================="
