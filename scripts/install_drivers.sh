#!/bin/bash

# Lenovo Legion 5 15IAX10 Driver Installation Script
# Clean and organized driver installer

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# Logging
LOGFILE=~/driver_installation_$(date +%F).log

log() { echo -e "${BLUE}[INFO]${NC} $1" | tee -a $LOGFILE; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1" | tee -a $LOGFILE; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1" | tee -a $LOGFILE; }
error() { echo -e "${RED}[ERROR]${NC} $1" | tee -a $LOGFILE; }

header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    error "Do not run as root. Use regular user with sudo privileges."
    exit 1
fi

header "LENOVO LEGION 5 15IAX10 DRIVER INSTALLER"

# Update system
log "Updating system packages..."
sudo apt update && sudo apt -y upgrade

# Install essential packages
log "Installing essential packages..."
sudo apt -y install \
    build-essential dkms linux-headers-$(uname -r) \
    ubuntu-drivers-common software-properties-common \
    curl wget git cmake

# Install NVIDIA drivers
log "Installing NVIDIA RTX 5070 drivers..."
sudo apt -y install \
    nvidia-driver-580 \
    nvidia-utils-580 \
    nvidia-settings \
    nvidia-cuda-toolkit

# Install Intel graphics
log "Installing Intel graphics drivers..."
sudo apt -y install \
    intel-media-va-driver-non-free \
    mesa-utils \
    mesa-vulkan-drivers

# Install network drivers
log "Installing network drivers..."
sudo apt -y install \
    linux-firmware \
    r8168-dkms \
    network-manager

# Install audio drivers
log "Installing audio drivers..."
sudo apt -y install \
    alsa-utils \
    pulseaudio \
    pipewire

# Install gaming packages
log "Installing gaming packages..."
sudo apt -y install \
    gamemode \
    steam-installer \
    vulkan-tools

# Install power management
log "Installing power management..."
sudo apt -y install \
    thermald \
    tlp \
    tlp-rdw

# Enable services
log "Enabling services..."
sudo systemctl enable thermald
sudo systemctl enable tlp

# Cleanup
log "Cleaning up..."
sudo apt -y autoremove
sudo apt -y autoclean

success "Driver installation complete!"
warning "REBOOT REQUIRED for all changes to take effect"
log "Installation log: $LOGFILE"
