#!/usr/bin/env bash
set -euo pipefail

# Simple setup script for the Tennis Court Detector
# Creates a Python virtual environment and installs dependencies.

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
error() { printf "\033[31m%s\033[0m\n" "$1" >&2; }

command_exists() { command -v "$1" >/dev/null 2>&1; }

bold "=== Tennis Court Detector Setup ==="

# Detect Raspberry Pi
IS_PI=false
if [ -f /proc/device-tree/model ] && grep -qi raspberry /proc/device-tree/model; then
    IS_PI=true
    bold "Raspberry Pi detected"
fi

# Ensure python3 exists
PYTHON=${PYTHON:-python3}
if ! command_exists "$PYTHON"; then
    error "python3 is required but was not found."
    exit 1
fi

PY_VERSION=$($PYTHON -c 'import sys;print("%d.%d" % sys.version_info[:2])')
bold "Using Python $PY_VERSION"

# Install system packages if apt-get is available
if command_exists apt-get; then
    bold "Installing system packages..."
    sudo apt-get update -y
    sudo apt-get install -y python3-venv python3-opencv libjpeg-dev \
        libtiff5-dev libopenblas-dev libatlas-base-dev libgl1 git curl
    if [ "$IS_PI" = true ]; then
        sudo apt-get install -y python3-picamera2 libcamera-dev libcamera-apps python3-libcamera
    fi
fi

# Create virtual environment with access to system packages
if [ ! -d venv ]; then
    bold "Creating virtual environment..."
    "$PYTHON" -m venv --system-site-packages venv
fi

# Activate environment
source venv/bin/activate

bold "Upgrading pip..."
pip install --upgrade pip

bold "Installing Python dependencies..."
pip install -r requirements.txt

bold "Setup complete!"
bold "Activate with: source venv/bin/activate"
