#!/usr/bin/env bash
set -euo pipefail
trap 'echo "Error on line $LINENO. Exiting." >&2; exit 1' ERR

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check if we have sudo privileges
check_sudo() {
  if ! command_exists sudo; then
    echo "sudo not found. Some features may be limited."
    return 1
  fi
  if ! sudo -v >/dev/null 2>&1; then
    echo "sudo access not available. Some features may be limited."
    return 1
  fi
  return 0
}

# Improved spinner function with basic ASCII characters
spinner() {
  local pid=$1
  local delay=0.1
  local spinstr='.oOo'
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    printf "\r\033[K%s [%c]" "$2" "${spinstr:$i:1}"
    i=$(( (i + 1) % ${#spinstr} ))
    sleep "$delay"
  done
  printf "\r\033[K"
}

# Function to run commands with spinner
run_with_spinner() {
  local cmd="$1"
  local msg="$2"
  local temp_file=$(mktemp)
  
  # Print the message in bold
  tput bold
  printf "%s... " "$msg"
  tput sgr0
  
  # Run the command and capture output
  if eval "$cmd" > "$temp_file" 2>&1 & spinner $! "$msg..."; then
    tput bold
    printf "\r%s... " "$msg"
    tput sgr0
    printf "✓\n"
    rm -f "$temp_file"
    return 0
  else
    tput bold
    printf "\r%s... " "$msg"
    tput sgr0
    printf "✗\n"
    if [ -s "$temp_file" ]; then
      printf "Error output:\n"
      cat "$temp_file"
    fi
    rm -f "$temp_file"
    return 1
  fi
}

# Banner
tput bold
printf "\n===============================================\n"
printf "      Tennis Court Detection Setup Script     \n"
printf "===============================================\n\n"
tput sgr0

# Check sudo access
HAS_SUDO=false
if check_sudo; then
  HAS_SUDO=true
fi

# Detect Raspberry Pi
IS_PI=false; IS_64BIT=false
if [ -f /proc/device-tree/model ]; then
  MODEL=$(< /proc/device-tree/model)
  if [[ $MODEL == *Raspberry* ]]; then
    IS_PI=true
    printf "Detected platform: %s\n" "$MODEL"
    if [ "$(uname -m)" = "aarch64" ]; then
      IS_64BIT=true
      printf "64-bit OS detected\n"
    else
      printf "32-bit OS detected\n"
    fi
    printf "\n"
  fi
fi

# Create directories
run_with_spinner "mkdir -p models images" "Creating directories"

# Python 3 check
tput bold
printf "\nChecking Python environment...\n"
tput sgr0
if ! command_exists python3; then
  printf "Python3 is required. Please install it.\n"
  exit 1
fi

PY=python3
PY_VERSION=$($PY --version)
printf "Found %s\n" "$PY_VERSION"

# Check Python version
PYV_MAJOR=$($PY -c 'import sys; print(sys.version_info.major)')
PYV_MINOR=$($PY -c 'import sys; print(sys.version_info.minor)')
if [ "$PYV_MAJOR" -lt 3 ] || ([ "$PYV_MAJOR" -eq 3 ] && [ "$PYV_MINOR" -lt 7 ]); then
  printf "Python 3.7 or higher is required. Found %s.%s\n" "$PYV_MAJOR" "$PYV_MINOR"
  exit 1
fi

# Virtual environment setup
if [ -d "venv" ]; then
  printf "\nVirtual environment already exists\n"
else
  run_with_spinner "$PY -m venv --system-site-packages venv" "Setting up virtual environment"
fi

# Activate virtual environment
source venv/bin/activate || {
  printf "Failed to activate virtual environment\n"
  exit 1
}

# Upgrade pip
run_with_spinner "pip install --upgrade pip setuptools wheel" "Upgrading pip"

# System dependencies
if command_exists apt-get && [ "$HAS_SUDO" = true ]; then
  tput bold
  printf "\nInstalling system dependencies...\n"
  tput sgr0
  run_with_spinner "sudo DEBIAN_FRONTEND=noninteractive apt-get update -y" "Updating package lists"
  
  # Install system packages
  run_with_spinner "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-opencv \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libtiff5-dev \
    libgl1-mesa-glx \
    git \
    curl" "Installing system packages"

  # Raspberry Pi camera support
  if [ "$IS_PI" = true ]; then
    run_with_spinner "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3-picamera2 \
      libcamera-dev \
      python3-libcamera" "Installing camera support"
  else
    printf "\nSkipping camera support (not a Raspberry Pi)\n"
  fi
else
  printf "\nSkipping system dependencies (apt-get not found or no sudo access)\n"
fi

# Python dependencies
tput bold
printf "\nInstalling Python dependencies...\n"
tput sgr0
if [ "$IS_PI" = true ]; then
  # Special handling for Raspberry Pi
  if [ "$PYV_MAJOR" -eq 3 ] && [ "$PYV_MINOR" -lt 10 ]; then
    run_with_spinner "pip install numpy==1.19.5" "Installing numpy for older Python"
  elif [ "$HAS_SUDO" = true ]; then
    run_with_spinner "sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-numpy" "Installing numpy"
  fi
  run_with_spinner "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu" "Installing PyTorch"
fi

# Install requirements
run_with_spinner "pip install -r requirements.txt" "Installing Python requirements"

# Install ultralytics
run_with_spinner "pip install ultralytics" "Installing ultralytics"

# Model downloads
tput bold
printf "\nModel downloads:\n"
tput sgr0
declare -A MODEL_URLS=(
  ["yolov5n"]="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt"
  ["yolov5s"]="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"
  ["yolov5m"]="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt"
  ["yolov5l"]="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt"
  ["yolov5x"]="https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt"
  ["yolov8n"]="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
  ["yolov8s"]="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
  ["yolov8m"]="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
  ["yolov8l"]="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
  ["yolov8x"]="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
)

for model in "${!MODEL_URLS[@]}"; do
  if [ -f "models/$model.pt" ]; then
    printf "  %s already downloaded\n" "$model"
  else
    read -rp "Download $model? [y/N]: " ynm
    if [[ $ynm =~ ^[Yy] ]]; then
      tput bold
      printf "  %s... " "$model"
      tput sgr0
      if curl -sL "${MODEL_URLS[$model]}" -o "models/$model.pt" & spinner $! "  $model..."; then
        tput bold
        printf "\r  %s... " "$model"
        tput sgr0
        printf "✓\n"
      else
        tput bold
        printf "\r  %s... " "$model"
        tput sgr0
        printf "✗\n"
        rm -f "models/$model.pt" 2>/dev/null
      fi
    fi
  fi
done

# Final setup
tput bold
printf "\nFinalizing setup...\n"
tput sgr0
run_with_spinner "pip list" "Verifying installations"

tput bold
printf "\nSetup complete!\n\n"
tput sgr0

# Post-install instructions
tput bold
printf "Post-install instructions:\n"
tput sgr0
printf "1. Activate virtual environment: source venv/bin/activate\n"
printf "2. Run detection: python main.py\n"
printf "3. For camera usage: python main.py --no-camera\n"
printf "4. For specific image: python main.py --input images/input.png\n\n"
