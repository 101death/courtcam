#!/usr/bin/env bash
set -euo pipefail
trap 'echo -e "\n\033[0;31mError on line $LINENO. Exiting.\033[0m" >&2; exit 1' ERR

# Color codes
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Function to check if a command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to check if we have sudo privileges
check_sudo() {
  if ! command_exists sudo; then
    echo -e "${YELLOW}sudo not found. Some features may be limited.${NC}"
    return 1
  fi
  if ! sudo -v >/dev/null 2>&1; then
    echo -e "${YELLOW}sudo access not available. Some features may be limited.${NC}"
    return 1
  fi
  return 0
}

# Improved spinner function
spinner() {
  local pid=$1
  local delay=0.1
  local spinstr='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
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
  
  # Print the message and start spinner
  echo -ne "${BOLD}${msg}...${NC}"
  
  # Run the command and capture output
  if eval "$cmd" > "$temp_file" 2>&1 & spinner $! "${BOLD}${msg}...${NC}"; then
    echo -e " ${GREEN}✓${NC}"
    rm -f "$temp_file"
    return 0
  else
    echo -e " ${RED}✗${NC}"
    if [ -s "$temp_file" ]; then
      echo -e "${RED}Error output:${NC}"
      cat "$temp_file"
    fi
    rm -f "$temp_file"
    return 1
  fi
}

# Banner
echo -e "\n${BLUE}${BOLD}===============================================${NC}"
echo -e "${BLUE}${BOLD}      Tennis Court Detection Setup Script     ${NC}"
echo -e "${BLUE}${BOLD}===============================================${NC}\n"

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
    echo -e "${GREEN}Detected platform: $MODEL${NC}"
    if [ "$(uname -m)" = "aarch64" ]; then
      IS_64BIT=true
      echo -e "${GREEN}64-bit OS detected${NC}"
    else
      echo -e "${GREEN}32-bit OS detected${NC}"
    fi
    echo
  fi
fi

# Create directories
run_with_spinner "mkdir -p models images" "Creating directories"

# Python 3 check
echo -e "\n${BOLD}Checking Python environment...${NC}"
if ! command_exists python3; then
  echo -e "${RED}Python3 is required. Please install it.${NC}"
  exit 1
fi

PY=python3
PY_VERSION=$($PY --version)
echo -e "${GREEN}Found $PY_VERSION${NC}"

# Check Python version
PYV_MAJOR=$($PY -c 'import sys; print(sys.version_info.major)')
PYV_MINOR=$($PY -c 'import sys; print(sys.version_info.minor)')
if [ "$PYV_MAJOR" -lt 3 ] || ([ "$PYV_MAJOR" -eq 3 ] && [ "$PYV_MINOR" -lt 7 ]); then
  echo -e "${RED}Python 3.7 or higher is required. Found $PYV_MAJOR.$PYV_MINOR${NC}"
  exit 1
fi

# Virtual environment setup
if [ -d "venv" ]; then
  echo -e "\n${YELLOW}Virtual environment already exists${NC}"
else
  run_with_spinner "$PY -m venv --system-site-packages venv" "Setting up virtual environment"
fi

# Activate virtual environment
source venv/bin/activate || {
  echo -e "${RED}Failed to activate virtual environment${NC}"
  exit 1
}

# Upgrade pip
run_with_spinner "pip install --upgrade pip setuptools wheel" "Upgrading pip"

# System dependencies
if command_exists apt-get && [ "$HAS_SUDO" = true ]; then
  echo -e "\n${BOLD}Installing system dependencies...${NC}"
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
    echo -e "\n${YELLOW}Skipping camera support (not a Raspberry Pi)${NC}"
  fi
else
  echo -e "\n${YELLOW}Skipping system dependencies (apt-get not found or no sudo access)${NC}"
fi

# Python dependencies
echo -e "\n${BOLD}Installing Python dependencies...${NC}"
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
echo -e "\n${BOLD}Model downloads:${NC}"
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
    echo -e "  ${YELLOW}$model already downloaded${NC}"
  else
    read -rp "Download $model? [y/N]: " ynm
    if [[ $ynm =~ ^[Yy] ]]; then
      echo -ne "  ${BOLD}$model...${NC}"
      if curl -sL "${MODEL_URLS[$model]}" -o "models/$model.pt" & spinner $! "  ${BOLD}$model...${NC}"; then
        echo -e " ${GREEN}✓${NC}"
      else
        echo -e " ${RED}✗${NC}"
        rm -f "models/$model.pt" 2>/dev/null
      fi
    fi
  fi
done

# Final setup
echo -e "\n${BOLD}Finalizing setup...${NC}"
run_with_spinner "pip list" "Verifying installations"

echo -e "\n${GREEN}${BOLD}Setup complete!${NC}\n"

# Post-install instructions
echo -e "${BOLD}Post-install instructions:${NC}"
echo -e "1. Activate virtual environment:${NC} source venv/bin/activate"
echo -e "2. Run detection:${NC} python main.py"
echo -e "3. For camera usage:${NC} python main.py --no-camera"
echo -e "4. For specific image:${NC} python main.py --input images/input.png\n"
