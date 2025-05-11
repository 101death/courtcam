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

# Spinner function for background tasks
spinner() {
  local pid=$1
  local delay=0.1
  local spinstr='|/-\\'
  while kill -0 "$pid" 2>/dev/null; do
    for ((i=0; i<${#spinstr}; i++)); do
      printf "\r [%c]" "${spinstr:$i:1}"
      sleep "$delay"
    done
  done
  printf "\r    \r"
}

# Banner
echo -e "${BLUE}${BOLD}===============================================${NC}"
echo -e "${BLUE}${BOLD}      Tennis Court Detection Setup Script     ${NC}"
echo -e "${BLUE}${BOLD}===============================================${NC}\n"

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
echo -e "${BOLD}Creating 'models/' and 'images/' directories...${NC}"
mkdir -p models images
echo -e "${GREEN}Directories ready${NC}\n"

# Python 3 check
echo -e "${BOLD}Checking for Python 3...${NC}"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
  PY_VERSION=$($PY --version)
  echo -e "${GREEN}Found $PY_VERSION${NC}\n"
else
  echo -e "${RED}Python3 is required. Please install it.${NC}"; exit 1
fi

# Check Python version
PYV_MAJOR=$($PY -c 'import sys; print(sys.version_info.major)')
PYV_MINOR=$($PY -c 'import sys; print(sys.version_info.minor)')
if [ "$PYV_MAJOR" -lt 3 ] || ([ "$PYV_MAJOR" -eq 3 ] && [ "$PYV_MINOR" -lt 7 ]); then
  echo -e "${RED}Python 3.7 or higher is required. Found $PYV_MAJOR.$PYV_MINOR${NC}"
  exit 1
fi

# Virtual environment setup
echo -ne "${BOLD}Setting up virtual environment...${NC}"
if [ -d "venv" ]; then
  echo -e " ${YELLOW}Virtual environment already exists${NC}"
else
  $PY -m venv --system-site-packages venv >/dev/null 2>&1 & spinner $!
  echo -e " ${GREEN}Done${NC}"
fi
source venv/bin/activate

# Upgrade pip
echo -ne "${BOLD}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}\n"

# System dependencies
if command -v apt-get >/dev/null 2>&1; then
  echo -ne "${BOLD}Installing system dependencies...${NC}"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-opencv \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libtiff5-dev \
    libgl1-mesa-glx \
    git \
    curl \
    >/dev/null 2>&1 & spinner $!
  echo -e " ${GREEN}Done${NC}\n"

  # Raspberry Pi camera support
  if [ "$IS_PI" = true ]; then
    echo -ne "${BOLD}Installing camera support...${NC}"
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3-picamera2 \
      libcamera-dev \
      python3-libcamera \
      >/dev/null 2>&1 & spinner $!
    echo -e " ${GREEN}Done${NC}\n"
  else
    echo -e "${YELLOW}Skipping camera support (not a Raspberry Pi)${NC}\n"
  fi
else
  echo -e "${YELLOW}Skipping system dependencies (apt-get not found)${NC}\n"
fi

# Python dependencies
echo -ne "${BOLD}Installing Python dependencies...${NC}"
if [ "$IS_PI" = true ]; then
  # Special handling for Raspberry Pi
  if [ "$PYV_MAJOR" -eq 3 ] && [ "$PYV_MINOR" -lt 10 ]; then
    pip install numpy==1.19.5 >/dev/null 2>&1
  else
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-numpy >/dev/null 2>&1
  fi
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1
fi

# Install all requirements
pip install -r requirements.txt >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}\n"

# Ultralytics (YOLOv8)
echo -ne "${BOLD}Installing ultralytics...${NC}"
pip install ultralytics >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}\n"

# Model downloads
echo -e "${BOLD}Model downloads:${NC}"
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
      echo -ne "  $model...${NC}"
      if curl -sL "${MODEL_URLS[$model]}" -o "models/$model.pt" & spinner $!; then
        echo -e " ${GREEN}Done${NC}"
      else
        echo -e " ${RED}Failed${NC}"
        rm -f "models/$model.pt" 2>/dev/null
      fi
    fi
  fi
done

echo
# Final spinner to wrap up
echo -ne "${BOLD}Finalizing setup...${NC}"
sleep 1 & spinner $!
echo -e " ${GREEN}Setup complete!${NC}\n"

# Post-install instructions
echo -e "${BOLD}Post-install instructions:${NC}"
echo -e "1. Activate virtual environment:${NC} source venv/bin/activate"
echo -e "2. Run detection:${NC} python main.py"
echo -e "3. For camera usage:${NC} python main.py --no-camera"
echo -e "4. For specific image:${NC} python main.py --input images/input.png\n"
