#!/usr/bin/env bash
# ===============================================
# Tennis Court Detection Setup Script (setup.sh)
# Raspberry Pi or Linux - with error handling & camera support
# ===============================================

set -euo pipefail
trap 'echo -e "\n\033[0;31mError on line $LINENO. Exiting.\033[0m" >&2; exit 1' ERR

# Text formatting
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Spinner for long-running tasks
spinner() {
  local pid=$1
  local delay=0.1
  local spinstr='|/-\\'
  printf ""
  while kill -0 $pid 2>/dev/null; do
    for i in {0..3}; do
      printf "\r[%c]" "${spinstr:i:1}"
      sleep $delay
    done
  done
  printf "\r    \r"
}

# Banner
echo -e "${BLUE}${BOLD}===============================================${NC}"
echo -e "${BLUE}${BOLD}      Tennis Court Detection Setup Script      ${NC}"
echo -e "${BLUE}${BOLD}===============================================${NC}\n"

# Detect Raspberry Pi
IS_PI=false; IS_64BIT=false
if [ -f /proc/device-tree/model ]; then
  MODEL=$(< /proc/device-tree/model)
  if [[ $MODEL == *Raspberry* ]]; then
    IS_PI=true
    echo -e "${GREEN}Detected platform: $MODEL${NC}"
    if [[ $(uname -m) == aarch64 ]]; then
      IS_64BIT=true
      echo -e "${GREEN}64-bit OS detected${NC}"
    else
      echo -e "${GREEN}32-bit OS detected${NC}"
    fi
    echo
  fi
fi

# Create directories
echo -e "${BOLD}Creating directories 'models/' and 'images/'...${NC}"
mkdir -p models images
echo -e "${GREEN}Directories ready${NC}\n"

# Ensure Python3
echo -e "${BOLD}Checking for Python 3...${NC}"
if command -v python3 &>/dev/null; then
  PY=python3
  echo -e "${GREEN}Python 3 found${NC}\n"
else
  echo -e "${RED}Python 3 is required. Please install and rerun.${NC}"; exit 1
fi

# Python version detection
PY_MAJOR=$($PY -c 'import sys; print(sys.version_info[0])')
PY_MINOR=$($PY -c 'import sys; print(sys.version_info[1])')
VER=$($PY --version 2>&1)
echo -e "${GREEN}Using $VER${NC}\n"

# Virtual environment setup
echo -ne "${BOLD}Creating venv...${NC}"
$PY -m venv --system-site-packages venv >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}"
echo -ne "${BOLD}Activating venv...${NC}"
source venv/bin/activate >/dev/null 2>&1
echo -e " ${GREEN}Done${NC}"
echo -ne "${BOLD}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}\n"

# Raspberry Pi camera support
echo -ne "${BOLD}Installing camera packages...${NC}"
if [ "$IS_PI" = true ]; then
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-picamera2 libcamera-dev python3-libcamera >/dev/null 2>&1 & spinner $!
  echo -e " ${GREEN}Done${NC}\n"
else
  echo -e "${YELLOW}Skipping camera packages${NC}\n"
fi

# Install common system packages
echo -ne "${BOLD}Installing system deps...${NC}"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv libopenblas-dev libatlas-base-dev libjpeg-dev libtiff5-dev libgl1-mesa-glx git curl >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}\n"

# Install Python dependencies
echo -ne "${BOLD}Installing Python deps...${NC}"
if [ "$IS_PI" = true ]; then
  if [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; then
    pip install numpy==1.19.5 >/dev/null 2>&1
  else
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-numpy >/dev/null 2>&1
  fi
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu opencv-python pandas shapely tqdm pillow matplotlib certifi >/dev/null 2>&1 & spinner $!
else
  pip install -r requirements.txt >/dev/null 2>&1 & spinner $!
fi
echo -e " ${GREEN}Done${NC}\n"

# Optional Ultralytics
echo -n "Install Ultralytics (YOLOv8)? [y/N]: "
read -r Y8
if [[ $Y8 =~ ^[Yy] ]]; then
  echo -ne "Installing ultralytics..."
  pip install ultralytics >/dev/null 2>&1 & spinner $!
  echo -e " ${GREEN}Done${NC}\n"
fifi

# Interactive model downloads
echo -e "${BOLD}Select YOLO models to download:${NC}"
read -p "Download YOLOv5n? [Y/n]: " DO_Y5N; DO_Y5N=${DO_Y5N:-N}
read -p "Download YOLOv5s? [Y/n]: " DO_Y5S; DO_Y5S=${DO_Y5S:-Y}
read -p "Download YOLOv5m? [Y/n]: " DO_Y5M; DO_Y5M=${DO_Y5M:-N}
read -p "Download YOLOv8n? [Y/n]: " DO_Y8N; DO_Y8N=${DO_Y8N:-Y}
read -p "Download YOLOv8s? [Y/n]: " DO_Y8S; DO_Y8S=${DO_Y8S:-N}

echo
[[ $DO_Y5N =~ ^[Yy] ]] && { echo -ne "Downloading YOLOv5n..."; curl -sL https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt -o models/yolov5n.pt & spinner $!; echo -e " ${GREEN}Done${NC}"; }
[[ $DO_Y5S =~ ^[Yy] ]] && { echo -ne "Downloading YOLOv5s..."; curl -sL https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -o models/yolov5s.pt & spinner $!; echo -e " ${GREEN}Done${NC}"; }
[[ $DO_Y5M =~ ^[Yy] ]] && { echo -ne "Downloading YOLOv5m..."; curl -sL https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt -o models/yolov5m.pt & spinner $!; echo -e " ${GREEN}Done${NC}"; }
[[ $DO_Y8N =~ ^[Yy] ]] && { echo -ne "Downloading YOLOv8n..."; curl -sL https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o models/yolov8n.pt & spinner $!; echo -e " ${GREEN}Done${NC}"; }
[[ $DO_Y8S =~ ^[Yy] ]] && { echo -ne "Downloading YOLOv8s..."; curl -sL https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -o models/yolov8s.pt & spinner $!; echo -e " ${GREEN}Done${NC}"; }

echo -e "${GREEN}Model downloads complete${NC}\n"

# Completion banner
echo -e "${BLUE}${BOLD}===============================================${NC}"
echo -e "${GREEN}${BOLD}Setup completed successfully!${NC}"
echo -e "${BLUE}${BOLD}===============================================${NC}\n"

echo -e "To activate venv: ${YELLOW}source venv/bin/activate${NC}"
echo -e "To run detection: ${YELLOW}python main.py --input images/input.png${NC}"
