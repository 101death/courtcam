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

echo
# Ensure Python3
echo -e "${BOLD}Checking for Python 3...${NC}"
if command -v python3 &>/dev/null; then
  PY=python3
  echo -e "${GREEN}Python 3 found${NC}"
else
  echo -e "${RED}Python 3 is required. Please install and rerun.${NC}"; exit 1
fi

# Python version
VER=$($PY --version 2>&1); echo -e "${GREEN}Using $VER${NC}\n"

# Virtual environment setup
echo -e "${BOLD}Setting up Python virtual environment...${NC}"
$PY -m venv --system-site-packages venv
echo -e "${GREEN}Created venv directory${NC}"
source venv/bin/activate
echo -e "${GREEN}Activated virtual environment${NC}"
pip install --upgrade pip setuptools wheel >/dev/null
echo -e "${GREEN}Upgraded pip, setuptools, wheel${NC}\n"

# Raspberry Pi camera support
echo -e "${BOLD}Installing Raspberry Pi camera support packages...${NC}"
if [ "$IS_PI" = true ]; then
  sudo apt update -y >/dev/null
  sudo apt install -y python3-picamera2 libcamera-dev python3-libcamera >/dev/null
  echo -e "${GREEN}Camera support packages installed${NC}\n"
else
  echo -e "${YELLOW}Skipping Raspberry Pi camera packages${NC}\n"
fi

# Install common system packages
echo -e "${BOLD}Installing common system packages...${NC}"
sudo apt update -y >/dev/null
sudo apt install -y python3-opencv libopenblas-dev libatlas-base-dev libjpeg-dev libtiff5-dev libgl1-mesa-glx git curl >/dev/null
echo -e "${GREEN}Common system packages installed${NC}\n"

# Install Python dependencies
echo -e "${BOLD}Installing Python dependencies...${NC}"
if [ "$IS_PI" = true ]; then
  pip install numpy==1.19.5 >/dev/null
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu >/dev/null
  pip install opencv-python pandas shapely tqdm pillow matplotlib certifi >/dev/null
  echo -e "${GREEN}Base Python packages installed${NC}"
  read -p "Install Ultralytics (YOLOv8 support)? [y/N]: " Y8
  if [[ $Y8 =~ ^[Yy] ]]; then
    pip install ultralytics >/dev/null
    echo -e "${GREEN}Ultralytics installed${NC}"
  else
    echo -e "${YELLOW}Skipping Ultralytics${NC}"
  fi
else
  pip install -r requirements.txt >/dev/null
  echo -e "${GREEN}Dependencies from requirements.txt installed${NC}"
fi

echo
# Interactive model downloads
echo -e "${BOLD}Select YOLO models to download:${NC}"
# YOLOv5
read -p "Download YOLOv5n? [Y/n]: " DO_Y5N; DO_Y5N=${DO_Y5N:-N}
read -p "Download YOLOv5s? [Y/n]: " DO_Y5S; DO_Y5S=${DO_Y5S:-Y}
read -p "Download YOLOv5m? [Y/n]: " DO_Y5M; DO_Y5M=${DO_Y5M:-N}
# YOLOv8
read -p "Download YOLOv8n? [Y/n]: " DO_Y8N; DO_Y8N=${DO_Y8N:-Y}
read -p "Download YOLOv8s? [Y/n]: " DO_Y8S; DO_Y8S=${DO_Y8S:-N}

echo
if [[ $DO_Y5N =~ ^[Yy] ]]; then
  echo -e "${YELLOW}→ Downloading YOLOv5n...${NC}"
  curl -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt -o models/yolov5n.pt
  echo -e "${GREEN}yolov5n.pt downloaded${NC}"
fi
if [[ $DO_Y5S =~ ^[Yy] ]]; then
  echo -e "${YELLOW}→ Downloading YOLOv5s...${NC}"
  curl -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -o models/yolov5s.pt
  echo -e "${GREEN}yolov5s.pt downloaded${NC}"
fi
if [[ $DO_Y5M =~ ^[Yy] ]]; then
  echo -e "${YELLOW}→ Downloading YOLOv5m...${NC}"
  curl -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt -o models/yolov5m.pt
  echo -e "${GREEN}yolov5m.pt downloaded${NC}"
fi
if [[ $DO_Y8N =~ ^[Yy] ]]; then
  echo -e "${YELLOW}→ Downloading YOLOv8n...${NC}"
  curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o models/yolov8n.pt
  echo -e "${GREEN}yolov8n.pt downloaded${NC}"
fi
if [[ $DO_Y8S =~ ^[Yy] ]]; then
  echo -e "${YELLOW}→ Downloading YOLOv8s...${NC}"
  curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -o models/yolov8s.pt
  echo -e "${GREEN}yolov8s.pt downloaded${NC}"
fi

echo -e "${GREEN}Model downloads complete${NC}\n"

# Completion banner
echo -e "${BLUE}${BOLD}===============================================${NC}"
echo -e "${GREEN}${BOLD}Setup completed successfully!${NC}"
echo -e "${BLUE}${BOLD}===============================================${NC}\n"

echo -e "To activate venv: ${YELLOW}source venv/bin/activate${NC}"
echo -e "To run detection: ${YELLOW}python main.py --input images/input.png${NC}"
