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
  echo -e "${GREEN}Found $(python3 --version)${NC}\n"
else
  echo -e "${RED}Python3 is required. Please install it.${NC}"; exit 1
fi

# Virtual environment setup
echo -ne "${BOLD}Setting up virtual environment...${NC}"
$PY -m venv --system-site-packages venv >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}"
source venv/bin/activate

# Upgrade pip
echo -ne "${BOLD}Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}\n"

# Raspberry Pi camera support
if [ "$IS_PI" = true ]; then
  echo -ne "${BOLD}Installing camera support...${NC}"
  sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-picamera2 libcamera-dev python3-libcamera >/dev/null 2>&1 & spinner $!
  echo -e " ${GREEN}Done${NC}\n"
else
  echo -e "${YELLOW}Skipping camera support (not a Raspberry Pi)${NC}\n"
fi

# System dependencies
echo -ne "${BOLD}Installing system dependencies...${NC}"
sudo DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-opencv libopenblas-dev libatlas-base-dev \
     libjpeg-dev libtiff5-dev libgl1-mesa-glx git curl >/dev/null 2>&1 & spinner $!
echo -e " ${GREEN}Done${NC}\n"

# Python dependencies
echo -ne "${BOLD}Installing Python dependencies...${NC}"
if [ "$IS_PI" = true ]; then
  PYV_MAJOR=$($PY -c 'import sys; print(sys.version_info.major)')
  PYV_MINOR=$($PY -c 'import sys; print(sys.version_info.minor)')
  if [ "$PYV_MAJOR" -eq 3 ] && [ "$PYV_MINOR" -lt 10 ]; then
    pip install numpy==1.19.5 >/dev/null 2>&1
  else
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y python3-numpy >/dev/null 2>&1
  fi
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
      opencv-python pandas shapely tqdm pillow matplotlib certifi >/dev/null 2>&1 & spinner $!
else
  pip install -r requirements.txt >/dev/null 2>&1 & spinner $!
fi
echo -e " ${GREEN}Done${NC}\n"

# Ultralytics (YOLOv8)
read -rp "Install ultralytics for YOLOv8 support? [y/N]: " yn
if [[ $yn =~ ^[Yy] ]]; then
  echo -ne "${BOLD}Installing ultralytics...${NC}"
  pip install ultralytics >/dev/null 2>&1 & spinner $!
  echo -e " ${GREEN}Done${NC}\n"
fi

# Model downloads
echo -e "${BOLD}Model downloads:${NC}"
for model in yolov5n yolov5s yolov5m yolov8n yolov8s; do
  read -rp "Download $model? [y/N]: " ynm
  if [[ $ynm =~ ^[Yy] ]]; then
    echo -ne "  $model...${NC}"
    case $model in
      yolov5n)   url=https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt ;;  
      yolov5s)   url=https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt ;;  
      yolov5m)   url=https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt ;;  
      yolov8n)   url=https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt ;;  
      yolov8s)   url=https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt ;;  
    esac
    curl -sL "$url" -o "models/$model.pt" & spinner $!
    echo -e " ${GREEN}Done${NC}"
  fi
done

echo
# Final spinner to wrap up
echo -ne "${BOLD}Finalizing setup...${NC}"
sleep 1 & spinner $!
echo -e " ${GREEN}Setup complete!${NC}\n"

# Post-install instructions
echo -e "To activate virtual environment:${NC} source venv/bin/activate"
echo -e "To run detection:${NC} python main.py --input images/input.png\n"
