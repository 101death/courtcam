#!/usr/bin/env bash
# -------------------------------------------------------------------------
# Interactive Tennis Court Detection System Installer (setup.sh)
# For Raspberry Pi Zero 2W - choose what to install & which YOLO model
# Outputs only script messages; command output is suppressed unless errors occur
# -------------------------------------------------------------------------

set -euo pipefail

# ANSI colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Error handler
error_exit() {
  echo -e "${RED}Error on line $1. Exiting.${NC}" 1>&2
  exit 1
}
trap 'error_exit $LINENO' ERR

# Welcome
echo -e "${GREEN}=== Tennis Court Detection System Installer ===${NC}"
echo

# 1) Choose installation scope
echo "Select installation scope:"
options=("Full install" "System dependencies only" "Python dependencies only" "Download models only" "Quit")
select opt in "${options[@]}"; do
  case $opt in
    "Full install")
      install_system=true; install_python=true; download_models=true; break;;
    "System dependencies only")
      install_system=true; install_python=false; download_models=false; break;;
    "Python dependencies only")
      install_system=false; install_python=true; download_models=false; break;;
    "Download models only")
      install_system=false; install_python=false; download_models=true; break;;
    "Quit")
      echo -e "${RED}Installation aborted by user.${NC}"; exit 0;;
    *) echo "Invalid choice. Please select a number.";;
  esac
done

echo
# 2) YOLO model selection
if [ "$download_models" = true ]; then
  echo "Select YOLO model to download:"
  model_opts=("YOLOv5s" "YOLOv8n" "Both" "None")
  select mopt in "${model_opts[@]}"; do
    case $mopt in
      "YOLOv5s") download_y5=true; download_y8=false; break;;
      "YOLOv8n") download_y5=false; download_y8=true; break;;
      "Both")      download_y5=true; download_y8=true; break;;
      "None")      download_y5=false; download_y8=false; break;;
      *) echo "Invalid choice.";;
    esac
done
  echo
fi

# Create dirs
echo -e "${GREEN}→ Creating directories 'models' and 'images'...${NC}"
mkdir -p models images

echo
# System dependencies
if [ "${install_system}" = true ]; then
  echo -e "${GREEN}→ Updating apt repositories...${NC}"
  sudo apt update -y > /dev/null

  echo -e "${GREEN}→ Installing system packages...${NC}"
  sudo apt install -y python3-venv python3-pip python3-picamera2 \
    libcamera-dev python3-libcamera libopenblas-dev libatlas-base-dev \
    libjpeg-dev libtiff5-dev libgl1-mesa-glx git curl > /dev/null

  echo -e "${GREEN}✔ System dependencies installed.${NC}"
fi

echo
# Python venv & dependencies
if [ "${install_python}" = true ]; then
  echo -e "${GREEN}→ Setting up Python virtual environment...${NC}"
  python3 -m venv --system-site-packages venv

  echo -e "${GREEN}→ Activating venv and upgrading pip...${NC}"
  source venv/bin/activate
  pip install --upgrade pip setuptools wheel > /dev/null

  echo -e "${GREEN}→ Installing Python dependencies...${NC}"
  pip install -r requirements.txt > /dev/null

  echo -e "${GREEN}✔ Python dependencies installed.${NC}"
else
  source venv/bin/activate 2>/dev/null || true
fi

echo
# Download models
if [ "${download_models}" = true ]; then
  if [ "$download_y5" = true ]; then
    echo -e "${YELLOW}→ Downloading YOLOv5s model...${NC}"
    curl -Ls https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -o models/yolov5s.pt
  fi
  if [ "$download_y8" = true ]; then
    echo -e "${YELLOW}→ Downloading YOLOv8n model...${NC}"
    curl -Ls https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o models/yolov8n.pt
  fi
  echo -e "${GREEN}✔ Model downloads complete.${NC}"
fi

echo
# Completion summary
echo -e "${GREEN}=== Setup Complete! ===${NC}"
if [ "${install_python}" = true ]; then
  echo -e "Activate environment: ${YELLOW}source venv/bin/activate${NC}"
fi
echo -e "Run detection: ${YELLOW}python main.py --input images/your_image.jpg${NC}"
if [ "$download_models" = true ]; then
  echo -e "Downloaded models:$( [ "$download_y5" = true ] && echo ' YOLOv5s')$( [ "$download_y8" = true ] && echo ' YOLOv8n')${NC}"
fi
