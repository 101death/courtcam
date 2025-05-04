#!/bin/bash
# Tennis Court Detection System - Linux/Raspberry Pi Dependency Installer
# This script installs all required dependencies for the tennis court detection system on Linux/Raspberry Pi

# Print colored output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Tennis Court Detection System - Linux/Raspberry Pi Installer${NC}"
echo -e "${YELLOW}This script will install all required dependencies for the Tennis Court Detection System.${NC}"
echo -e "${YELLOW}It may take some time to complete. Please be patient.${NC}"
echo ""

# Create necessary directories
echo -e "${GREEN}Creating necessary directories...${NC}"
mkdir -p models
mkdir -p images

# Update system packages
echo -e "${GREEN}Updating system packages...${NC}"
sudo apt update -y || { echo -e "${RED}Failed to update system packages.${NC}"; exit 1; }

# Install required system packages
echo -e "${GREEN}Installing required system packages...${NC}"
sudo apt install -y \
    python3-pip \
    python3-opencv \
    python3-numpy \
    libopenblas-dev \
    libatlas-base-dev \
    libjpeg-dev \
    libopenjp2-7-dev \
    libtiff5-dev \
    libgl1-mesa-glx \
    || { echo -e "${RED}Failed to install system packages.${NC}"; exit 1; }

# Install Python dependencies
echo -e "${GREEN}Installing Python dependencies...${NC}"
pip3 install --upgrade pip

# Install optimized PyTorch for Raspberry Pi
echo -e "${GREEN}Installing optimized PyTorch for Raspberry Pi...${NC}"
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
echo -e "${GREEN}Installing other Python dependencies...${NC}"
pip3 install \
    opencv-python \
    numpy \
    pandas \
    shapely \
    tqdm \
    pillow \
    matplotlib \
    || { echo -e "${RED}Failed to install Python dependencies.${NC}"; exit 1; }

# Install Ultralytics for YOLOv5
echo -e "${GREEN}Installing Ultralytics for YOLOv5...${NC}"
pip3 install ultralytics || { echo -e "${RED}Failed to install Ultralytics.${NC}"; exit 1; }

# Download YOLOv5 model
echo -e "${GREEN}Downloading YOLOv5 model...${NC}"
if [ ! -f "models/yolov5s.pt" ]; then
    echo -e "${YELLOW}Downloading YOLOv5s model (this may take a while)...${NC}"
    curl -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -o models/yolov5s.pt || \
    { echo -e "${RED}Failed to download YOLOv5 model.${NC}"; exit 1; }
else
    echo -e "${YELLOW}YOLOv5s model already exists.${NC}"
fi

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}You can now run the tennis court detection system:${NC}"
echo -e "python3 main.py --input images/your_image.jpg"
echo ""
echo -e "${YELLOW}For better performance on Raspberry Pi, you can use:${NC}"
echo -e "python3 main.py --device cpu"
echo "" 
