#!/bin/bash
# Tennis Court Detection Setup Script
# This script sets up the environment for tennis court detection on Raspberry Pi

# Text formatting
BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}${BOLD}===============================================${NC}"
echo -e "${BLUE}${BOLD}      Tennis Court Detection Setup Script      ${NC}"
echo -e "${BLUE}${BOLD}===============================================${NC}"
echo

# Detect Raspberry Pi model
PI_MODEL=""
IS_RASPBERRY_PI=false
IS_64BIT=false

if [ -f /proc/device-tree/model ]; then
    PI_MODEL=$(cat /proc/device-tree/model)
    if [[ $PI_MODEL == *"Raspberry Pi"* ]]; then
        IS_RASPBERRY_PI=true
        echo -e "${GREEN}Detected: $PI_MODEL${NC}"
        
        # Check if running 64-bit OS
        if [ $(uname -m) == "aarch64" ]; then
            IS_64BIT=true
            echo -e "${GREEN}Detected 64-bit OS${NC}"
        else
            echo -e "${GREEN}Detected 32-bit OS${NC}"
        fi
    fi
else
    echo -e "${YELLOW}Not running on a Raspberry Pi${NC}"
fi

# Create required directories
echo -e "\n${BOLD}Creating required directories...${NC}"
mkdir -p models
mkdir -p images

# Check for Python
echo -e "\n${BOLD}Checking Python installation...${NC}"
if command -v python3 &>/dev/null; then
    PYTHON_CMD=python3
    echo -e "${GREEN}Python 3 found${NC}"
else
    echo -e "${RED}Python 3 not found. Please install Python 3.${NC}"
    exit 1
fi

# Python version
PYTHON_VERSION=$($PYTHON_CMD --version | cut -d " " -f 2)
echo -e "${GREEN}Python version: $PYTHON_VERSION${NC}"

# Ask if user wants to use virtual environment
echo -e "\n${BOLD}Virtual Environment Setup${NC}"
read -p "Do you want to use a virtual environment? (Recommended) [Y/n]: " USE_VENV
USE_VENV=${USE_VENV:-Y}

if [[ $USE_VENV =~ ^[Yy]$ ]]; then
    echo -e "\n${BOLD}Setting up virtual environment...${NC}"
    
    # Check if virtualenv is installed
    if ! $PYTHON_CMD -m pip show virtualenv &>/dev/null; then
        echo "Installing virtualenv..."
        $PYTHON_CMD -m pip install virtualenv
    fi
    
    # Create and activate virtual environment
    $PYTHON_CMD -m virtualenv venv
    
    if [ -f venv/bin/activate ]; then
        source venv/bin/activate
        echo -e "${GREEN}Virtual environment activated${NC}"
        PYTHON_CMD="python"  # Use just 'python' now that we're in the venv
    else
        echo -e "${YELLOW}Could not activate virtual environment, continuing with system Python${NC}"
    fi
fi

# Install dependencies based on platform
echo -e "\n${BOLD}Installing dependencies...${NC}"

if [ "$IS_RASPBERRY_PI" = true ]; then
    echo -e "${BOLD}Installing Raspberry Pi specific dependencies...${NC}"
    
    # Check for camera module
    echo "Checking for camera module..."
    if vcgencmd get_camera | grep -q "detected=1"; then
        echo -e "${GREEN}Camera module detected${NC}"
        
        # Install camera libraries based on OS
        if [ "$IS_64BIT" = true ]; then
            echo -e "${BOLD}Installing picamera2 for 64-bit OS...${NC}"
            echo -e "${YELLOW}Note: picamera2 should be installed via apt, not pip${NC}"
            echo "You may need to run: sudo apt install -y python3-picamera2 libcamera-dev"
            
            # Ask if user wants to install system packages
            read -p "Do you want to install required system packages for camera support? (requires sudo) [Y/n]: " INSTALL_CAMERA
            INSTALL_CAMERA=${INSTALL_CAMERA:-Y}
            
            if [[ $INSTALL_CAMERA =~ ^[Yy]$ ]]; then
                sudo apt update
                sudo apt install -y python3-picamera2 libcamera-dev python3-libcamera
            fi
        else
            echo -e "${BOLD}Installing picamera for 32-bit OS...${NC}"
            $PYTHON_CMD -m pip install "picamera>=1.13"
            
            # Ask if user wants to install system packages
            read -p "Do you want to install required system packages for camera support? (requires sudo) [Y/n]: " INSTALL_CAMERA
            INSTALL_CAMERA=${INSTALL_CAMERA:-Y}
            
            if [[ $INSTALL_CAMERA =~ ^[Yy]$ ]]; then
                sudo apt update
                sudo apt install -y libraspberrypi-dev
            fi
        fi
    else
        echo -e "${YELLOW}No camera module detected or camera not enabled${NC}"
        echo "Run 'sudo raspi-config' to enable camera interface if needed"
    fi
    
    # Install specific NumPy version to avoid binary incompatibility
    echo -e "${BOLD}Installing numpy==1.19.5 for compatibility...${NC}"
    $PYTHON_CMD -m pip install numpy==1.19.5
    
    # Install other dependencies with lower requirements for Pi
    echo -e "${BOLD}Installing other dependencies...${NC}"
    $PYTHON_CMD -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
    # Install remaining packages from requirements.txt
    $PYTHON_CMD -m pip install opencv-python pandas shapely tqdm Pillow matplotlib certifi
    
    # Optional: ultralytics for YOLOv8+
    read -p "Do you want to install ultralytics for YOLOv8 support? (slower on Pi) [y/N]: " INSTALL_ULTRALYTICS
    INSTALL_ULTRALYTICS=${INSTALL_ULTRALYTICS:-N}
    
    if [[ $INSTALL_ULTRALYTICS =~ ^[Yy]$ ]]; then
        echo -e "${BOLD}Installing ultralytics...${NC}"
        $PYTHON_CMD -m pip install ultralytics
    fi
else
    # For non-Raspberry Pi systems, install from requirements.txt
    echo -e "${BOLD}Installing dependencies from requirements.txt...${NC}"
    $PYTHON_CMD -m pip install -r requirements.txt
fi

# Download YOLOv5 model
echo -e "\n${BOLD}Downloading YOLOv5 model...${NC}"
if [ ! -f models/yolov5s.pt ]; then
    echo "Downloading YOLOv5s model..."
    curl -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -o models/yolov5s.pt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Model downloaded successfully${NC}"
    else
        echo -e "${RED}Failed to download model${NC}"
    fi
else
    echo -e "${GREEN}YOLOv5s model already exists${NC}"
fi

# Done
echo -e "\n${BLUE}${BOLD}===============================================${NC}"
echo -e "${GREEN}${BOLD}Setup completed!${NC}"
echo -e "${BLUE}${BOLD}===============================================${NC}"

if [[ $USE_VENV =~ ^[Yy]$ ]]; then
    echo -e "\n${BOLD}Virtual environment usage:${NC}"
    echo "  - To activate: source venv/bin/activate"
    echo "  - To deactivate: deactivate"
    echo -e "\n${YELLOW}Note: If you encounter NumPy compatibility issues, try:${NC}"
    echo "  1. Run outside the virtual environment: deactivate"
    echo "  2. Or reinstall numpy: pip install numpy==1.19.5"
fi

echo -e "\n${BOLD}To run the program:${NC}"
echo "  python main.py"
echo
echo -e "${BOLD}To test camera only:${NC}"
echo "  python camera.py"
echo
