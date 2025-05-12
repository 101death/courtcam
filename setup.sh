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
# Filter potential null bytes from version string
PY_VERSION=$($PY --version | tr -d '\0')
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

# Download default model (yolov8x)
MODEL_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
TARGET_FILE="models/yolov8x.pt"

if [ -f "$TARGET_FILE" ]; then
  printf "Default model yolov8x already downloaded.\n"
else
  printf "Downloading default model (yolov8x)...\n"
  if curl -sL "$MODEL_URL" -o "$TARGET_FILE" & spinner $! "Downloading yolov8x..."; then
    printf "✓ Downloaded yolov8x successfully\n"
  else
    printf "✗ Failed to download yolov8x\n"
    rm -f "$TARGET_FILE" 2>/dev/null # Clean up partial download
  fi
fi

# --- Configuration Section ---
tput bold
printf "\nConfiguring default settings (config.json)...\n"
tput sgr0

CONFIG_FILE="config.json"

# If config doesn't exist, proceed with simplified setup
# This printf is now more of a general statement as we always configure.
printf "Loading existing settings from %s if present, then configuring.\n" "$CONFIG_FILE"

# Helper functions for user input (only select_from_list needed now)
select_from_list() {
    local prompt="$1"
    shift
    local options=("$@")
    local default_selection="$1" # Default is the first option passed if no explicit default
    
    if [ $# -eq 0 ]; then
      printf "No options available for %s.\n" "$prompt"
      echo ""
      return 1
    fi

    printf "%s\n" "$prompt"
    for i in "${!options[@]}"; do
      printf "  %d. %s\n" "$((i+1))" "${options[$i]}"
    done
    
    local choice
    while true; do
      read -rp "Enter number (default: 1. ${options[0]}): " choice
      choice=${choice:-1}
      if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#options[@]}" ]; then
        echo "${options[$((choice-1))]}"
        return 0
      else
        printf "Invalid selection. Please enter a number between 1 and %s.\n" "${#options[@]}"
      fi
    done
}

# Default values
CAM_WIDTH=1280
CAM_HEIGHT=720
MODEL_NAME="yolov8x"
MODEL_CONFIDENCE=0.25
MODEL_IOU=0.45
MODEL_CLASSES="[0]"
OUTPUT_VERBOSE="false"
OUTPUT_SUPER_QUIET="false"
DEBUG_MODE="false"
MULTIPROC_ENABLED="true"
MULTIPROC_NUM_PROCESSES=4

# Load existing config if present (this part is crucial and remains)
if [ -f "$CONFIG_FILE" ]; then
  printf "Found existing %s. Loading current values for applicable settings.\n" "$CONFIG_FILE"
  # Basic parsing - assumes jq is not available for robustness
  # This is a simplified parser. A more robust solution might use Python or jq.
  # Camera settings will use script defaults unless overwritten by a future, more complex setup
  # CAM_WIDTH, CAM_HEIGHT will use the new script defaults.

  CURRENT_MODEL_NAME=$(grep '"NAME"' "$CONFIG_FILE" | sed -e 's/.*: *\"([^\"]*)\".*/\1/' || echo "$MODEL_NAME")
  MODEL_NAME=${CURRENT_MODEL_NAME:-$MODEL_NAME}
  
  # MODEL_CONFIDENCE will use the new script default.
  # OUTPUT_VERBOSE will use the new script default.
  # DEBUG_MODE will use the new script default.
  # MULTIPROC_ENABLED and NUM_PROCESSES will use new script defaults.
fi

printf "\n--- Camera Settings ---\n"
RESOLUTIONS=(
  "1280x720 (HD - Recommended)"
  "1920x1080 (Full HD)"
  "640x480 (VGA)"
  "800x600 (SVGA)"
  "1024x768 (XGA)"
)

printf "Select camera resolution:\n"
for i in "${!RESOLUTIONS[@]}"; do
  printf "  %d. %s\n" "$((i+1))" "${RESOLUTIONS[$i]}"
done

while true; do
  read -rp "Enter number (default: 1. ${RESOLUTIONS[0]}): " choice
  choice=${choice:-1}
  if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#RESOLUTIONS[@]}" ]; then
    selected_res="${RESOLUTIONS[$((choice-1))]}"
    # Extract resolution values using regex
    if [[ $selected_res =~ ([0-9]+)x([0-9]+) ]]; then
      CAM_WIDTH="${BASH_REMATCH[1]}"
      CAM_HEIGHT="${BASH_REMATCH[2]}"
      printf "Camera resolution set to %sx%s.\n" "$CAM_WIDTH" "$CAM_HEIGHT"
      break
    fi
  else
    printf "Invalid selection. Please enter a number between 1 and %s.\n" "${#RESOLUTIONS[@]}"
  fi
done

printf "\n--- Model Settings ---\n"
MODELS_DIR="models"
MODEL_NAME="yolov8x"
printf "Using default model: %s\n" "$MODEL_NAME"
printf "Model confidence threshold set to %s\n" "$MODEL_CONFIDENCE"

printf "\n--- Output Settings ---\n"
printf "Using standard output settings\n"

printf "\n--- Debug Settings ---\n"
printf "Debug mode disabled\n"

printf "\n--- Performance Settings ---\n"
printf "Using multiprocessing with 4 processes\n"

# Write to config.json
cat > "$CONFIG_FILE" << EOF
{
  "Camera": {
    "width": $CAM_WIDTH,
    "height": $CAM_HEIGHT
  },
  "Model": {
    "NAME": "$MODEL_NAME",
    "CONFIDENCE": $MODEL_CONFIDENCE,
    "IOU": $MODEL_IOU,
    "CLASSES": $MODEL_CLASSES
  },
  "Output": {
    "VERBOSE": $OUTPUT_VERBOSE,
    "SUPER_QUIET": $OUTPUT_SUPER_QUIET,
    "SUMMARY_ONLY": false,
    "EXTRA_VERBOSE": false
  },
  "DEBUG_MODE": $DEBUG_MODE,
  "MultiProcessing": {
    "ENABLED": $MULTIPROC_ENABLED,
    "NUM_PROCESSES": $MULTIPROC_NUM_PROCESSES
  }
}
EOF

printf "\nConfiguration saved to %s\n" "$CONFIG_FILE"

# --- End Configuration Section ---

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
