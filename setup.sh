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

# --- Configuration Section ---
tput bold
printf "\nConfiguring default settings (config.json)...\n"
tput sgr0

CONFIG_FILE="config.json"

# Helper functions for user input
ask_yes_no() {
  local prompt="$1"
  local default="$2"
  local answer
  while true; do
    read -rp "$prompt [Y/n] " answer
    answer=${answer:-$default}
    case "$answer" in
      [Yy]* ) return 0;;
      [Nn]* ) return 1;;
      * ) printf "Please answer yes or no.\n";;
    esac
  done
}

get_int_input() {
  local prompt="$1"
  local default="$2"
  local value
  while true; do
    read -rp "$prompt (default: $default): " value
    value=${value:-$default}
    if [[ "$value" =~ ^[0-9]+$ ]]; then
      echo "$value"
      return 0
    else
      printf "Invalid input. Please enter a number.\n"
    fi
  done
}

get_float_input() {
  local prompt="$1"
  local default="$2"
  local value
  while true; do
    read -rp "$prompt (default: $default): " value
    value=${value:-$default}
    if [[ "$value" =~ ^[0-9]*\.?[0-9]+$ ]]; then
      echo "$value"
      return 0
    else
      printf "Invalid input. Please enter a floating point number (e.g., 0.1).\n"
    fi
  done
}

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
MODEL_CONFIDENCE=0.1
MODEL_IOU=0.45 # Not currently in config.json structure from Python, but good to have
MODEL_CLASSES="[0]" # Represent as string for now, will be array in JSON
OUTPUT_VERBOSE="true"
OUTPUT_SUPER_QUIET="false"
DEBUG_MODE="false"
MULTIPROC_ENABLED="true"
MULTIPROC_NUM_PROCESSES=4

# Load existing config if present
if [ -f "$CONFIG_FILE" ]; then
  printf "Found existing %s. Loading current values.\n" "$CONFIG_FILE"
  # Basic parsing - assumes jq is not available for robustness
  # This is a simplified parser. A more robust solution might use Python or jq.
  CURRENT_CAM_WIDTH=$(grep '"width"' "$CONFIG_FILE" | sed 's/.*: *\([0-9]*\).*/\1/' || echo "$CAM_WIDTH")
  CAM_WIDTH=${CURRENT_CAM_WIDTH:-$CAM_WIDTH}
  
  CURRENT_CAM_HEIGHT=$(grep '"height"' "$CONFIG_FILE" | sed 's/.*: *\([0-9]*\).*/\1/' || echo "$CAM_HEIGHT")
  CAM_HEIGHT=${CURRENT_CAM_HEIGHT:-$CAM_HEIGHT}
  
  CURRENT_MODEL_NAME=$(grep '"NAME"' "$CONFIG_FILE" | sed -e 's/.*: *"\([^\"]*\)".*/\1/' || echo "$MODEL_NAME")
  MODEL_NAME=${CURRENT_MODEL_NAME:-$MODEL_NAME}
  
  CURRENT_MODEL_CONFIDENCE=$(grep '"CONFIDENCE"' "$CONFIG_FILE" | sed 's/.*: *\([0-9.]*\).*/\1/' || echo "$MODEL_CONFIDENCE")
  MODEL_CONFIDENCE=${CURRENT_MODEL_CONFIDENCE:-$MODEL_CONFIDENCE}

  CURRENT_OUTPUT_VERBOSE=$(grep '"VERBOSE"' "$CONFIG_FILE" | sed 's/.*: *\(true\|false\).*/\1/' || echo "$OUTPUT_VERBOSE")
  OUTPUT_VERBOSE=${CURRENT_OUTPUT_VERBOSE:-$OUTPUT_VERBOSE}

  CURRENT_DEBUG_MODE=$(grep '"DEBUG_MODE"' "$CONFIG_FILE" | sed 's/.*: *\(true\|false\).*/\1/' || echo "$DEBUG_MODE")
  DEBUG_MODE=${CURRENT_DEBUG_MODE:-$DEBUG_MODE}

  CURRENT_MULTIPROC_ENABLED=$(grep '"ENABLED"' "$CONFIG_FILE" | sed 's/.*"MultiProcessing".*"ENABLED": *\(true\|false\).*/\1/' || echo "$MULTIPROC_ENABLED")
  MULTIPROC_ENABLED=${CURRENT_MULTIPROC_ENABLED:-$MULTIPROC_ENABLED}

  CURRENT_MULTIPROC_NUM_PROCESSES=$(grep '"NUM_PROCESSES"' "$CONFIG_FILE" | sed 's/.*"NUM_PROCESSES": *\([0-9]*\).*/\1/' || echo "$MULTIPROC_NUM_PROCESSES")
  MULTIPROC_NUM_PROCESSES=${CURRENT_MULTIPROC_NUM_PROCESSES:-$MULTIPROC_NUM_PROCESSES}
fi

printf "\n--- Camera Settings ---\n"
RESOLUTIONS=("Low (640x480)" "HD (1280x720)" "Full HD (1920x1080)" "Custom")
SELECTED_RES_KEY=$(select_from_list "Select camera capture resolution:" "${RESOLUTIONS[@]}")

case "$SELECTED_RES_KEY" in
  "Low (640x480)") CAM_WIDTH=640; CAM_HEIGHT=480 ;; 
  "HD (1280x720)") CAM_WIDTH=1280; CAM_HEIGHT=720 ;; 
  "Full HD (1920x1080)") CAM_WIDTH=1920; CAM_HEIGHT=1080 ;; 
  "Custom") 
    CAM_WIDTH=$(get_int_input "Enter custom width:" "$CAM_WIDTH")
    CAM_HEIGHT=$(get_int_input "Enter custom height:" "$CAM_HEIGHT")
    ;;
esac

printf "\n--- Model Settings ---\n"
MODELS_DIR="models"
AVAILABLE_MODELS=()
if [ -d "$MODELS_DIR" ]; then
  for f in "$MODELS_DIR"/*.pt; do
    [ -e "$f" ] && AVAILABLE_MODELS+=("$(basename "$f")")
  done
fi

COMMON_MODELS=("yolov8n.pt" "yolov8s.pt" "yolov8m.pt" "yolov8x.pt" "yolov5s.pt" "yolov5m.pt")
ALL_CHOICES=()
TEMP_CHOICES=("${AVAILABLE_MODELS[@]}" "${COMMON_MODELS[@]}")
# Deduplicate and sort
# shellcheck disable=SC2207
ALL_CHOICES=( $(printf "%s\n" "${TEMP_CHOICES[@]}" | sort -u) )

if [ ${#ALL_CHOICES[@]} -eq 0 ]; then
  printf "No models found in ./models/ or common models list.\n"
  read -rp "Enter desired model name (e.g., yolov8x, default: $MODEL_NAME): " model_name_input
  MODEL_NAME=${model_name_input:-$MODEL_NAME}
  # Ensure .pt is not part of the name for config, main.py adds it if needed for download
  MODEL_NAME=${MODEL_NAME%.pt}
else
  SELECTED_MODEL_FILE=$(select_from_list "Select default YOLO model (current: $MODEL_NAME.pt):" "${ALL_CHOICES[@]}")
  MODEL_NAME=${SELECTED_MODEL_FILE%.pt} # Store without .pt
fi

MODEL_CONFIDENCE=$(get_float_input "Set model confidence threshold (0.0-1.0, current: $MODEL_CONFIDENCE):" "$MODEL_CONFIDENCE")

printf "\n--- Output Settings ---\n"
if ask_yes_no "Enable verbose output (current: $OUTPUT_VERBOSE)?" "y"; then OUTPUT_VERBOSE="true"; else OUTPUT_VERBOSE="false"; fi
if [ "$OUTPUT_VERBOSE" = "false" ]; then
  if ask_yes_no "Enable super quiet mode (minimal output)?" "n"; then OUTPUT_SUPER_QUIET="true"; else OUTPUT_SUPER_QUIET="false"; fi
else
  OUTPUT_SUPER_QUIET="false"
fi

printf "\n--- Debug Settings ---\n"
if ask_yes_no "Enable debug mode (saves intermediate images, current: $DEBUG_MODE)?" "n"; then DEBUG_MODE="true"; else DEBUG_MODE="false"; fi

printf "\n--- Performance Settings ---\n"
if ask_yes_no "Enable multiprocessing (current: $MULTIPROC_ENABLED)?" "y"; then MULTIPROC_ENABLED="true"; else MULTIPROC_ENABLED="false"; fi
if [ "$MULTIPROC_ENABLED" = "true" ]; then
  CPU_CORES=$(nproc --all || echo 4) # Get CPU cores, default 4
  MULTIPROC_NUM_PROCESSES=$(get_int_input "Number of processes to use (recommended: up to $CPU_CORES, current: $MULTIPROC_NUM_PROCESSES):" "$MULTIPROC_NUM_PROCESSES")
fi

# Write to config.json
printf "{
  \"Camera\": {
    \"width\": %d,
    \"height\": %d
  },
  \"Model\": {
    \"NAME\": \"%s\",
    \"CONFIDENCE\": %s,
    \"IOU\": %s,
    \"CLASSES\": %s
  },
  \"Output\": {
    \"VERBOSE\": %s,
    \"SUPER_QUIET\": %s,
    \"SUMMARY_ONLY\": false, 
    \"EXTRA_VERBOSE\": false 
  },
  \"DEBUG_MODE\": %s,
  \"MultiProcessing\": {
    \"ENABLED\": %s,
    \"NUM_PROCESSES\": %d
  }
}
" "$CAM_WIDTH" "$CAM_HEIGHT" \
  "$MODEL_NAME" "$MODEL_CONFIDENCE" "$MODEL_IOU" "$MODEL_CLASSES" \
  "$OUTPUT_VERBOSE" "$OUTPUT_SUPER_QUIET" \
  "$DEBUG_MODE" \
  "$MULTIPROC_ENABLED" "$MULTIPROC_NUM_PROCESSES" > "$CONFIG_FILE"

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
