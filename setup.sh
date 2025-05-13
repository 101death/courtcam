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
printf "Creating or updating configuration in %s\n" "$CONFIG_FILE"

# Helper functions for user input
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

get_user_input() {
    local prompt="$1"
    local default="$2"
    local result
    
    read -rp "$prompt [$default]: " result
    echo "${result:-$default}"
}

get_boolean_input() {
    local prompt="$1"
    local default="$2"
    local choice
    
    while true; do
        read -rp "$prompt (yes/no) [$default]: " choice
        choice=${choice:-$default}
        case "$choice" in
            [Yy]|[Yy][Ee][Ss])
                echo "true"
                return 0
                ;;
            [Nn]|[Nn][Oo])
                echo "false"
                return 0
                ;;
            *)
                printf "Please answer 'yes' or 'no'.\n"
                ;;
        esac
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
OUTPUT_SUMMARY_ONLY="false"
OUTPUT_EXTRA_VERBOSE="false"
DEBUG_MODE="false"
MULTIPROC_ENABLED="true"
MULTIPROC_NUM_PROCESSES=4

# Load existing config if present
if [ -f "$CONFIG_FILE" ]; then
  printf "Found existing %s. Loading current values for applicable settings.\n" "$CONFIG_FILE"
  
  # Extract values using grep/sed - this is basic parsing
  # We only extract values we need; for more complex parsing you might use jq or python
  
  # Camera settings
  CURRENT_WIDTH=$(grep -o '"width": *[0-9]\+' "$CONFIG_FILE" | grep -o '[0-9]\+' || echo "$CAM_WIDTH")
  CAM_WIDTH=${CURRENT_WIDTH:-$CAM_WIDTH}
  
  CURRENT_HEIGHT=$(grep -o '"height": *[0-9]\+' "$CONFIG_FILE" | grep -o '[0-9]\+' || echo "$CAM_HEIGHT")
  CAM_HEIGHT=${CURRENT_HEIGHT:-$CAM_HEIGHT}
  
  # Model settings
  CURRENT_MODEL_NAME=$(grep -o '"NAME": *"[^"]*"' "$CONFIG_FILE" | grep -o '"[^"]*"$' | tr -d '"' || echo "$MODEL_NAME")
  MODEL_NAME=${CURRENT_MODEL_NAME:-$MODEL_NAME}
  
  CURRENT_MODEL_CONFIDENCE=$(grep -o '"CONFIDENCE": *[0-9.]\+' "$CONFIG_FILE" | grep -o '[0-9.]\+$' || echo "$MODEL_CONFIDENCE")
  MODEL_CONFIDENCE=${CURRENT_MODEL_CONFIDENCE:-$MODEL_CONFIDENCE}
  
  CURRENT_MODEL_IOU=$(grep -o '"IOU": *[0-9.]\+' "$CONFIG_FILE" | grep -o '[0-9.]\+$' || echo "$MODEL_IOU")
  MODEL_IOU=${CURRENT_MODEL_IOU:-$MODEL_IOU}
  
  # Extract CLASSES array - simplified by always using [0] if parsing fails
  CURRENT_MODEL_CLASSES=$(grep -A 5 '"CLASSES":' "$CONFIG_FILE" | grep -o '\[[^]]*\]' || echo "$MODEL_CLASSES")
  MODEL_CLASSES=${CURRENT_MODEL_CLASSES:-$MODEL_CLASSES}
  
  # Output settings
  CURRENT_OUTPUT_VERBOSE=$(grep -o '"VERBOSE": *\(true\|false\)' "$CONFIG_FILE" | grep -o '\(true\|false\)$' || echo "$OUTPUT_VERBOSE")
  OUTPUT_VERBOSE=${CURRENT_OUTPUT_VERBOSE:-$OUTPUT_VERBOSE}
  
  CURRENT_OUTPUT_SUPER_QUIET=$(grep -o '"SUPER_QUIET": *\(true\|false\)' "$CONFIG_FILE" | grep -o '\(true\|false\)$' || echo "$OUTPUT_SUPER_QUIET")
  OUTPUT_SUPER_QUIET=${CURRENT_OUTPUT_SUPER_QUIET:-$OUTPUT_SUPER_QUIET}
  
  CURRENT_OUTPUT_SUMMARY_ONLY=$(grep -o '"SUMMARY_ONLY": *\(true\|false\)' "$CONFIG_FILE" | grep -o '\(true\|false\)$' || echo "$OUTPUT_SUMMARY_ONLY")
  OUTPUT_SUMMARY_ONLY=${CURRENT_OUTPUT_SUMMARY_ONLY:-$OUTPUT_SUMMARY_ONLY}
  
  CURRENT_OUTPUT_EXTRA_VERBOSE=$(grep -o '"EXTRA_VERBOSE": *\(true\|false\)' "$CONFIG_FILE" | grep -o '\(true\|false\)$' || echo "$OUTPUT_EXTRA_VERBOSE")
  OUTPUT_EXTRA_VERBOSE=${CURRENT_OUTPUT_EXTRA_VERBOSE:-$OUTPUT_EXTRA_VERBOSE}
  
  # Debug mode
  CURRENT_DEBUG_MODE=$(grep -o '"DEBUG_MODE": *\(true\|false\)' "$CONFIG_FILE" | grep -o '\(true\|false\)$' || echo "$DEBUG_MODE")
  DEBUG_MODE=${CURRENT_DEBUG_MODE:-$DEBUG_MODE}
  
  # MultiProcessing settings
  CURRENT_MULTIPROC_ENABLED=$(grep -o '"ENABLED": *\(true\|false\)' "$CONFIG_FILE" | grep -o '\(true\|false\)$' || echo "$MULTIPROC_ENABLED")
  MULTIPROC_ENABLED=${CURRENT_MULTIPROC_ENABLED:-$MULTIPROC_ENABLED}
  
  CURRENT_MULTIPROC_NUM_PROCESSES=$(grep -o '"NUM_PROCESSES": *[0-9]\+' "$CONFIG_FILE" | grep -o '[0-9]\+$' || echo "$MULTIPROC_NUM_PROCESSES")
  MULTIPROC_NUM_PROCESSES=${CURRENT_MULTIPROC_NUM_PROCESSES:-$MULTIPROC_NUM_PROCESSES}
fi

printf "\n--- Camera Settings ---\n"
# Resolutions ordered by size with human-readable naming
RESOLUTIONS=(
  "640x480 (VGA - Standard definition)"
  "1280x720 (HD - High definition, balanced)"
  "1920x1080 (Full HD - 1080p)"
  "2304x1296 (3MP - Higher resolution)"
  "4608x2592 (12MP - Maximum Pi Camera 3 resolution, may impact performance)"
)

printf "Select camera resolution for Pi Camera 3:\n"
for i in "${!RESOLUTIONS[@]}"; do
  printf "  %d. %s\n" "$((i+1))" "${RESOLUTIONS[$i]}"
done

# Find current resolution in the list
current_res_index=4  # Default to HD (1280x720)
for i in "${!RESOLUTIONS[@]}"; do
  if [[ "${RESOLUTIONS[$i]}" =~ ${CAM_WIDTH}x${CAM_HEIGHT} ]]; then
    current_res_index=$i
    break
  fi
done

while true; do
  read -rp "Enter number (default: $((current_res_index+1)). ${RESOLUTIONS[$current_res_index]}): " choice
  choice=${choice:-$((current_res_index+1))}
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
MODELS=(
  "yolov8n (Nano - Smallest and fastest, lower accuracy)"
  "yolov8s (Small - Good balance of speed and accuracy)"
  "yolov8m (Medium - Better accuracy, moderate speed)"
  "yolov8l (Large - High accuracy, slower)"
  "yolov8x (XLarge - Highest accuracy, slowest)"
)

# Find current model in the list
model_index=4  # Default to XLarge
for i in "${!MODELS[@]}"; do
  if [[ "${MODELS[$i]}" =~ ${MODEL_NAME} ]]; then
    model_index=$i
    break
  fi
done

printf "Select YOLO model size for people detection:\n"
for i in "${!MODELS[@]}"; do
  printf "  %d. %s\n" "$((i+1))" "${MODELS[$i]}"
done

while true; do
  read -rp "Enter number (default: $((model_index+1)). ${MODELS[$model_index]}): " choice
  choice=${choice:-$((model_index+1))}
  if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#MODELS[@]}" ]; then
    selected_model="${MODELS[$((choice-1))]}"
    # Extract model name
    if [[ $selected_model =~ (yolov[0-9][nsmplx]+) ]]; then
      MODEL_NAME="${BASH_REMATCH[1]}"
      printf "Model set to %s.\n" "$MODEL_NAME"
      break
    fi
  else
    printf "Invalid selection. Please enter a number between 1 and %s.\n" "${#MODELS[@]}"
  fi
done

# Model confidence threshold
MODEL_CONFIDENCE=$(get_user_input "Enter detection confidence threshold (0.1-0.9)" "$MODEL_CONFIDENCE")
printf "Model confidence threshold set to %s.\n" "$MODEL_CONFIDENCE"

printf "\n--- Output Settings ---\n"
OUTPUT_VERBOSE=$(get_boolean_input "Enable verbose output" "$OUTPUT_VERBOSE")
OUTPUT_SUPER_QUIET=$(get_boolean_input "Enable super quiet mode (minimal output)" "$OUTPUT_SUPER_QUIET")
OUTPUT_SUMMARY_ONLY=$(get_boolean_input "Show only summary results" "$OUTPUT_SUMMARY_ONLY")
OUTPUT_EXTRA_VERBOSE=$(get_boolean_input "Enable extra verbose output (debugging)" "$OUTPUT_EXTRA_VERBOSE")

printf "\n--- Debug Settings ---\n"
DEBUG_MODE=$(get_boolean_input "Enable debug mode" "$DEBUG_MODE")

printf "\n--- Performance Settings ---\n"
MULTIPROC_ENABLED=$(get_boolean_input "Enable multiprocessing" "$MULTIPROC_ENABLED")

if [ "$MULTIPROC_ENABLED" = "true" ]; then
  # Determine CPU count
  CPU_COUNT=4
  if command_exists nproc; then
    CPU_COUNT=$(nproc)
  elif [ -f /proc/cpuinfo ]; then
    CPU_COUNT=$(grep -c processor /proc/cpuinfo)
  elif command_exists sysctl && sysctl -n hw.ncpu >/dev/null 2>&1; then
    CPU_COUNT=$(sysctl -n hw.ncpu)
  fi
  
  # Recommend a reasonable number of processes
  RECOMMENDED_PROCS=$((CPU_COUNT - 1))
  [ "$RECOMMENDED_PROCS" -lt 1 ] && RECOMMENDED_PROCS=1
  
  printf "System has %d CPU cores. Recommended processes: %d\n" "$CPU_COUNT" "$RECOMMENDED_PROCS"
  MULTIPROC_NUM_PROCESSES=$(get_user_input "Number of processes for multiprocessing" "$RECOMMENDED_PROCS")
else
  MULTIPROC_NUM_PROCESSES=1
fi

# Object detection classes (usually just people = 0)
printf "\nDetection classes setting: %s (0=person, default is person only)\n" "$MODEL_CLASSES"
printf "To detect only people, use [0]. To detect multiple classes, use format [0,1,2]\n"
MODEL_CLASSES=$(get_user_input "Detection classes" "$MODEL_CLASSES")

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
    "SUMMARY_ONLY": $OUTPUT_SUMMARY_ONLY,
    "EXTRA_VERBOSE": $OUTPUT_EXTRA_VERBOSE
  },
  "DEBUG_MODE": $DEBUG_MODE,
  "MultiProcessing": {
    "ENABLED": $MULTIPROC_ENABLED,
    "NUM_PROCESSES": $MULTIPROC_NUM_PROCESSES
  }
}
EOF

printf "\nConfiguration saved to %s\n" "$CONFIG_FILE"
printf "You can edit this file directly anytime to adjust settings.\n"

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
printf "3. For camera usage: python main.py\n"
printf "4. To skip camera and use test image: python main.py --no-camera\n"
printf "5. Edit config.json anytime to change settings\n\n"
