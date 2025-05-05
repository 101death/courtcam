#!/bin/bash

# Function to run Python setup
python_setup() {
    python3 -c "
import sys
sys.path.append('.')
from main import OutputManager
import os

# Function to run command and capture output
def run_command(cmd):
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

# Check Python version
python_version = sys.version_info
if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
    OutputManager.log('Python 3.7 or higher is required', 'ERROR')
    sys.exit(1)

# Create virtual environment
venv_path = 'venv'
if not os.path.exists(venv_path):
    OutputManager.log('Creating virtual environment...', 'INFO')
    success, output = run_command('python3 -m venv venv')
    if not success:
        OutputManager.log(f'Failed to create virtual environment: {output}', 'ERROR')
        sys.exit(1)
    OutputManager.log('Virtual environment created successfully', 'SUCCESS')

# Activate virtual environment and install requirements
OutputManager.log('Installing requirements...', 'INFO')
success, output = run_command('. venv/bin/activate && pip install -r requirements.txt')
if not success:
    OutputManager.log(f'Failed to install requirements: {output}', 'ERROR')
    sys.exit(1)
OutputManager.log('Requirements installed successfully', 'SUCCESS')

# Create necessary directories
for dir in ['images', 'models']:
    if not os.path.exists(dir):
        OutputManager.log(f'Creating directory: {dir}', 'INFO')
        os.makedirs(dir, exist_ok=True)
        OutputManager.log(f'Directory {dir} created successfully', 'SUCCESS')

# Check for camera support
if os.path.exists('/proc/device-tree/model'):
    with open('/proc/device-tree/model', 'r') as f:
        if 'Raspberry Pi' in f.read():
            OutputManager.log('Raspberry Pi detected', 'INFO')
            
            # Check for picamera2 (newer systems)
            success, output = run_command('dpkg -l | grep python3-picamera2')
            if success:
                OutputManager.log('picamera2 is installed', 'SUCCESS')
            else:
                OutputManager.log('picamera2 is not installed', 'WARNING')
                OutputManager.log('To install picamera2, run: sudo apt install python3-picamera2 libcamera-dev', 'INFO')
            
            # Check for legacy picamera
            success, output = run_command('pip list | grep picamera')
            if success:
                OutputManager.log('legacy picamera is installed', 'SUCCESS')
            else:
                OutputManager.log('legacy picamera is not installed', 'WARNING')
                OutputManager.log('To install legacy picamera, run: pip install picamera', 'INFO')
        else:
            OutputManager.log('Not running on Raspberry Pi - camera support will be limited', 'WARNING')
    else:
        OutputManager.log('Not running on Raspberry Pi - camera support will be limited', 'WARNING')

OutputManager.log('Setup completed successfully', 'SUCCESS')
"
}

# Run the Python setup
python_setup

# Print post-install instructions
echo
echo "To activate the virtual environment, run:"
echo "source venv/bin/activate"
echo
echo "To run the detection, use:"
echo "python main.py"
echo
