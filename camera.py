import numpy as np
import time
import os
import sys
import subprocess
import io
import re
import contextlib
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

# --- Output Styling (mimicking main.py's OutputManager) ---
# ANSI color codes
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_GRAY = "\033[90m" # For DEBUG
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"

# Symbols - Match exactly with main.py
SYMBOL_SUCCESS = "✓"
SYMBOL_INFO = "ℹ"
SYMBOL_WARNING = "⚠"
SYMBOL_ERROR = "✗"
SYMBOL_STATUS = "→"
SYMBOL_DEBUG = "•"

def _log_camera_message(message, level="INFO"):
    """Internal logger for camera.py with consistent styling matching main.py's OutputManager."""
    color = COLOR_RESET
    symbol = ""

    if level == "INFO":
        color = COLOR_BLUE
        symbol = SYMBOL_INFO
    elif level == "SUCCESS":
        color = COLOR_GREEN
        symbol = SYMBOL_SUCCESS
    elif level == "WARNING":
        color = COLOR_YELLOW
        symbol = SYMBOL_WARNING
    elif level == "ERROR":
        color = COLOR_RED
        symbol = SYMBOL_ERROR
    elif level == "STATUS":
        color = COLOR_CYAN
        symbol = SYMBOL_STATUS
    elif level == "DEBUG":
        color = COLOR_GRAY
        symbol = SYMBOL_DEBUG
    
    # Add timestamp to match main.py format
    timestamp = datetime.now().strftime("%H:%M:%S") + " " 
    
    # Print format exactly matching main.py OutputManager
    sys.stdout.write(f"{timestamp}{color}{symbol} {message}{COLOR_RESET}\n")
    sys.stdout.flush()

# Context manager to suppress stdout and stderr 
class suppress_stdout_stderr:
    """
    Context manager to suppress standard output and error streams.
    
    Usage:
        with suppress_stdout_stderr():
            # code that might print unwanted output
    """
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr

# Define platform and architecture detection
IS_RASPBERRY_PI = False
IS_64BIT = False
PI_MODEL = "unknown"
PI_CAMERA_VERSION = None

# Standard resolutions for reference - specific to Raspberry Pi Camera 3
CAMERA_RESOLUTIONS = {
    "VGA": (640, 480),      # Standard Definition
    "HD": (1280, 720),      # High Definition
    "FULL_HD": (1920, 1080), # Full HD (1080p)
    "3MP": (2304, 1296),    # 3 Megapixel
    "12MP": (4608, 2592)    # 12 Megapixel (Camera 3 full resolution)
}

# Default HD resolution (recommended balance of quality and performance)
DEFAULT_RESOLUTION = CAMERA_RESOLUTIONS["HD"]

try:
    # Check if running on Raspberry Pi
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                IS_RASPBERRY_PI = True
                PI_MODEL = model.strip('\0')
except Exception: # General exception for safety in detection
    pass

# Check if we're on 64-bit architecture
IS_64BIT = sys.maxsize > 2**32

# Import camera modules conditionally
# Suppress their native print statements during import
_picamera2_available = False
_picamera_legacy_available = False

with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        from picamera2 import Picamera2
        _picamera2_available = True
    except ImportError:
        pass # Handled below
    except Exception: # Catch any other import error from picamera2
        pass

    if not _picamera2_available:
        try:
            import picamera
            from picamera.array import PiRGBArray
            _picamera_legacy_available = True
        except ImportError:
            pass # Handled below
        except Exception: # Catch any other import error from picamera
            pass

if _picamera2_available:
    PI_CAMERA_VERSION = 2
    _log_camera_message("Picamera2 module found and will be used.", "INFO")
elif _picamera_legacy_available:
    PI_CAMERA_VERSION = 1
    _log_camera_message("Legacy Picamera module found and will be used.", "INFO")
else:
    PI_CAMERA_VERSION = None
    if IS_RASPBERRY_PI:
        _log_camera_message("No PiCamera module (Picamera2 or legacy) found. Camera functionality disabled.", "WARNING")
    # else: not on RPi, so no message needed about camera modules

# Function to capture and format camera output
def format_camera_output(func):
    """
    Decorator to format camera output consistently with main.py style.
    """
    def wrapper(*args, **kwargs):
        # Capture stdout and stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            result = func(*args, **kwargs)
        
        # Process captured output
        stdout_output = stdout_buffer.getvalue()
        stderr_output = stderr_buffer.getvalue()
        combined_output = stdout_output + stderr_output
        
        # Filter and format the messages with consistent style
        if combined_output:
            lines = combined_output.strip().split('\n')
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Format based on message type
                if 'ERROR' in line or 'Error' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    cleaned_line = re.sub(pattern, '', line)
                    _log_camera_message(cleaned_line, "ERROR")
                elif 'WARN' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    cleaned_line = re.sub(pattern, '', line)
                    _log_camera_message(cleaned_line, "WARNING")
                elif 'INFO' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    cleaned_line = re.sub(pattern, '', line)
                    _log_camera_message(cleaned_line, "INFO")
                else:
                    _log_camera_message(line, "SUCCESS")
        
        return result
    
    return wrapper

# Context manager version for inline usage
class CameraOutputFormatter:
    """
    Context manager to format camera output consistently with main.py style.
    
    Usage example:
    with CameraOutputFormatter():
        # code that generates camera output
    """
    def __enter__(self):
        # Set up buffers for capturing output
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.stdout_buffer
        sys.stderr = self.stderr_buffer
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original stdout/stderr
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
        # Process captured output
        stdout_output = self.stdout_buffer.getvalue()
        stderr_output = self.stderr_buffer.getvalue()
        combined_output = stdout_output + stderr_output
        
        # Filter and format the messages with consistent style
        if combined_output:
            lines = combined_output.strip().split('\n')
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Format based on message type
                if 'ERROR' in line or 'Error' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    cleaned_line = re.sub(pattern, '', line)
                    _log_camera_message(cleaned_line, "ERROR")
                elif 'WARN' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    cleaned_line = re.sub(pattern, '', line)
                    _log_camera_message(cleaned_line, "WARNING")
                elif 'INFO' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    cleaned_line = re.sub(pattern, '', line)
                    _log_camera_message(cleaned_line, "INFO")
                else:
                    _log_camera_message(line, "SUCCESS")

# Function to validate and correct resolution
def validate_resolution(width, height):
    """
    Validates camera resolution and returns appropriate values.
    If invalid values are provided, defaults to HD (1280x720).
    
    Args:
        width: Requested width in pixels
        height: Requested height in pixels
        
    Returns:
        tuple: (width, height) validated resolution
    """
    # Handle invalid values
    try:
        width = int(width)
        height = int(height)
    except (ValueError, TypeError):
        _log_camera_message(f"Invalid resolution values: {width}x{height}, using default {DEFAULT_RESOLUTION}", "WARNING")
        return DEFAULT_RESOLUTION
    
    # Ensure minimum resolution
    if width < 160 or height < 120:
        _log_camera_message(f"Resolution too small: {width}x{height}, using minimum 160x120", "WARNING")
        return (160, 120)
    
    # Warn about extremely high resolutions on Raspberry Pi
    if IS_RASPBERRY_PI and (width > 2560 or height > 1440):
        _log_camera_message(f"High resolution {width}x{height} may cause performance issues on Raspberry Pi", "WARNING")
    
    return (width, height)

# Keep the decorator for when this module is used directly
@format_camera_output
def takePhoto(output_dir='output', output_filename='input.png', width=1280, height=720):
    """
    Take a photo using the Raspberry Pi camera.
    
    Args:
        output_dir: Directory to save the captured image.
        output_filename: Filename for the captured image.
        width: Width of the photo resolution.
        height: Height of the photo resolution.
        
    Returns:
        bool: True if capture was successful, False otherwise.
    """
    if not IS_RASPBERRY_PI:
        _log_camera_message("Not running on a Raspberry Pi. Photo capture skipped.", "INFO")
        return False
    if PI_CAMERA_VERSION is None:
        _log_camera_message("No PiCamera module available. Cannot take photo.", "ERROR")
        return False

    try:
        # Combine output_dir and output_filename
        output_file = os.path.join(output_dir, output_filename)
        resolution = validate_resolution(width, height)
        
        _log_camera_message(f"Preparing to capture photo at {resolution} to {output_file}", "STATUS")
        
        # Ensure output directory exists
        if not os.path.exists(output_dir) and output_dir != "": # Check if output_dir is not empty string
            try:
                os.makedirs(output_dir, exist_ok=True)
                _log_camera_message(f"Created output directory: {output_dir}", "DEBUG")
            except OSError as e:
                _log_camera_message(f"Failed to create directory {output_dir}: {e}", "ERROR")
                return False
    except Exception as e:
        _log_camera_message(f"Error setting up output path {output_file}: {e}", "ERROR")
        return False

    capture_success = False
    # Use stronger suppression for camera libraries
    with suppress_stdout_stderr():
        try:
            if PI_CAMERA_VERSION == 2:
                camera = Picamera2()
                # Create a configuration dictionary
                config = camera.create_still_configuration(main={"size": resolution})
                camera.configure(config)
                
                # Start and wait for camera
                camera.start()
                time.sleep(0.5) # Give camera time to adjust
                
                # Capture image
                camera.capture_file(output_file)
                camera.stop()
                camera.close()
                capture_success = True
                
            elif PI_CAMERA_VERSION == 1:
                with picamera.PiCamera() as camera:
                    camera.resolution = resolution
                    # Add more camera settings from config.json if needed in future
                    camera.start_preview()
                    time.sleep(0.5) # Give camera time to adjust
                    camera.capture(output_file)
                    capture_success = True
                    
        except Exception as e:
            _log_camera_message(f"Error during photo capture: {e}", "ERROR")
            return False

    if capture_success:
        _log_camera_message(f"Photo captured successfully at resolution {resolution[0]}x{resolution[1]}: {output_file}", "SUCCESS")
        return True
    else:
        _log_camera_message("Photo capture failed", "ERROR")
        return False

# Example of calling the function (will run when this script is executed directly)
if __name__ == "__main__":
    _log_camera_message("Camera script running directly.", "INFO")
    
    # Print available resolutions
    _log_camera_message("Available standard resolutions:", "INFO")
    for name, res in CAMERA_RESOLUTIONS.items():
        _log_camera_message(f"  {name}: {res[0]}x{res[1]}", "INFO")
    
    current_resolution = DEFAULT_RESOLUTION
    output_dir = "images"
    output_filename = "capture_test.png" # Default for direct run

    # Allow command-line arguments: python camera.py [output_path] [width,height]
    if len(sys.argv) > 1:
        # Check if first argument specifies a named resolution
        if sys.argv[1].upper() in CAMERA_RESOLUTIONS:
            current_resolution = CAMERA_RESOLUTIONS[sys.argv[1].upper()]
            _log_camera_message(f"Using {sys.argv[1].upper()} resolution: {current_resolution[0]}x{current_resolution[1]}", "INFO")
        elif os.path.dirname(sys.argv[1]): # If first arg includes a directory path
            output_dir = os.path.dirname(sys.argv[1])
            output_filename = os.path.basename(sys.argv[1])
            _log_camera_message(f"Output specified: dir={output_dir}, file={output_filename}", "INFO")
        else:
            # Just the filename specified
            output_filename = sys.argv[1]
            _log_camera_message(f"Output filename specified: {output_filename}", "INFO")
            
        # Check for resolution as second argument
        if len(sys.argv) > 2:
            if sys.argv[2].upper() in CAMERA_RESOLUTIONS:
                # Use named resolution
                current_resolution = CAMERA_RESOLUTIONS[sys.argv[2].upper()]
                _log_camera_message(f"Using {sys.argv[2].upper()} resolution: {current_resolution[0]}x{current_resolution[1]}", "INFO")
            elif "," in sys.argv[2]:
                # Parse custom resolution
                try:
                    w_str, h_str = sys.argv[2].split(',')
                    w, h = int(w_str), int(h_str)
                    if w > 0 and h > 0:
                        current_resolution = (w, h)
                        _log_camera_message(f"Using custom resolution: {current_resolution[0]}x{current_resolution[1]}", "INFO")
                    else:
                        _log_camera_message(f"Invalid resolution values: {w_str},{h_str}. Using default.", "WARNING")
                except ValueError:
                    _log_camera_message(f"Could not parse resolution: {sys.argv[2]}. Using default.", "WARNING")
    else:
        _log_camera_message(f"Using default output: {output_dir}/{output_filename} and resolution: {current_resolution[0]}x{current_resolution[1]}", "INFO")

    _log_camera_message("Attempting to take photo...", "STATUS")
    
    # Test with specified or default parameters
    capture_success = takePhoto(output_dir=output_dir, output_filename=output_filename, width=current_resolution[0], height=current_resolution[1])
    
    if capture_success:
        _log_camera_message("Test photo capture successful.", "SUCCESS")
    else:
        _log_camera_message("Test photo capture failed.", "ERROR")
        if not IS_RASPBERRY_PI:
             _log_camera_message("Note: Script is not running on a Raspberry Pi.", "INFO")
        elif PI_CAMERA_VERSION is None:
             _log_camera_message("Note: No Raspberry Pi camera Python libraries (Picamera2 or legacy Picamera) were found.", "INFO")
        sys.exit(1)