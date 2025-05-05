import numpy as np
import time
import os
import sys
import subprocess
import io
import re
import contextlib
from contextlib import redirect_stdout, redirect_stderr

# Define platform and architecture detection
IS_RASPBERRY_PI = False
IS_64BIT = False
PI_MODEL = "unknown"
PI_CAMERA_VERSION = None

# Lower default resolution for better performance
DEFAULT_RESOLUTION = (640, 480)  # Reduced from 1920x1080 to 640x480 for better performance

try:
    # Check if running on Raspberry Pi
    if os.path.exists('/proc/device-tree/model'):
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            if 'Raspberry Pi' in model:
                IS_RASPBERRY_PI = True
                PI_MODEL = model.strip('\0')
except:
    pass

# Check if we're on 64-bit architecture
IS_64BIT = sys.maxsize > 2**32

# Function to capture and format camera output
def format_camera_output(func):
    """
    Decorator to format camera output with emojis and remove timestamps.
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
        
        # Filter and format the messages with emojis
        if combined_output:
            lines = combined_output.strip().split('\n')
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Format based on message type
                if 'ERROR' in line or 'Error' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    print(f"❌ {re.sub(pattern, '', line)}")
                elif 'WARN' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    print(f"⚠️ {re.sub(pattern, '', line)}")
                elif 'INFO' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    print(f"ℹ️ {re.sub(pattern, '', line)}")
                else:
                    print(f"✓ {line}")
        
        return result
    
    return wrapper

# Context manager version for inline usage
class CameraOutputFormatter:
    """
    Context manager to format camera output with emojis.
    
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
        
        # Filter and format the messages with emojis
        if combined_output:
            lines = combined_output.strip().split('\n')
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Format based on message type
                if 'ERROR' in line or 'Error' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    print(f"❌ {re.sub(pattern, '', line)}")
                elif 'WARN' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    print(f"⚠️ {re.sub(pattern, '', line)}")
                elif 'INFO' in line:
                    pattern = r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+'
                    print(f"ℹ️ {re.sub(pattern, '', line)}")
                else:
                    print(f"✓ {line}")

# Import camera modules conditionally
try:
    from picamera2 import Picamera2
    PI_CAMERA_VERSION = 2
    print("✓ Using picamera2 module")
except ImportError:
    try:
        import picamera
        from picamera.array import PiRGBArray
        PI_CAMERA_VERSION = 1
        print("✓ Using legacy picamera module")
    except ImportError:
        PI_CAMERA_VERSION = None
        # No verbose output here - will be handled by main program

# Keep the decorator for when this module is used directly
@format_camera_output
def takePhoto(resolution=DEFAULT_RESOLUTION, output_file='images/input.png'):
    """
    Take a photo using the Raspberry Pi camera.
    
    Args:
        resolution: Tuple of (width, height) for the photo resolution
        output_file: Path to save the captured image
        
    Returns:
        Boolean indicating success or failure
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # If not on Raspberry Pi or camera modules not available, return False
    if not IS_RASPBERRY_PI or PI_CAMERA_VERSION is None:
        return False

    # Attempt to take photo using available camera module
    try:
        if PI_CAMERA_VERSION == 2:
            # PiCamera2 approach
            camera = Picamera2()
            camera_config = camera.create_still_configuration(main={"size": resolution})
            camera.configure(camera_config)
            camera.start()
            time.sleep(1)  # Let camera warm up
            camera.capture_file(output_file)
            camera.close()
            return True
            
        elif PI_CAMERA_VERSION == 1:
            # Legacy PiCamera approach
            with picamera.PiCamera() as camera:
                camera.resolution = resolution
                camera.start_preview()
                time.sleep(1)  # Let camera warm up
                camera.capture(output_file)
                return True
    except Exception as e:
        # No verbose output here - will be handled by main program
        return False
    
    return False

# Example of calling the function (will run when this script is executed directly)
if __name__ == "__main__":
    print("✓ Running takePhoto function directly...")
    
    # Check for custom resolution in arguments
    resolution = DEFAULT_RESOLUTION  # Use lower default resolution
    if len(sys.argv) > 1 and "," in sys.argv[1]:
        try:
            w, h = sys.argv[1].split(",")
            resolution = (int(w), int(h))
            print(f"✓ Using custom resolution: {resolution}")
        except Exception as e:
            print(f"❌ Error parsing resolution: {e}")
    
    success = takePhoto(resolution=resolution)
    if success:
        print("✓ takePhoto function completed successfully.")
    else:
        print("❌ takePhoto function completed with errors.")
        sys.exit(1)