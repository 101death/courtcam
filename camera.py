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
    Decorator to format camera output with consistent styling.
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
        
        # Filter and format the messages
        if combined_output:
            lines = combined_output.strip().split('\n')
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Remove timestamps and process IDs
                clean_line = re.sub(r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+', '', line)
                
                # Format based on message type
                if 'ERROR' in line or 'Error' in line:
                    from main import OutputManager
                    OutputManager.log(clean_line, "ERROR")
                elif 'WARN' in line:
                    from main import OutputManager
                    OutputManager.log(clean_line, "WARNING")
                elif 'INFO' in line:
                    from main import OutputManager
                    OutputManager.log(clean_line, "INFO")
                else:
                    from main import OutputManager
                    OutputManager.log(clean_line, "INFO")
        
        return result
    
    return wrapper

# Context manager version for inline usage
class CameraOutputFormatter:
    """
    Context manager to format camera output with consistent styling.
    
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
        
        # Filter and format the messages
        if combined_output:
            lines = combined_output.strip().split('\n')
            for line in lines:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Remove timestamps and process IDs
                clean_line = re.sub(r'^\[\d+:\d+:\d+\.\d+\]\s+\[\d+\]\s+\w+\s+', '', line)
                
                # Format based on message type
                if 'ERROR' in line or 'Error' in line:
                    from main import OutputManager
                    OutputManager.log(clean_line, "ERROR")
                elif 'WARN' in line:
                    from main import OutputManager
                    OutputManager.log(clean_line, "WARNING")
                elif 'INFO' in line:
                    from main import OutputManager
                    OutputManager.log(clean_line, "INFO")
                else:
                    from main import OutputManager
                    OutputManager.log(clean_line, "INFO")

# Import camera modules conditionally
try:
    from picamera2 import Picamera2
    PI_CAMERA_VERSION = 2
    from main import OutputManager
    OutputManager.log("Using picamera2 module", "INFO")
except ImportError:
    try:
        import picamera
        from picamera.array import PiRGBArray
        PI_CAMERA_VERSION = 1
        from main import OutputManager
        OutputManager.log("Using legacy picamera module", "INFO")
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
        from main import OutputManager
        OutputManager.log("Camera not available on this system", "WARNING")
        return False

    # Attempt to take photo using available camera module
    try:
        if PI_CAMERA_VERSION == 2:
            # PiCamera2 approach
            from main import OutputManager
            OutputManager.log("Initializing PiCamera2...", "INFO")
            camera = Picamera2()
            camera_config = camera.create_still_configuration(main={"size": resolution})
            camera.configure(camera_config)
            camera.start()
            time.sleep(1)  # Let camera warm up
            OutputManager.log("Capturing photo...", "INFO")
            camera.capture_file(output_file)
            camera.close()
            OutputManager.log(f"Photo saved to {output_file}", "SUCCESS")
            return True
            
        elif PI_CAMERA_VERSION == 1:
            # Legacy PiCamera approach
            from main import OutputManager
            OutputManager.log("Initializing legacy PiCamera...", "INFO")
            with picamera.PiCamera() as camera:
                camera.resolution = resolution
                camera.start_preview()
                time.sleep(1)  # Let camera warm up
                OutputManager.log("Capturing photo...", "INFO")
                camera.capture(output_file)
                OutputManager.log(f"Photo saved to {output_file}", "SUCCESS")
                return True
    except Exception as e:
        from main import OutputManager
        OutputManager.log(f"Camera error: {str(e)}", "ERROR")
        return False
    
    return False

# Example of calling the function (will run when this script is executed directly)
if __name__ == "__main__":
    from main import OutputManager
    OutputManager.log("Running takePhoto function directly...", "INFO")
    
    # Check for custom resolution in arguments
    resolution = DEFAULT_RESOLUTION  # Use lower default resolution
    if len(sys.argv) > 1 and "," in sys.argv[1]:
        try:
            w, h = sys.argv[1].split(",")
            resolution = (int(w), int(h))
            OutputManager.log(f"Using custom resolution: {resolution}", "INFO")
        except Exception as e:
            OutputManager.log(f"Error parsing resolution: {e}", "ERROR")
    
    success = takePhoto(resolution=resolution)
    if success:
        OutputManager.log("Photo capture completed successfully", "SUCCESS")
    else:
        OutputManager.log("Photo capture failed", "ERROR")
        sys.exit(1)