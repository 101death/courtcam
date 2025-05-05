import numpy as np
import time
import os
import sys
import subprocess

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
                print(f"Detected Raspberry Pi: {PI_MODEL}")
                
                # Check if running 64-bit OS
                if '64' in os.uname().machine:
                    IS_64BIT = True
                    print("Detected 64-bit OS")
                
                # Try to detect Camera Module version
                try:
                    # Use v4l2-ctl to detect camera properties if available
                    result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                           capture_output=True, text=True, check=False)
                    
                    if "Camera 3" in result.stdout:
                        PI_CAMERA_VERSION = 3
                        print("Detected Raspberry Pi Camera 3")
                    elif "Camera 2" in result.stdout:
                        PI_CAMERA_VERSION = 2
                        print("Detected Raspberry Pi Camera 2")
                    elif "Camera Module" in result.stdout:
                        PI_CAMERA_VERSION = 1
                        print("Detected Raspberry Pi Camera 1")
                except Exception as e:
                    # Can't detect camera version from v4l2-ctl
                    print(f"Unable to determine exact camera version: {e}")
except Exception as e:
    print(f"Error detecting Raspberry Pi: {e}")

# Track camera errors and status
camera_errors = []  # Track any errors to report later

# Try to determine if camera is connected using vcgencmd if on a Raspberry Pi
CAMERA_CONNECTED = False
if IS_RASPBERRY_PI:
    try:
        result = subprocess.run(['vcgencmd', 'get_camera'], capture_output=True, text=True)
        if result.returncode == 0 and 'detected=1' in result.stdout:
            CAMERA_CONNECTED = True
            print("Camera hardware detected via vcgencmd")
    except Exception as e:
        print(f"Error checking camera hardware: {e}")

# Attempt to import PiCamera libraries, handle multiple possibilities
try:
    # Try picamera2 first (preferred for Raspberry Pi OS Bullseye/Bookworm and 64-bit systems)
    from picamera2 import Picamera2, Preview # type: ignore
    PICAMERA_LIB = "picamera2"
    print("Using picamera2 library (recommended for Raspberry Pi OS Bullseye/Bookworm)")
except ImportError as e:
    camera_errors.append(f"picamera2 import error: {str(e)}")
    print(f"Could not import picamera2: {e}")
    print("Trying legacy picamera library...")
    
    try:
        # Try legacy picamera (better for older Raspberry Pi OS)
        from picamera.array import PiRGBArray # type: ignore
        from picamera import PiCamera # type: ignore
        PICAMERA_LIB = "picamera"
        print("Using legacy picamera library")
        
        # Warning for 64-bit systems using legacy picamera
        if IS_64BIT:
            print("WARNING: Using legacy picamera on 64-bit OS may cause issues.")
            print("It's recommended to install picamera2 for 64-bit systems:")
            print("sudo apt update && sudo apt install -y python3-picamera2")
    except ImportError as e:
        camera_errors.append(f"picamera import error: {str(e)}")
        PICAMERA_LIB = None
        print(f"Could not import legacy picamera: {e}")
    except OSError as e:
        # This catches errors like missing shared libraries (libmmal.so, libbcm_host.so)
        camera_errors.append(f"Camera library dependency error: {str(e)}")
        PICAMERA_LIB = None
        print(f"Error loading camera libraries: {str(e)}")
        print("\nMissing system libraries. Run the following command to install required dependencies:")
        
        if IS_64BIT:
            print("For 64-bit OS (recommended):")
            print("sudo apt update && sudo apt install -y python3-picamera2 libcamera-dev python3-libcamera")
        else:
            print("For 32-bit OS:")
            print("sudo apt update && sudo apt install -y libraspberrypi-dev libcamera-dev python3-libcamera python3-picamera")
        
        print("\nYou can also run the script with --no-camera to continue without camera support.")

if PICAMERA_LIB is None:
    print("Camera functionality will be simulated.")
    
    # Print detailed troubleshooting based on Pi model and OS
    if IS_RASPBERRY_PI:
        print("\nTroubleshooting camera issues on your Raspberry Pi:")
        print("1. Ensure the camera is properly connected (check the ribbon cable)")
        print("2. Enable the camera interface:")
        print("   sudo raspi-config → Interface Options → Camera → Enable")
        print("   (Reboot required after enabling)")
        
        if PI_CAMERA_VERSION == 3:
            print("\nSpecific instructions for Raspberry Pi Camera 3:")
            print("1. Ensure you have the latest OS and libraries:")
            print("   sudo apt update && sudo apt upgrade")
            print("2. Install libcamera and picamera2:")
            print("   sudo apt install -y python3-picamera2 libcamera-apps")
            print("3. The Camera 3 works best with picamera2, NOT legacy picamera")
        elif "Zero 2" in PI_MODEL:
            print("\nSpecific instructions for Raspberry Pi Zero 2W:")
            if IS_64BIT:
                print("For 64-bit OS on Pi Zero 2W:")
                print("1. Install picamera2 (recommended for 64-bit):")
                print("   sudo apt update && sudo apt install -y python3-picamera2 libcamera-dev")
            else:
                print("For 32-bit OS on Pi Zero 2W:")
                print("1. You can use either picamera2 (newer) or picamera (legacy):")
                print("   sudo apt update && sudo apt install -y python3-picamera2 python3-picamera")
            print("2. The Pi Zero 2W has limited resources, so you may need to optimize:")
            print("   - Set smaller resolution: python main.py --camera-resolution 640,480")
        else:
            # Generic RPi instructions
            if IS_64BIT:
                print("\nFor 64-bit Raspberry Pi OS:")
                print("1. Install picamera2 (recommended for 64-bit):")
                print("   sudo apt update && sudo apt install -y python3-picamera2 libcamera-dev")
            else:
                print("\nFor 32-bit Raspberry Pi OS:")
                print("1. For legacy camera support:")
                print("   sudo apt update && sudo apt install -y python3-picamera")
                print("2. For new camera stack (recommended):")
                print("   sudo apt update && sudo apt install -y python3-picamera2 libcamera-dev")
    
    print("\nAlternative options:")
    print("1. Run with --no-camera to skip camera functionality:")
    print("   python main.py --no-camera")
    print("2. Place an image manually in the images/ directory:")
    print("   mkdir -p images && cp your_image.jpg images/input.png")

# Import PIL for saving image regardless of camera library
try:
    from PIL import Image
except ImportError:
    # If PIL is not available, we'll fall back to OpenCV for image saving
    print("PIL (Pillow) not found, will try OpenCV for image saving if needed.")

def takePhoto(resolution=DEFAULT_RESOLUTION, output_file='images/input.png'):
    """
    Captures a photo using the available Raspberry Pi camera library (picamera2 or legacy picamera).
    Saves the image to the specified output file.
    If no camera library is found, it simulates the action.
    
    Args:
        resolution: Tuple of (width, height) for the captured image
        output_file: Path where the image will be saved
        
    Returns:
        True if capture was successful, False otherwise
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created directory: {output_dir}")
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}")
            # If directory can't be created, we cannot save the image
            return False

    # Check if camera is physically connected (for better error messages)
    if IS_RASPBERRY_PI and not CAMERA_CONNECTED:
        print("No camera hardware detected. Make sure camera is connected and enabled.")
        print("Run 'vcgencmd get_camera' to check camera status.")
        print("Run 'sudo raspi-config' to enable camera interface if needed.")
        create_dummy_image(resolution, output_file)
        return False

    # Use the available camera library
    if PICAMERA_LIB == "picamera2":
        try:
            # Initialize Picamera2
            picam2 = Picamera2()
            
            # Configure the camera for still capture
            # Apply resolution optimizations based on detected hardware
            actual_resolution = resolution
            
            # Special handling for Pi Camera 3
            if PI_CAMERA_VERSION == 3:
                # Camera 3 works best with specific configurations
                print(f"Using optimized settings for Pi Camera 3")
                # Use a modest resolution that works well with Camera 3
                if resolution[0] > 640:
                    actual_resolution = (640, 480)
                    print(f"Using lower resolution {actual_resolution} for Camera 3 for better performance")
                
                # Create configuration with additional Camera 3 specific tuning
                config = picam2.create_still_configuration(
                    main={"size": actual_resolution},
                    lores={"size": (320, 240)},  # Low-res stream for viewfinder
                    display="lores",              # Use low-res for display
                    buffer_count=1               # Minimize memory usage
                )
            elif "Zero" in PI_MODEL and resolution[0] > 800:
                # Use a smaller resolution for Pi Zero to conserve resources
                actual_resolution = (800, 600)
                print(f"Reducing resolution to {actual_resolution} for Pi Zero")
                config = picam2.create_still_configuration(main={"size": actual_resolution})
            else:
                # Standard configuration for other Pi models
                config = picam2.create_still_configuration(main={"size": actual_resolution})
                
            picam2.configure(config)
            
            # Start the camera
            picam2.start()
            
            # Allow camera to adjust - shorter time for Pi Zero to reduce memory usage
            warm_up_time = 0.5 if "Zero" in PI_MODEL else 1.0
            time.sleep(warm_up_time)
            
            # Capture the image (returns a numpy array)
            print("Capturing image...")
            image_array = picam2.capture_array("main")
            
            # Stop the camera
            picam2.stop()
            picam2.close()

            # Resize the image if needed
            if actual_resolution != resolution:
                try:
                    from PIL import Image
                    # Convert numpy array to PIL image
                    pil_image = Image.fromarray(image_array)
                    # Resize to requested resolution
                    pil_image = pil_image.resize(resolution)
                    pil_image.save(output_file)
                    print(f"Image resized to {resolution} and saved as {output_file}")
                except Exception as e:
                    print(f"Error resizing image: {e}")
                    # Save without resizing
                    try:
                        Image.fromarray(image_array).save(output_file)
                    except:
                        import cv2
                        cv2.imwrite(output_file, image_array)
            else:
                # Save the image using PIL if available
                try:
                    image = Image.fromarray(image_array)
                    image.save(output_file)
                except NameError:
                    # PIL not available, try OpenCV
                    import cv2
                    cv2.imwrite(output_file, image_array)
                
            print(f"Image saved as {output_file} using picamera2")
            return True

        except Exception as e:
            print(f"Error using picamera2: {e}")
            print("Please ensure the camera is connected and enabled in raspi-config.")
            
            # Attempt to clean up if error occurred mid-process
            if 'picam2' in locals() and hasattr(picam2, 'started') and picam2.started:
                try: picam2.stop()
                except: pass
                try: picam2.close()
                except: pass
            
            # Create a dummy image as fallback
            create_dummy_image(resolution, output_file)
            return False

    elif PICAMERA_LIB == "picamera":
        camera = None  # Ensure camera is defined for finally block
        try:
            # Skip attempting to use legacy picamera with Camera 3
            if PI_CAMERA_VERSION == 3:
                print("WARNING: Raspberry Pi Camera 3 is not compatible with legacy picamera.")
                print("Please install and use picamera2 instead:")
                print("sudo apt update && sudo apt install -y python3-picamera2 libcamera-dev")
                create_dummy_image(resolution, output_file)
                return False
                
            # Initialize the legacy camera
            camera = PiCamera()
            
            # Use a lower resolution for Pi Zero or if requested resolution is high
            actual_resolution = resolution
            if "Zero" in PI_MODEL and resolution[0] > 800:
                actual_resolution = (800, 600)
                print(f"Reducing resolution to {actual_resolution} for Pi Zero")
            
            camera.resolution = actual_resolution
            camera.framerate = 1  # Lower framerate for stills
            
            # Allow the camera to warm up - shorter time for Pi Zero
            warm_up_time = 0.5 if "Zero" in PI_MODEL else 1.0
            print(f"Warming up camera for {warm_up_time} seconds...")
            time.sleep(warm_up_time)
            
            # For Pi Zero, use direct capture to file to reduce memory usage
            if "Zero" in PI_MODEL:
                print("Using direct capture to file for Pi Zero")
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                camera.capture(output_file)
                print(f"Image captured directly to {output_file}")
            else:
                # For other Pi models, capture to memory first
                rawCapture = PiRGBArray(camera, size=actual_resolution)
                camera.capture(rawCapture, format="bgr")  # BGR format for OpenCV compatibility
                image_array = rawCapture.array
                
                # Resize if needed
                if actual_resolution != resolution:
                    try:
                        from PIL import Image
                        # Convert to PIL and resize
                        pil_image = Image.fromarray(image_array)
                        pil_image = pil_image.resize(resolution)
                        pil_image.save(output_file)
                    except Exception as e:
                        print(f"Error resizing: {e}")
                        # Save without resizing
                        try:
                            Image.fromarray(image_array).save(output_file)
                        except:
                            import cv2
                            cv2.imwrite(output_file, image_array)
                else:
                    # Save without resizing
                    try:
                        image = Image.fromarray(image_array)
                        image.save(output_file)
                    except NameError:
                        import cv2
                        cv2.imwrite(output_file, image_array)
                
            print(f"Image saved as {output_file} using legacy picamera")
            return True

        except Exception as e:
            print(f"Error using legacy picamera: {e}")
            print("Please ensure the camera is connected, enabled in raspi-config,")
            print("and that the necessary libraries are installed.")
            
            if IS_64BIT:
                print("\nNOTE: Legacy picamera often has issues with 64-bit OS.")
                print("Try installing picamera2 instead:")
                print("sudo apt update && sudo apt install -y python3-picamera2 libcamera-dev")
            else:
                print("Try: sudo apt update && sudo apt install -y libraspberrypi-dev python3-picamera")
            
            # Create a dummy image as fallback
            create_dummy_image(resolution, output_file)
            return False
            
        finally:
            # Ensure camera resources are released
            if camera is not None:
                try:
                    camera.close()
                except Exception as close_e:
                    print(f"Error closing legacy picamera: {close_e}")

    else:
        # No camera library found - simulate
        create_dummy_image(resolution, output_file)
        return False


def create_dummy_image(resolution=(640, 480), output_file='images/input.png'):
    """Create a dummy black image when camera capture fails"""
    print(f"Creating a placeholder black image at {output_file}")
    try:
        # Try using PIL
        try:
            from PIL import Image
            image = Image.new('RGB', resolution, color='black')
            image.save(output_file)
        except ImportError:
            # Use OpenCV instead
            import cv2
            dummy_image = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
            cv2.imwrite(output_file, dummy_image)
        print(f"Created placeholder image at {output_file}")
        return True
    except Exception as write_error:
        print(f"Error creating placeholder image: {write_error}")
        return False


# Example of calling the function (will run when this script is executed directly)
if __name__ == "__main__":
    print("Running takePhoto function directly...")
    
    # Check for custom resolution in arguments
    resolution = DEFAULT_RESOLUTION  # Use lower default resolution
    if len(sys.argv) > 1 and "," in sys.argv[1]:
        try:
            w, h = sys.argv[1].split(",")
            resolution = (int(w), int(h))
            print(f"Using custom resolution: {resolution}")
        except Exception as e:
            print(f"Error parsing resolution: {e}")
    
    success = takePhoto(resolution=resolution)
    if success:
        print("takePhoto function completed successfully.")
    else:
        print("takePhoto function completed with errors.")
        sys.exit(1)