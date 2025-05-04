import numpy as np
import time
import os

# Attempt to import PiCamera libraries, handle multiple possibilities
try:
    # Try picamera2 first (recommended for newer RPi OS)
    from picamera2 import Picamera2, Preview # type: ignore
    PICAMERA_LIB = "picamera2"
    print("Using picamera2 library.")
except ImportError:
    try:
        # Try legacy picamera
        from picamera.array import PiRGBArray # type: ignore
        from picamera import PiCamera # type: ignore
        PICAMERA_LIB = "picamera"
        print("Using legacy picamera library.")
    except ImportError:
        # If neither works, set to None
        PICAMERA_LIB = None
        print("Warning: Neither picamera2 nor picamera library found.")
        print("Camera functionality will be simulated.")

# Import PIL for saving image regardless of camera library
try:
    from PIL import Image
except ImportError:
    # If PIL is not available, we'll fall back to OpenCV for image saving
    print("PIL (Pillow) not found, will try OpenCV for image saving if needed.")

def takePhoto(resolution=(1920, 1080), output_file='images/input.png'):
    """
    Captures a photo using the available Raspberry Pi camera library (picamera2 or legacy picamera).
    Saves the image to the specified output file.
    If no camera library is found, it simulates the action.
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
            return

    if PICAMERA_LIB == "picamera2":
        try:
            # Initialize Picamera2
            picam2 = Picamera2()
            # Configure the camera for still capture
            config = picam2.create_still_configuration(main={"size": resolution})
            picam2.configure(config)
            
            # Start the camera
            picam2.start()
            
            # Allow camera to adjust
            time.sleep(1) # Increased sleep time for better auto-adjustments
            
            # Capture the image (returns a numpy array)
            image_array = picam2.capture_array("main") # Capture from the 'main' stream
            
            # Stop the camera
            picam2.stop()
            picam2.close() # Release camera resources

            # Save the image using PIL if available
            try:
                image = Image.fromarray(image_array)
                image.save(output_file)
            except NameError:
                # PIL not available, try OpenCV
                import cv2
                cv2.imwrite(output_file, image_array)
                
            print(f"Image saved as {output_file} using picamera2")

        except Exception as e:
            print(f"Error using picamera2: {e}")
            print("Please ensure the camera is connected and enabled in raspi-config.")
            # Attempt to clean up if error occurred mid-process
            if 'picam2' in locals() and hasattr(picam2, 'started') and picam2.started:
                 try: picam2.stop()
                 except: pass
                 try: picam2.close()
                 except: pass

    elif PICAMERA_LIB == "picamera":
        camera = None # Ensure camera is defined for finally block
        try:
            # Initialize the legacy camera
            camera = PiCamera()
            camera.resolution = resolution
            camera.framerate = 1 # Lower framerate for stills
            rawCapture = PiRGBArray(camera, size=resolution)
            
            # Allow the camera to warm up
            time.sleep(1)
            
            # Capture to array
            camera.capture(rawCapture, format="bgr") # BGR format for OpenCV compatibility
            image_array = rawCapture.array
            
            # Save the image using PIL if available
            try:
                image = Image.fromarray(image_array)
                image.save(output_file)
            except NameError:
                # PIL not available, try OpenCV
                import cv2
                cv2.imwrite(output_file, image_array)
                
            print(f"Image saved as {output_file} using legacy picamera")

        except Exception as e:
            print(f"Error using legacy picamera: {e}")
            print("Please ensure the camera is connected, enabled in raspi-config,")
            print("and that the necessary libraries (like libbcm_host.so) are installed.")
            print("Try: sudo apt update && sudo apt install -y libraspberrypi-dev")
        finally:
            # Ensure camera resources are released
            if camera is not None:
                try:
                    camera.close()
                except Exception as close_e:
                     print(f"Error closing legacy picamera: {close_e}")

    else:
        # No camera library found - simulate
        print(f"Simulation: No camera library found. Creating a blank image at {output_file}")
        # Create a placeholder black image
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
        except Exception as write_error:
            print(f"Error creating placeholder image: {write_error}")


# Example of calling the function (will run when this script is executed directly)
if __name__ == "__main__":
     print("Running takePhoto function directly...")
     takePhoto()
     print("takePhoto function finished.")