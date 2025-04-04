import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
from PIL import Image
import time

def takePhoto(resolution=(1920, 1080), output_file='images/input.png'):
    # Initialize the camera
    camera = PiCamera()
    camera.resolution = resolution  # Set resolution
    camera.framerate = 1
    rawCapture = PiRGBArray(camera, size=resolution)
    
    # Allow the camera to warm up
    time.sleep(0.5)
    
    # Capture a frame from the camera
    camera.capture(rawCapture, format="bgr")
    image = rawCapture.array
    
    # Save the image
    image = Image.fromarray(image)
    image.save(output_file)
    print(f"Image saved as {output_file}")

# Example of calling the function
takePhoto()