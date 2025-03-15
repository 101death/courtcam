import cv2
import numpy as np
import json
import os
import urllib.request
import certifi
import ssl

# Terminal styling functions
def print_header(text):
    print(f"\n{'=' * 50}\n{text.center(50)}\n{'=' * 50}")

def print_step(text):
    print(f"[+] {text}")

def print_error(text):
    print(f"[ERROR] {text}")

def print_success(text):
    print(f"[SUCCESS] {text}")

# Function to ensure model files are present
def ensure_model_files():
    files = {
        'yolov3.cfg': 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg',
        'coco.names': 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names',
        'yolov3.weights': 'https://pjreddie.com/media/files/yolov3.weights'
    }
    for filename, url in files.items():
        if not os.path.exists(filename):
            print_step(f"{filename} not found. Downloading from {url}...")
            try:
                context = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(url, context=context) as response:
                    with open(filename, 'wb') as out_file:
                        out_file.write(response.read())
                print_success(f"Downloaded {filename} successfully.")
            except Exception as e:
                print_error(f"Failed to download {filename}: {e}")
                print_error("Please check your internet connection and try again.")
                exit(1)

# Function to detect court using multiple color ranges
def detect_court(image, config, advanced_config):
    print_step("Detecting tennis court using advanced line detection and AI methods...")
    
    # Get the court color from config
    court_color = config['court_color']
    width_flexibility = config['width_flexibility']
    length_flexibility = config['length_flexibility']
    
    # Get output folder from config
    output_folder = config['output_folder']
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Tennis court detection via line segments
    # First, detect lines which are common in tennis courts
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Canny edge detector
    edges = cv2.Canny(blurred, 50, 150)
    
    # Save edge detection result for debugging
    cv2.imwrite(os.path.join(output_folder, 'edges.png'), edges)
    
    # Use HoughLinesP to detect line segments
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
    
    # Create a blank image to draw the detected lines
    line_image = np.zeros_like(gray)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    
    cv2.imwrite(os.path.join(output_folder, 'line_image.png'), line_image)
    
    # Step 2: Combine with color detection
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a mask for the court based on color
    color_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    # Process inside, outside, and lines court areas
    for area_type in ['inside', 'outside', 'lines']:
        if area_type in advanced_config[court_color]:
            lower_hsv = np.array(advanced_config[court_color][area_type]['lower_hsv'])
            upper_hsv = np.array(advanced_config[court_color][area_type]['upper_hsv'])
            
            # Create mask for this color range
            area_mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
            
            # Add to combined mask
            color_mask = cv2.bitwise_or(color_mask, area_mask)
    
    # Save color mask for debugging
    cv2.imwrite(os.path.join(output_folder, 'color_mask.png'), color_mask)
    cv2.imwrite(os.path.join(output_folder, 'line_image.png'), line_image)
    
    # Since color_mask.png is working well, skip the refinement and just use it directly
    print_step("Using direct color mask without further processing")
    
    # Just for debug information, find the contours in the color mask to report dimensions
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the largest contour for debugging info
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Debug information
        print_step(f"Court detected at: x={x}, y={y}, width={w}, height={h}")
        print_step(f"Court aspect ratio: {w/h if h > 0 else 0:.2f}")
    else:
        print_step("No contours found in the color mask")
    
    # Save the color mask as the final mask for consistency in output
    cv2.imwrite(os.path.join(output_folder, 'final_mask.png'), color_mask)
    
    # Return the direct color mask as our court detection result
    return color_mask

# Main script starts here
print_header("Court Camera Analysis")

print_step("Ensuring model files are present...")
ensure_model_files()
print_success("Model files are ready.")

# Check for input image
if not os.path.exists('input.png'):
    print_error("input.png not found. Please provide the input image.")
    exit(1)

# Load configuration files
print_step("Loading configuration files...")
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print_error("config.json not found. Please provide the configuration file.")
    exit(1)
except json.JSONDecodeError:
    print_error("config.json is not a valid JSON file.")
    exit(1)

try:
    with open('advanced.json', 'r') as f:
        advanced_config = json.load(f)
except FileNotFoundError:
    print_error("advanced.json not found. Please provide the advanced configuration file.")
    exit(1)
except json.JSONDecodeError:
    print_error("advanced.json is not a valid JSON file.")
    exit(1)

# Ensure config has output_folder
if 'output_folder' not in config:
    print_step("No output folder specified in config. Using default 'output'")
    config['output_folder'] = 'output'

# Create output folder if it doesn't exist
os.makedirs(config['output_folder'], exist_ok=True)

# Load and process the image
print_step("Loading and processing input image...")
image = cv2.imread('input.png')
if image is None:
    print_error("Could not load input.png. Please ensure the file is a valid image.")
    exit(1)

# Use improved court detection function
adjusted_mask = detect_court(image, config, advanced_config)

# Load YOLOv3 model
print_step("Loading YOLOv3 model for person detection...")
try:
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
except Exception as e:
    print_error(f"Error loading YOLOv3 model: {e}")
    exit(1)

with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Detect people
print_step("Detecting people in the image...")
height, width = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

boxes = []
confidences = []
class_ids = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and classes[class_id] == 'person':
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Annotate the output image
print_step("Annotating the output image...")
output_image = image.copy()

# Draw court overlay
court_overlay = np.zeros_like(image)
court_overlay[adjusted_mask == 255] = (0, 255, 255)  # Yellow overlay for court
alpha = 0.3
cv2.addWeighted(court_overlay, alpha, output_image, 1 - alpha, 0, output_image)

# Draw bounding boxes and labels for people
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        feet_x = x + w // 2
        feet_y = y + h
        inside = (0 <= feet_x < width and 0 <= feet_y < height and adjusted_mask[feet_y, feet_x] == 255)
        color = (0, 255, 0) if inside else (0, 0, 255)  # Green for inside, Red for outside
        label = 'Inside' if inside else 'Outside'
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the output and debug visualizations
print_step("Saving output and debug images...")

# Create a visualization of the detection process
debug_image = np.hstack([
    cv2.imread(os.path.join(config['output_folder'], 'edges.png')) if os.path.exists(os.path.join(config['output_folder'], 'edges.png')) else np.zeros((300, 300, 3), dtype=np.uint8),
    cv2.imread(os.path.join(config['output_folder'], 'line_image.png')) if os.path.exists(os.path.join(config['output_folder'], 'line_image.png')) else np.zeros((300, 300, 3), dtype=np.uint8),
])
debug_image2 = np.hstack([
    cv2.imread(os.path.join(config['output_folder'], 'color_mask.png')) if os.path.exists(os.path.join(config['output_folder'], 'color_mask.png')) else np.zeros((300, 300, 3), dtype=np.uint8),
    cv2.imread(os.path.join(config['output_folder'], 'final_mask.png')) if os.path.exists(os.path.join(config['output_folder'], 'final_mask.png')) else np.zeros((300, 300, 3), dtype=np.uint8),
])

# Save the final output image
cv2.imwrite(os.path.join(config['output_folder'], 'output.png'), output_image)
cv2.imwrite(os.path.join(config['output_folder'], 'debug_process.png'), debug_image)
cv2.imwrite(os.path.join(config['output_folder'], 'debug_masks.png'), debug_image2)

print_success(f"Output image saved as '{os.path.join(config['output_folder'], 'output.png')}'.")
print_success(f"Debug visualizations saved as '{os.path.join(config['output_folder'], 'debug_process.png')}' and '{os.path.join(config['output_folder'], 'debug_masks.png')}'.")
print_header("Analysis Complete")
