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

# Improved function to detect court using only color mask and straight line filtering
def detect_court(image, config, advanced_config):
    print_step("Detecting tennis court using color mask and straight line filtering...")
    
    # Get the court color from config
    court_color = config['court_color']
    
    # Get output folder from config
    output_folder = config['output_folder']
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Create color mask based on HSV ranges
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
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
    
    # Save initial color mask for debugging
    cv2.imwrite(os.path.join(output_folder, 'initial_color_mask.png'), color_mask)
    
    # Step 2: Apply morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Save cleaned mask
    cv2.imwrite(os.path.join(output_folder, 'cleaned_mask.png'), color_mask)
    
    # Step 3: Find contours and filter non-straight lines
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    straight_mask = np.zeros_like(color_mask)
    
    if contours:
        # Process each contour
        for contour in contours:
            # Filter small contours
            if cv2.contourArea(contour) < 500:  # Adjust threshold as needed
                continue
                
            # Approximate the contour with a polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)  # Adjust tolerance for "wiggly" lines
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Draw the approximated contour (straighter lines)
            cv2.drawContours(straight_mask, [approx], 0, 255, -1)
    
    # Save the straight line mask
    cv2.imwrite(os.path.join(output_folder, 'straight_mask.png'), straight_mask)
    
    # Step 4: Fill holes in the mask to create the final court area
    final_mask = straight_mask.copy()
    
    # Find all contours in the straight mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Fill in the largest contour (which should be the court)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest_contour], 0, 255, -1)
    
    # Save the final mask
    cv2.imwrite(os.path.join(output_folder, 'final_mask.png'), final_mask)
    
    # Debug information for the largest area
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        print_step(f"Court detected at: x={x}, y={y}, width={w}, height={h}")
        print_step(f"Court aspect ratio: {w/h if h > 0 else 0:.2f}")
    
    return final_mask

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
court_mask = detect_court(image, config, advanced_config)

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
court_overlay[court_mask == 255] = (0, 255, 255)  # Yellow overlay for court
alpha = 0.3
cv2.addWeighted(court_overlay, alpha, output_image, 1 - alpha, 0, output_image)

# Draw bounding boxes and labels for people
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        feet_x = x + w // 2
        feet_y = y + h
        inside = (0 <= feet_x < width and 0 <= feet_y < height and court_mask[feet_y, feet_x] == 255)
        color = (0, 255, 0) if inside else (0, 0, 255)  # Green for inside, Red for outside
        label = 'Inside' if inside else 'Outside'
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(output_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save the output image
print_step("Saving output image...")
cv2.imwrite(os.path.join(config['output_folder'], 'output.png'), output_image)

# Create a visualization of the mask processing steps
mask_process = np.hstack([
    cv2.imread(os.path.join(config['output_folder'], 'initial_color_mask.png')) if os.path.exists(os.path.join(config['output_folder'], 'initial_color_mask.png')) else np.zeros((300, 300), dtype=np.uint8),
    cv2.imread(os.path.join(config['output_folder'], 'cleaned_mask.png')) if os.path.exists(os.path.join(config['output_folder'], 'cleaned_mask.png')) else np.zeros((300, 300), dtype=np.uint8),
    cv2.imread(os.path.join(config['output_folder'], 'straight_mask.png')) if os.path.exists(os.path.join(config['output_folder'], 'straight_mask.png')) else np.zeros((300, 300), dtype=np.uint8),
    cv2.imread(os.path.join(config['output_folder'], 'final_mask.png')) if os.path.exists(os.path.join(config['output_folder'], 'final_mask.png')) else np.zeros((300, 300), dtype=np.uint8)
])

# Save the debug visualization
cv2.imwrite(os.path.join(config['output_folder'], 'mask_process.png'), mask_process)

print_success(f"Output image saved as '{os.path.join(config['output_folder'], 'output.png')}'.")
print_success(f"Mask processing steps saved as '{os.path.join(config['output_folder'], 'mask_process.png')}'.")
print_header("Analysis Complete")