#!/usr/bin/env python3
"""
Tennis Court Detector - Detects people on tennis courts using YOLOv5
"""
import os
import cv2 # type: ignore
import numpy as np # type: ignore
import json
import torch # type: ignore
from shapely.geometry import Polygon, Point # type: ignore
import sys
import ssl
import argparse
import time
import io
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import contextlib

# === CONFIGURATION SETTINGS ===
class Config:
    # Color settings for court detection
    COURT_COLORS = {
        "blue": {
            "lower": [90, 40, 40],
            "upper": [120, 255, 255]
        },
        "green": {
            "lower": [40, 40, 40],
            "upper": [80, 255, 255]
        },
        "red": {
            "lower": [0, 50, 50],
            "upper": [10, 255, 255],
            "lower2": [170, 50, 50],
            "upper2": [180, 255, 255]
        }
    }
    
    # Court detection parameters
    class Court:
        MIN_AREA = 5000              # Minimum court area
        MAX_AREA = 150000            # Maximum court area
        MIN_SCORE = 0.5              # Minimum score for a valid court
        MIN_ASPECT_RATIO = 1.2       # Minimum aspect ratio (width/height)
        MAX_ASPECT_RATIO = 3.0       # Maximum aspect ratio (width/height)
        MIN_BLUE_RATIO = 0.3         # Minimum ratio of blue pixels
        MIN_GREEN_RATIO = 0.05       # Minimum ratio of green pixels
    
    # Morphological operation settings
    class Morphology:
        KERNEL_SIZE = 5              # Kernel size for morphological operations
        ITERATIONS = 2               # Number of iterations for closing operations
    
    # Court area definitions
    IN_BOUNDS_COLOR = "blue"         # Color that represents in-bounds
    OUT_BOUNDS_COLOR = "green"       # Color that represents out-of-bounds
    
    # Visualization settings
    class Visual:
        COURT_OUTLINE_COLOR = (0, 255, 0)        # Green
        COURT_OUTLINE_THICKNESS = 3              # Line thickness
        PERSON_IN_BOUNDS_COLOR = (0, 255, 0)     # Green for people in court
        PERSON_OUT_BOUNDS_COLOR = (0, 165, 255)  # Orange for people near court
        PERSON_OFF_COURT_COLOR = (0, 0, 255)     # Red for people off court
        TEXT_COLOR = (255, 255, 255)             # White
        FONT_SCALE = 0.5                         # Text size
        TEXT_THICKNESS = 2                       # Text thickness
        DRAW_COURT_OUTLINE = True                # Whether to draw court outline
        SHOW_COURT_NUMBER = False                # Whether to show court number in labels
    
    # Terminal output settings
    class Output:
        VERBOSE = True               # Show detailed output
        USE_COLOR_OUTPUT = True      # Use colored terminal output
        SHOW_TIMESTAMP = True        # Show timestamps in output
        SUPER_QUIET = False          # Super quiet mode (almost no output)
        SUMMARY_ONLY = False         # Only show summary of results
        
        # ANSI color codes for terminal output
        COLORS = {
            "INFO": "\033[94m",      # Blue
            "SUCCESS": "\033[92m",   # Green
            "WARNING": "\033[93m",   # Yellow
            "ERROR": "\033[91m",     # Red
            "DEBUG": "\033[90m",     # Gray
            "RESET": "\033[0m",      # Reset
            "BOLD": "\033[1m",       # Bold
            "UNDERLINE": "\033[4m"   # Underline
        }
    
    # Debug mode
    DEBUG_MODE = False              # Detailed debug output mode
    
    # Paths and directories
    class Paths:
        IMAGES_DIR = "images"
        INPUT_IMAGE = "input.png"
        OUTPUT_IMAGE = "output.png"
        MODELS_DIR = "models"
        
        @classmethod
        def input_path(cls):
            return os.path.join(cls.IMAGES_DIR, cls.INPUT_IMAGE)
            
        @classmethod
        def output_path(cls):
            return os.path.join(cls.IMAGES_DIR, cls.OUTPUT_IMAGE)
            
        @classmethod
        def debug_dir(cls):
            return os.path.join(os.path.dirname(cls.output_path()), "debug")
    
    # Model settings
    class Model:
        NAME = "yolov5s"             # YOLOv5 model size (yolov5s, yolov5m, yolov5l, etc.)
        CONFIDENCE = 0.3             # Detection confidence threshold
        IOU = 0.45                   # IoU threshold
        CLASSES = [0]                # Only detect people (class 0)

class OutputManager:
    """
    Centralized output manager for all terminal output.
    Provides pretty formatting and suppresses unwanted messages.
    """
    # ANSI color codes
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
    
    # Symbol map for message types
    SYMBOLS = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "DEBUG": "🔍",
    }
    
    @staticmethod
    def log(message, level="INFO"):
        """Print formatted log messages with consistent, clean formatting"""
        # In super quiet mode, only show errors and success messages
        if Config.Output.SUPER_QUIET and level not in ["ERROR", "SUCCESS"]:
            return
            
        if not Config.Output.VERBOSE and level == "DEBUG":
            return
        
        # Get color based on level
        color = ""
        if Config.Output.USE_COLOR_OUTPUT:
            if level == "INFO":
                color = OutputManager.BLUE
            elif level == "SUCCESS":
                color = OutputManager.GREEN
            elif level == "WARNING":
                color = OutputManager.YELLOW
            elif level == "ERROR":
                color = OutputManager.RED
            elif level == "DEBUG":
                color = OutputManager.GRAY
        
        # Format timestamp with consistent width
        timestamp = ""
        if Config.Output.SHOW_TIMESTAMP:
            timestamp = f"{OutputManager.GRAY}[{datetime.now().strftime('%H:%M:%S')}]{OutputManager.RESET} "
        
        # Create symbol prefix
        symbol = OutputManager.SYMBOLS.get(level, "")
        
        # Format the message with appropriate styling
        formatted_message = f"{timestamp}{color}{symbol} {message}{OutputManager.RESET}"
        
        print(formatted_message)
    
    @staticmethod
    def summarize_detections(courts, people, people_locations):
        """Summarize detection results in a nice format"""
        if Config.Output.SUPER_QUIET:
            return
            
        OutputManager.log(f"{OutputManager.BOLD}Detection Summary:{OutputManager.RESET}", "INFO")
        OutputManager.log(f"Found {len(courts)} tennis courts", "SUCCESS")
        OutputManager.log(f"Found {len(people)} people in the image", "SUCCESS")
        
        # Count people by location
        in_bounds_count = sum(1 for _, area_type in people_locations if area_type == 'in_bounds')
        out_bounds_count = sum(1 for _, area_type in people_locations if area_type == 'out_bounds')
        off_court_count = sum(1 for _, area_type in people_locations if area_type == 'off_court')
        
        OutputManager.log(f"  - {in_bounds_count} people in-bounds", "INFO")
        OutputManager.log(f"  - {out_bounds_count} people out-of-bounds", "INFO")
        OutputManager.log(f"  - {off_court_count} people off court", "INFO")

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output"""
    # Save the original stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Create string buffers to capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Redirect output to the buffers
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer
    
    try:
        yield  # Code inside the with block runs with redirected stdout/stderr
    finally:
        # Restore the original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        # You can access captured output if needed via stdout_buffer.getvalue() 
        # and stderr_buffer.getvalue()

def create_blue_mask(image):
    """Create a mask for blue areas in the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get blue range from config
    blue_range = Config.COURT_COLORS["blue"]
    lower = np.array(blue_range["lower"])
    upper = np.array(blue_range["upper"])
    
    # Create mask
    blue_mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up mask
    kernel = np.ones((Config.Morphology.KERNEL_SIZE, Config.Morphology.KERNEL_SIZE), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=Config.Morphology.ITERATIONS)
    
    return blue_mask

def create_green_mask(image):
    """Create a mask for green areas in the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get green range from config
    green_range = Config.COURT_COLORS["green"]
    lower = np.array(green_range["lower"])
    upper = np.array(green_range["upper"])
    
    # Create mask
    green_mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up mask
    kernel = np.ones((Config.Morphology.KERNEL_SIZE, Config.Morphology.KERNEL_SIZE), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=Config.Morphology.ITERATIONS)
    
    return green_mask

def is_sky_region(contour, image_height, image_width):
    """Check if a contour is likely to be sky based on position and characteristics"""
    x, y, w, h = cv2.boundingRect(contour)
    
    # Sky is usually at the top of the image
    is_at_top = y < image_height * 0.15
    
    # Sky is usually wide
    is_wide = w > image_width * 0.5
    
    # Sky usually has a small height
    is_short = h < image_height * 0.2
    
    # Check if the contour is likely to be sky
    return is_at_top and (is_wide or is_short)

def detect_tennis_court(image, debug_folder=None):
    """
    Detect tennis courts in an image using color masking and contour analysis.
    Simplified approach: Every blue area next to green is a court.
    Returns list of tennis court contours.
    """
    height, width = image.shape[:2]
    
    # Create masks
    OutputManager.log("Creating blue and green masks...", "DEBUG")
    blue_mask = create_blue_mask(image)
    green_mask = create_green_mask(image)
    
    # Save raw masks for debugging
    if debug_folder and Config.Output.VERBOSE:
        cv2.imwrite(os.path.join(debug_folder, "blue_mask_raw.png"), blue_mask)
        cv2.imwrite(os.path.join(debug_folder, "green_mask_raw.png"), green_mask)
    
    # Find blue contours (potential courts)
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each blue contour as a potential court
    valid_courts = []
    
    for contour in blue_contours:
        area = cv2.contourArea(contour)
        # Filter by minimum area to avoid noise
        if area < Config.Court.MIN_AREA:
            continue
            
        # Get bounding box for aspect ratio check
        x, y, w, h = cv2.boundingRect(contour)
        
        # Create slightly dilated mask for this blue region to check if it's next to green
        region_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(region_mask, [contour], -1, 255, -1)
        
        # Dilate the region mask slightly to check for adjacent green pixels
        kernel = np.ones((15, 15), np.uint8)
        dilated_region = cv2.dilate(region_mask, kernel, iterations=1)
        
        # Check if there's green adjacent to this blue region
        green_nearby = cv2.bitwise_and(green_mask, dilated_region)
        green_nearby_pixels = cv2.countNonZero(green_nearby)
        
        # If there's no green nearby, it's not a court
        if green_nearby_pixels < 100:  # Threshold for minimum green pixels needed
            OutputManager.log(f"Rejecting blue area: no green nearby", "DEBUG")
            continue
        
        # Get the blue area itself
        blue_area = cv2.bitwise_and(blue_mask, region_mask)
        blue_pixels = cv2.countNonZero(blue_area)
        
        # Get convex hull for better shape
        hull = cv2.convexHull(contour)
        
        # Approximate polygon
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
        
        # Save court info
        court_info = {
            'contour': contour,
            'approx': approx,
            'hull': hull,
            'area': area,
            'blue_ratio': 1.0,  # This is a blue region, so ratio is 1
            'green_ratio': green_nearby_pixels / area,
            'blue_mask': blue_area,
            'green_mask': green_nearby,
            'blue_pixels': blue_pixels,
            'green_pixels': green_nearby_pixels
        }
        
        valid_courts.append(court_info)
        OutputManager.log(f"Court {len(valid_courts)} accepted: Area={area:.1f}, Green nearby pixels={green_nearby_pixels}", "SUCCESS")
    
    # Save debug visualizations
    if debug_folder and Config.Output.VERBOSE:
        # Create visualization of masks
        masks_viz = np.zeros((height, width, 3), dtype=np.uint8)
        masks_viz[blue_mask > 0] = [255, 0, 0]  # Blue
        masks_viz[green_mask > 0] = [0, 255, 0]  # Green
        cv2.imwrite(os.path.join(debug_folder, "color_masks.png"), masks_viz)
        
        # Create visualization of all courts
        if valid_courts:
            courts_viz = image.copy()
            for i, court in enumerate(valid_courts):
                # Draw court outline
                cv2.drawContours(courts_viz, [court['approx']], 0, Config.Visual.COURT_OUTLINE_COLOR, 2)
                
                # Add court number
                x, y, w, h = cv2.boundingRect(court['approx'])
                cv2.putText(courts_viz, f"Court {i+1}", (x + w//2 - 40, y + h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(debug_folder, "courts_detected.png"), courts_viz)
    
    OutputManager.log(f"Found {len(valid_courts)} tennis courts", "SUCCESS")
    return valid_courts

def detect_people(image):
    """
    Detect people in an image using YOLOv5
    Returns a list of dictionaries with person positions and bounding boxes
    """
    # Set the yolov5 model path
    yolov5_path = os.path.join(Config.Paths.MODELS_DIR, f'{Config.Model.NAME}.pt')
    
    # Suppress YOLOv5 output
    with suppress_stdout_stderr():
        # Load the model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolov5_path, verbose=False)
        
        # Set model parameters
        model.conf = Config.Model.CONFIDENCE
        model.iou = Config.Model.IOU
        model.classes = Config.Model.CLASSES
        
        # Perform detection
        results = model(image)
    
    # Extract people detections
    people = []
    
    # Get pandas dataframe from results
    df = results.pandas().xyxy[0]
    
    # Filter for person class (0)
    df = df[df['class'] == 0]
    
    # Process each detection
    for _, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Calculate center point and foot position
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        foot_x = center_x
        foot_y = y2  # Bottom of bounding box represents feet
        
        # Add to people list
        people.append({
            'position': (center_x, center_y),
            'foot_position': (foot_x, foot_y),
            'bbox': (x1, y1, x2, y2),
            'confidence': row['confidence']
        })
    
    return people

def is_person_on_court(person, courts):
    """
    Determine if a person is on a tennis court.
    Returns (court_index, area_type) where area_type is 'in_bounds', 'out_bounds', or 'off_court'
    Uses foot position for more accurate placement.
    """
    # Extract foot position
    if 'foot_position' in person:
        foot_x, foot_y = person['foot_position']
    else:
        # Fall back to center position if foot position isn't available
        foot_x, foot_y = person['position']
    
    foot_point = Point(foot_x, foot_y)
    
    # Check each court
    for court_idx, court in enumerate(courts):
        # Get the court polygon
        approx = court['approx']
        points = approx.reshape(-1, 2)
        court_polygon = Polygon(points)
        
        # Check if the foot position is inside the court
        if court_polygon.contains(foot_point):
            # Person is on this court - now determine if they're on blue (in-bounds) or green (out-bounds)
            x, y = foot_x, foot_y
            
            # Check if the foot is on blue area (in-bounds)
            blue_mask = court['blue_mask']
            if y < blue_mask.shape[0] and x < blue_mask.shape[1] and blue_mask[y, x] > 0:
                return court_idx, 'in_bounds'
            
            # Check if the foot is on green area (out-bounds)
            green_mask = court['green_mask']
            if y < green_mask.shape[0] and x < green_mask.shape[1] and green_mask[y, x] > 0:
                return court_idx, 'out_bounds'
            
            # If not specifically on blue or green, consider it in-bounds if the court has more blue than green
            if court['blue_ratio'] > court['green_ratio']:
                return court_idx, 'in_bounds'
            else:
                return court_idx, 'out_bounds'
    
    # If we reached here, the person is not on any court
    return -1, 'off_court'

def main():
    """Main function"""
    # Load image
    input_path = Config.Paths.input_path()
    image = cv2.imread(input_path)
    if image is None:
        OutputManager.log(f"Could not load image from {input_path}", "ERROR")
        return 1
    
    # Set up debug folder
    debug_folder = Config.Paths.debug_dir()
    os.makedirs(debug_folder, exist_ok=True)
    
    # Detect tennis courts
    OutputManager.log("Analyzing image for tennis courts...", "INFO")
    courts = detect_tennis_court(image, debug_folder)
    
    # Detect people
    OutputManager.log("Looking for people in the image...", "INFO")
    people = detect_people(image)
    
    # Determine if each person is on a court
    people_locations = []
    for person_idx, person in enumerate(people):
        court_idx, area_type = is_person_on_court(person, courts)
        people_locations.append((court_idx, area_type))
    
    # Display summary
    OutputManager.summarize_detections(courts, people, people_locations)
    
    # Make a copy for drawing results
    output_image = image.copy()
    
    # Draw court outlines
    if Config.Visual.DRAW_COURT_OUTLINE:
        for court_idx, court in enumerate(courts):
            cv2.drawContours(output_image, [court['approx']], 0, 
                            Config.Visual.COURT_OUTLINE_COLOR, 
                            Config.Visual.COURT_OUTLINE_THICKNESS)
    
    # Draw people and their locations
    for person_idx, person in enumerate(people):
        court_idx, area_type = people_locations[person_idx]
        
        # Draw bounding box and label
        x1, y1, x2, y2 = person['bbox']
        
        # Choose color based on location
        if court_idx >= 0:
            if area_type == 'in_bounds':
                color = Config.Visual.PERSON_IN_BOUNDS_COLOR
                label = f"Court {court_idx+1} IN"
            else:  # out_bounds
                color = Config.Visual.PERSON_OUT_BOUNDS_COLOR
                label = f"Court {court_idx+1} OUT"
        else:
            color = Config.Visual.PERSON_OFF_COURT_COLOR
            label = "OFF COURT"
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with black background for readability
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                   Config.Visual.FONT_SCALE, 
                                   Config.Visual.TEXT_THICKNESS)[0]
        cv2.rectangle(output_image, (x1, y1 - text_size[1] - 5), 
                     (x1 + text_size[0], y1), color, -1)
        cv2.putText(output_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   Config.Visual.FONT_SCALE, 
                   Config.Visual.TEXT_COLOR, 
                   Config.Visual.TEXT_THICKNESS)
    
    # Save output image
    output_path = Config.Paths.output_path()
    cv2.imwrite(output_path, output_image)
    OutputManager.log(f"Output image saved to {output_path}", "SUCCESS")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
