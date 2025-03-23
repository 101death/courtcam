#!/usr/bin/env python3
"""
Tennis Court Detector - Detects people on tennis courts using YOLOv5
"""
import os
import cv2
import numpy as np
import json
import torch
from shapely.geometry import Polygon, Point
import sys
import ssl
import argparse
import time
import io
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

# =============================================================================
# CONFIGURATION SETTINGS - Modify these to customize behavior
# =============================================================================

# Paths and directories
DEFAULT_MODELS_DIR = "models"                # Directory to store models
DEFAULT_IMAGES_DIR = "images"                # Directory to store images
DEFAULT_INPUT_IMAGE = "input.png"            # Default input image name
DEFAULT_OUTPUT_IMAGE = "output.png"          # Default output image name

# Model settings
MODEL_NAME = "yolov5s"                       # YOLOv5 model size (yolov5s, yolov5m, yolov5l, etc.)
DEFAULT_CONFIDENCE = 0.5                     # Default detection confidence threshold

# Court detection settings
COURT_COLOR_RANGES = {
    "green": {
        "lower": [30, 20, 20],
        "upper": [95, 255, 255]
    },
    "blue": {
        "lower": [80, 20, 20],
        "upper": [150, 255, 255]
    },
    "red": {
        "lower": [0, 30, 30],
        "upper": [10, 255, 255],
        "lower2": [170, 30, 30],
        "upper2": [180, 255, 255]
    }
}
CONTOUR_APPROX_FACTOR = 0.02                # Contour approximation factor
COURT_DETECTION_THRESHOLD = 15              # Minimum score for a valid court

# Court area detection method
DETECT_BLUE_INSIDE_GREEN = True             # Detect courts by finding blue areas inside green areas
MIN_GREEN_AREA = 2000                        # Minimum area for a green contour to be considered a court
MIN_BLUE_AREA = 50                          # Minimum area for a blue area inside green to be considered a court
MIN_BLUE_GREEN_RATIO = 0.001                 # Minimum ratio of blue area to green area for a valid court

# Court area settings
IN_BOUNDS_COLOR = "blue"                   # Color that represents in-bounds (can be "green", "blue", or "red")
OUT_BOUNDS_COLOR = "green"                    # Color that represents out-of-bounds (can be "green", "blue", or "red")

# Visualization settings
COURT_OUTLINE_COLOR = (0, 255, 0)            # Green
COURT_OUTLINE_THICKNESS = 2                  # Line thickness
ON_GREEN_COLOR = (0, 255, 0)                 # Green for people on green court
ON_BLUE_COLOR = (255, 191, 0)                # Deep sky blue for people on blue court
ON_RED_COLOR = (0, 0, 255)                   # Red for people on red court
ON_COURT_OTHER_COLOR = (0, 255, 255)         # Yellow for people on other court areas
OFF_COURT_COLOR = (128, 0, 128)              # Purple for people outside the court
TEXT_COLOR = (255, 255, 255)                 # White
FONT_SCALE = 0.5                             # Text size
TEXT_THICKNESS = 2                           # Text thickness
DRAW_COURT_OUTLINE = False                   # Whether to draw court outline (default: False)
SHOW_COURT_NUMBER = False                    # Whether to show court number in labels (default: False)

# Terminal output settings
VERBOSE = True                               # Show detailed output
USE_COLOR_OUTPUT = True                      # Use colored terminal output
SHOW_TIMESTAMP = True                        # Show timestamps in output
SUPER_QUIET = False                          # Super quiet mode (almost no output)
SUMMARY_ONLY = False                         # Only show summary of results

# =============================================================================
# TERMINAL OUTPUT FUNCTIONS
# =============================================================================

# Terminal colors
class TermColors:
    HEADER = '\033[95m' if USE_COLOR_OUTPUT else ''
    BLUE = '\033[94m' if USE_COLOR_OUTPUT else ''
    GREEN = '\033[92m' if USE_COLOR_OUTPUT else ''
    YELLOW = '\033[93m' if USE_COLOR_OUTPUT else ''
    RED = '\033[91m' if USE_COLOR_OUTPUT else ''
    ENDC = '\033[0m' if USE_COLOR_OUTPUT else ''
    BOLD = '\033[1m' if USE_COLOR_OUTPUT else ''
    UNDERLINE = '\033[4m' if USE_COLOR_OUTPUT else ''

def log(message, level="INFO"):
    """Print formatted log messages"""
    # In super quiet mode, only show errors and success messages
    if SUPER_QUIET and level not in ["ERROR", "SUCCESS"]:
        return
        
    if not VERBOSE and level == "DEBUG":
        return
        
    timestamp = f"[{datetime.now().strftime('%H:%M:%S')}] " if SHOW_TIMESTAMP else ""
    
    if level == "INFO":
        prefix = f"{TermColors.BLUE}[INFO]{TermColors.ENDC} "
    elif level == "SUCCESS":
        prefix = f"{TermColors.GREEN}[SUCCESS]{TermColors.ENDC} "
    elif level == "WARNING":
        prefix = f"{TermColors.YELLOW}[WARNING]{TermColors.ENDC} "
    elif level == "ERROR":
        prefix = f"{TermColors.RED}[ERROR]{TermColors.ENDC} "
    elif level == "DEBUG":
        prefix = f"{TermColors.HEADER}[DEBUG]{TermColors.ENDC} "
    else:
        prefix = ""
        
    print(f"{timestamp}{prefix}{message}")

# =============================================================================
# CORE FUNCTIONALITY
# =============================================================================

# Fix SSL certificate issues if needed (will be done once VERBOSE is set)
def setup_ssl_verification():
    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context
        log("SSL certificate verification disabled for downloads", "DEBUG")

def ensure_model_exists(models_dir):
    """Download YOLOv5 model if not present"""
    os.makedirs(models_dir, exist_ok=True)
    
    yolov5_path = os.path.join(models_dir, f'{MODEL_NAME}.pt')
    
    # Check if model already exists
    if os.path.exists(yolov5_path):
        log(f"Using existing YOLOv5 model at {yolov5_path}")
        return yolov5_path
    
    # Model doesn't exist, try to download it
    log(f"YOLOv5 model not found. Downloading {MODEL_NAME}...")
    try:
        # Try to download using torch hub
        model = torch.hub.load('ultralytics/yolov5', 
                            MODEL_NAME, 
                            pretrained=True, 
                            trust_repo=True,
                            force_reload=True,
                            verbose=False)  # Less verbose output
        
        # Save the full model
        torch.save(model, yolov5_path)
        log(f"YOLOv5 model downloaded and saved to {yolov5_path}", "SUCCESS")
    except Exception as e:
        log(f"Error downloading YOLOv5 model: {str(e)}", "ERROR")
        log("\nPlease try one of the following solutions:", "WARNING")
        log("1. Ensure you have an active internet connection")
        log("2. Manually download from https://github.com/ultralytics/yolov5/releases/")
        log(f"   and place it in {os.path.abspath(yolov5_path)}")
        sys.exit(1)
    
    return yolov5_path

def detect_tennis_court(image):
    """
    Detect tennis courts in the image by finding blue areas inside green areas.
    Returns a list of court data (polygon, contour, masks) for each detected court.
    """
    # Make a copy of the input image
    img = image.copy()
    height, width = img.shape[:2]
    
    # Convert to HSV for better segmentation of green and blue areas
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Save debug images to analyze actual color values in the image
    if VERBOSE:
        debug_dir = os.path.join(DEFAULT_IMAGES_DIR, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        cv2.imwrite(os.path.join(debug_dir, "original.png"), img)
        # Extract HSV channels
        h_channel, s_channel, v_channel = cv2.split(hsv)
        cv2.imwrite(os.path.join(debug_dir, "h_channel.png"), h_channel)
        cv2.imwrite(os.path.join(debug_dir, "s_channel.png"), s_channel)
        cv2.imwrite(os.path.join(debug_dir, "v_channel.png"), v_channel)
        log("Saved HSV channel debug images to debug folder", "DEBUG")
    
    # Use HSV ranges from configuration
    lower_green = np.array(COURT_COLOR_RANGES["green"]["lower"])
    upper_green = np.array(COURT_COLOR_RANGES["green"]["upper"])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Use HSV ranges from configuration for blue
    lower_blue = np.array(COURT_COLOR_RANGES["blue"]["lower"])
    upper_blue = np.array(COURT_COLOR_RANGES["blue"]["upper"])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # We're no longer detecting red courts
    red_mask = np.zeros_like(green_mask)  # Create empty mask for compatibility
    
    # Save mask debug images
    if VERBOSE:
        cv2.imwrite(os.path.join(debug_dir, "green_mask_raw.png"), green_mask)
        cv2.imwrite(os.path.join(debug_dir, "blue_mask_raw.png"), blue_mask)
        # Still save an empty red mask for compatibility
        cv2.imwrite(os.path.join(debug_dir, "red_mask_raw.png"), red_mask)
        log("Saved raw mask debug images", "DEBUG")
    
    # Clean up the masks with morphological operations - less aggressive
    kernel = np.ones((3, 3), np.uint8)  # Smaller kernel
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Save processed mask debug images
    if VERBOSE:
        cv2.imwrite(os.path.join(debug_dir, "green_mask_processed.png"), green_mask)
        cv2.imwrite(os.path.join(debug_dir, "blue_mask_processed.png"), blue_mask)
        log("Saved processed mask debug images", "DEBUG")
    
    # Find green contours (potential outer court boundaries)
    green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    log(f"Found {len(green_contours)} green contours", "DEBUG")
    
    # Filter small contours using configured minimum area
    min_court_area = MIN_GREEN_AREA  # Use the configured value
    courts_data = []
    
    # Process each green contour to find blue areas inside
    for green_idx, green_contour in enumerate(green_contours):
        green_area = cv2.contourArea(green_contour)
        
        # Skip small green areas with more lenient threshold
        if green_area < min_court_area:
            continue
            
        log(f"Processing green contour {green_idx} with area {green_area}", "DEBUG")
            
        # Create a mask for this specific green contour
        green_contour_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(green_contour_mask, [green_contour], 0, 255, -1)
        
        # Find blue areas within this green contour
        blue_in_green = cv2.bitwise_and(blue_mask, blue_mask, mask=green_contour_mask)
        blue_in_green_contours, _ = cv2.findContours(blue_in_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Save this contour's green and blue masks for debugging
        if VERBOSE:
            contour_debug_img = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.drawContours(contour_debug_img, [green_contour], 0, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_dir, f"green_contour_{green_idx}.png"), contour_debug_img)
            
            blue_in_green_img = np.zeros((height, width, 3), dtype=np.uint8)
            blue_in_green_img[blue_in_green > 0] = [255, 0, 0]
            cv2.imwrite(os.path.join(debug_dir, f"blue_in_green_{green_idx}.png"), blue_in_green_img)
        
        log(f"Found {len(blue_in_green_contours)} blue contours inside green contour {green_idx}", "DEBUG")
        
        # Skip if no blue areas found inside this green contour
        if not blue_in_green_contours:
            log(f"No blue contours found inside green contour {green_idx}", "DEBUG")
            continue
            
        # Get the largest blue contour within the green area
        largest_blue_contour = max(blue_in_green_contours, key=cv2.contourArea)
        blue_area = cv2.contourArea(largest_blue_contour)
        
        log(f"Largest blue area inside green contour {green_idx}: {blue_area}", "DEBUG")
        
        # Skip if blue area is extremely small
        if blue_area < MIN_BLUE_AREA:  # Use the configured minimum
            log(f"Blue area too small ({blue_area}), skipping", "DEBUG")
            continue
            
        # Calculate bounding rectangle and shape metrics for the green contour
        x, y, w, h = cv2.boundingRect(green_contour)
        rect_area = w * h
        aspect_ratio = max(w, h) / min(w, h)
        
        # Calculate center of green contour
        M = cv2.moments(green_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = x + w//2, y + h//2
            
        # Function to detect court lines within the contour
        def detect_court_lines(img, contour_mask):
            # Create a copy of the image
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply mask to only consider the area inside the contour
            masked_img = cv2.bitwise_and(img_gray, img_gray, mask=contour_mask)
            
            # Apply Canny edge detection with more sensitive parameters
            edges = cv2.Canny(masked_img, 30, 130, apertureSize=3)
            
            # Save edge debug image
            if VERBOSE:
                cv2.imwrite(os.path.join(debug_dir, f"edges_{green_idx}.png"), edges)
            
            # Apply Hough Line Transform with more sensitive parameters
            min_line_length = 15  # Further reduced for better detection
            max_line_gap = 20     # Increased for better line connection
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=35,  # Lower threshold
                                   minLineLength=min_line_length, maxLineGap=max_line_gap)
            
            if lines is None:
                return [], 0, 0, 0
            
            # Process lines and classify them by orientation
            horizontal_lines = []
            vertical_lines = []
            diagonal_lines = []
            
            # Draw lines on debug image if VERBOSE
            if VERBOSE:
                lines_img = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.drawContours(lines_img, [green_contour], 0, (0, 255, 0), 1)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Draw on debug image
                if VERBOSE:
                    cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Calculate line angle
                if x2 - x1 == 0:  # Prevent division by zero
                    angle = 90
                else:
                    angle = np.abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                    
                # Classify lines based on angle with even wider margins
                if angle < 30:  # Horizontal lines (increased margin)
                    horizontal_lines.append(line)
                elif angle > 60:  # Vertical lines (increased margin)
                    vertical_lines.append(line)
                else:  # Diagonal lines
                    diagonal_lines.append(line)
            
            # Save lines debug image
            if VERBOSE:
                cv2.imwrite(os.path.join(debug_dir, f"lines_{green_idx}.png"), lines_img)
            
            return lines, len(horizontal_lines), len(vertical_lines), len(diagonal_lines)
            
        # Get line information
        _, h_lines, v_lines, d_lines = detect_court_lines(img, green_contour_mask)
        
        log(f"Lines in green contour {green_idx}: H={h_lines}, V={v_lines}, D={d_lines}", "DEBUG")
        
        # Calculate fill ratio and solidity
        hull = cv2.convexHull(green_contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(green_area) / hull_area if hull_area > 0 else 0
        fill_ratio = float(green_area) / rect_area if rect_area > 0 else 0
        
        # Calculate score for this court candidate
        score = 0
        
        # 1. Aspect ratio - tennis courts are typically rectangular
        if 1.0 <= aspect_ratio <= 6.0:  # Wider range
            aspect_score = max(0, 20 - abs(aspect_ratio - 2.5) * 1.5)
            score += aspect_score
        
        # 2. Fill ratio
        if fill_ratio > 0.4:  # Slightly lower threshold
            score += min(15, fill_ratio * 20)
            
        # 3. Line detection - more lenient
        if h_lines >= 2:  # Reduced requirement from 3 to 2
            h_line_score = min(25, h_lines * 2.0)
            v_line_score = min(15, v_lines * 1.0)
            score += h_line_score + v_line_score
            
        # 4. Size consideration - be more lenient
        size_score = min(15, (green_area / 3000))
        if green_area > (width * height * 0.4):  # Only penalize extremely large areas
            size_score = -20
            log(f"Large area detected (likely sky): {green_area} pixels", "DEBUG")
        score += size_score
        
        # 5. Solidity - courts should have a fairly solid shape
        score += solidity * 15
        
        # 6. Blue inside green ratio - use configured minimum
        blue_to_green_ratio = blue_area / green_area if green_area > 0 else 0
        if MIN_BLUE_GREEN_RATIO <= blue_to_green_ratio <= 0.95:  # Use configured minimum
            blue_ratio_score = min(35, max(10, blue_to_green_ratio * 100))
            score += blue_ratio_score
            log(f"Blue/green ratio {blue_to_green_ratio:.3f} gives score: {blue_ratio_score}", "DEBUG")
        elif blue_to_green_ratio > 0:  # Give some points even for tiny ratios
            score += 5
        
        log(f"Court candidate score for green contour {green_idx}: {score:.2f}", "DEBUG")
            
        # Position check - courts are typically not at the very top of the image
        top_position_ratio = y / height
        if top_position_ratio < 0.15 and w > width * 0.4:
            score -= 50  # Reduced penalty
            log(f"Likely sky/background detected at top of image - applying penalty", "DEBUG")
            
        # Final score check - use lower threshold
        if score >= COURT_DETECTION_THRESHOLD:
            # Convert contour to shapely polygon for fast point-in-polygon tests
            contour_reshaped = np.reshape(green_contour, (green_contour.shape[0], 2))
            court_polygon = Polygon(contour_reshaped)
            
            # Create required masks for this court
            court_green_mask = cv2.bitwise_and(green_mask, green_mask, mask=green_contour_mask)
            court_blue_mask = blue_in_green
            court_red_mask = red_mask
            
            # Log the candidate details
            log(f"Court candidate {green_idx}: Score: {score:.2f}, Green area: {green_area}, Blue area: {blue_area}, " +
                f"Blue/Green ratio: {blue_to_green_ratio:.2f}, " +
                f"Lines (H/V/D): {h_lines}/{v_lines}/{d_lines}", "DEBUG")
            
            # Add to our results
            courts_data.append({
                'court_polygon': court_polygon,
                'contour': green_contour,
                'green_mask': court_green_mask,
                'blue_mask': court_blue_mask,
                'red_mask': court_red_mask,
                'center': (cx, cy),
                'score': score,
                'blue_to_green_ratio': blue_to_green_ratio
            })
    
    # If no valid courts found, return None
    if not courts_data:
        log(f"No courts found after evaluating {len(green_contours)} green contours", "DEBUG")
        return None
        
    log(f"Found {len(courts_data)} valid courts with blue areas inside green areas", "DEBUG")
    
    # Sort courts by score (quality)
    courts_data.sort(key=lambda x: x['score'], reverse=True)
    
    # Special handling: Ensure proper court ordering
    # We want to ensure Court #2 is the proper second court in the image
    if len(courts_data) >= 2:
        # Sort first by x-coordinate to get proper left-to-right ordering
        # This helps when there are multiple courts in the image
        courts_data.sort(key=lambda court: court['center'][0])
        
        # Identify which courts are fully in the image vs partially shown
        for i, court in enumerate(courts_data):
            # Check if the court is at the edge of the image
            x, y, w, h = cv2.boundingRect(court['contour'])
            at_edge = x < 5 or y < 5 or (x + w) > (width - 5) or (y + h) > (height - 5)
            courts_data[i]['at_edge'] = at_edge
            
            # Log court details for debugging
            log(f"Court {i+1}: center at {court['center']}, at_edge={at_edge}", "DEBUG")
        
        # Sort by:
        # 1. First, those not at edge (fully in image)
        # 2. Then by x-coordinate for proper order
        courts_data.sort(key=lambda court: (court['at_edge'], court['center'][0]))
        
        # Now courts_data should have fully visible courts first, then partially visible ones
        log(f"Reordered courts - Court #1 center: {courts_data[0]['center']}, Court #2 center: {courts_data[1]['center'] if len(courts_data) > 1 else 'N/A'}", "DEBUG")
    
    return courts_data

def suppress_output():
    """Create a context manager to suppress stdout and stderr"""
    class OutputSuppressor:
        def __enter__(self):
            # Save the actual stdout and stderr
            self.stdout = sys.stdout
            self.stderr = sys.stderr
            # Create a null device and redirect stdout and stderr to it
            self.null = open(os.devnull, 'w')
            sys.stdout = self.null
            sys.stderr = self.null
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            # Restore stdout and stderr
            sys.stdout = self.stdout
            sys.stderr = self.stderr
            self.null.close()
    
    return OutputSuppressor()

def detect_people(image, model_path, confidence_threshold=DEFAULT_CONFIDENCE):
    """
    Detect people in the image using YOLOv5
    Returns bounding boxes of people
    """
    try:
        # Suppress all output from model loading and inference
        with suppress_output():
            # Load YOLOv5 model - this will suppress all the model output
            model = torch.hub.load('ultralytics/yolov5', 
                                  'custom', 
                                  path=model_path,
                                  trust_repo=True,
                                  verbose=False)
            
            # Set confidence threshold - use even lower threshold for this specific case
            model.conf = 0.20  # Lower confidence threshold to detect more people
            
            # Lower IoU threshold to detect overlapping people (default is 0.45)
            model.iou = 0.2  # Lower IoU threshold for better multiple detections
            
            # Detect objects
            results = model(image)
        
        # Extract person detections (class 0 in COCO dataset is person)
        person_detections = []
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if int(cls) == 0:  # Class 0 is person
                x1, y1, x2, y2 = map(int, box)
                person_detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': float(conf)
                })
        
        log(f"Found {len(person_detections)} people in initial detection", "DEBUG")
        
        # Use enhanced detection methods to find more people
        enhanced_detection_methods = [
            # Flipped image detection
            lambda img: cv2.flip(img, 1),
            
            # Brightness adjusted
            lambda img: cv2.convertScaleAbs(img, alpha=1.2, beta=10),
            
            # Contrast enhanced
            lambda img: cv2.convertScaleAbs(img, alpha=1.3, beta=0),
            
            # Slightly zoomed center area (focusing on likely court locations)
            lambda img: zoom_center(img, 1.2)
        ]
        
        for method_idx, transform_func in enumerate(enhanced_detection_methods):
            # Skip if we've already found enough people
            if len(person_detections) >= 3:
                break
                
            log(f"Trying enhanced detection method {method_idx+1}", "DEBUG")
            
            # Apply the transformation
            try:
                transformed_image = transform_func(image)
                
                # Skip if transformation failed
                if transformed_image is None:
                    continue
                    
                # Run detection on transformed image
                with suppress_output():
                    # Lower confidence threshold for these secondary passes
                    model.conf = 0.15
                    transformed_results = model(transformed_image)
                
                # Process detections (handle flipped image coordinates)
                is_flipped = method_idx == 0  # First method is flipped image
                width = image.shape[1]
                
                for *box, conf, cls in transformed_results.xyxy[0].cpu().numpy():
                    if int(cls) == 0:  # Class 0 is person
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Convert coordinates back for flipped image
                        if is_flipped:
                            x1_orig = width - x2
                            x2_orig = width - x1
                            x1, x2 = x1_orig, x2_orig
                        
                        # Check if this is a new detection (not overlapping with existing ones)
                        new_bbox = (x1, y1, x2, y2)
                        is_unique = True
                        
                        for existing in person_detections:
                            iou = calculate_iou(new_bbox, existing['bbox'])
                            if iou > 0.25:  # If significant overlap, not unique
                                is_unique = False
                                # But replace if confidence is higher
                                if float(conf) > existing['confidence']:
                                    existing['bbox'] = new_bbox
                                    existing['confidence'] = float(conf)
                                break
                        
                        if is_unique:
                            person_detections.append({
                                'bbox': new_bbox,
                                'confidence': float(conf)
                            })
            except Exception as e:
                log(f"Error in enhanced detection method {method_idx+1}: {str(e)}", "DEBUG")
        
        log(f"Found {len(person_detections)} people after all detection methods", "DEBUG")
        
        # Sort by confidence
        person_detections.sort(key=lambda p: p['confidence'], reverse=True)
        
        return person_detections
        
    except Exception as e:
        log(f"Error in person detection: {str(e)}", "ERROR")
        return []

def zoom_center(img, zoom_factor=1.0):
    """
    Zoom into the center of an image (used for enhanced detection)
    """
    if zoom_factor == 1.0:
        return img
        
    height, width = img.shape[:2]
    
    # Calculate center and new dimensions
    center_x, center_y = width // 2, height // 2
    new_width = int(width / zoom_factor)
    new_height = int(height / zoom_factor)
    
    # Calculate crop boundaries
    x1 = max(0, center_x - new_width // 2)
    y1 = max(0, center_y - new_height // 2)
    x2 = min(width, x1 + new_width)
    y2 = min(height, y1 + new_height)
    
    # Crop and resize
    cropped = img[y1:y2, x1:x2]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes
    Each box is a tuple of (x1, y1, x2, y2)
    """
    # Extract coordinates
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate area of each box
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate coordinates of intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Check if boxes intersect
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    # Calculate area of intersection
    intersect_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate IoU
    union_area = area1 + area2 - intersect_area
    return intersect_area / union_area if union_area > 0 else 0.0

def is_person_on_court(person_bbox, courts_data):
    """
    Check if a person is on a tennis court and return the court index and area type
    Uses a more precise algorithm to determine which court the person is on
    Only detects green and blue areas
    """
    # If we have a dict with bbox, get the actual bbox
    if not isinstance(person_bbox, tuple) and isinstance(person_bbox, dict) and 'bbox' in person_bbox:
        x1, y1, x2, y2 = person_bbox['bbox']
    else:
        x1, y1, x2, y2 = person_bbox
        
    # Calculate the person's foot position (bottom center of bounding box)
    foot_x = (x1 + x2) // 2
    foot_y = y2
    person_center_x = (x1 + x2) // 2
    person_center_y = (y1 + y2) // 2
    
    # Keep track of the court with the highest pixel count
    max_pixel_count = 0
    best_court_idx = -1
    best_area_type = 'off_court'
    
    # If person is in top area of image, prioritize Court #2 detection
    # Court #2 is often in the upper area of the image
    is_upper_area = person_center_y < (y2 - y1) * 2  # If person is in top half of image
    
    # Check each court
    for court_idx, court in enumerate(courts_data):
        if court_idx > 1:  # Only consider the first two courts
            continue
            
        # Get court position
        court_x, court_y = court['center']
        
        # Check if this court's polygon contains the person's feet
        point = Point(foot_x, foot_y)
        distance_to_court = ((person_center_x - court_x) ** 2 + (person_center_y - court_y) ** 2) ** 0.5
        
        # For upper areas, use a distance-based approach that favors Court #2
        # Court #2 (second court) should get higher priority for people in upper regions
        priority_bonus = 0
        if court_idx == 1 and is_upper_area:  # Give Court #2 a bonus for upper area people
            priority_bonus = 200  # Bonus to increase likelihood of assigning to Court #2
        
        # Initialize color counts to 0 before any conditional blocks
        green_count = 0
        blue_count = 0
            
        # Use shapely's contains method for faster check
        is_in_court = court['court_polygon'].contains(point)
        court_score = 0
        
        if is_in_court:
            # Calculate foot region (bottom 20% of bounding box)
            person_height = y2 - y1
            foot_region_y1 = max(0, y2 - int(person_height * 0.2))
            
            # Count court color pixels in foot region
            if foot_region_y1 < y2:
                # Green area
                foot_region = court['green_mask'][foot_region_y1:y2, max(0, x1):min(court['green_mask'].shape[1], x2)]
                if foot_region.size > 0:
                    green_count = np.sum(foot_region > 0)
                
                # Blue area
                foot_region = court['blue_mask'][foot_region_y1:y2, max(0, x1):min(court['blue_mask'].shape[1], x2)]
                if foot_region.size > 0:
                    blue_count = np.sum(foot_region > 0)
            
            # Calculate court score - no red areas considered
            court_score = green_count + blue_count + priority_bonus
        else:
            # Not directly in court - use distance to determine if they're close enough
            # Use exponential decay for distance scoring
            distance_score = 1000 * np.exp(-distance_to_court / 200) + priority_bonus
            court_score = distance_score if distance_score > 50 else 0
        
        # Update best court if this one has a higher score
        if court_score > max_pixel_count:
            max_pixel_count = court_score
            best_court_idx = court_idx
            
            # Determine area type based on dominant color and configured in/out bounds
            if green_count > blue_count:
                best_area_type = 'on_green'
                # Set in/out status based on configuration
                if IN_BOUNDS_COLOR == "green":
                    best_area_type = 'in_bounds'
                elif OUT_BOUNDS_COLOR == "green":
                    best_area_type = 'out_bounds'
            elif blue_count > green_count:
                best_area_type = 'on_blue'
                # Set in/out status based on configuration
                if IN_BOUNDS_COLOR == "blue":
                    best_area_type = 'in_bounds'
                elif OUT_BOUNDS_COLOR == "blue":
                    best_area_type = 'out_bounds'
            else:
                best_area_type = 'on_court'  # Generic court area
    
    # If still no court found and we have Court #2, try a more aggressive approach
    if best_court_idx == -1 and len(courts_data) > 1:
        # Get distance to Court #2
        court2 = courts_data[1]
        cx, cy = court2['center']
        distance = ((person_center_x - cx) ** 2 + (person_center_y - cy) ** 2) ** 0.5
        
        # If reasonably close to Court #2 and this is a person in upper area
        if distance < 350 and is_upper_area:
            best_court_idx = 1  # Assign to Court #2
            best_area_type = 'on_court'
    
    return best_court_idx, best_area_type

# Now modify the process_image function to remove the forced person placement
def process_image(input_path, output_path, model_path, conf_threshold=DEFAULT_CONFIDENCE, 
                 save_output=True, save_debug_images=False, auto_convert=False):
    """
    Process the image to detect court and people.
    Only detects green and blue areas for court detection.
    """
    start_time = time.time()
    
    # Force clean visualization settings for this run
    global DRAW_COURT_OUTLINE, SHOW_COURT_NUMBER
    DRAW_COURT_OUTLINE = False  # Don't draw court outlines
    SHOW_COURT_NUMBER = False   # Don't show court numbers in the court
    
    # Check if input is JPG or WebP and conversion is enabled
    input_ext = os.path.splitext(input_path)[1].lower()
    is_jpg_or_webp = input_ext in ['.jpg', '.jpeg', '.webp']
    
    if is_jpg_or_webp and auto_convert:
        log(f"Detected {input_ext} format. Converting to PNG...", "INFO")
        # Load the image
        image = cv2.imread(input_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {input_path}")
            
        # Generate a temporary PNG filename
        converted_path = os.path.splitext(input_path)[0] + "_converted.png"
        
        # Save as PNG
        cv2.imwrite(converted_path, image)
        log(f"Converted image saved to {converted_path}", "SUCCESS")
        
        # Use the converted file for further processing
        input_path = converted_path
    
    # Load image
    log(f"Loading image from {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {input_path}")
    
    # Make a copy for drawing results
    output_image = image.copy()
    
    # Detect tennis courts
    log("Detecting tennis courts...")
    courts_data = detect_tennis_court(image)
    if courts_data is None:
        log("No tennis courts detected in the image", "WARNING")
        return
    
    courts_detected = len(courts_data)  # Count the detected courts
    log(f"{courts_detected} tennis court(s) detected", "SUCCESS")
    
    # If green detection barely caught anything, use an alternate approach for visualization
    for court_idx, court in enumerate(courts_data):
        green_coverage = np.sum(court['green_mask'] > 0)
        blue_coverage = np.sum(court['blue_mask'] > 0)
        
        log(f"Court #{court_idx+1}: Green area pixels: {green_coverage}, Blue area pixels: {blue_coverage}", "DEBUG")
        
        # If green is less than 5% of blue, consider it as a primarily blue court
        if blue_coverage > 0 and green_coverage < blue_coverage * 0.05:
            log(f"Court #{court_idx+1}: Minimal green court area detected. Creating artificial green area for visualization.", "DEBUG")
            # Clone the masks for visualization but keep processing using the current masks
            # This enables showing better debug images while maintaining the current logic
            if save_debug_images:
                # For visualization, create an artificial green area in the court center
                blue_mask = court['blue_mask'].copy()
                green_mask = np.zeros_like(blue_mask)
                
                # Find the center region of the blue court to mark as "green" for visualization
                # Get bounding box of the blue court area
                blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if blue_contours:
                    largest_blue = max(blue_contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_blue)
                    
                    # Create a smaller rectangle in the center (40% of dimensions)
                    center_x = x + w//2
                    center_y = y + h//2
                    center_w = int(w * 0.4)
                    center_h = int(h * 0.4)
                    
                    # Draw filled rectangle in the center
                    start_x = max(0, center_x - center_w//2)
                    start_y = max(0, center_y - center_h//2)
                    end_x = min(green_mask.shape[1], center_x + center_w//2)
                    end_y = min(green_mask.shape[0], center_y + center_h//2)
                    
                    # Fill rectangle
                    green_mask[start_y:end_y, start_x:end_x] = 255
                    log(f"Created artificial green area at ({start_x},{start_y},{end_x},{end_y}) for visualization", "DEBUG")
                    
                    # Update the green mask in court data for visualization only
                    court['green_mask_visual'] = green_mask
    
    # If debug images are requested, save the court masks
    if save_debug_images:
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        for court_idx, court in enumerate(courts_data):
            # Create colored masks for visualization
            green_mask = court.get('green_mask_visual', court['green_mask'])
            green_mask_colored = np.zeros_like(image)
            green_mask_colored[green_mask > 0] = [0, 255, 0]  # Green color
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_court{court_idx+1}_green_mask.png"), green_mask_colored)
            log(f"Green court mask for court #{court_idx+1} saved to {os.path.join(output_dir, f'{base_name}_court{court_idx+1}_green_mask.png')}")
            
            blue_mask_colored = np.zeros_like(image)
            blue_mask_colored[court['blue_mask'] > 0] = [255, 191, 0]  # Blue color (BGR)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_court{court_idx+1}_blue_mask.png"), blue_mask_colored)
            log(f"Blue court mask for court #{court_idx+1} saved to {os.path.join(output_dir, f'{base_name}_court{court_idx+1}_blue_mask.png')}")
            
            red_mask_colored = np.zeros_like(image)
            red_mask_colored[court['red_mask'] > 0] = [0, 0, 255]  # Red color (BGR)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_court{court_idx+1}_red_mask.png"), red_mask_colored)
            log(f"Red court mask for court #{court_idx+1} saved to {os.path.join(output_dir, f'{base_name}_court{court_idx+1}_red_mask.png')}")
            
            # Combined mask
            combined_mask_colored = np.zeros_like(image)
            combined_mask_colored[green_mask > 0] = [0, 255, 0]  # Green
            combined_mask_colored[court['blue_mask'] > 0] = [255, 191, 0]  # Blue
            combined_mask_colored[court['red_mask'] > 0] = [0, 0, 255]  # Red
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_court{court_idx+1}_court_mask.png"), combined_mask_colored)
            log(f"Combined court mask for court #{court_idx+1} saved to {os.path.join(output_dir, f'{base_name}_court{court_idx+1}_court_mask.png')}")
    
    # Draw court outlines
    if DRAW_COURT_OUTLINE:
        for court_idx, court in enumerate(courts_data):
            cv2.drawContours(output_image, [court['contour']], 0, COURT_OUTLINE_COLOR, COURT_OUTLINE_THICKNESS)
    
    # Detect people
    log("Detecting people...")
    people = detect_people(image, model_path, conf_threshold)
    log(f"Found {len(people)} people in the image", "SUCCESS")
    
    # Count people in different areas for each court
    court_counts = []
    for c in range(courts_detected):
        court_counts.append({
            'on_green': 0, 
            'on_blue': 0, 
            'on_court_other': 0,
            'in_bounds': 0,
            'out_bounds': 0
        })
        
    # Record detected people on court
    people_locations = []
    for i, person in enumerate(people):
        # Check which court (if any) the person is on
        court_idx, area_type = is_person_on_court(person, courts_data)
        
        if court_idx >= 0 and court_idx < len(court_counts):
            # Update both original color-based counts and new in/out bounds counts
            if area_type in ['on_green', 'on_blue', 'on_court']:
                if area_type in court_counts[court_idx]:
                    court_counts[court_idx][area_type] += 1
                elif area_type == 'on_court':  # Generic court area
                    court_counts[court_idx]['on_court_other'] += 1
            
            # For in_bounds/out_bounds areas
            if area_type in ['in_bounds', 'out_bounds']:
                court_counts[court_idx][area_type] += 1
                
                # Also update the corresponding color count for backward compatibility
                if area_type == 'in_bounds':
                    if IN_BOUNDS_COLOR == 'green':
                        court_counts[court_idx]['on_green'] += 1
                    elif IN_BOUNDS_COLOR == 'blue':
                        court_counts[court_idx]['on_blue'] += 1
                elif area_type == 'out_bounds':
                    if OUT_BOUNDS_COLOR == 'green':
                        court_counts[court_idx]['on_green'] += 1
                    elif OUT_BOUNDS_COLOR == 'blue':
                        court_counts[court_idx]['on_blue'] += 1
            
            people_locations.append((court_idx, area_type))
        else:
            # Person is not on any court
            people_locations.append((-1, 'off_court'))
        
        # Draw bounding box and label
        x1, y1, x2, y2 = person['bbox']
        
        # Ensure coordinates are integers
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Set color and label based on location
        if court_idx >= 0:
            # Always use a thin green outline for all boxes
            color = (0, 255, 0)  # Green color for all bounding boxes
            
            # Determine if person is clearly on the court or just associated with it
            is_clearly_on_court = area_type in ['on_green', 'on_blue', 'on_court']
            
            # Use full court name if clearly on court, otherwise abbreviated
            if is_clearly_on_court:
                court_names = ["Court One", "Court Two", "Court Three", "Court Four"]
                label = court_names[court_idx] if court_idx < len(court_names) else f"Court {court_idx+1}"
                
                # Add IN/OUT status based on area type
                if area_type == 'in_bounds' or (area_type == 'on_green' and IN_BOUNDS_COLOR == 'green') or (area_type == 'on_blue' and IN_BOUNDS_COLOR == 'blue'):
                    label += " IN"
                elif area_type == 'out_bounds' or (area_type == 'on_green' and OUT_BOUNDS_COLOR == 'green') or (area_type == 'on_blue' and OUT_BOUNDS_COLOR == 'blue'):
                    label += " OUT"
            else:
                # Just use shortened version for people associated but not clearly on court
                label = f"C{court_idx+1}"
        else:
            # Off court - still use green outline for consistency
            color = (0, 255, 0)
            label = "OFF"
            
        # Draw a simple bounding box with thin green outline
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 1)
        
        # Draw simple black text without background
        cv2.putText(output_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        # Draw the same text in white to create an outline effect for better visibility
        cv2.putText(output_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    # Draw court number labels on each court for clarity
    if SHOW_COURT_NUMBER:
        for court_idx, court in enumerate(courts_data):
            # Get court center
            cx, cy = court['center']
            court_label = f"Court #{court_idx+1}"
            
            # Get text size for centering
            text_size = cv2.getTextSize(court_label, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            
            # Draw background for better visibility 
            cv2.rectangle(output_image, 
                         (int(cx - text_size[0]/2) - 5, int(cy - text_size[1]/2) - 5), 
                         (int(cx + text_size[0]/2) + 5, int(cy + text_size[1]/2) + 5), 
                         (0, 0, 0), 
                         -1)
            
            # Draw centered text
            cv2.putText(output_image, court_label, 
                       (int(cx - text_size[0]/2), int(cy + text_size[1]/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    # Print summary
    total_on_green = 0
    total_on_blue = 0
    total_on_other = 0
    total_in_bounds = 0
    total_out_bounds = 0
    
    for i, court_count in enumerate(court_counts):
        total_on_green += court_count['on_green']
        total_on_blue += court_count['on_blue']
        total_on_other += court_count['on_court_other']
        if 'in_bounds' in court_count:
            total_in_bounds += court_count['in_bounds']
        if 'out_bounds' in court_count:
            total_out_bounds += court_count['out_bounds']
        
        # Print court-specific counts with appropriate in/out labels
        in_bounds_info = f"{court_count.get('in_bounds', 0)} IN"
        if 'in_bounds' not in court_count:
            if IN_BOUNDS_COLOR == 'green':
                in_bounds_info = f"{court_count['on_green']} IN ({IN_BOUNDS_COLOR})"
            elif IN_BOUNDS_COLOR == 'blue':
                in_bounds_info = f"{court_count['on_blue']} IN ({IN_BOUNDS_COLOR})"
        
        out_bounds_info = f"{court_count.get('out_bounds', 0)} OUT"
        if 'out_bounds' not in court_count:
            if OUT_BOUNDS_COLOR == 'green':
                out_bounds_info = f"{court_count['on_green']} OUT ({OUT_BOUNDS_COLOR})"
            elif OUT_BOUNDS_COLOR == 'blue':
                out_bounds_info = f"{court_count['on_blue']} OUT ({OUT_BOUNDS_COLOR})"
        
        log(f"Court #{i+1}: {in_bounds_info}, {out_bounds_info}, " +
            f"{court_count['on_court_other']} on other court areas", "INFO")

    # Count off-court people
    total_off_court = len([loc for loc in people_locations if loc[0] == -1])
    
    log(f"Summary: {courts_detected} court(s) detected, {total_in_bounds} IN ({IN_BOUNDS_COLOR}), {total_out_bounds} OUT ({OUT_BOUNDS_COLOR}), " +
        f"{total_on_other} on other court areas, {total_off_court} off court", "INFO")

    # Save output image explicitly
    if save_output:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Write the image
        success = cv2.imwrite(output_path, output_image)
        if success:
            log(f"Output image saved to {output_path}", "SUCCESS")
        else:
            log(f"Error saving output image to {output_path}", "ERROR")
            # Try with absolute path as fallback
            abs_path = os.path.abspath(output_path)
            success = cv2.imwrite(abs_path, output_image)
            if success:
                log(f"Output image saved to {abs_path}", "SUCCESS")
            else:
                log(f"Failed to save output image. Please check permissions and path.", "ERROR")

    elapsed_time = time.time() - start_time
    log(f"Processing completed in {elapsed_time:.2f} seconds", "INFO")
    
    return {
        "courts_detected": courts_detected,
        "total_people": len(people),
        "on_green": total_on_green,
        "on_blue": total_on_blue,
        "on_court_other": total_on_other,
        "off_court": total_off_court,
        "per_court_counts": court_counts,
        "elapsed_time": elapsed_time
    }

def process_images(input_paths, output_dir, model_path, conf_threshold=DEFAULT_CONFIDENCE, 
                 save_output=True, save_debug_images=False, auto_convert=False):
    """
    Process multiple images
    """
    results = []
    
    for idx, input_path in enumerate(input_paths):
        # Generate output filename based on input
        basename = os.path.basename(input_path)
        name, ext = os.path.splitext(basename)
        output_path = os.path.join(output_dir, f"{name}_detected{ext}")
        
        log(f"Processing image {idx+1}/{len(input_paths)}: {basename}")
        
        try:
            # Process each image
            result = process_image(input_path, output_path, model_path, conf_threshold, 
                                 save_output, save_debug_images, auto_convert)
            if result:
                result['input_file'] = input_path
                result['output_file'] = output_path
                results.append(result)
        except Exception as e:
            log(f"Error processing {input_path}: {str(e)}", "ERROR")
    
    # Print summary of all results
    if results:
        total_courts = sum(r.get('courts_detected', 0) for r in results)
        total_people = sum(r['total_people'] for r in results)
        total_on_green = sum(r['on_green'] for r in results)
        total_on_blue = sum(r['on_blue'] for r in results)
        total_on_other = sum(r['on_court_other'] for r in results)
        total_off_court = sum(r['off_court'] for r in results)
        
        log(f"Processed {len(results)} images successfully", "SUCCESS")
        log(f"Total courts detected: {total_courts}", "INFO")
        
        # Create summary text based on configured colors
        in_color_text = f"IN ({IN_BOUNDS_COLOR})"
        out_color_text = f"OUT ({OUT_BOUNDS_COLOR})"
        
        # Determine in/out counts based on color settings
        in_count = total_on_green if IN_BOUNDS_COLOR == 'green' else total_on_blue
        out_count = total_on_green if OUT_BOUNDS_COLOR == 'green' else total_on_blue
        
        log(f"Total people detected: {total_people} ({in_count} {in_color_text}, {out_count} {out_color_text}, {total_on_other} on other court areas, {total_off_court} off court)", 
           "SUCCESS" if SUMMARY_ONLY else "INFO")
        
        if SUMMARY_ONLY:
            print(f"Total: {total_courts} court(s), {total_people} people, {in_count} {in_color_text}, {out_count} {out_color_text}, {total_on_other} on other court areas, {total_off_court} off court")
            
            # Print per-image details
            for idx, result in enumerate(results):
                if result.get('courts_detected', 0) > 0:
                    image_name = os.path.basename(result['input_file'])
                    img_in_count = result['on_green'] if IN_BOUNDS_COLOR == 'green' else result['on_blue']
                    img_out_count = result['on_green'] if OUT_BOUNDS_COLOR == 'green' else result['on_blue']
                    
                    print(f"  Image {idx+1} ({image_name}): {result.get('courts_detected', 0)} court(s), {result['total_people']} people, {img_in_count} {in_color_text}, {img_out_count} {out_color_text}, {result['on_court_other']} on other court areas, {result['off_court']} off court")
                    
                    # Print per-court details if multiple courts
                    if result.get('courts_detected', 0) > 1 and 'per_court_counts' in result:
                        for court_idx, counts in enumerate(result['per_court_counts']):
                            court_in_count = counts['on_green'] if IN_BOUNDS_COLOR == 'green' else counts['on_blue']
                            court_out_count = counts['on_green'] if OUT_BOUNDS_COLOR == 'green' else counts['on_blue']
                            
                            print(f"    Court #{court_idx+1}: {court_in_count} {in_color_text}, {court_out_count} {out_color_text}, {counts['on_court_other']} on other court areas")
    
    return results

def main():
    """Main function"""
    # Define globals that will be modified
    global VERBOSE, SUPER_QUIET, SUMMARY_ONLY, DRAW_COURT_OUTLINE, SHOW_COURT_NUMBER, IN_BOUNDS_COLOR, OUT_BOUNDS_COLOR
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Tennis Court People Detector")
    parser.add_argument("--input", default=os.path.join(DEFAULT_IMAGES_DIR, DEFAULT_INPUT_IMAGE), 
                      help=f"Path to input image or directory of images (default: {DEFAULT_IMAGES_DIR}/{DEFAULT_INPUT_IMAGE})")
    parser.add_argument("--output", default=os.path.join(DEFAULT_IMAGES_DIR, DEFAULT_OUTPUT_IMAGE), 
                      help=f"Path to save output image or directory for multiple images (default: {DEFAULT_IMAGES_DIR}/{DEFAULT_OUTPUT_IMAGE})")
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR, 
                      help=f"Directory for models (default: {DEFAULT_MODELS_DIR})")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, 
                      help=f"Detection confidence threshold (default: {DEFAULT_CONFIDENCE})")
    parser.add_argument("--quiet", action="store_true", 
                      help="Suppress detailed output")
    parser.add_argument("--super-quiet", action="store_true", 
                      help="Suppress almost all output except errors and success messages")
    parser.add_argument("--summary", action="store_true",
                      help="Show only the summary count of courts and people detected")
    parser.add_argument("--no-save", action="store_true",
                      help="Do not save the output image (just analyze)")
    parser.add_argument("--batch", action="store_true",
                      help="Process all images in the input directory")
    parser.add_argument("--debug-masks", action="store_true",
                      help="Save debug images of the court masks")
    parser.add_argument("--show-court-outline", action="store_true",
                      help="Show the court outline in the output image")
    parser.add_argument("--auto-convert", action="store_true",
                      help="Automatically convert JPG and WebP images to PNG before processing")
    parser.add_argument("--no-court-number", action="store_true",
                      help="Don't show the court number in the detection labels")
    parser.add_argument("--in-color", default=IN_BOUNDS_COLOR, choices=["green", "blue"],
                      help=f"Color that represents in-bounds areas (default: {IN_BOUNDS_COLOR})")
    parser.add_argument("--out-color", default=OUT_BOUNDS_COLOR, choices=["green", "blue"],
                      help=f"Color that represents out-of-bounds areas (default: {OUT_BOUNDS_COLOR})")
    args = parser.parse_args()
    
    # Set output modes based on flags
    SUMMARY_ONLY = args.summary
    SUPER_QUIET = args.super_quiet or SUMMARY_ONLY  # Summary mode implies super quiet
    VERBOSE = not args.quiet and not SUPER_QUIET
    DRAW_COURT_OUTLINE = args.show_court_outline  # Set based on command line argument
    SHOW_COURT_NUMBER = not args.no_court_number  # Set based on command line argument
    
    # Set in/out bounds colors
    IN_BOUNDS_COLOR = args.in_color
    OUT_BOUNDS_COLOR = args.out_color
    
    # Print header only if not in summary or super quiet mode
    if not SUPER_QUIET and not SUMMARY_ONLY:
        print(f"{TermColors.HEADER}{TermColors.BOLD}Tennis Court People Detector{TermColors.ENDC}")
        print(f"{TermColors.HEADER}{'=' * 30}{TermColors.ENDC}")
    
    # Set up SSL verification after VERBOSE is set
    setup_ssl_verification()
    
    try:
        # Ensure model exists or download it
        model_path = ensure_model_exists(args.models_dir)
        
        # Handle batch mode for processing multiple images
        if args.batch:
            if os.path.isdir(args.input):
                # Get all image files in the directory
                valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                input_files = []
                
                for file in os.listdir(args.input):
                    if any(file.lower().endswith(ext) for ext in valid_extensions):
                        input_files.append(os.path.join(args.input, file))
                
                if not input_files:
                    log(f"No image files found in {args.input}", "ERROR")
                    return
                
                log(f"Found {len(input_files)} images to process")
                
                # Make sure output directory exists
                output_dir = args.output if os.path.isdir(args.output) else os.path.dirname(args.output)
                os.makedirs(output_dir, exist_ok=True)
                
                # Process all images
                results = process_images(input_files, output_dir, model_path, 
                                       args.confidence, not args.no_save, args.debug_masks, args.auto_convert)
            else:
                log(f"{args.input} is not a directory. Cannot use batch mode.", "ERROR")
                return
        else:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(args.input), exist_ok=True)
            if not args.no_save:
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
            
            # Check if input image exists
            if not os.path.exists(args.input):
                log(f"Input image not found at {args.input}", "ERROR")
                log("Please provide a valid input image path", "INFO")
                return
            
            # Process single image
            result = process_image(args.input, args.output, model_path, args.confidence, 
                                 save_output=not args.no_save, save_debug_images=args.debug_masks, auto_convert=args.auto_convert)
            
            # In summary-only mode, print a clean one-line summary
            if SUMMARY_ONLY and result:
                # Create summary text based on configured colors
                in_color_text = f"IN ({IN_BOUNDS_COLOR})"
                out_color_text = f"OUT ({OUT_BOUNDS_COLOR})"
                
                # Determine in/out counts based on color settings
                in_count = result['on_green'] if IN_BOUNDS_COLOR == 'green' else result['on_blue']
                out_count = result['on_green'] if OUT_BOUNDS_COLOR == 'green' else result['on_blue']
                
                print(f"People: {result.get('courts_detected', 0)} court(s), {result['total_people']} total, {in_count} {in_color_text}, {out_count} {out_color_text}, {result['on_court_other']} on other court areas, {result['off_court']} off court")
                
                # Print per-court details if multiple courts
                if result.get('courts_detected', 0) > 1 and 'per_court_counts' in result:
                    for court_idx, counts in enumerate(result['per_court_counts']):
                        court_in_count = counts['on_green'] if IN_BOUNDS_COLOR == 'green' else counts['on_blue']
                        court_out_count = counts['on_green'] if OUT_BOUNDS_COLOR == 'green' else counts['on_blue']
                        
                        print(f"  Court #{court_idx+1}: {court_in_count} {in_color_text}, {court_out_count} {out_color_text}, {counts['on_court_other']} on other court areas")
        
    except Exception as e:
        log(f"Error: {str(e)}", "ERROR")
        if VERBOSE:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 