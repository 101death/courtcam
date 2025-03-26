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
        SHOW_DETAILED_LABELS = False             # Whether to show detailed labels on output image
    
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
    
    # Symbol map for message types - using simpler Unicode symbols
    SYMBOLS = {
        "INFO": "→",
        "SUCCESS": "✓",
        "WARNING": "!",
        "ERROR": "✗",
        "DEBUG": "•",
        "FATAL": "✗✗",
    }
    
    # Track messages for summary
    warnings = []
    errors = []
    successes = []
    info = []
    
    @classmethod
    def reset_logs(cls):
        """Reset all tracked messages"""
        cls.warnings = []
        cls.errors = []
        cls.successes = []
        cls.info = []
    
    @classmethod
    def log(cls, message, level="INFO"):
        """Print formatted log messages with consistent, clean formatting and track for summary"""
        # In super quiet mode, only show errors and success messages
        if Config.Output.SUPER_QUIET and level not in ["ERROR", "SUCCESS", "FATAL"]:
            return
            
        if not Config.Output.VERBOSE and level == "DEBUG":
            return
        
        # Get color based on level
        color = ""
        if Config.Output.USE_COLOR_OUTPUT:
            if level == "INFO":
                cls.info.append(message)
            elif level == "SUCCESS":
                color = cls.GREEN
                cls.successes.append(message)
            elif level == "WARNING":
                color = cls.YELLOW
                cls.warnings.append(message)
            elif level == "ERROR" or level == "FATAL":
                color = cls.RED
                cls.errors.append(message)
            elif level == "DEBUG":
                color = cls.GRAY
        
        # Format timestamp with consistent width
        timestamp = ""
        if Config.Output.SHOW_TIMESTAMP:
            timestamp = f"{cls.GRAY}[{datetime.now().strftime('%H:%M:%S')}]{cls.RESET} "
        
        # Create symbol prefix
        symbol = cls.SYMBOLS.get(level, "")
        
        # Format the message with appropriate styling
        formatted_message = f"{timestamp}{color}{symbol} {message}{cls.RESET}"
        
        print(formatted_message)
    
    @classmethod
    def summarize_detections(cls, courts, people, people_locations):
        """Summarize detection results in a more concise format"""
        if Config.Output.SUPER_QUIET:
            return {}
            
        cls.log(f"{cls.BOLD}Detection Summary:{cls.RESET}", "INFO")
        cls.log(f"Found {len(courts)} tennis courts", "SUCCESS")
        cls.log(f"Found {len(people)} people in the image", "SUCCESS")
        
        # Count people by court
        court_counts = {}
        for court_idx, area_type in people_locations:
            if court_idx >= 0:
                court_num = court_idx + 1
                if court_num not in court_counts:
                    court_counts[court_num] = 0
                court_counts[court_num] += 1
        
        # Print simplified court-specific counts
        for court_num in sorted(court_counts.keys()):
            cls.log(f"Court {court_num}: {court_counts[court_num]} people", "INFO")
        
        return court_counts
    
    @classmethod
    def create_final_summary(cls, people_count, court_counts, output_path=None):
        """Create a final summary line with errors/warnings if any"""
        # Base summary parts
        summary_parts = []
        
        # Start with what was successful
        if people_count is not None:
            # Include court count in the summary
            court_count = len(court_counts) if court_counts else 0
            
            # Format the main summary line
            if people_count == 0:
                summary_parts.append(f"detected {court_count} courts, no people")
            elif court_count == 0:
                summary_parts.append(f"detected {people_count} {'person' if people_count == 1 else 'people'}, no courts")
            else:
                # Format as "detected X courts, N on court 1, M on court 2"
                court_details = []
                for court_num in sorted(court_counts.keys()):
                    count = court_counts[court_num]
                    if count > 0:
                        court_details.append(f"{count} on court {court_num}")
                
                summary = f"detected {court_count} {'court' if court_count == 1 else 'courts'}"
                if court_details:
                    summary += f", {', '.join(court_details)}"
                summary_parts.append(summary)
        
        # Create the base summary text
        final_summary = " ".join(summary_parts)
        
        # Add output path in a separate line for better presentation
        if output_path:
            final_summary += f"\n\nOutput image saved"
        
        # Add errors/warnings differently - in a more natural way
        if cls.errors or cls.warnings:
            final_summary += "\n\n"
            
            if cls.errors:
                final_summary += cls.get_potential_fixes()
            elif cls.warnings and not cls.errors:
                if len(cls.warnings) == 1:
                    final_summary += f"Just a heads up: {cls.warnings[0]}"
                else:
                    warning_texts = [f"• {w}" for w in cls.warnings]
                    final_summary += f"A few things to note:\n" + "\n".join(warning_texts)
        
        return final_summary
    
    @classmethod
    def get_potential_fixes(cls):
        """Generate more natural language error messages"""
        messages = []
        
        if not cls.errors:
            return ""
            
        # Look for common error patterns and extract details
        for error in cls.errors:
            # Try to extract variable name for undefined variables
            if "is not defined" in error:
                var_name = error.split("'")[1] if "'" in error else "a variable"
                messages.append(f"Looks like {var_name} isn't defined in the code. Check the detection code around line 559.")
            
            # Image loading errors
            elif "Unable to open" in error or "load image" in error:
                messages.append(f"I couldn't find the image file. Make sure it exists at {Config.Paths.input_path()}.")
                
            # Model loading issues
            elif "model" in error.lower():
                messages.append(f"There was a problem with the YOLOv5 model. You might need to download it first.")
                
            # Permission issues
            elif any(word in error.lower() for word in ["permission", "access", "denied"]):
                messages.append(f"I don't have permission to access some files. Try running with higher permissions.")
                
            # For any other errors, give a simpler message
            else:
                messages.append(f"There was an error: {error}")
        
        # Create a more natural message
        if len(messages) == 1:
            return messages[0]
        else:
            return "I ran into a few issues:\n• " + "\n• ".join(messages)
            
        return "\n".join(messages)

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

def assign_court_numbers(blue_mask_connected):
    """
    Assign court numbers by clustering blue regions
    Returns a labeled mask where each court has a unique number
    """
    # Find all connected components in the blue mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask_connected, connectivity=8)
    
    # The first label (0) is the background, so we start from 1
    courts = []
    
    # Filter out small components (noise)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= Config.Court.MIN_AREA:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            courts.append({
                'id': i,
                'area': area,
                'bbox': (x, y, w, h),
                'centroid': centroids[i]
            })
    
    # Sort courts by x-coordinate to assign numbers from left to right
    courts.sort(key=lambda c: c['centroid'][0])
    
    # Create a renumbered mask
    court_mask = np.zeros_like(blue_mask_connected, dtype=np.uint8)
    
    # Assign new court numbers (1, 2, 3, ...) to each court based on sorted order
    for i, court in enumerate(courts):
        court_id = i + 1  # Start numbering from 1
        court['court_number'] = court_id
        # Extract original label mask and assign new number
        court_region = (labels == court['id']).astype(np.uint8) * court_id
        court_mask = cv2.add(court_mask, court_region)
    
    return court_mask, courts

def main():
    """Main function"""
    # Reset any previously tracked logs
    OutputManager.reset_logs()
    
    # Load image
    input_path = Config.Paths.input_path()
    try:
        image = cv2.imread(input_path)
        if image is None:
            OutputManager.log(f"Unable to open the image at {input_path}", "ERROR")
            # Show final summary with error and exit
            final_summary = OutputManager.create_final_summary(
                people_count=None, 
                court_counts={}, 
                output_path=None
            )
            OutputManager.log(f"{OutputManager.BOLD}{final_summary}{OutputManager.RESET}", "FATAL")
            return 1
    except Exception as e:
        OutputManager.log(f"Problem loading the image: {str(e)}", "ERROR")
        final_summary = OutputManager.create_final_summary(
            people_count=None, 
            court_counts={}, 
            output_path=None
        )
        OutputManager.log(f"{OutputManager.BOLD}{final_summary}{OutputManager.RESET}", "FATAL")
        return 1
    
    # Set up debug folder
    try:
        debug_folder = Config.Paths.debug_dir()
        os.makedirs(debug_folder, exist_ok=True)
    except Exception as e:
        OutputManager.log(f"Can't create debug folder: {str(e)}", "WARNING")
        # Continue execution even if debug folder can't be created
    
    # Create color masks for court detection
    OutputManager.log("Analyzing court colors in the image...", "INFO")
    blue_mask = create_blue_mask(image)
    green_mask = create_green_mask(image)
    
    # Do NOT apply additional morphological operations to connect blue regions
    # Use the raw blue mask to avoid connecting unrelated areas like the sky
    blue_mask_raw = blue_mask.copy()
    
    # Create court mask where green overrides blue
    height, width = image.shape[:2]
    court_mask = np.zeros((height, width), dtype=np.uint8)
    court_mask[blue_mask_raw > 0] = 1  # Blue areas
    court_mask[green_mask > 0] = 0     # Green areas override blue
    
    # Filter out blue regions that don't have any green nearby (like sky)
    # Find all connected components in the blue mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask_raw, connectivity=8)
    
    # For each blue region, check if there's green nearby
    filtered_court_mask = np.zeros_like(court_mask)
    for i in range(1, num_labels):
        region = (labels == i).astype(np.uint8)
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Skip very small regions
        if area < Config.Court.MIN_AREA:
            continue
        
        # Dilate the region to check for nearby green
        kernel = np.ones((15, 15), np.uint8)
        dilated_region = cv2.dilate(region, kernel, iterations=1)
        
        # Check if there's green nearby this blue region
        green_nearby = cv2.bitwise_and(green_mask, dilated_region)
        green_nearby_pixels = cv2.countNonZero(green_nearby)
        
        # Only keep blue regions that have at least some green nearby
        if green_nearby_pixels > 50:  # Minimum threshold for green pixels
            # This is likely a court (not sky) - keep it
            filtered_court_mask[region > 0] = court_mask[region > 0]
    
    # Use the filtered court mask for further processing
    court_mask = filtered_court_mask
    
    # Save raw masks for debugging
    try:
        cv2.imwrite(os.path.join(debug_folder, "blue_mask_raw.png"), blue_mask_raw)
        cv2.imwrite(os.path.join(debug_folder, "green_mask.png"), green_mask)
        cv2.imwrite(os.path.join(debug_folder, "filtered_court_mask.png"), court_mask * 255)
    except Exception as e:
        OutputManager.log(f"Couldn't save debug masks: {str(e)}", "WARNING")
    
    # Create colored visualization of masks
    court_mask_viz = np.zeros((height, width, 3), dtype=np.uint8)
    court_mask_viz[blue_mask_raw > 0] = [255, 0, 0]  # Blue for all blue areas
    court_mask_viz[green_mask > 0] = [0, 255, 0]     # Green areas override blue
    
    # Highlight filtered courts in a brighter blue
    filtered_blue = np.zeros_like(court_mask_viz)
    filtered_blue[court_mask > 0] = [255, 127, 0]  # Bright blue for valid courts
    cv2.addWeighted(court_mask_viz, 1, filtered_blue, 0.7, 0, court_mask_viz)
    
    try:
        cv2.imwrite(os.path.join(debug_folder, "color_masks.png"), court_mask_viz)
    except Exception as e:
        OutputManager.log(f"Couldn't save color visualization: {str(e)}", "WARNING")
    
    # Assign court numbers to each separate blue region
    court_numbers_mask, courts = assign_court_numbers(court_mask)
    
    if len(courts) == 0:
        OutputManager.log("Couldn't find any tennis courts in this image", "WARNING")
    else:
        OutputManager.log(f"Found {len(courts)} tennis court{'s' if len(courts) > 1 else ''}", "SUCCESS")
    
    # Create a color-coded court mask for visualization
    court_viz = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign different colors to each court
    court_colors = [
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 0, 255),  # Purple
        (0, 255, 255)   # Yellow
    ]
    
    # Draw each court with a different color
    for court in courts:
        court_id = court['court_number']
        color_idx = (court_id - 1) % len(court_colors)
        court_color = court_colors[color_idx]
        
        # Extract court mask
        court_mask_individual = (court_numbers_mask == court_id).astype(np.uint8) * 255
        # Find contours of the court
        court_contours, _ = cv2.findContours(court_mask_individual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the court area
        court_area = np.zeros_like(court_viz)
        court_area[court_mask_individual > 0] = court_color
        cv2.addWeighted(court_viz, 1, court_area, 0.7, 0, court_viz)
        
        # Draw court number at center only if enabled in debug visualizations too
        if Config.Visual.SHOW_COURT_LABELS:
            cx, cy = int(court['centroid'][0]), int(court['centroid'][1])
            cv2.putText(court_viz, f"Court {court_id}", (cx-40, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Save court visualization
    try:
        cv2.imwrite(os.path.join(debug_folder, "courts_numbered.png"), court_viz)
    except Exception as e:
        OutputManager.log(f"Couldn't save court visualization: {str(e)}", "WARNING")
    
    # Create a semi-transparent overlay of the masks on the original image
    alpha = 0.5  # Transparency factor
    mask_overlay = image.copy()
    # Apply the colored masks with transparency
    cv2.addWeighted(court_mask_viz, alpha, mask_overlay, 1 - alpha, 0, mask_overlay)
    
    # Detect people
    OutputManager.log("Looking for people in the image...", "INFO")
    try:
        people = detect_people(image)
    except Exception as e:
        OutputManager.log(f"Problem detecting people: {str(e)}", "ERROR")
        # Continue with empty people list
        people = []
    
    if len(people) == 0:
        OutputManager.log("No people appear to be in this image", "WARNING")
    else:
        OutputManager.log(f"Found {len(people)} {'person' if len(people) == 1 else 'people'} in the image", "SUCCESS")
    
    # Determine if each person is on a court based on masks directly
    people_locations = []
    
    for person in people:
        # Get foot position
        if 'foot_position' in person:
            foot_x, foot_y = person['foot_position']
        else:
            foot_x, foot_y = person['position']
        
        # Check if foot is on blue (in-bounds) or green (out-bounds)
        if foot_y < height and foot_x < width:
            # Check which court number the person is on
            if blue_mask_raw[foot_y, foot_x] > 0 and green_mask[foot_y, foot_x] == 0 and court_mask[foot_y, foot_x] > 0:
                # On blue area and not on green area (in-bounds)
                # Get the court number
                court_number = court_numbers_mask[foot_y, foot_x]
                if court_number > 0:
                    court_idx = court_number - 1  # Convert court number to zero-based index
                    area_type = 'in_bounds'
                else:
                    court_idx = -1
                    area_type = 'off_court'
            elif green_mask[foot_y, foot_x] > 0:
                # On green area (out-bounds)
                # Check nearby blue to determine the court number
                nearby_mask = np.zeros((height, width), dtype=np.uint8)
                cv2.circle(nearby_mask, (foot_x, foot_y), 20, 255, -1)
                nearby_courts = cv2.bitwise_and(court_numbers_mask, court_numbers_mask, mask=nearby_mask)
                unique_courts = np.unique(nearby_courts)
                if len(unique_courts) > 1:  # First value is 0 (background)
                    court_number = unique_courts[1]
                    court_idx = court_number - 1
                    area_type = 'out_bounds'
                else:
                    court_idx = -1
                    area_type = 'off_court'
            else:
                # Not on any colored area
                court_idx = -1
                area_type = 'off_court'
        else:
            court_idx = -1
            area_type = 'off_court'
            
        people_locations.append((court_idx, area_type))
    
    # Display summary and get court counts - simplify to avoid repetition
    court_counts = {}
    for court_idx, area_type in people_locations:
        if court_idx >= 0:
            court_num = court_idx + 1
            if court_num not in court_counts:
                court_counts[court_num] = 0
            court_counts[court_num] += 1
    
    # Log court counts without repeating the detection summary
    for court_num in sorted(court_counts.keys()):
        OutputManager.log(f"Court {court_num}: {court_counts[court_num]} people", "INFO")
    
    # Create debug visualization showing foot positions on mask
    debug_foot_positions = court_viz.copy()
    for person_idx, person in enumerate(people):
        if 'foot_position' in person:
            foot_x, foot_y = person['foot_position']
            # Draw foot position marker (circle)
            cv2.circle(debug_foot_positions, (foot_x, foot_y), 10, (255, 255, 255), -1)
            cv2.circle(debug_foot_positions, (foot_x, foot_y), 10, (0, 0, 0), 2)
            # Label with person index
            cv2.putText(debug_foot_positions, f"P{person_idx+1}", (foot_x+15, foot_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    try:
        cv2.imwrite(os.path.join(debug_folder, "foot_positions_debug.png"), debug_foot_positions)
    except Exception as e:
        OutputManager.log(f"Couldn't save foot positions debug image: {str(e)}", "WARNING")
    
    # Draw people and their locations
    output_image = image.copy()
    
    # Draw court outlines with different colors
    for court in courts:
        court_id = court['court_number']
        color_idx = (court_id - 1) % len(court_colors)
        court_color = court_colors[color_idx]
        
        # Extract court mask
        court_mask_individual = (court_numbers_mask == court_id).astype(np.uint8) * 255
        # Find contours of the court
        court_contours, _ = cv2.findContours(court_mask_individual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw the court outline
        cv2.drawContours(output_image, court_contours, -1, court_color, 2)
        
        # Draw court number at center only if enabled
        if Config.Visual.SHOW_COURT_LABELS:
            cx, cy = int(court['centroid'][0]), int(court['centroid'][1])
            cv2.putText(output_image, f"Court {court_id}", (cx-40, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Draw people and their locations
    for person_idx, person in enumerate(people):
        court_idx, area_type = people_locations[person_idx]
        
        # Draw bounding box and label
        x1, y1, x2, y2 = person['bbox']
        
        # Choose color based on location
        if court_idx >= 0:
            court_number = court_idx + 1
            if area_type == 'in_bounds':
                color = Config.Visual.PERSON_IN_BOUNDS_COLOR
                label = f"Court {court_number}" if Config.Visual.SHOW_DETAILED_LABELS else ""
            else:  # out_bounds
                color = Config.Visual.PERSON_OUT_BOUNDS_COLOR
                label = f"Court {court_number} • Sideline" if Config.Visual.SHOW_DETAILED_LABELS else ""
        else:
            color = Config.Visual.PERSON_OFF_COURT_COLOR
            label = "Not on court" if Config.Visual.SHOW_DETAILED_LABELS else ""
        
        # Draw bounding box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw foot position marker - smaller and less intrusive
        foot_x, foot_y = person['foot_position']
        cv2.circle(output_image, (foot_x, foot_y), 3, color, -1)
        
        # Only draw text labels if specified
        if Config.Visual.SHOW_DETAILED_LABELS and label:
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
            
            # Add person index number
            cv2.putText(output_image, f"Person {person_idx+1}", (x1, y2 + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, Config.Visual.FONT_SCALE, 
                        color, Config.Visual.TEXT_THICKNESS)
        else:
            # Just add a small number indicator for simpler display
            cv2.putText(output_image, f"{person_idx+1}", (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       Config.Visual.FONT_SCALE, 
                       Config.Visual.TEXT_COLOR, 
                       Config.Visual.TEXT_THICKNESS)
    
    # Save the final output image
    output_path = Config.Paths.output_path()
    try:
        cv2.imwrite(output_path, output_image)
        OutputManager.log(f"Output image saved", "SUCCESS")
    except Exception as e:
        OutputManager.log(f"Error saving output image: {str(e)}", "ERROR")
        output_path = None
    
    # Create the adaptive final summary
    final_summary = OutputManager.create_final_summary(
        people_count=len(people),
        court_counts=court_counts,
        output_path=output_path
    )
    
    # Print the final summary with decorative borders
    border_width = 80
    top_border = "╭" + "─" * (border_width - 2) + "╮"
    middle_border = "├" + "─" * (border_width - 2) + "┤"
    bottom_border = "╰" + "─" * (border_width - 2) + "╯"
    
    print("\n" + top_border)
    print("│ " + "RESULTS SUMMARY".center(border_width - 4) + " │")
    print(middle_border)
    print("│ " + " " * (border_width - 4) + " │")
    
    # Split the summary by lines and format each line
    summary_lines = final_summary.split('\n')
    for line in summary_lines:
        # Handle empty lines
        if not line.strip():
            print("│ " + " " * (border_width - 4) + " │")
        else:
            # Format line to fit within border
            print("│ " + line.ljust(border_width - 4)[:border_width - 4] + " │")
    
    print("│ " + " " * (border_width - 4) + " │")
    print(bottom_border)
    
    # If there were errors that didn't cause a fatal exit, still indicate an error status
    if OutputManager.errors:
        return 1
    
    return 0

# Add a backward compatibility wrapper for the old log function
def log(message, level="INFO"):
    """Wrapper for backward compatibility with the old log function"""
    OutputManager.log(message, level)

if __name__ == "__main__":
    # Add command-line arguments for easier use
    parser = argparse.ArgumentParser(description="Tennis Court Detection System")
    parser.add_argument("--input", type=str, help="Path to input image", default=Config.Paths.input_path())
    parser.add_argument("--output", type=str, help="Path for output image", default=Config.Paths.output_path())
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional outputs")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    parser.add_argument("--show-labels", action="store_true", help="Show detailed labels on output image")
    parser.add_argument("--show-court-labels", action="store_true", help="Show court numbers on output image")
    args = parser.parse_args()
    
    # Update config based on arguments
    if args.input != Config.Paths.input_path():
        Config.Paths.INPUT_IMAGE = os.path.basename(args.input)
        Config.Paths.IMAGES_DIR = os.path.dirname(args.input)
    
    if args.output != Config.Paths.output_path():
        Config.Paths.OUTPUT_IMAGE = os.path.basename(args.output)
        Config.Paths.IMAGES_DIR = os.path.dirname(args.output)
    
    Config.DEBUG_MODE = args.debug
    Config.Output.VERBOSE = not args.quiet
    Config.Visual.SHOW_DETAILED_LABELS = args.show_labels
    Config.Visual.SHOW_COURT_LABELS = args.show_court_labels
    
    sys.exit(main())
