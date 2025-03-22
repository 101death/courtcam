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
        "lower": [30, 25, 25],              # Lower HSV range for green courts (much broader)
        "upper": [90, 255, 255]             # Upper HSV range for green courts (much broader)
    },
    "blue": {
        "lower": [90, 40, 40],              # Lower HSV range for blue courts
        "upper": [130, 255, 255]            # Upper HSV range for blue courts
    }
}
CONTOUR_APPROX_FACTOR = 0.02                 # Contour approximation factor

# Visualization settings
COURT_OUTLINE_COLOR = (0, 255, 0)            # Green
COURT_OUTLINE_THICKNESS = 2                  # Line thickness
ON_GREEN_COLOR = (0, 255, 0)                 # Green for people on green court
ON_BLUE_COLOR = (255, 191, 0)                # Deep sky blue for people on blue court
ON_COURT_OTHER_COLOR = (0, 255, 255)         # Yellow for people on other court areas
OFF_COURT_COLOR = (0, 0, 255)                # Red for people outside the court
TEXT_COLOR = (255, 255, 255)                 # White
FONT_SCALE = 0.5                             # Text size
TEXT_THICKNESS = 2                           # Text thickness

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
    Detect tennis court in the image
    Returns the court polygon and separate masks for green and blue areas
    """
    # Convert to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create masks for green and blue courts using the configured settings
    mask_green = cv2.inRange(
        hsv, 
        np.array(COURT_COLOR_RANGES["green"]["lower"]), 
        np.array(COURT_COLOR_RANGES["green"]["upper"])
    )
    
    mask_blue = cv2.inRange(
        hsv, 
        np.array(COURT_COLOR_RANGES["blue"]["lower"]), 
        np.array(COURT_COLOR_RANGES["blue"]["upper"])
    )
    
    # Apply morphological operations to clean up each mask individually
    # Create larger kernels for more aggressive morphology
    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open = np.ones((5, 5), np.uint8)
    
    # More aggressive morphology for green mask
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, kernel_close)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel_open)
    mask_green = cv2.dilate(mask_green, kernel_open, iterations=1)  # Additional dilation
    
    # Standard morphology for blue mask
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel_close)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel_open)
    
    # Combine masks for overall court detection
    court_mask = cv2.bitwise_or(mask_green, mask_blue)
    
    # Apply additional morphology to close gaps in the combined mask
    court_mask = cv2.morphologyEx(court_mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    
    # Find contours on combined mask
    contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # If no contours found, return None
    if not contours:
        return None
    
    # Find the largest contour (likely the court)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to simplify
    epsilon = CONTOUR_APPROX_FACTOR * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to Shapely polygon for easier operations
    court_polygon = Polygon([(p[0][0], p[0][1]) for p in approx])
    
    return {
        'court_polygon': court_polygon,
        'court_contour': approx,
        'green_mask': mask_green,
        'blue_mask': mask_blue
    }

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
            
            # Set confidence threshold
            model.conf = confidence_threshold
            
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
        
        return person_detections
    except Exception as e:
        log(f"Error in person detection: {str(e)}", "ERROR")
        raise

def is_person_on_court(person_bbox, court_data):
    """
    Determine if a person is on the tennis court and if they're on green or blue areas
    """
    x1, y1, x2, y2 = person_bbox
    
    # Use the bottom center point of the bounding box (person's feet)
    foot_x = int((x1 + x2) / 2)
    foot_y = y2
    foot_point = Point(foot_x, foot_y)
    
    # Check if this point is inside the court polygon
    if not court_data['court_polygon'].contains(foot_point):
        return 'off_court'
    
    # Check multiple points around the feet to make a more robust detection
    # This handles cases where the exact foot point might be on a line
    height = 10
    width = 20
    points_to_check = [
        (foot_x, foot_y),  # Bottom center
        (max(x1, foot_x - width), foot_y),  # Bottom left
        (min(x2, foot_x + width), foot_y),  # Bottom right
        (foot_x, max(y1, foot_y - height))  # Slightly above
    ]
    
    # Check each point and record what it's on
    on_green = False
    on_blue = False
    
    for px, py in points_to_check:
        # Make sure we don't go outside image bounds
        if 0 <= py < court_data['green_mask'].shape[0] and 0 <= px < court_data['green_mask'].shape[1]:
            if court_data['green_mask'][py, px] > 0:
                on_green = True
            elif court_data['blue_mask'][py, px] > 0:
                on_blue = True
    
    # If there's very little green in the entire court (less than 5% of blue),
    # consider the player to be "on blue" for this particular court
    green_total = np.sum(court_data['green_mask'] > 0)
    blue_total = np.sum(court_data['blue_mask'] > 0)
    
    if blue_total > 0 and green_total < blue_total * 0.05 and on_blue:
        return 'on_blue'
    
    # Return based on priority (green takes precedence if standing on both)
    if on_green:
        return 'on_green'
    elif on_blue:
        return 'on_blue'
    else:
        # Inside court polygon but not directly on colored area 
        # (might be on line or transitional area)
        return 'on_court'

def process_image(input_path, output_path, model_path, conf_threshold=DEFAULT_CONFIDENCE, 
                 save_output=True, save_debug_images=False):
    """
    Process the image to detect court and people
    """
    start_time = time.time()
    
    # Load image
    log(f"Loading image from {input_path}")
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {input_path}")
    
    # Make a copy for drawing results
    output_image = image.copy()
    
    # Detect tennis court
    log("Detecting tennis court...")
    court_data = detect_tennis_court(image)
    if court_data is None:
        log("No tennis court detected in the image", "WARNING")
        return
    
    log(f"Tennis court detected with {len(court_data['court_contour'])} points", "SUCCESS")
    
    # If green detection barely caught anything, use an alternate approach
    green_coverage = np.sum(court_data['green_mask'] > 0)
    blue_coverage = np.sum(court_data['blue_mask'] > 0)
    
    log(f"Green area pixels: {green_coverage}, Blue area pixels: {blue_coverage}", "DEBUG")
    
    # If green is less than 5% of blue, consider it as a primarily blue court
    if blue_coverage > 0 and green_coverage < blue_coverage * 0.05:
        log("Minimal green court area detected. Creating artificial green area for visualization.", "DEBUG")
        # Clone the masks for visualization but keep processing using the current masks
        # This enables showing better debug images while maintaining the current logic
        if save_debug_images:
            # For visualization, create an artificial green area in the court center
            # This is just for visualization to help users understand the detection
            blue_mask = court_data['blue_mask'].copy()
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
                court_data['green_mask_visual'] = green_mask
    
    # If debug images are requested, save the court masks
    if save_debug_images:
        output_dir = os.path.dirname(output_path)
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        
        # Create colored masks for visualization
        green_mask = court_data.get('green_mask_visual', court_data['green_mask'])
        green_mask_colored = np.zeros_like(image)
        green_mask_colored[green_mask > 0] = [0, 255, 0]  # Green color
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_green_mask.png"), green_mask_colored)
        log(f"Green court mask saved to {os.path.join(output_dir, f'{base_name}_green_mask.png')}")
        
        blue_mask_colored = np.zeros_like(image)
        blue_mask_colored[court_data['blue_mask'] > 0] = [255, 191, 0]  # Blue color (BGR)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_blue_mask.png"), blue_mask_colored)
        log(f"Blue court mask saved to {os.path.join(output_dir, f'{base_name}_blue_mask.png')}")
        
        # Combined mask
        combined_mask_colored = np.zeros_like(image)
        combined_mask_colored[green_mask > 0] = [0, 255, 0]  # Green
        combined_mask_colored[court_data['blue_mask'] > 0] = [255, 191, 0]  # Blue
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_court_mask.png"), combined_mask_colored)
        log(f"Combined court mask saved to {os.path.join(output_dir, f'{base_name}_court_mask.png')}")
    
    # Draw court outline
    cv2.drawContours(output_image, [court_data['court_contour']], 0, COURT_OUTLINE_COLOR, COURT_OUTLINE_THICKNESS)
    
    # Detect people
    log("Detecting people...")
    people = detect_people(image, model_path, conf_threshold)
    log(f"Found {len(people)} people in the image", "SUCCESS")
    
    # Count people in different areas
    on_green_count = 0
    on_blue_count = 0
    on_court_other = 0  # On court but not specifically on green or blue
    off_court_count = 0
    
    # Process each person
    for person in people:
        bbox = person['bbox']
        x1, y1, x2, y2 = bbox
        
        # Check person's location
        location = is_person_on_court(bbox, court_data)
        
        # Increment counters
        if location == 'on_green':
            on_green_count += 1
            color = ON_GREEN_COLOR
            label = f"On Green ({person['confidence']:.2f})"
        elif location == 'on_blue':
            on_blue_count += 1
            color = ON_BLUE_COLOR
            label = f"On Blue ({person['confidence']:.2f})"
        elif location == 'on_court':
            on_court_other += 1
            color = ON_COURT_OTHER_COLOR
            label = f"On Court ({person['confidence']:.2f})"
        else:  # off_court
            off_court_count += 1
            color = OFF_COURT_COLOR
            label = f"Off Court ({person['confidence']:.2f})"
        
        # Draw bounding box with appropriate color
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
        
        # Add label with confidence
        cv2.putText(output_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, TEXT_THICKNESS)
    
    # Save output image if requested
    if save_output:
        cv2.imwrite(output_path, output_image)
        log(f"Output image saved to {output_path}", "SUCCESS")
    
    # Add summary text
    elapsed_time = time.time() - start_time
    summary = f"Summary: {on_green_count} on green, {on_blue_count} on blue, {on_court_other} on other court areas, {off_court_count} off court"
    log(summary, "SUCCESS" if SUMMARY_ONLY else "INFO")
    
    if not SUMMARY_ONLY:
        log(f"Processing completed in {elapsed_time:.2f} seconds", "INFO")
    
    return {
        "total_people": len(people),
        "on_green": on_green_count,
        "on_blue": on_blue_count,
        "on_court_other": on_court_other,
        "off_court": off_court_count,
        "elapsed_time": elapsed_time
    }

def process_images(input_paths, output_dir, model_path, conf_threshold=DEFAULT_CONFIDENCE, 
                 save_output=True, save_debug_images=False):
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
                                 save_output, save_debug_images)
            if result:
                result['input_file'] = input_path
                result['output_file'] = output_path
                results.append(result)
        except Exception as e:
            log(f"Error processing {input_path}: {str(e)}", "ERROR")
    
    # Print summary of all results
    if results:
        total_people = sum(r['total_people'] for r in results)
        total_on_green = sum(r['on_green'] for r in results)
        total_on_blue = sum(r['on_blue'] for r in results)
        total_on_court_other = sum(r['on_court_other'] for r in results)
        total_off_court = sum(r['off_court'] for r in results)
        
        log(f"Processed {len(results)} images successfully", "SUCCESS")
        log(f"Total people detected: {total_people} ({total_on_green} on green, {total_on_blue} on blue, {total_on_court_other} on other court areas, {total_off_court} off court)", 
           "SUCCESS" if SUMMARY_ONLY else "INFO")
        
        if SUMMARY_ONLY:
            print(f"Total: {total_people} people, {total_on_green} on green, {total_on_blue} on blue, {total_on_court_other} on other court areas, {total_off_court} off court")
    
    return results

def main():
    """Main function"""
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
                      help="Show only the summary count of people detected")
    parser.add_argument("--no-save", action="store_true",
                      help="Do not save the output image (just analyze)")
    parser.add_argument("--batch", action="store_true",
                      help="Process all images in the input directory")
    parser.add_argument("--debug-masks", action="store_true",
                      help="Save debug images of the court masks")
    args = parser.parse_args()
    
    # Set output modes based on flags
    global VERBOSE, SUPER_QUIET, SUMMARY_ONLY
    SUMMARY_ONLY = args.summary
    SUPER_QUIET = args.super_quiet or SUMMARY_ONLY  # Summary mode implies super quiet
    VERBOSE = not args.quiet and not SUPER_QUIET
    
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
                                       args.confidence, not args.no_save, args.debug_masks)
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
                                 save_output=not args.no_save, save_debug_images=args.debug_masks)
            
            # In summary-only mode, print a clean one-line summary
            if SUMMARY_ONLY and result:
                print(f"People: {result['total_people']} total, {result['on_green']} on green, {result['on_blue']} on blue, {result['on_court_other']} on other court areas, {result['off_court']} off court")
        
    except Exception as e:
        log(f"Error: {str(e)}", "ERROR")
        if VERBOSE:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 