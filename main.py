import cv2
import numpy as np
import sys
import os
import json
from ultralytics import YOLO
from PIL import Image

# ------------------- Load Config -------------------
def load_config(config_path="config.json"):
    """
    Loads a simple JSON config or uses fallback defaults.
    """
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading config.json: {e}")
            sys.exit(1)
    else:
        # Fallback config if file not found
        return {
            "court": {
                "inside_color": "blue",
                "line_color": "white"
            },
            "processing": {},
            "yolo": {
                "model_path": "yolov8n.pt"
            }
        }

# ------------------- Read Image -------------------
def read_image(image_path):
    """
    Reads an image using OpenCV. Converts AVIF to PNG if needed.
    """
    if not os.path.exists(image_path):
        print(f"Error: Input image '{image_path}' does not exist!")
        sys.exit(1)
    
    if image_path.lower().endswith(".avif"):
        try:
            avif_image = Image.open(image_path)
            converted_path = "converted_image.png"
            avif_image.save(converted_path, format="PNG")
            image_path = converted_path
        except Exception as e:
            print(f"Error converting AVIF: {e}")
            sys.exit(1)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image '{image_path}'!")
        sys.exit(1)
    return img

# ------------------- Court Mask -------------------
def create_court_mask(image, inside_color):
    """
    Creates a binary mask for the court color (blue/green/red).
    Improved with adaptive thresholding and multiple color ranges for detection.
    """
    # Enhanced HSV ranges for better court detection
    COLOR_RANGES = {
        "blue": [
            ([90, 50, 50], [130, 255, 255]),   # Standard blue
            ([100, 40, 40], [140, 255, 255]),  # Darker blue variation
            ([80, 40, 40], [110, 255, 255])    # Lighter blue variation
        ],
        "green": [
            ([35, 50, 50], [85, 255, 255]),    # Standard green
            ([30, 40, 40], [90, 255, 255]),    # Wider green range
            ([40, 30, 40], [80, 255, 255])     # Another green variation
        ],
        "red": [
            ([0, 50, 50], [10, 255, 255]),     # Lower red hue
            ([160, 50, 50], [180, 255, 255]),  # Upper red hue (red wraps around)
            ([0, 40, 40], [20, 255, 255])      # Wider lower red
        ]
    }
    
    color_ranges = COLOR_RANGES.get(inside_color, COLOR_RANGES["blue"])
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create combined mask from all ranges for the given color
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    
    for lower_hsv, upper_hsv in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # Improved morphological operations with adaptive kernel size
    height, width = image.shape[:2]
    kernel_size = max(5, min(15, int(min(height, width) / 100)))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Close operation to fill small holes
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Open operation to remove small noise
    smaller_kernel = np.ones((kernel_size//2, kernel_size//2), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, smaller_kernel)
    
    return combined_mask

# ------------------- Court Detection -------------------
def detect_largest_court_contour(mask):
    """
    Finds the largest contour in the mask that matches court aspect ratio.
    Returns the contour if found, else None.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Sort contours by area, largest first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Try to find a contour with reasonable court aspect ratio
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000:  # Skip too small contours
            continue
            
        # Check if the contour has a reasonable aspect ratio for a tennis court
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Tennis courts typically have an aspect ratio around 1.8-2.4 (length/width)
        # Be a bit lenient with the range
        if 1.5 <= aspect_ratio <= 3.0 or 0.33 <= aspect_ratio <= 0.67:
            return contour
    
    # If no contour with proper aspect ratio found, return the largest one
    if contours:
        return contours[0]
    
    return None

def approximate_polygon(contour):
    """
    Approximates a polygon from the largest contour with dynamic epsilon.
    """
    if contour is None:
        return None
        
    perimeter = cv2.arcLength(contour, True)
    # Dynamic epsilon calculation based on perimeter
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # For tennis courts, we expect 4-6 corners after approximation
    # If we got too many points, increase epsilon
    if len(approx) > 8:
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
    
    return approx

# ------------------- Line Detection -------------------
def enhance_court_lines(image):
    """
    Enhances white lines in the court to improve detection.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to find potential lines
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 11, 2)
    
    # Create a mask of white regions
    white_mask = cv2.inRange(image, np.array([180, 180, 180]), np.array([255, 255, 255]))
    
    # Combine with thresholded image
    combined = cv2.bitwise_and(thresh, white_mask)
    
    # Small kernel for removing noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    
    return cleaned

# ------------------- Court Refinement -------------------
def refine_court_detection(image, mask, polygon):
    """
    Refines court detection by incorporating line detection.
    """
    if polygon is None or len(polygon) < 4:
        return polygon
        
    # Enhance lines
    line_mask = enhance_court_lines(image)
    
    # Use Hough lines to detect straight lines
    lines = cv2.HoughLinesP(line_mask, 1, np.pi/180, threshold=100,
                           minLineLength=100, maxLineGap=20)
                           
    # If no lines detected, return original polygon
    if lines is None:
        return polygon
        
    # Draw lines on a blank image
    h, w = image.shape[:2]
    line_image = np.zeros((h, w), dtype=np.uint8)
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), 255, 2)
    
    # Combine with original mask
    refined_mask = cv2.bitwise_or(mask, line_image)
    
    # Find contours on refined mask
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return polygon
        
    largest_contour = max(contours, key=cv2.contourArea)
    refined_polygon = approximate_polygon(largest_contour)
    
    # If refined polygon looks reasonable, use it, otherwise keep original
    if refined_polygon is not None and 4 <= len(refined_polygon) <= 8:
        return refined_polygon
    
    return polygon

# ------------------- Gradient Overlay -------------------
def apply_gradient_overlay(image, poly, max_alpha=0.6):
    """
    Applies a vertical gradient overlay to the bounding rectangle of the polygon.
    """
    if poly is None or len(poly) < 3:
        return image
    
    x, y, w, h = cv2.boundingRect(poly)
    x_end = min(x + w, image.shape[1])
    y_end = min(y + h, image.shape[0])
    
    # Extract region of interest
    roi = image[y:y_end, x:x_end].copy().astype(np.float32)
    overlay = np.zeros_like(roi, dtype=np.float32)
    
    # Create gradient from top (0) to bottom (max_alpha)
    height = roi.shape[0]
    width = roi.shape[1]
    grad = np.tile(np.linspace(0, max_alpha, height).reshape(-1, 1), (1, width))
    grad = np.repeat(grad[:, :, np.newaxis], 3, axis=2)
    
    blended = roi * (1 - grad) + overlay * grad
    image[y:y_end, x:x_end] = blended.astype(np.uint8)
    return image

# ------------------- YOLO Detection -------------------
def detect_people_yolo(image, model_path="yolov8n.pt", conf_thresh=0.5):
    """
    Detects people in the image using YOLOv8.
    Returns a list of bounding boxes [x1, y1, x2, y2, confidence].
    """
    model = YOLO(model_path)
    results = model(image)
    boxes_out = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # 'person'
                conf = float(box.conf[0])
                if conf >= conf_thresh:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes_out.append([x1, y1, x2, y2, conf])
    return boxes_out

# ------------------- Check if Inside Court -------------------
def is_person_inside_court(x1, y1, x2, y2, polygon):
    """
    Uses the bottom-center (feet) of the bounding box to determine if inside the polygon.
    """
    if polygon is None or len(polygon) < 3:
        return False
    px = x1 + (x2 - x1) // 2
    py = y2  # feet position
    # Reshape polygon to Nx2 if needed
    pts = polygon.reshape(-1, 2)
    result = cv2.pointPolygonTest(pts, (px, py), False)
    return result >= 0

# ------------------- Main Script -------------------
def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <input_image> <output_image>")
        sys.exit(1)
    
    input_image_path = sys.argv[1]
    output_image_path = sys.argv[2]
    
    # Load config
    config = load_config("config.json")
    
    # Read image
    original_img = read_image(input_image_path)
    annotated_img = original_img.copy()
    
    # Get parameters from config
    inside_color = config["court"].get("inside_color", "blue")
    model_path = config["yolo"].get("model_path", "yolov8n.pt")
    
    # 1. Create mask for the court color
    court_mask = create_court_mask(original_img, inside_color)
    
    # Optional: Save the mask for debugging
    cv2.imwrite("court_mask.png", court_mask)
    
    # 2. Find largest contour
    court_contour = detect_largest_court_contour(court_mask)
    if court_contour is None:
        print("No court contour found.")
        cv2.imwrite(output_image_path, annotated_img)
        return
    
    # 3. Approximate polygon
    court_poly = approximate_polygon(court_contour)
    
    # 4. Refine court detection using line information
    refined_poly = refine_court_detection(original_img, court_mask, court_poly)
    
    # 5. Draw polygon with VERY noticeable, thick lines
    # Use bright magenta color for maximum visibility regardless of background
    court_outline_color = (255, 0, 255)  # Bright magenta in BGR
    
    # Draw the main polygon with thick lines (5px)
    cv2.polylines(annotated_img, [refined_poly], True, court_outline_color, 5)
    
    # Add a contrasting outline (yellow) to make it stand out even more
    cv2.polylines(annotated_img, [refined_poly], True, (0, 255, 255), 2)
    
    # 6. Apply gradient overlay to bounding rectangle of the court polygon
    annotated_img = apply_gradient_overlay(annotated_img, refined_poly, max_alpha=0.6)
    
    # 7. Detect people with YOLO
    people_boxes = detect_people_yolo(original_img, model_path=model_path, conf_thresh=0.5)
    
    # 8. Check if each person is inside the court
    court_occupied = 0
    total_people = 0
    for (x1, y1, x2, y2, conf) in people_boxes:
        total_people += 1
        inside = is_person_inside_court(x1, y1, x2, y2, refined_poly)
        if inside:
            court_occupied = 1
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        label = f"Inside Court ({conf:.2f})" if inside else f"Outside Court ({conf:.2f})"
        cv2.putText(annotated_img, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 9. Save output
    cv2.imwrite(output_image_path, annotated_img)
    
    # 10. Save a clean version with ONLY the court polygon
    court_only_img = original_img.copy()
    # Draw a very bold, highly visible court outline
    cv2.polylines(court_only_img, [refined_poly], True, (255, 0, 255), 8)  # Thick magenta
    cv2.polylines(court_only_img, [refined_poly], True, (0, 255, 255), 3)  # Yellow border
    
    # Draw dots at each vertex for clarity
    for point in refined_poly:
        x, y = point[0]
        cv2.circle(court_only_img, (x, y), 5, (0, 0, 255), -1)  # Red dots at vertices
        
    cv2.imwrite("court_outline_only.png", court_only_img)
    
    # Save other debug images
    cv2.imwrite("court_mask_debug.png", court_mask)
    
    # 11. Print results
    print(f"Court Occupied: {court_occupied}")
    print(f"Total People Detected: {total_people}")
    print(f"Court Polygon Points: {len(refined_poly)}")

if __name__ == "__main__":
    main()