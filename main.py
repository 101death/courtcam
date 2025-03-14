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
    Adjust HSV or morphological parameters as needed.
    """
    # Default HSV ranges (tweak as needed)
    COLOR_RANGES = {
        "blue": ([90, 50, 50], [130, 255, 255]),
        "green": ([35, 50, 50], [85, 255, 255]),
        "red": ([0, 50, 50], [10, 255, 255])
    }
    
    lower_hsv, upper_hsv = COLOR_RANGES.get(inside_color, COLOR_RANGES["blue"])
    
    # Convert to HSV and apply threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
    
    # Morphological operations (hardcoded defaults)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask

# ------------------- Court Detection -------------------
def detect_largest_court_contour(mask):
    """
    Finds the largest contour in the mask. We assume it's the court.
    Returns the contour if found, else None.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 5000:  # Arbitrary min area
        return None
    return max_contour

def approximate_polygon(contour):
    """
    Approximates a polygon from the largest contour.
    """
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

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
    
    # Court color from config
    inside_color = config["court"].get("inside_color", "blue")
    # YOLO model path from config
    model_path = config["yolo"].get("model_path", "yolov8n.pt")
    
    # 1. Create mask for the court color
    court_mask = create_court_mask(original_img, inside_color)
    
    # 2. Find largest contour
    court_contour = detect_largest_court_contour(court_mask)
    if court_contour is None:
        print("No court contour found.")
        cv2.imwrite(output_image_path, annotated_img)
        return
    
    # 3. Approximate polygon
    court_poly = approximate_polygon(court_contour)
    
    # 4. Draw polygon (in white, from config)
    line_color = config["court"].get("line_color", "white")
    # Convert color name to BGR if needed, here we assume white => (255,255,255)
    color_map = {
        "white": (255, 255, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "red": (0, 0, 255)
    }
    bgr_line_color = color_map.get(line_color, (255, 255, 255))
    cv2.polylines(annotated_img, [court_poly], True, bgr_line_color, 2)
    
    # 5. Apply gradient overlay to bounding rectangle of the court polygon
    annotated_img = apply_gradient_overlay(annotated_img, court_poly, max_alpha=0.6)
    
    # 6. Detect people with YOLO
    people_boxes = detect_people_yolo(original_img, model_path=model_path, conf_thresh=0.5)
    
    # 7. Check if each person is inside the court
    court_occupied = 0
    total_people = 0
    for (x1, y1, x2, y2, conf) in people_boxes:
        total_people += 1
        inside = is_person_inside_court(x1, y1, x2, y2, court_poly)
        if inside:
            court_occupied = 1
        color = (0, 255, 0) if inside else (0, 0, 255)
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        label = "Inside Court" if inside else "Outside Court"
        cv2.putText(annotated_img, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # 8. Save output
    cv2.imwrite(output_image_path, annotated_img)
    
    # 9. Print results
    print(f"Court Occupied: {court_occupied}")
    print(f"Total People Detected: {total_people}")

if __name__ == "__main__":
    main()
