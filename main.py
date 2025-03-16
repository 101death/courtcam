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

# Helper: safely read an image or return a blank image if not found.
def safe_imread(path, shape=(300,300,3)):
    img = cv2.imread(path)
    if img is None:
        return np.zeros(shape, dtype=np.uint8)
    return img

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

# ----- Geometry helper functions -----
def compute_line_params(x1, y1, x2, y2):
    """
    Returns slope (m) and intercept (b).
    If vertical line, returns (None, x).
    """
    if abs(x2 - x1) < 1e-6:
        return None, (x1 + x2) / 2.0
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return m, b

def line_angle_and_length(x1, y1, x2, y2):
    """
    Returns (angle_degrees, length) for the line segment.
    Angle is in degrees from -180..+180. We'll mostly see -90..+90 for typical slope usage.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = np.hypot(dx, dy)
    angle_deg = np.degrees(np.arctan2(dy, dx))  # range -180..180
    return angle_deg, length

def extend_line(line, width, height):
    """
    Extend a line (dict with keys: m, b or x) to the image borders (0..width, 0..height).
    """
    if line["m"] is None:
        # Vertical line
        x = line["x"]
        return (int(x), 0), (int(x), height)
    else:
        m = line["m"]
        b = line["b"]
        pts = []
        # Left border (x=0)
        y = m * 0 + b
        if 0 <= y <= height:
            pts.append((0, int(y)))
        # Right border (x=width)
        y = m * width + b
        if 0 <= y <= height:
            pts.append((width, int(y)))
        # Top border (y=0)
        if abs(m) > 1e-6:
            x = -b / m
            if 0 <= x <= width:
                pts.append((int(x), 0))
        # Bottom border (y=height)
        if abs(m) > 1e-6:
            x = (height - b) / m
            if 0 <= x <= width:
                pts.append((int(x), height))
        if len(pts) >= 2:
            # pick the pair that yields the maximum distance
            max_dist = 0
            best_pair = (pts[0], pts[1])
            for i in range(len(pts)):
                for j in range(i+1, len(pts)):
                    dist = np.hypot(pts[i][0]-pts[j][0], pts[i][1]-pts[j][1])
                    if dist > max_dist:
                        max_dist = dist
                        best_pair = (pts[i], pts[j])
            return best_pair
        else:
            # fallback: original endpoints
            return (int(line["x1"]), int(line["y1"])), (int(line["x2"]), int(line["y2"]))

def line_intersection(lineA, lineB):
    """
    Compute intersection of two lines (each dict with keys: m, b or x).
    Return (x, y) or None if parallel / no intersection.
    """
    if lineA["m"] is None and lineB["m"] is None:
        return None
    if lineA["m"] is None:
        x = lineA["x"]
        y = lineB["m"] * x + lineB["b"]
        return (x, y)
    if lineB["m"] is None:
        x = lineB["x"]
        y = lineA["m"] * x + lineA["b"]
        return (x, y)
    if abs(lineA["m"] - lineB["m"]) < 1e-6:
        return None
    x = (lineB["b"] - lineA["b"]) / (lineA["m"] - lineB["m"])
    y = lineA["m"] * x + lineA["b"]
    return (x, y)

def merge_collinear_lines(lines, angle_thresh, gap_thresh):
    """
    Repeatedly merges lines that are close and have angle difference < angle_thresh.
    gap_thresh = how close endpoints must be to merge lines in the same group.
    """

    def can_merge(L1, L2):
        # They must have a small angle difference
        if abs(L1["angle"] - L2["angle"]) > angle_thresh:
            return False
        # Check if the lines are "close" in the plane
        endpoints1 = [(L1["x1"], L1["y1"]), (L1["x2"], L1["y2"])]
        endpoints2 = [(L2["x1"], L2["y1"]), (L2["x2"], L2["y2"])]
        for p1 in endpoints1:
            for p2 in endpoints2:
                dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
                if dist < gap_thresh:
                    return True
        return False

    merged = True
    while merged:
        merged = False
        new_lines = []
        used = [False]*len(lines)
        for i in range(len(lines)):
            if used[i]:
                continue
            L1 = lines[i]
            found_merge = False
            for j in range(i+1, len(lines)):
                if used[j]:
                    continue
                L2 = lines[j]
                if can_merge(L1, L2):
                    used[i] = True
                    used[j] = True
                    # Merge them by re-fitting a line through the furthest pair of endpoints
                    all_points = [
                        (L1["x1"], L1["y1"]), (L1["x2"], L1["y2"]),
                        (L2["x1"], L2["y1"]), (L2["x2"], L2["y2"])
                    ]
                    best_dist = 0
                    best_pair = (all_points[0], all_points[1])
                    for p1 in all_points:
                        for p2 in all_points:
                            d = np.hypot(p1[0]-p2[0], p1[1]-p2[1])
                            if d > best_dist:
                                best_dist = d
                                best_pair = (p1, p2)
                    x1m, y1m = best_pair[0]
                    x2m, y2m = best_pair[1]
                    m_merged, b_merged = compute_line_params(x1m, y1m, x2m, y2m)
                    angle_merged, length_merged = line_angle_and_length(x1m, y1m, x2m, y2m)

                    if m_merged is None:
                        # vertical
                        new_lines.append({
                            "x1": x1m, "y1": y1m,
                            "x2": x2m, "y2": y2m,
                            "m": None,
                            "x": b_merged,  # store the vertical x
                            "b": None,
                            "angle": angle_merged,
                            "length": length_merged
                        })
                    else:
                        new_lines.append({
                            "x1": x1m, "y1": y1m,
                            "x2": x2m, "y2": y2m,
                            "m": m_merged,
                            "b": b_merged,
                            "angle": angle_merged,
                            "length": length_merged
                        })
                    found_merge = True
                    merged = True
                    break
            if not found_merge:
                new_lines.append(L1)
                used[i] = True
        lines = new_lines
    return lines

def fill_polygon_nonconvex(intersections, width, height, output_folder):
    """
    Instead of taking a convex hull, we sort the intersection points by angle
    around their centroid and fill that polygon. This can produce a concave shape
    that more accurately matches the real court if lines define such a shape.
    """
    if len(intersections) < 3:
        return np.zeros((height, width), dtype=np.uint8)

    cx = np.mean([p[0] for p in intersections])
    cy = np.mean([p[1] for p in intersections])

    def angle_from_centroid(pt):
        return np.arctan2(pt[1] - cy, pt[0] - cx)

    sorted_pts = sorted(intersections, key=angle_from_centroid)
    polygon = np.array(sorted_pts, dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    # For debugging, you can save the polygon image
    # But let's do it by default:
    cv2.imwrite(os.path.join(output_folder, 'final_mask_nonconvex.png'), mask)
    return mask

# ----- Hybrid approach with geometry-based bridging + optional line skipping -----
def detect_court_fixed(image, config, advanced_config):
    """
    1) Filter out non-court lines (threshold + morphological opening).
    2) HoughLinesP to find raw lines.
    3) (Optional) skip lines above a top_cutoff or near the net if remove_net_lines = True
    4) Keep lines whose midpoints lie in the largest contour.
    5) Geometric bridging (merge collinear lines).
    6) Extend lines, find intersections, build a non-convex polygon (sorted by angle).
    """
    output_folder = config['output_folder']
    os.makedirs(output_folder, exist_ok=True)
    height, width = image.shape[:2]

    # 1) Preprocessing
    print_step("Preprocessing & Edge Detection")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(output_folder, '01_grayscale.png'), gray)

    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    cv2.imwrite(os.path.join(output_folder, '02_bilateral_filtered.png'), filtered)

    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cv2.imwrite(os.path.join(output_folder, '03_adaptive_threshold.png'), thresh)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imwrite(os.path.join(output_folder, '04_morphological_opening.png'), opening)

    edges = cv2.Canny(opening, 50, 150, apertureSize=3)
    cv2.imwrite(os.path.join(output_folder, '05_canny_edges.png'), edges)

    # 2) Detect Raw Lines
    print_step("Detecting Raw Lines (HoughLinesP)")
    ls_params = advanced_config["line_settings"]
    raw = cv2.HoughLinesP(
        edges, 
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=ls_params.get("min_line_length", 100),
        maxLineGap=ls_params.get("max_line_gap", 20)
    )
    if raw is None:
        print_error("No lines detected at all.")
        return np.zeros((height, width), dtype=np.uint8)

    raw_line_mask = np.zeros_like(gray)
    lines_list = []
    for line in raw:
        x1, y1, x2, y2 = line[0]
        cv2.line(raw_line_mask, (x1, y1), (x2, y2), 255, 2)
        m, b = compute_line_params(x1, y1, x2, y2)
        angle, length = line_angle_and_length(x1, y1, x2, y2)
        if m is None:
            # vertical
            lines_list.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "m": None,
                "x": b,   # for vertical lines, b is actually x
                "b": None,
                "angle": angle,
                "length": length
            })
        else:
            lines_list.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "m": m,
                "b": b,
                "angle": angle,
                "length": length
            })
    cv2.imwrite(os.path.join(output_folder, '06_raw_line_mask.png'), raw_line_mask)

    # (Optional) skip top lines or net lines
    top_cutoff = advanced_config["optional_filters"].get("top_cutoff", 0)  # e.g., 50 or 100
    remove_net_lines = advanced_config["optional_filters"].get("remove_net_lines", False)
    net_angle_thresh = advanced_config["optional_filters"].get("net_angle_thresh", 5)
    net_y_thresh = advanced_config["optional_filters"].get("net_y_thresh", 30)

    filtered_lines_stage1 = []
    for L in lines_list:
        midx = (L["x1"] + L["x2"]) / 2.0
        midy = (L["y1"] + L["y2"]) / 2.0

        # 1) If top_cutoff > 0, skip lines whose midpoint is above that y
        if top_cutoff > 0 and midy < top_cutoff:
            continue

        # 2) If remove_net_lines, skip lines that are near horizontal around mid-court
        # e.g., if the net is near height/2, we skip lines with angle ~ 0 deg and
        # midpoint near height/2 ± net_y_thresh
        if remove_net_lines:
            # angle near 0 => abs(L["angle"]) < net_angle_thresh
            # midpoint near mid => abs(midy - height/2) < net_y_thresh
            if abs(L["angle"]) < net_angle_thresh and abs(midy - (height/2)) < net_y_thresh:
                continue

        filtered_lines_stage1.append(L)

    # 3) Keep lines whose midpoints lie in largest contour
    print_step("Finding largest contour in raw_line_mask, filtering lines by midpoint location")
    contours, _ = cv2.findContours(raw_line_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print_error("No contours found in line mask.")
        return np.zeros((height, width), dtype=np.uint8)
    biggest = max(contours, key=cv2.contourArea)
    contour_mask = np.zeros_like(gray)
    cv2.drawContours(contour_mask, [biggest], -1, 255, -1)
    cv2.imwrite(os.path.join(output_folder, '07_court_contour_mask.png'), contour_mask)

    final_lines = []
    for L in filtered_lines_stage1:
        midx = (L["x1"] + L["x2"]) / 2.0
        midy = (L["y1"] + L["y2"]) / 2.0
        if cv2.pointPolygonTest(biggest, (midx, midy), False) >= 0:
            final_lines.append(L)

    if len(final_lines) < 2:
        print_error("Not enough lines after contour + optional filters.")
        return np.zeros((height, width), dtype=np.uint8)

    # 4) Geometric bridging
    print_step("Merging collinear lines (geometric bridging, no morphological bridging)")
    angle_threshold = ls_params.get("angle_threshold", 5)
    gap_threshold   = ls_params.get("gap_threshold", 30)
    merged_lines = merge_collinear_lines(final_lines, angle_threshold, gap_threshold)

    # Debug: show merged lines
    merged_mask = np.zeros_like(gray)
    for L in merged_lines:
        cv2.line(merged_mask, (int(L["x1"]), int(L["y1"])), (int(L["x2"]), int(L["y2"])), 255, 2)
    cv2.imwrite(os.path.join(output_folder, '08_merged_lines.png'), merged_mask)

    # 5) Extend lines, find intersections, build non-convex polygon
    print_step("Extending lines to borders, computing intersections, filling polygon (non-convex).")
    extended = []
    for L in merged_lines:
        pt1, pt2 = extend_line(L, width, height)
        m_ext, b_ext = compute_line_params(pt1[0], pt1[1], pt2[0], pt2[1])
        angle_ext, length_ext = line_angle_and_length(pt1[0], pt1[1], pt2[0], pt2[1])
        if m_ext is None:
            # vertical
            extended.append({
                "x1": pt1[0], "y1": pt1[1],
                "x2": pt2[0], "y2": pt2[1],
                "m": None,
                "x": b_ext,
                "b": None,
                "angle": angle_ext,
                "length": length_ext
            })
        else:
            extended.append({
                "x1": pt1[0], "y1": pt1[1],
                "x2": pt2[0], "y2": pt2[1],
                "m": m_ext,
                "b": b_ext,
                "angle": angle_ext,
                "length": length_ext
            })

    # Gather all intersection points
    intersections = []
    for i in range(len(extended)):
        for j in range(i+1, len(extended)):
            inter = line_intersection(extended[i], extended[j])
            if inter is not None:
                x, y = inter
                if 0 <= x <= width and 0 <= y <= height:
                    intersections.append((int(x), int(y)))

    if len(intersections) < 3:
        print_error("Not enough intersections to form a polygon.")
        return np.zeros((height, width), dtype=np.uint8)

    # Non-convex fill
    final_mask = fill_polygon_nonconvex(intersections, width, height, output_folder)
    return final_mask

# ----- Main script -----
print_header("Improved Court Camera Analysis")

print_step("Ensuring model files are present...")
ensure_model_files()
print_success("Model files are ready.")

if not os.path.exists('input.png'):
    print_error("input.png not found. Please provide the input image.")
    exit(1)

print_step("Loading configuration files...")
try:
    with open('config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print_error("config.json not found. Creating default config.")
    config = {
        "court_color": "white",
        "output_folder": "images",
        "high_contrast_mode": True
    }
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)

try:
    with open('advanced.json', 'r') as f:
        advanced_config = json.load(f)
except FileNotFoundError:
    print_error("advanced.json not found. Creating default config.")
    advanced_config = {
        "line_settings": {
            "min_line_length": 100,
            "max_line_gap": 20,
            "angle_threshold": 5,
            "gap_threshold": 30
        },
        "optional_filters": {
            "top_cutoff": 0,
            "remove_net_lines": False,
            "net_angle_thresh": 5,
            "net_y_thresh": 30
        }
    }
    with open('advanced.json', 'w') as f:
        json.dump(advanced_config, f, indent=2)

if 'output_folder' not in config:
    config['output_folder'] = 'images'
os.makedirs(config['output_folder'], exist_ok=True)

print_step("Loading input image...")
image = cv2.imread('input.png')
if image is None:
    print_error("Could not load input.png. Please ensure it's a valid image.")
    exit(1)

print_step("Detecting court with advanced line filtering and non-convex fill...")
court_mask = detect_court_fixed(image, config, advanced_config)

print_step("Loading YOLOv3 model for person detection...")
try:
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
except Exception as e:
    print_error(f"Error loading YOLOv3 model: {e}")
    exit(1)

with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')
output_layers = net.getUnconnectedOutLayersNames()

print_step("Detecting people in the image...")
height, width = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0,0,0), True, crop=False)
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
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

print_step("Annotating output image...")
output_image = image.copy()
court_overlay = np.zeros_like(image)
court_overlay[court_mask == 255] = (0, 255, 255)
alpha = 0.3
cv2.addWeighted(court_overlay, alpha, output_image, 1 - alpha, 0, output_image)

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        feet_x = x + w//2
        feet_y = y + h
        inside = (0 <= feet_x < width and 0 <= feet_y < height and court_mask[feet_y, feet_x] == 255)
        color = (0,255,0) if inside else (0,0,255)
        label = "Inside" if inside else "Outside"
        cv2.rectangle(output_image, (x,y), (x+w,y+h), color, 2)
        cv2.putText(output_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

print_step("Saving output image...")
cv2.imwrite(os.path.join(config['output_folder'], 'output.png'), output_image)

# Build a debug visualization of intermediate steps.
process_images = [
    safe_imread(os.path.join(config['output_folder'], '01_grayscale.png')),
    safe_imread(os.path.join(config['output_folder'], '05_canny_edges.png')),
    safe_imread(os.path.join(config['output_folder'], '06_raw_line_mask.png')),
    safe_imread(os.path.join(config['output_folder'], '08_merged_lines.png')),
    safe_imread(os.path.join(config['output_folder'], 'final_mask_nonconvex.png'))
]

h0, w0 = process_images[0].shape[:2]
for i in range(len(process_images)):
    if process_images[i].shape[:2] != (h0, w0):
        process_images[i] = cv2.resize(process_images[i], (w0, h0))
horizontal_stack = np.hstack(process_images)
cv2.imwrite(os.path.join(config['output_folder'], 'process_steps.png'), horizontal_stack)

print_success(f"Output image saved to {os.path.join(config['output_folder'], 'output.png')}")
print_success(f"Debug steps saved to {os.path.join(config['output_folder'], 'process_steps.png')}")
print_header("Analysis Complete")
