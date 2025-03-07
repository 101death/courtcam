def detect_court_lines(frame, court_roi_info, court_roi, court_mask_roi):
    if court_roi is None or court_roi.shape[0] == 0 or court_roi.shape[1] == 0:
        print("Court ROI is invalid or empty.")
        return frame, []

    roi_x, roi_y, roi_w, roi_h = court_roi_info
    hsv_roi = cv2.cvtColor(court_roi, cv2.COLOR_BGR2HSV)
    lower_white, upper_white = WHITE_LINE_COLOR
    white_mask = cv2.inRange(hsv_roi, np.array(lower_white), np.array(upper_white))
    
    # Morphological cleaning and enhancement
    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.bilateralFilter(white_mask, 9, 75, 75)
    white_mask = cv2.dilate(white_mask, np.ones((5, 5), np.uint8), iterations=1)
    
    # Use adaptive thresholding for better edge extraction
    adaptive_thresh = cv2.adaptiveThreshold(white_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    edges = cv2.Canny(adaptive_thresh, config["canny_threshold1"], config["canny_threshold2"])
    
    # Option 1: Probabilistic Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, config["hough_threshold"], 
                            minLineLength=config["min_line_length"], 
                            maxLineGap=config["max_line_gap"])
    
    # Option 2: Alternatively, use LSD
    # lsd = cv2.createLineSegmentDetector(0)
    # lines_lsd, _, _, _ = lsd.detect(white_mask)
    # lines = np.array([[[int(x1), int(y1), int(x2), int(y2)]] for [[x1, y1, x2, y2]] in lines_lsd]) if lines_lsd is not None else None
    
    # Process detected lines as before, categorizing horizontal/vertical, etc.
    horizontal_lines = []
    vertical_lines = []
    all_lines = []
    top_line = bottom_line = left_line = right_line = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 > x2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            line_info = {'points': (x1, y1, x2, y2), 'angle': angle, 'length': length,
                         'y_avg': (y1 + y2) / 2, 'x_avg': (x1 + x2) / 2}
            if angle < 30 or angle > 150:
                horizontal_lines.append(line_info)
            elif 60 < angle < 120:
                vertical_lines.append(line_info)
            all_lines.append(line_info)
        
        # Process horizontal lines
        if horizontal_lines:
            horizontal_lines.sort(key=lambda x: x['length'], reverse=True)
            max_length = horizontal_lines[0]['length']
            long_h_lines = [l for l in horizontal_lines if l['length'] > 0.5 * max_length]
            long_h_lines.sort(key=lambda x: x['y_avg'])
            top_line = long_h_lines[0]['points']
            if len(long_h_lines) > 1:
                bottom_line = long_h_lines[-1]['points']
        
        # Process vertical lines
        if vertical_lines:
            vertical_lines.sort(key=lambda x: x['length'], reverse=True)
            max_length = vertical_lines[0]['length']
            long_v_lines = [l for l in vertical_lines if l['length'] > 0.5 * max_length]
            long_v_lines.sort(key=lambda x: x['x_avg'])
            left_line = long_v_lines[0]['points']
            if len(long_v_lines) > 1:
                right_line = long_v_lines[-1]['points']
    
    # Create an overlay for drawing
    boundary_image = np.zeros_like(court_roi)
    court_boundary_points = []
    
    if top_line:
        cv2.line(boundary_image, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (0, 255, 0), config["line_thickness"])
        court_boundary_points += [(top_line[0] + roi_x, top_line[1] + roi_y), (top_line[2] + roi_x, top_line[3] + roi_y)]
    if bottom_line:
        cv2.line(boundary_image, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (0, 255, 0), config["line_thickness"])
        court_boundary_points += [(bottom_line[0] + roi_x, bottom_line[1] + roi_y), (bottom_line[2] + roi_x, bottom_line[3] + roi_y)]
    if left_line:
        cv2.line(boundary_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 0, 255), config["line_thickness"])
        court_boundary_points += [(left_line[0] + roi_x, left_line[1] + roi_y), (left_line[2] + roi_x, left_line[3] + roi_y)]
    if right_line:
        cv2.line(boundary_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), config["line_thickness"])
        court_boundary_points += [(right_line[0] + roi_x, right_line[1] + roi_y), (right_line[2] + roi_x, right_line[3] + roi_y)]
    
    # Optionally, connect the estimated corners
    if top_line and bottom_line and left_line and right_line:
        top_left = (left_line[0], top_line[1])
        top_right = (right_line[0], top_line[1])
        bottom_left = (left_line[0], bottom_line[1])
        bottom_right = (right_line[0], bottom_line[1])
        cv2.line(boundary_image, top_left, top_right, (255, 0, 0), config["line_thickness"])
        cv2.line(boundary_image, bottom_left, bottom_right, (255, 0, 0), config["line_thickness"])
        cv2.line(boundary_image, top_left, bottom_left, (255, 0, 0), config["line_thickness"])
        cv2.line(boundary_image, top_right, bottom_right, (255, 0, 0), config["line_thickness"])
    
    lines_detected = cv2.addWeighted(court_roi, 1, boundary_image, 1, 0)
    frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = lines_detected
    cv2.putText(frame, "Tennis Court Boundary", (roi_x, roi_y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
    
    return frame, court_boundary_points
