# Tennis Court Detector

Detect tennis courts and people playing on them using computer vision and YOLOv8 object detection.

## Quick Start

1. Run the setup script to install dependencies and configure settings:

```bash
chmod +x setup.sh
./setup.sh
```

2. Activate the virtual environment:

```bash
source venv/bin/activate
```

3. Run the detector with camera:

```bash
python main.py
```

4. Run with test image instead of camera:

```bash
python main.py --no-camera
```
5. Run the test suite:

```bash
pytest
```


## Configuration

The `config.json` file contains all customizable settings for the application. You can edit this file directly or run the setup script to configure it interactively.

### Camera Settings

```json
"Camera": {
  "width": 1280,
  "height": 720
}
```

**Raspberry Pi Camera 3 Supported Resolutions:**
- 640x480 (VGA - Standard definition)
- 1280x720 (HD - High definition, recommended for most uses)
- 1920x1080 (Full HD - 1080p)
- 2304x1296 (3MP - Higher resolution)
- 4608x2592 (12MP - Maximum resolution, may impact performance)

Note: These resolution settings are specifically optimized for the Raspberry Pi Camera 3. Other camera modules may support different resolutions.

### Model Settings

```json
"Model": {
  "NAME": "yolov8n",
  "CONFIDENCE": 0.1,
  "IOU": 0.45,
  "CLASSES": [0]
}
```

- `NAME`: Model size - smaller is faster, larger is more accurate
  - `yolov8n` - Nano (smallest and fastest)
  - `yolov8s` - Small (good balance)
  - `yolov8m` - Medium
  - `yolov8l` - Large
  - `yolov8x` - XLarge (most accurate but slowest)
  
- `CONFIDENCE`: Detection threshold (0.1-0.9)
  - Lower values detect more objects but with more false positives
  - Higher values are more selective but might miss objects
  
- `CLASSES`: Object classes to detect 
  - `[0]` - People only
  - `[0, 2, 5]` - Multiple classes (see COCO dataset for IDs)

### Output Settings

```json
"Output": {
  "VERBOSE": false,
  "SUPER_QUIET": false,
  "SUMMARY_ONLY": false,
  "EXTRA_VERBOSE": false
}
```

- `VERBOSE`: Show detailed output during processing
- `SUPER_QUIET`: Minimal output (errors only)
- `SUMMARY_ONLY`: Show only the final results summary
- `EXTRA_VERBOSE`: Show maximum debugging information

### Debug Mode

```json
"DEBUG_MODE": false
```

When enabled, saves additional debug images and information to the debug folder.

### Multiprocessing

```json
"MultiProcessing": {
  "ENABLED": true,
  "NUM_PROCESSES": 4
}
```

- `ENABLED`: Enable/disable parallel processing
- `NUM_PROCESSES`: Number of CPU cores to use (recommended: number of cores - 1)

### Visual Settings

```json
"Visual": {
  "COURT_OUTLINE_COLOR": [0, 255, 0],
  "COURT_OUTLINE_THICKNESS": 4,
  "PERSON_IN_BOUNDS_COLOR": [0, 255, 0],
  "PERSON_OUT_BOUNDS_COLOR": [0, 165, 255],
  "PERSON_OFF_COURT_COLOR": [0, 0, 255],
  "DRAW_COURT_OUTLINE": true,
  "SHOW_COURT_NUMBER": true,
  "SHOW_DETAILED_LABELS": true
}
```

These settings control the appearance of the output image, including colors and labels.

### Court Detection Settings

```json
"Court": {
  "MIN_AREA": 3000,
  "MAX_AREA": 200000,
  "MIN_SCORE": 0.5,
  "MIN_ASPECT_RATIO": 1.0,
  "MAX_ASPECT_RATIO": 4.0
}
```

These settings control the sensitivity of court detection. Adjust if courts are not being detected correctly.

## Command-line Arguments

The main program supports various command-line arguments:

```
python main.py [options]
```

- `--no-camera`: Skip camera capture and use default input image
- `--input PATH`: Specify input image path
- `--output PATH`: Specify output image path
- `--model NAME`: Specify YOLO model to use
- `--debug`: Enable debug mode
- `--quiet`: Reduce console output
- `--show-labels`: Show detailed labels on output image
- `--processes NUM`: Set number of processes for multiprocessing
- `--court-positions LIST`: Manually set court corners as `x1,y1,...,x8,y8` pairs separated by semicolons
- `--set-courts-gui`: Launch a Tkinter-based editor. A sidebar shows permanent
  instructions. Use the `+` and `-` buttons to add or remove courts. Points are
  placed by clicking on the image and the shape automatically closes when the
  final click is near the first. Buttons include simple hover animations and
  `Done`/`Finish` complete the selection. The program loads the image and exits
  after saving the positions. The editor scales the image to fit within an
  800x600 window so the whole frame is visible even on smaller screens.

When court positions are provided or saved in `config.json`, court detection is skipped on subsequent runs.

### Performance Testing

The program includes a comprehensive performance testing mode to find the fastest configuration:

```
python main.py --test-mode [options]
```

Performance test options:
- `--test-mode`: Run performance tests to find the fastest configuration
- `--test-output-dir DIR`: Directory to save test results (default: test_results)
- `--test-quick`: Run a quick test with limited models and configurations
- `--test-models LIST`: Comma-separated list of models to test (e.g., "yolov8n,yolov5n")
- `--test-with-resolution WxH`: Test only with the specified resolution (e.g., "1280x720")
- `--test-with-device cpu|cuda`: Test only with the specified device

Example:
```bash
# Quick test with smallest models
python main.py --test-mode --test-quick --input images/input.png

# Test specific models at a specific resolution on specific device
python main.py --test-mode --test-models yolov8n,yolov5n --test-with-resolution 640x480 --test-with-device cpu
```

After testing, the program will offer to apply the fastest configuration and save it to `config.json` for future use.

## Direct Camera Usage

You can test the camera directly:

```bash
python camera.py [output_file] [resolution]
```

Example:
```bash
python camera.py test.jpg HD
```

Available resolution shortcuts: VGA, HD, FULL_HD, 3MP, 12MP

## Requirements

- Python 3.7+
- OpenCV
- PyTorch
- Ultralytics (for YOLOv8)
- Raspberry Pi Camera module (for Raspberry Pi)


## API Usage

See [API.md](API.md) for instructions on using the FastAPI server. In addition to the `/courts` endpoint there are `/status` and `/court_count` endpoints for device info and just the number of courts.
