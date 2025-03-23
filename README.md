# Tennis Court Camera System

This system detects people on a tennis court using computer vision and YOLOv5 object detection, with visual highlights to show who is on or off the court.

## Setup

1. Ensure you have Python 3.6+ installed
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Place your tennis court image in the `images` folder with the name `input.png`
2. Run the script:
   ```
   python main.py
   ```
3. The script will automatically download the YOLOv5 model if needed
4. The output image with detections will be saved as `images/output.png`

## Configuration

The script has many configurable settings at the top of the file that you can easily modify:

### Paths and Directories
```python
DEFAULT_MODELS_DIR = "models"                # Directory to store models
DEFAULT_IMAGES_DIR = "images"                # Directory to store images
DEFAULT_INPUT_IMAGE = "input.png"            # Default input image name
DEFAULT_OUTPUT_IMAGE = "output.png"          # Default output image name
```

### Model Settings
```python
MODEL_NAME = "yolov5s"                       # YOLOv5 model size (yolov5s, yolov5m, yolov5l, etc.)
DEFAULT_CONFIDENCE = 0.5                     # Default detection confidence threshold
```

### Court Detection Settings
```python
COURT_COLOR_RANGES = {
    "green": {
        "lower": [40, 40, 40],              # Lower HSV range for green courts
        "upper": [80, 255, 255]             # Upper HSV range for green courts
    },
    "blue": {
        "lower": [90, 40, 40],              # Lower HSV range for blue courts
        "upper": [130, 255, 255]            # Upper HSV range for blue courts
    }
}
CONTOUR_APPROX_FACTOR = 0.02                 # Contour approximation factor
```

### Visualization Settings
```python
COURT_OUTLINE_COLOR = (0, 255, 0)            # Green
COURT_OUTLINE_THICKNESS = 2                  # Line thickness
ON_GREEN_COLOR = (0, 255, 0)                 # Green for people on green court
ON_BLUE_COLOR = (255, 191, 0)                # Deep sky blue for people on blue court
ON_COURT_OTHER_COLOR = (0, 255, 255)         # Yellow for people on other court areas
OFF_COURT_COLOR = (0, 0, 255)                # Red for people outside the court
TEXT_COLOR = (255, 255, 255)                 # White
FONT_SCALE = 0.5                             # Text size
TEXT_THICKNESS = 2                           # Text thickness
DRAW_COURT_OUTLINE = False                   # Whether to draw court outline (default: False)
```

### Terminal Output Settings
```python
VERBOSE = True                               # Show detailed output
USE_COLOR_OUTPUT = True                      # Use colored terminal output
SHOW_TIMESTAMP = True                        # Show timestamps in output
```

## Command Line Options

You can customize the behavior with these arguments:

```
python main.py --input your_image.jpg --output results.jpg --confidence 0.6 --quiet
```

Available options:
- `--input`: Path to input image or directory of images (default: images/input.png)
- `--output`: Path to save output image or directory for multiple images (default: images/output.png)
- `--models-dir`: Directory for storing models (default: models)
- `--confidence`: Detection confidence threshold (default: 0.5)
- `--quiet`: Suppress detailed console output
- `--super-quiet`: Suppress almost all output except errors and success messages
- `--summary`: Show only the summary count of people detected (one-line output)
- `--no-save`: Do not save the output image (just analyze)
- `--batch`: Process all images in the input directory
- `--debug-masks`: Save visualization images of the detected green and blue court areas
- `--show-court-outline`: Display the green court outline in the output image
- `--auto-convert`: Automatically convert JPG and WebP images to PNG format before processing

### Output Modes

The script has several output modes:
1. **Normal** - Shows all information, progress updates, and timestamps
2. **Quiet** (`--quiet`) - Shows less information but keeps important status updates
3. **Super Quiet** (`--super-quiet`) - Shows only errors and success messages
4. **Summary** (`--summary`) - Shows just a single line with the count of people detected

### Examples

Basic usage with default settings:
```
python main.py
```

Analyze an image without saving output:
```
python main.py --input my_image.jpg --no-save
```

Quick people count only:
```
python main.py --summary
```

Custom model directory and confidence:
```
python main.py --models-dir /path/to/models --confidence 0.7
```

Process all images in a directory:
```
python main.py --input images/folder --output results/ --batch
```

Process a batch of images with summary only:
```
python main.py --input images/folder --batch --summary
```

Process an image and save debug visualization of court areas:
```
python main.py --debug-masks
```

Process with court outline shown:
```
python main.py --show-court-outline
```

Process a JPG image with automatic conversion to PNG:
```
python main.py --input images/court.jpg --auto-convert
```

## How It Works

1. The system detects the tennis court using color segmentation in HSV space, identifying both green and blue areas
2. It detects people using YOLOv5 object detection
3. It determines where people are located with detailed classification:
   - On green court area
   - On blue court area
   - On other court areas (lines, net, etc.)
   - Off court completely
4. It generates an output image with color-coded bounding boxes:
   - Green: People on green court areas
   - Blue/Cyan: People on blue court areas
   - Yellow: People on other court areas (lines, transitions)
   - Red: People completely outside the court
5. Each detection shows the confidence score and specific location

## Troubleshooting

If you encounter issues with model downloading:
1. Ensure you have a working internet connection
2. If SSL issues persist, the script temporarily disables SSL verification
3. You can manually download the YOLOv5 model from https://github.com/ultralytics/yolov5/releases/ and place it in the `models` directory

If the court detection doesn't work well:
- Use the `--debug-masks` option to visualize exactly what areas are being detected as green or blue
- Adjust the HSV color ranges in the COURT_COLOR_RANGES settings to match your court colors
- For indoor vs outdoor courts with different lighting, you may need different settings 