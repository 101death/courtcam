# Tennis Court Detection System

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python: 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)
![OpenCV](https://img.shields.io/badge/CV-OpenCV-green.svg)
![YOLOv5](https://img.shields.io/badge/Model-YOLOv5-lightgrey.svg)

<div align="center">
  <img src="images/output.png" alt="Tennis Court Detection Example" width="600">
  <p><i>Example tennis court and people detection</i></p>
</div>

## Overview

This system uses computer vision and machine learning to detect tennis courts and track people on them, providing real-time analysis of who is in-bounds or out-of-bounds on each court. Perfect for tennis facilities, coaches, and event organizers who need automated court monitoring.

### Key Features

- **Court Detection**: Automatically identifies tennis courts using color analysis
- **People Tracking**: Detects and tracks people using YOLOv5 object detection
- **Position Analysis**: Determines if people are in-bounds, out-of-bounds, or off-court
- **Court Numbering**: Numbers each court and tracks people per court
- **Rich Visualization**: Outputs images with color-coded courts and player positions
- **Configuration**: Highly customizable with easy-to-modify settings

## Quick Start

### Prerequisites

- Python 3.6+
- CUDA-compatible GPU (recommended for faster processing)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/tennis-court-detection.git
   cd tennis-court-detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Basic Usage

1. Place your tennis court image in the `images` folder with the name `input.png`
2. Run the main script:
   ```bash
   python main.py
   ```
3. Find the output in `images/output.png`

## Example Commands

### Process a single image with default settings
```bash
python main.py
```

### Process a specific image with custom output path
```bash
python main.py --input images/my_court.png --output results/analyzed_court.png
```

### Increase detection confidence for more precise results
```bash
python main.py --confidence 0.65
```

### Process all images in a directory
```bash
python main.py --input images/matches/ --output results/ --batch
```

### Generate only summary statistics without visual output
```bash
python main.py --summary --no-save
```

## Configuration

The system is highly configurable through the `Config` class in `main.py`. Key configuration sections include:

### Court Detection

```python
# Color settings for court detection
COURT_COLORS = {
    "blue": {
        "lower": [90, 40, 40],   # Lower HSV range for blue courts (in-bounds)
        "upper": [120, 255, 255] # Upper HSV range for blue courts
    },
    "green": {
        "lower": [40, 40, 40],   # Lower HSV range for green areas (out-of-bounds)
        "upper": [80, 255, 255]  # Upper HSV range for green areas
    }
}

# Court detection parameters
class Court:
    MIN_AREA = 5000              # Minimum court area in pixels
    MAX_AREA = 150000            # Maximum court area in pixels
```

### Visualization

```python
class Visual:
    PERSON_IN_BOUNDS_COLOR = (0, 255, 0)     # Green for people in-bounds
    PERSON_OUT_BOUNDS_COLOR = (0, 165, 255)  # Orange for people out-of-bounds
    PERSON_OFF_COURT_COLOR = (0, 0, 255)     # Red for people off court
    FONT_SCALE = 0.5                         # Text size for labels
```

### Output Settings

```python
class Output:
    VERBOSE = True               # Show detailed output
    USE_COLOR_OUTPUT = True      # Use colored terminal output
    SHOW_TIMESTAMP = True        # Show timestamps in output
    SUPER_QUIET = False          # Super quiet mode (almost no output)
```

## How It Works

1. **Color Masking**: The system creates masks for blue areas (in-bounds) and green areas (out-of-bounds) using HSV color thresholds.

2. **Court Identification**: 
   - Blue regions adjacent to green are identified as potential courts
   - Small isolated blue regions and those without green nearby are filtered out
   - Each court is assigned a number for tracking

3. **People Detection**:
   - Uses YOLOv5 to detect people in the image
   - Calculates foot position for each person

4. **Position Analysis**:
   - Determines if each person is on a blue area (in-bounds)
   - Checks if they're on a green area (out-of-bounds)
   - Maps each person to a specific court number

5. **Visualization**:
   - Draws court outlines with unique colors per court
   - Labels each person with their court number and status
   - Creates debugging images for analysis

6. **Summary Reporting**:
   - Reports counts of people by court and status
   - Provides overall statistics

## Output Examples

### Terminal Output

```
[12:45:32] ℹ️ Creating color masks for court detection...
[12:45:33] ✅ Found 2 tennis courts
[12:45:34] ℹ️ Looking for people in the image...
[12:45:35] ✅ Found 8 people in the image
[12:45:35] ℹ️ Detection Summary:
[12:45:35] ✅ Found 2 tennis courts
[12:45:35] ✅ Found 8 people in the image
[12:45:35] ℹ️ Court 1: 3 people
[12:45:35] ℹ️ Court 2: 3 people
[12:45:36] ✅ Output image with detection results saved to images/output.png
[12:45:36] ✅ Found 8 people, 3 in court 1, 3 in court 2, image saved to images/output.png
```

### Visual Output

The system generates several output files:

- **Main output**: `images/output.png` - Original image with court outlines and people labeled
- **Debug files** (in `images/debug/`):
  - `blue_mask_raw.png` - Raw blue areas detection
  - `green_mask.png` - Green areas detection
  - `filtered_court_mask.png` - Courts after filtering
  - `courts_numbered.png` - Each court with a unique color and number
  - `foot_positions_debug.png` - People's positions on the courts

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input image | `images/input.png` |
| `--output` | Path for output image | `images/output.png` |
| `--confidence` | Detection confidence threshold | `0.3` |
| `--batch` | Process all images in input directory | `False` |
| `--quiet` | Reduce console output | `False` |
| `--super-quiet` | Show only critical messages | `False` |
| `--summary` | Show only detection summary | `False` |
| `--no-save` | Don't save output images | `False` |

## Troubleshooting

### Court Detection Issues

- **Sky detected as courts**: Ensure the green threshold check is working properly
- **Courts not detected**: Adjust HSV ranges in `Config.COURT_COLORS` to match your court colors
- **Courts merged together**: The green areas between courts should be properly detected

### People Detection Issues

- **Missing detections**: Decrease the confidence threshold with `--confidence 0.25`
- **False positives**: Increase the confidence threshold with `--confidence 0.5`
- **Wrong position classification**: Check the debug images to verify color masks

## Advanced Usage

### Processing Multiple Courts

The system automatically detects and numbers multiple courts in a single image. Each court gets a unique color and ID.

### Customizing Color Thresholds

For courts with different colors:

```python
# Example for red clay courts
"red_clay": {
    "lower": [0, 100, 100],
    "upper": [10, 255, 255],
    "lower2": [170, 100, 100],  # Wrap-around for red hue
    "upper2": [180, 255, 255]
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or support, please open an issue on the GitHub repository.

---

<div align="center">
  <p>Made with love for tennis enthusiasts</p>
</div>