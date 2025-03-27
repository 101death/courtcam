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

### Features

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

1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/tennis-court-detection.git
   cd tennis-court-detection
   ```

2. Create virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Place tennis court image in the `images` folder with the name `input.png`
2. Run script:
   ```bash
   python main.py
   ```
3. Find the output in `images/output.png`

## Command-Line Arguments

Command line arguments to change stuff temporarily:

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input image | `images/input.png` |
| `--output` | Path for output image | `images/output.png` |
| `--debug` | Enable debug mode with additional outputs | `False` |
| `--quiet` | Reduce console output | `False` |
| `--show-labels` | Show detailed labels on output image | `False` |
| `--show-court-labels` | Show court numbers on output image | `False` |

### Example Commands

#### Process a single image with default settings
```bash
python main.py
```

#### Process a specific image with custom output path
```bash
python main.py --input my_courts.png --output results.png
```

#### Enable debug mode for additional visualization outputs
```bash
python main.py --debug
```

#### Generate clean output without detailed labels
```bash
python main.py  # Labels are hidden by default
```

#### Show detailed labels on the output image
```bash
python main.py --show-labels
```

#### Show court numbers on the output image
```bash
python main.py --show-court-labels
```

## Output Visualization

The output visualization is clean and easy to understand:

- **Default Mode**: Shows court outlines, people with color-coded bounding boxes, and minimal labels
- **Detailed Mode**: Add `--show-labels` to see detailed information about each person and their court position

### Color Coding

- **Green**: People in-bounds on a court
- **Orange**: People on court sidelines (out-of-bounds)
- **Red**: People not on any court

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
    SHOW_DETAILED_LABELS = False             # Whether to show detailed labels
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
   - Shows people with color-coded bounding boxes
   - Optional detailed labels with court information

6. **Summary Reporting**:
   - Reports counts of people by court and status in a clean box format
   - Provides overall statistics

## Troubleshooting

### Court Detection Issues

- **Sky detected as courts**: Ensure the green threshold check is working properly
- **Courts not detected**: Adjust HSV ranges in `Config.COURT_COLORS` to match your court colors
- **Courts merged together**: The green areas between courts should be properly detected

### People Detection Issues

- **Missing detections**: Lower detection threshold or check YOLOv5 model setup
- **Wrong position classification**: Use `--debug` to check color masks

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

<div align="center">
  <p>Made with love for tennis enthusiasts</p>
</div>