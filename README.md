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
- CUDA-compatible GPU (recommended for faster processing, but not required)
- The following Python packages:
  - torch>=1.9.0
  - torchvision>=0.10.0
  - opencv-python>=4.5.0
  - numpy>=1.20.0
  - pandas>=2.0.0
  - shapely>=1.7.0
  - ultralytics>=8.0.0 (For YOLOv5)
  - tqdm>=4.65.0 (For progress bars)
  - Pillow>=10.0.0 (For image processing)
  - matplotlib>=3.7.0 (For visualization)

### Installation

#### Standard Installation (Windows/Mac/Linux)

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

#### Raspberry Pi Installation

For Raspberry Pi, we provide an optimized installation script:

1. Clone repository:
   ```bash
   git clone https://github.com/yourusername/tennis-court-detection.git
   cd tennis-court-detection
   ```

2. Run the Raspberry Pi installer script:
   ```bash
   chmod +x install_dependencies_raspi.sh
   ./install_dependencies_raspi.sh
   ```

The script will:
- Install required system packages using apt
- Set up optimized PyTorch for Raspberry Pi
- Install all Python dependencies
- Download the YOLOv5 model
- Create necessary directories

### Running the System

#### Basic Usage

1. Place a tennis court image in the `images` folder with the name `input.png`
2. Run the script:
   ```bash
   python main.py
   ```
3. Find the output in `images/output.png`

#### Using a Custom Image

Run with a specific input image:
```bash
python main.py --input path/to/your/image.jpg
```

#### Specify Output Location

Save the result to a custom location:
```bash
python main.py --output path/to/save/result.png
```

#### Raspberry Pi Specific

On Raspberry Pi, use python3 explicitly:
```bash
python3 main.py --input images/your_tennis_court.jpg
```

For better performance on Raspberry Pi, you can use the CPU device flag:
```bash
python3 main.py --device cpu
```

If you encounter SSL errors during model download:
```bash
python3 main.py --disable-ssl-verify
```

## Command-Line Arguments

The system supports various command-line arguments for customization:

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to input image | `images/input.png` |
| `--output` | Path for output image | `images/output.png` |
| `--debug` | Enable debug mode with additional outputs | `False` |
| `--quiet` | Reduce console output | `False` |
| `--show-labels` | Show detailed labels on output image | `False` |
| `--show-court-labels` | Show court numbers on output image | `False` |
| `--device` | Device to use for inference (`cpu` or `cuda`) | auto-detect |
| `--disable-ssl-verify` | Disable SSL verification for downloads | `False` |

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

#### Force CPU usage even if CUDA is available
```bash
python main.py --device cpu
```

## Troubleshooting

### Common Issues and Solutions

#### Module Not Found Errors
If you encounter "No module named X" errors:

- For standard systems:
  ```bash
  pip install -r requirements.txt
  ```

- For Raspberry Pi:
  ```bash
  bash install_dependencies_raspi.sh
  ```
  
  Or install specific modules:
  ```bash
  sudo apt update && sudo apt install -y python3-opencv python3-numpy python3-shapely python3-pip
  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```

#### YOLOv5 Model Missing
If you get a model missing error:
```bash
mkdir -p models
curl -L https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -o models/yolov5s.pt
```

#### CUDA / Memory Errors
If you encounter CUDA out-of-memory errors:
```bash
python main.py --device cpu
```

#### SSL Certificate Errors
If you encounter SSL verification issues:
```bash
python main.py --disable-ssl-verify
```

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

## Debug Mode

When running with the `--debug` flag, the system generates visualizations in the `images/debug/` directory to help understand the detection process:

- **blue_mask_raw.png**: Raw blue court area detection
- **green_mask.png**: Green out-of-bounds area detection
- **filtered_court_mask.png**: Filtered court mask after processing
- **courts_numbered.png**: Courts with assigned numbers
- **foot_positions_debug.png**: Debug visualization of detected people's positions
- **color_masks.png**: Combined visualization of all color masks

These visualizations are invaluable for troubleshooting detection issues and understanding the internal workings of the system.

## Model Requirements

The system uses YOLOv5s by default for people detection. The model is automatically downloaded during installation but requires the following:

- **Model File**: `models/yolov5s.pt` (14MB)
- **Minimum RAM**: 2GB recommended
- **GPU**: Optional but recommended for faster processing

You can switch to smaller models like YOLOv5n for resource-constrained devices by modifying the `Config.Model.NAME` setting in the code.