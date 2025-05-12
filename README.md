# Tennis Court Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](#) [![OpenCV](https://img.shields.io/badge/CV-OpenCV-green.svg)](#) [![YOLO](https://img.shields.io/badge/Models-YOLOv5_|_YOLOv8+-lightgrey.svg)](#)

> Automated tennis court and player detection leveraging OpenCV and YOLO models. Optimized for Raspberry Pi but runs on standard Linux, macOS, and Windows.

---

## 📋 Table of Contents

* [Installation](#installation)
* [Usage](#usage)
* [Features](#features)
* [Configuration](#configuration)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)

---

## 🚀 Installation

Follow these steps to get the system running on your machine.

**Prerequisites:**

*   **Python:** Version 3.8 or later is recommended.
*   **Git:** For cloning the repository.
*   **pip:** Python package installer (usually comes with Python).
*   **(Optional) Raspberry Pi Camera:** If you intend to capture images directly from a Pi.
*   **Internet Connection:** Required for downloading dependencies and models.

**Installation Steps:**

1.  **Clone the Repository:**
    Open your terminal or command prompt and run:
    ```bash
    git clone https://github.com/101death/courtcam.git
    cd courtcam
    ```

2.  **Create a Virtual Environment (Recommended):**
    Using a virtual environment keeps dependencies isolated.
    ```bash
    # For Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows (Command Prompt/PowerShell)
    python -m venv venv
    .\venv\Scripts\activate 
    ```
    *You should see `(venv)` appear at the beginning of your terminal prompt.* 

3.  **Install Python Dependencies:**
    Install the required Python libraries using pip and the `requirements.txt` file.
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *This step might take a few minutes as it downloads and installs packages like OpenCV, PyTorch, and Ultralytics.* 

4.  **Download Default YOLO Model:**
    The script will attempt to download the default model (`yolov8x.pt`) automatically on the first run if it's missing. However, you can pre-download it:
    ```bash
    mkdir -p models 
    # Download the model (example using wget)
    # Replace with curl or manual download if needed
    wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt -O models/yolov8x.pt 
    ```
    *You can specify other models using the `--model` flag during usage (see below).* 

5.  **(Raspberry Pi Only) Camera Setup:**
    *   Ensure your camera is physically connected correctly.
    *   Enable the camera interface:
        ```bash
        sudo raspi-config
        ```
        Navigate to `Interface Options` -> `Camera` -> `Enable`. Reboot if prompted.
    *   The necessary Python libraries (`picamera2` or legacy `picamera`) should have been installed in Step 3 if required by `requirements.txt`. If you encounter issues, try installing them manually:
        ```bash
        # Example for Picamera2 (Bullseye/Bookworm OS)
        sudo apt update
        sudo apt install -y python3-picamera2
        ```

**Installation Complete!** You should now be ready to use the system.

---

## ▶️ Usage

Activate your virtual environment (`source venv/bin/activate` or `.\venv\Scripts\activate`) if you haven't already.

The basic command structure is:

```bash
python main.py [OPTIONS]
```

**Common Examples:**

*   **Analyze a specific image:**
    ```bash
    python main.py --input /path/to/your/image.jpg
    ```
    *(If `--input` is omitted, it defaults to `images/input.png`)*

*   **Specify an output filename:**
    ```bash
    python main.py --output /path/to/save/result.png
    ```
    *(If `--output` is omitted, it defaults to `images/output.png`)*

*   **Use a different detection model:**
    ```bash
    # Example using a smaller/faster model
    python main.py --model yolov8s 
    # Example using YOLOv5
    python main.py --model yolov5s
    ```
    *(The specified model will be downloaded automatically if not found in the `models/` directory)*

*   **(Raspberry Pi) Capture image from camera:**
    ```bash
    python camera.py images/live_capture.png 
    python main.py --input images/live_capture.png
    ```
    *(Use `camera.py` to capture, then `main.py` to process)*

*   **Get help on all options:**
    ```bash
    python main.py --help
    ```

---

## ✨ Features

*   Automatic tennis court detection using color analysis.
*   Player detection using YOLOv5 and YOLOv8+ models.
*   Analysis of player positions (on court, sideline).
*   Support for multiple courts with basic numbering.
*   Configurable via command-line flags.
*   Optimized output style for readability.
*   Includes Raspberry Pi camera support (`camera.py`).

---

## ⚙️ Key Configuration Flags

Run `python main.py --help` for a full list. Key flags include:

| Flag                  | Description                                     | Default             |
| --------------------- | ----------------------------------------------- | ------------------- |
| `--input`             | Path to input image                             | `images/input.png`  |
| `--output`            | Path for output image                           | `images/output.png` |
| `--model`             | YOLO model name (e.g., `yolov8s`, `yolov5s`)    | `yolov8x`           |
| `--device`            | Inference device (`cpu` or `cuda`)              | Auto-detected       |
| `--disable-ssl-verify`| Disable SSL check for model downloads           | Off                 |
| `--debug`             | Enable debug mode (saves intermediate images)   | Off                 |
| `--quiet`             | Reduce console output                           | Off                 |

---

## 🔧 Troubleshooting

*   **Missing Modules (ImportError):** Ensure you activated the virtual environment (`source venv/bin/activate`) and installed requirements (`pip install -r requirements.txt`).
*   **Model Download Failure:** Check internet connection. Try the `--disable-ssl-verify` flag. Manually download the `.pt` file to the `models/` directory.
*   **Camera Issues (Raspberry Pi):** Verify connection. Ensure camera is enabled via `sudo raspi-config`. Check if `libcamera-dev` and `python3-picamera2` (or legacy `python3-picamera`) are installed.
*   **Permission Denied:** You might need `sudo` for system-wide installs (like `apt`) or if writing to protected directories.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss changes or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.