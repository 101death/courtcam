# Tennis Court Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Python 3.6+](https://img.shields.io/badge/Python-3.6+-blue.svg)](#) [![OpenCV](https://img.shields.io/badge/CV-OpenCV-green.svg)](#) [![YOLO](https://img.shields.io/badge/Models-YOLOv5%20%7C%20YOLOv8-lightgrey.svg)](#)

> Automated tennis court & player detection leveraging OpenCV and YOLO on Raspberry Pi or desktop environments.

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

### Prerequisites

* **Python** 3.6 or later
* **git**, **curl**
* **Raspberry Pi Zero 2W** (if using Pi camera)
* **Internet connection** for initial dependency & model downloads

<details>
<summary><strong>Raspberry Pi (Full automated setup)</strong></summary>

```bash
# 1. Clone the repo & navigate in
git clone https://github.com/101death/courtcam.git
cd courtcam

# 2. Make and run the interactive installer
chmod +x setup.sh
./setup.sh

# 3. Activate the virtual environment
source venv/bin/activate
```

</details>

<details>
<summary><strong>Manual / Other Linux & macOS</strong></summary>

```bash
# 1. Clone and enter
git clone https://github.com/101death/courtcam.git
cd courtcam

# 2. Create Python venv & activate
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4. (Optional) Install system libs for Pi camera
sudo apt update
sudo apt install -y python3-picamera2 libcamera-dev python3-libcamera
```

</details>

> **Note:** On Windows, use PowerShell to run `python -m venv venv` and adjust activation command to `venv\Scripts\activate`.

---

## ▶️ Usage

```bash
# With an existing image
python main.py --input images/input.png

# Using the Pi camera
python main.py --camera

# Specify model & device
python main.py --model yolov5s --device cpu

# Advanced flags
python main.py --output results.png --show-labels --debug
```

All outputs are saved to the `images/` folder by default.

---

## ✨ Features

* **Automatic Court Detection** based on color analysis
* **YOLOv5 & YOLOv8** support for player detection
* **In‑bounds Analysis**: Precisely determines player positions
* **Multi‑court** numbering & tracking
* **Configurable** via command‑line flags & config file
* **Raspberry Pi** optimized installation & camera capture

---

## ⚙️ Configuration

Adjust default settings in `config.py` or via CLI arguments:

| Option                | Description                             | Default             |
| --------------------- | --------------------------------------- | ------------------- |
| `--input`             | Path to input image                     | `images/input.png`  |
| `--output`            | Path for output image                   | `images/output.png` |
| `--model`             | YOLO model (`yolov5s`, `yolov8n`, etc.) | `yolov8x`           |
| `--device`            | Inference device (`cpu` or `cuda`)      | auto                |
| `--show-labels`       | Overlay player labels                   | off                 |
| `--show-court-labels` | Overlay court numbers                   | off                 |
| `--camera`            | Use Pi camera for capture               | off                 |
| `--debug`             | Write debug masks to `images/debug/`    | off                 |

---

## 🔧 Troubleshooting

* **`setup.sh` errors**: ensure executable (`chmod +x setup.sh`) and internet access.
* **Camera not detected**: enable via `sudo raspi-config → Interface Options → Camera` and reboot.
* **Missing model**: re-run setup with “Download models only” or manually create `models/` and download.
* **Python errors**: activate venv and `pip install -r requirements.txt`.

---

## 🤝 Contributing

Contributions welcome! Please open issues or pull requests with descriptive titles.

1. Fork the repo
2. Create feature branch (`git checkout -b feature/...`)
3. Commit your changes (`git commit -m "..."`)
4. Push to branch (`git push origin feature/...`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.