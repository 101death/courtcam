@echo off
:: Tennis Court Detection System - Windows Dependency Installer
:: This script installs all required dependencies for the tennis court detection system on Windows

echo Tennis Court Detection System - Windows Installer
echo This script will install all required dependencies for the Tennis Court Detection System.
echo It may take some time to complete. Please be patient.
echo.

:: Create necessary directories
echo Creating necessary directories...
if not exist models mkdir models
if not exist images mkdir images

:: Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in the PATH. Please install Python 3.6+ and try again.
    exit /b 1
)

:: Install Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip

:: Install PyTorch
echo Installing PyTorch...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

:: Install other Python dependencies
echo Installing other Python dependencies...
python -m pip install ^
    opencv-python ^
    numpy ^
    pandas ^
    shapely ^
    tqdm ^
    pillow ^
    matplotlib
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install Python dependencies.
    exit /b 1
)

:: Install Ultralytics for YOLOv5
echo Installing Ultralytics for YOLOv5...
python -m pip install ultralytics
if %ERRORLEVEL% NEQ 0 (
    echo Failed to install Ultralytics.
    exit /b 1
)

:: Download YOLOv5 model if it doesn't exist
echo Checking for YOLOv5 model...
if not exist models\yolov5s.pt (
    echo Downloading YOLOv5s model (this may take a while)...
    powershell -Command "& {Invoke-WebRequest -Uri 'https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt' -OutFile 'models\yolov5s.pt'}"
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to download YOLOv5 model.
        exit /b 1
    )
) else (
    echo YOLOv5s model already exists.
)

echo.
echo Installation complete!
echo You can now run the tennis court detection system:
echo python main.py --input images\your_image.jpg
echo.
pause 