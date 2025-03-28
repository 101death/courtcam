#!/usr/bin/env python3
"""
Utility script to install ultralytics package for YOLOv8 models
"""

import sys
import subprocess
import os

def main():
    print("======================================================")
    print("  Installing ultralytics package for YOLOv8 models")
    print("======================================================")
    
    try:
        # Check if pip is available
        subprocess.check_call([sys.executable, "-m", "pip", "--version"])
        
        # Install ultralytics
        print("\nInstalling ultralytics package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        
        # Verify installation
        print("\nVerifying installation...")
        subprocess.check_call([sys.executable, "-c", "import ultralytics; print(f'Ultralytics version: {ultralytics.__version__}')"])
        
        print("\n✅ Installation successful!")
        print("\nYou can now use YOLOv8 models with the main script.")
        print("Try running: python main.py --model yolov8x")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nManual installation instructions:")
        print("1. Open a terminal or command prompt")
        print("2. Run: pip install ultralytics")
        print("3. Then run the main script with: python main.py --model yolov8x")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 