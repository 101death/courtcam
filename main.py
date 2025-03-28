#!/usr/bin/env python3
"""
Tennis Court Detector - Detects people on tennis courts using YOLOv5
"""
import os
import cv2 # type: ignore
import numpy as np # type: ignore
import json
import torch # type: ignore
from shapely.geometry import Polygon, Point # type: ignore
import sys
import ssl
import argparse
import time
import io
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
import contextlib
import threading
from multiprocessing import cpu_count, Pool
from functools import partial
import urllib.request

# Global variables
args = None  # Will store command-line arguments

# Fix macOS SSL certificate issues
if sys.platform == 'darwin':
    # Check if we're running on macOS and set default SSL context
    try:
        # Try to import certifi for better certificate handling
        import certifi # type: ignore
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        # If certifi isn't available, try the default macOS certificate location
        ssl._create_default_https_context = lambda: ssl.create_default_context()

# === CONFIGURATION SETTINGS ===
class Config:
    # Color settings for court detection
    COURT_COLORS = {
        "blue": {
            "lower": [90, 40, 40],
            "upper": [120, 255, 255]
        },
        "green": {
            "lower": [40, 40, 40],
            "upper": [80, 255, 255]
        },
        "red": {
            "lower": [0, 50, 50],
            "upper": [10, 255, 255],
            "lower2": [170, 50, 50],
            "upper2": [180, 255, 255]
        }
    }
    
    # Court detection parameters
    class Court:
        MIN_AREA = 3000              # Reduced from 5000 to detect smaller courts
        MAX_AREA = 200000            # Increased from 150000 to detect larger courts
        MIN_SCORE = 0.5              # Minimum score for a valid court
        MIN_ASPECT_RATIO = 1.0       # Reduced from 1.2 to allow more court shapes
        MAX_ASPECT_RATIO = 4.0       # Increased from 3.0 to allow wider courts
        MIN_BLUE_RATIO = 0.2         # Reduced from 0.3 to be more lenient
        MIN_GREEN_RATIO = 0.02       # Reduced from 0.05 to be more lenient
    
    # Morphological operation settings
    class Morphology:
        KERNEL_SIZE = 5              # Kernel size for morphological operations
        ITERATIONS = 2               # Number of iterations for closing operations
    
    # Court area definitions
    IN_BOUNDS_COLOR = "blue"         # Color that represents in-bounds
    OUT_BOUNDS_COLOR = "green"       # Color that represents out-of-bounds
    
    # Visualization settings
    class Visual:
        COURT_OUTLINE_COLOR = (0, 255, 0)        # Green
        COURT_OUTLINE_THICKNESS = 4              # Line thickness
        PERSON_IN_BOUNDS_COLOR = (0, 255, 0)     # Green for people in court
        PERSON_OUT_BOUNDS_COLOR = (0, 165, 255)  # Orange for people near court
        PERSON_OFF_COURT_COLOR = (0, 0, 255)     # Red for people off court
        TEXT_COLOR = (255, 255, 255)             # White
        FONT_SCALE = 0.5                         # Text size
        TEXT_THICKNESS = 2                       # Text thickness
        DRAW_COURT_OUTLINE = True                # Whether to draw court outline
        SHOW_COURT_NUMBER = True                # Whether to show court number in labels
        SHOW_DETAILED_LABELS = True             # Whether to show detailed labels on output image
    
    # Terminal output settings
    class Output:
        VERBOSE = True               # Show detailed output
        USE_COLOR_OUTPUT = True      # Use colored terminal output
        SHOW_TIMESTAMP = True        # Show timestamps in output
        SUPER_QUIET = False          # Super quiet mode (almost no output)
        SUMMARY_ONLY = False         # Only show summary of results
        EXTRA_VERBOSE = True         # Show extra detailed output for Raspberry Pi
        
        # ANSI color codes for terminal output
        COLORS = {
            "INFO": "\033[94m",      # Blue
            "SUCCESS": "\033[92m",   # Green
            "WARNING": "\033[93m",   # Yellow
            "ERROR": "\033[91m",     # Red
            "DEBUG": "\033[90m",     # Gray
            "RESET": "\033[0m",      # Reset
            "BOLD": "\033[1m",       # Bold
            "UNDERLINE": "\033[4m"   # Underline
        }
    
    # Debug mode
    DEBUG_MODE = False              # Detailed debug output mode
    
    # Paths and directories
    class Paths:
        IMAGES_DIR = "images"
        INPUT_IMAGE = "input.png"
        OUTPUT_IMAGE = "output.png"
        MODELS_DIR = "models"
        
        @classmethod
        def input_path(cls):
            return os.path.join(cls.IMAGES_DIR, cls.INPUT_IMAGE)
            
        @classmethod
        def output_path(cls):
            return os.path.join(cls.IMAGES_DIR, cls.OUTPUT_IMAGE)
            
        @classmethod
        def debug_dir(cls):
            return os.path.join(os.path.dirname(cls.output_path()), "debug")
    
    # Model settings
    class Model:
        NAME = "yolov8x"             # YOLOv5 model size (yolov5s, yolov5m, yolov5l, etc.)
        CONFIDENCE = 0.3             # Detection confidence threshold
        IOU = 0.45                   # IoU threshold
        CLASSES = [0]                # Only detect people (class 0)
        
        # YOLOv5 model URLs - add more as needed
        MODEL_URLS = {
            "yolov5n": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt",
            "yolov5s": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt",
            "yolov5m": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5m.pt",
            "yolov5l": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5l.pt",
            "yolov5x": "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5x.pt",
            # YOLOv6 models
            "yolov6n": "https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.pt",
            "yolov6s": "https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.pt",
            "yolov6m": "https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.pt",
            "yolov6l": "https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.pt",
            # YOLOv7 models
            "yolov7": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
            "yolov7-tiny": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt",
            "yolov7x": "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt",
            # YOLOv8 models
            "yolov8n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
            "yolov8s": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
            "yolov8m": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
            "yolov8l": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt",
            "yolov8x": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt",
        }
        
        @classmethod
        def get_model_url(cls, model_name):
            # Default to yolov5s if model not found
            if model_name.lower() not in cls.MODEL_URLS:
                OutputManager.log(f"Model {model_name} not found in known models. Defaulting to yolov5s.", "WARNING")
                return cls.MODEL_URLS["yolov5s"]
            return cls.MODEL_URLS[model_name.lower()]
    
    # Multiprocessing settings
    class MultiProcessing:
        ENABLED = True              # Enable multiprocessing
        NUM_PROCESSES = 3           # Number of CPU cores to use
        CHUNK_SIZE = 10             # Chunk size for processing

class OutputManager:
    """
    Centralized output manager for all terminal output.
    Provides professional formatting, reliable animations, and clean output management.
    """
    # ANSI color and style codes
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"
    
    # Symbols for different message types
    SYMBOLS = {
        "INFO": "ℹ",
        "SUCCESS": "✓",
        "WARNING": "⚠",
        "ERROR": "✗",
        "DEBUG": "•",
        "FATAL": "☠",
        "STATUS": "→",
    }
    
    # Track messages for summary
    warnings = []
    errors = []
    successes = []
    info = []
    
    # Animation state
    _animation_active = False
    _animation_thread = None
    _stop_animation = False
    _progress_total = 0
    _progress_current = 0
    
    @classmethod
    def reset_logs(cls):
        """Reset all tracked messages"""
        cls.warnings = []
        cls.errors = []
        cls.successes = []
        cls.info = []
    
    @classmethod
    def _ensure_animation_stopped(cls):
        """Ensure any running animation is stopped and cleaned up"""
        if cls._animation_active:
            cls._stop_animation = True
            cls._animation_active = False
            if cls._animation_thread and cls._animation_thread.is_alive():
                cls._animation_thread.join(timeout=0.1)
            
            # Clear the current line
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
    
    @classmethod
    def log(cls, message, level="INFO"):
        """Print formatted log messages with consistent, clean formatting"""
        # In super quiet mode, only show errors and success messages
        if Config.Output.SUPER_QUIET and level not in ["ERROR", "SUCCESS", "FATAL", "STATUS"]:
            return
            
        if not Config.Output.VERBOSE and level == "DEBUG":
            return
        
        # Stop any running animation
        cls._ensure_animation_stopped()
        
        # Track messages for summary
        if level == "INFO":
            cls.info.append(message)
        elif level == "SUCCESS":
            cls.successes.append(message)
        elif level == "WARNING":
            cls.warnings.append(message)
        elif level == "ERROR" or level == "FATAL":
            cls.errors.append(message)
        
        # Get color based on level
        color = ""
        if Config.Output.USE_COLOR_OUTPUT:
            if level == "INFO":
                color = cls.BLUE
            elif level == "SUCCESS":
                color = cls.GREEN
            elif level == "WARNING":
                color = cls.YELLOW
            elif level == "ERROR" or level == "FATAL":
                color = cls.RED
            elif level == "DEBUG":
                color = cls.GRAY
            elif level == "STATUS":
                color = cls.CYAN
        
        # Format timestamp with consistent width
        timestamp = ""
        if Config.Output.SHOW_TIMESTAMP:
            timestamp = f"{cls.GRAY}[{datetime.now().strftime('%H:%M:%S')}]{cls.RESET} "
        
        # Get symbol based on message level
        symbol = cls.SYMBOLS.get(level, "")
        
        # Format the message with appropriate styling
        formatted_message = f"{timestamp}{color}{symbol} {message}{cls.RESET}"
        
        # Print the message
        print(formatted_message)
        
        # Flush stdout to ensure immediate display on Raspberry Pi
        sys.stdout.flush()
    
    @classmethod
    def status(cls, message):
        """Log a status update message - replacement for animations"""
        cls.log(message, "STATUS")
    
    @classmethod
    def _run_animation_thread(cls, animate_func):
        """Run an animation thread with the given animation function"""
        # Ensure any previous animation is stopped
        cls._ensure_animation_stopped()
        
        # Set up new animation
        cls._stop_animation = False
        cls._animation_active = True
        
        # Start the animation thread
        cls._animation_thread = threading.Thread(target=animate_func)
        cls._animation_thread.daemon = True
        cls._animation_thread.start()
    
    @classmethod
    def animate(cls, message, is_progress=False, total=20):
        """
        Display a simple spinning animation next to text
        
        Args:
            message: Message to display with the animation
            is_progress: Ignored (maintained for compatibility)
            total: Ignored (maintained for compatibility)
        """
        # Classic spinning animation with clear visibility
        frames = ["|", "/", "-", "\\"]
        
        def animate():
            idx = 0
            
            # Continue spinning until stopped
            while not cls._stop_animation and cls._animation_active:
                # Get the current frame
                frame = frames[idx % len(frames)]
                
                # Format timestamp with consistent width
                timestamp = ""
                if Config.Output.SHOW_TIMESTAMP:
                    timestamp = f"{cls.GRAY}[{datetime.now().strftime('%H:%M:%S')}]{cls.RESET} "
                
                # Clear the line and print the new status with spinner
                sys.stdout.write(f"\r\033[K{timestamp}{message} {cls.BOLD}{cls.CYAN}{frame}{cls.RESET}")
                sys.stdout.flush()
                
                # Update the index for next frame and sleep briefly
                idx = (idx + 1) % len(frames)
                time.sleep(0.08)  # Fast animation for clear spinning movement
        
        # Run the animation in a separate thread
        cls._run_animation_thread(animate)
    
    @classmethod
    def set_progress(cls, value):
        """Set the current progress value (0.0 to 1.0)"""
        cls._progress_current = min(cls._progress_total, max(0, value * cls._progress_total))
    
    @classmethod
    def stop_animation(cls, success=True):
        """Stop the current animation with optional completion indicator"""
        if cls._animation_active:
            # First stop the animation thread
            cls._stop_animation = True
            cls._animation_active = False
            if cls._animation_thread and cls._animation_thread.is_alive():
                cls._animation_thread.join(timeout=0.1)
            
            # Clear the line
            sys.stdout.write("\r\033[K")
            sys.stdout.flush()
            
            # No completion message - the calling code will do this
    
    @classmethod
    def summarize_detections(cls, courts, people, people_locations):
        """Summarize detection results in a more concise format"""
        if Config.Output.SUPER_QUIET:
            return {}
            
        cls.log(f"{cls.BOLD}Detection Summary:{cls.RESET}", "INFO")
        cls.log(f"Found {len(courts)} tennis courts", "SUCCESS")
        cls.log(f"Found {len(people)} {'person' if len(people) == 1 else 'people'} in the image", "SUCCESS")
        
        # Count people by court
        court_counts = {}
        for court_idx, area_type in people_locations:
            if court_idx >= 0:
                court_num = court_idx + 1
                if court_num not in court_counts:
                    court_counts[court_num] = 0
                court_counts[court_num] += 1
        
        # Print simplified court-specific counts
        for court_num in sorted(court_counts.keys()):
            cls.log(f"Court {court_num}: {court_counts[court_num]} {'person' if court_counts[court_num] == 1 else 'people'}", "INFO")
        
        return court_counts
    
    @classmethod
    def fancy_summary(cls, title, content, processing_time=None):
        """
        Display a fancy boxed summary with animations
        
        Args:
            title: Title of the summary box
            content: List of content lines or a string with newlines
            processing_time: Optional processing time to display
        """
        # Stop any running animation
        cls._ensure_animation_stopped()
        
        # Helper function to wrap text by words
        def wrap_text(text, width):
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + (1 if current_length > 0 else 0) <= width:
                    # Add word to current line
                    if current_length > 0:
                        current_length += 1  # For the space
                        current_line.append(" ")
                    current_line.append(word)
                    current_length += len(word)
                else:
                    # Line is full, start a new one
                    if current_line:
                        lines.append("".join(current_line))
                    current_line = [word]
                    current_length = len(word)
            
            # Add the last line if it exists
            if current_line:
                lines.append("".join(current_line))
            
            return lines
        
        # Process the content
        max_width = 70  # Maximum characters per line
        content_lines = []
        
        if isinstance(content, str):
            # Split the content by newlines first
            for paragraph in content.split('\n'):
                if not paragraph:
                    # Keep empty lines
                    content_lines.append("")
                else:
                    # Wrap the paragraph
                    wrapped_lines = wrap_text(paragraph, max_width)
                    content_lines.extend(wrapped_lines)
        else:
            # Handle list input
            for paragraph in content:
                if not paragraph:
                    content_lines.append("")
                else:
                    wrapped_lines = wrap_text(paragraph, max_width)
                    content_lines.extend(wrapped_lines)
        
        # Add processing time if provided
        if processing_time is not None:
            content_lines.append("")
            content_lines.append(f"Processing time: {processing_time:.2f} seconds")
        
        # Calculate box width based on content
        content_width = max(len(line) for line in content_lines) if content_lines else 0
        box_width = max(content_width + 4, len(title) + 10, 50)
        
        # Box drawing characters
        top_left = "╭"
        top_right = "╮"
        bottom_left = "╰"
        bottom_right = "╯"
        horizontal = "─"
        vertical = "│"
        
        # Build the box parts
        top_border = f"{top_left}{horizontal * (box_width - 2)}{top_right}"
        title_line = f"{vertical} {cls.BOLD}{title.center(box_width - 4)}{cls.RESET} {vertical}"
        divider = f"├{horizontal * (box_width - 2)}┤"
        empty_line = f"{vertical}{' ' * (box_width - 2)}{vertical}"
        bottom_border = f"{bottom_left}{horizontal * (box_width - 2)}{bottom_right}"
        
        # Display the summary box with animation
        def display_box():
            # Print top border
            print(top_border)
            
            # Print title
            print(title_line)
            
            # Print divider
            print(divider)
            
            # Print empty line
            print(empty_line)
            
            # Print content
            for line in content_lines:
                # Pad the line to fit the box
                padded_line = line.ljust(box_width - 4)[:box_width - 4]
                print(f"{vertical} {padded_line} {vertical}")
            
            # Print empty line
            print(empty_line)
            
            # Print bottom border
            print(bottom_border)
        
        # Display the box
        display_box()
    
    @classmethod
    def create_final_summary(cls, people_count, court_counts, output_path=None, processing_time=None, total_courts=None):
        """Create a final summary for display"""
        # Base summary parts
        summary_parts = []
        
        # Start with what was successful
        if people_count is not None:
            # Use the total number of courts passed in
            court_count = total_courts if total_courts is not None else 0
            
            # Format the main summary line
            if people_count == 0:
                summary_parts.append(f"{court_count} {'court was' if court_count == 1 else 'courts were'} detected, no people present")
            elif court_count == 0:
                summary_parts.append(f"{people_count} {'person was' if people_count == 1 else 'people were'} detected, no tennis courts in frame")
            else:
                # Format details for each court
                court_details = []
                total_on_courts = 0
                
                for court_num in sorted(court_counts.keys()):
                    count = court_counts[court_num]
                    total_on_courts += count
                    if count > 0:
                        court_details.append(f"{count} on court {court_num}")
                
                # Create the court details string with "and" between the last two items
                court_details_str = ""
                if len(court_details) > 1:
                    court_details_str = " with " + " and ".join([", ".join(court_details[:-1]), court_details[-1]])
                elif len(court_details) == 1:
                    court_details_str = f" with {court_details[0]}"
                
                summary = f"{court_count} {'court was' if court_count == 1 else 'courts were'} detected{court_details_str}"
                summary_parts.append(summary)
        
        # Create the base summary text
        final_summary = " ".join(summary_parts)
        
        # Add output path in a separate line for better presentation
        if output_path:
            final_summary += f"\n\nOutput image saved"
        
        # Add errors/warnings differently - in a more natural way
        if cls.errors or cls.warnings:
            final_summary += "\n\n"
            
            if cls.errors:
                error_fixes = cls.get_potential_fixes()
                if "\n" in error_fixes:
                    final_summary += f"{cls.RED}{cls.BOLD}ERRORS DETECTED:{cls.RESET}\n{error_fixes}"
                else:
                    final_summary += f"{cls.RED}{cls.BOLD}ERROR:{cls.RESET} {error_fixes}"
            elif cls.warnings and not cls.errors:
                if len(cls.warnings) == 1:
                    final_summary += f"{cls.YELLOW}{cls.BOLD}NOTE:{cls.RESET} {cls.warnings[0]}"
                else:
                    warning_header = f"{cls.YELLOW}{cls.BOLD}NOTES:{cls.RESET}"
                    warning_texts = [f"• {w}" for w in cls.warnings]
                    final_summary += f"{warning_header}\n" + "\n".join(warning_texts)
        
        return final_summary
    
    @classmethod
    def get_potential_fixes(cls):
        """Generate detailed solutions for common errors with specific commands to run"""
        messages = []
        
        if not cls.errors:
            return ""
        
        # Quick commands for common problems
        for error in cls.errors:
            # YOLOv8 specific errors
            if "'list' object has no attribute 'pandas'" in error or "'boxes'" in error or "YOLOv8 result format error" in error:
                messages.append("YOLOv8 compatibility error\nYou're using a YOLOv8 model but the script encountered format issues. Try:\n1. Run with a YOLOv5 model: python main.py --model yolov5s\n2. Or install ultralytics package: pip install ultralytics")
            
            # Module not found errors
            elif "ModuleNotFoundError" in error or "No module named" in error:
                module_name = error.split("'")[1] if "'" in error else "unknown"
                if module_name == "cv2":
                    messages.append(f"Missing OpenCV: {module_name}\nRun: pip install opencv-python")
                elif module_name == "torch":
                    messages.append(f"Missing PyTorch: {module_name}\nRun: pip install torch torchvision")
                elif module_name == "numpy":
                    messages.append(f"Missing NumPy: {module_name}\nRun: pip install numpy")
                elif module_name == "shapely":
                    messages.append(f"Missing Shapely: {module_name}\nRun: pip install shapely")
                elif module_name == "ultralytics" and "yolov8" in error.lower():
                    messages.append(f"Missing Ultralytics package needed for YOLOv8\nRun: pip install ultralytics\nOr use YOLOv5 model: python main.py --model yolov5s")
                else:
                    messages.append(f"Missing module: {module_name}\nRun: pip install {module_name}\nOr: pip install -r requirements.txt")
                
            # Torch/CUDA errors
            elif "CUDA" in error or "cuda" in error:
                if "out of memory" in error.lower():
                    messages.append("CUDA out of memory error\nTry: Reduce batch size or image size\nOr: Use --device cpu flag to use CPU instead")
                elif "not available" in error.lower():
                    messages.append("CUDA not available\nRun: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118\nOr: Use CPU with --device cpu")
                else:
                    messages.append("CUDA error detected\nRun: pip install -r requirements.txt\nOr: Use CPU with --device cpu")
                
            # OpenCV errors
            elif "cv2" in error or "OpenCV" in error:
                if "cannot read frame" in error.lower() or "empty frame" in error.lower():
                    messages.append("OpenCV cannot read the image\nCheck that your image is valid and not corrupted")
                else:
                    messages.append("OpenCV error detected\nRun: pip install opencv-python")
                
            # YOLOv5/YOLOv8 model errors
            elif "model" in error.lower() and ("yolo" in error.lower() or "not found" in error.lower()):
                if "models directory" in error.lower():
                    messages.append(f"Models directory not found\nRun: mkdir -p models")
                elif "yolov8" in error.lower():
                    messages.append(f"YOLOv8 model not found or error\n1. Try YOLOv5 instead: python main.py --model yolov5s\n2. Install ultralytics: pip install ultralytics\n3. Or manually download YOLOv8: mkdir -p models && wget -q https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -O models/yolov8s.pt")
                else:
                    messages.append(f"YOLO model not found\nRun: mkdir -p models && wget -q https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -O models/yolov5s.pt")
            
            # macOS SSL Certificate errors
            elif "ssl" in error.lower() and "certificate" in error.lower() and ("darwin" in error.lower() or "macos" in error.lower()):
                messages.append("macOS SSL Certificate verification issue\nRunning one of these commands should fix it:\n1. Run with: python main.py --disable-ssl-verify\n2. Run with: python main.py --force-macos-cert-install\n3. Or manually install certificates by running /Applications/Python 3.x/Install Certificates.command")
                
            # Permission errors
            elif any(word in error.lower() for word in ["permission", "access", "denied"]):
                if sys.platform == "win32":
                    messages.append("Permission denied\nRun the script as Administrator or check file permissions\nRight-click Command Prompt/PowerShell and select 'Run as Administrator'")
                else:
                    messages.append("Permission denied\nRun: chmod +x main.py && sudo ./main.py")
                
            # Shapely errors
            elif "shapely" in error.lower():
                messages.append("Shapely error detected\nRun: pip install shapely")
                
            # Image errors
            elif any(word in error.lower() for word in ["unable to open", "load image", "read image", "no such file"]):
                messages.append(f"Image not found at {Config.Paths.input_path()}\nEnsure the image exists:\n  - Run: mkdir -p images && cp your_image.jpg images/input.png\n  - Or use: python main.py --input /path/to/your/image.jpg")
                
            # Directory errors
            elif any(word in error.lower() for word in ["no such file or directory", "directory", "folder", "path"]):
                if "images" in error.lower():
                    messages.append("Images directory not found\nRun: mkdir -p images && cp your_image.jpg images/input.png")
                elif "models" in error.lower():
                    messages.append("Models directory not found\nRun: mkdir -p models && wget -q https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -O models/yolov5s.pt")
                else:
                    dir_name = error.split("'")[1] if "'" in error else "directory"
                    messages.append(f"Directory error: {dir_name}\nCheck that all required directories exist\nRun: mkdir -p images models")
            
            # Internet connection errors during YOLOv5 download
            elif any(word in error.lower() for word in ["connection", "timeout", "network", "internet", "download"]):
                messages.append("Network error detected\nCheck your internet connection and try again\nIf downloading YOLO model fails, manually download from: https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt\nThen place it in the models/ directory")
            
            # Memory errors
            elif "memory" in error.lower():
                messages.append("Memory error detected\nTry with a smaller image or on a machine with more RAM")
                
            # SSL errors
            elif "ssl" in error.lower():
                if sys.platform == "darwin":  # macOS
                    messages.append("SSL error detected on macOS\nTry: python main.py --force-macos-cert-install\nOr: python main.py --disable-ssl-verify")
                else:
                    messages.append("SSL error detected when downloading model\nTry: pip install certifi --upgrade\nOr: Run with --disable-ssl-verify flag")
                
            # Disk space errors
            elif any(word in error.lower() for word in ["disk", "space", "storage", "write"]):
                messages.append("Disk space or write error\nCheck that you have sufficient disk space and write permissions in the current directory")
               
            # For any other errors, give a more detailed message
            else:
                messages.append(f"Error: {error}\n\nGeneral troubleshooting:\n1. Ensure all dependencies are installed: pip install -r requirements.txt\n2. Check for proper YOLO model: models/yolov5s.pt\n3. Verify input image exists and is readable\n4. Try running with --debug flag for more information")
        
        # Create a direct message with commands
        if len(messages) == 1:
            return messages[0]
        else:
            return "Multiple issues detected:\n\n" + "\n\n".join(f"ISSUE {i+1}:\n{msg}" for i, msg in enumerate(messages))
        
        return "\n".join(messages)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output"""
    # Save the original stdout and stderr
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    # Create string buffers to capture output
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    
    # Redirect output to the buffers
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer
    
    try:
        yield  # Code inside the with block runs with redirected stdout/stderr
    finally:
        # Restore the original stdout and stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def create_blue_mask(image):
    """Create a mask for blue areas in the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get blue range from config
    blue_range = Config.COURT_COLORS["blue"]
    lower = np.array(blue_range["lower"])
    upper = np.array(blue_range["upper"])
    
    # Create mask
    blue_mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up mask
    kernel = np.ones((Config.Morphology.KERNEL_SIZE, Config.Morphology.KERNEL_SIZE), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=Config.Morphology.ITERATIONS)
    
    return blue_mask

def create_green_mask(image):
    """Create a mask for green areas in the image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Get green range from config
    green_range = Config.COURT_COLORS["green"]
    lower = np.array(green_range["lower"])
    upper = np.array(green_range["upper"])
    
    # Create mask
    green_mask = cv2.inRange(hsv, lower, upper)
    
    # Clean up mask
    kernel = np.ones((Config.Morphology.KERNEL_SIZE, Config.Morphology.KERNEL_SIZE), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=Config.Morphology.ITERATIONS)
    
    return green_mask

def is_sky_region(contour, image_height, image_width):
    """Check if a contour is likely to be sky based on position and characteristics"""
    x, y, w, h = cv2.boundingRect(contour)
    
    # Sky is usually at the top of the image
    is_at_top = y < image_height * 0.15
    
    # Sky is usually wide
    is_wide = w > image_width * 0.5
    
    # Sky usually has a small height
    is_short = h < image_height * 0.2
    
    # Check if the contour is likely to be sky
    return is_at_top and (is_wide or is_short)

def process_court_contour(contour, blue_mask, green_mask, height, width):
    """Process a single court contour - designed for multiprocessing"""
    area = cv2.contourArea(contour)
    # Filter by minimum area to avoid noise
    if area < Config.Court.MIN_AREA:
        return None
        
    # Get bounding box for aspect ratio check
    x, y, w, h = cv2.boundingRect(contour)
    
    # Create slightly dilated mask for this blue region to check if it's next to green
    region_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(region_mask, [contour], -1, 255, -1)
    
    # Dilate the region mask slightly to check for adjacent green pixels
    kernel = np.ones((15, 15), np.uint8)
    dilated_region = cv2.dilate(region_mask, kernel, iterations=1)
    
    # Check if there's green adjacent to this blue region
    green_nearby = cv2.bitwise_and(green_mask, dilated_region)
    green_nearby_pixels = cv2.countNonZero(green_nearby)
    
    # If there's no green nearby, it's not a court
    if green_nearby_pixels < 100:  # Threshold for minimum green pixels needed
        return None
    
    # Get the blue area itself
    blue_area = cv2.bitwise_and(blue_mask, region_mask)
    blue_pixels = cv2.countNonZero(blue_area)
    
    # Get convex hull for better shape
    hull = cv2.convexHull(contour)
    
    # Approximate polygon
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
    
    # Calculate centroid
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = x + w//2, y + h//2
    
    # Save court info with all required keys
    court_info = {
        'contour': contour,
        'approx': approx,
        'hull': hull,
        'area': area,
        'blue_ratio': 1.0,  # This is a blue region, so ratio is 1
        'green_ratio': green_nearby_pixels / area,
        'blue_mask': blue_area,
        'green_mask': green_nearby,
        'blue_pixels': blue_pixels,
        'green_pixels': green_nearby_pixels,
        'centroid': (cx, cy),
        'bbox': (x, y, w, h)
    }
    
    return court_info

def check_person_on_court(person_data):
    """
    Process a single person to determine court position - designed for multiprocessing
    person_data: tuple of (person, courts)
    Returns: (court_index, area_type)
    """
    person, courts = person_data
    
    # Get the bounding box coordinates
    x1, y1, x2, y2 = person['bbox']
    
    # Calculate the bottom half of the bounding box
    bottom_y1 = y1 + (y2 - y1) // 2  # Start from middle of box
    bottom_y2 = y2  # End at bottom of box
    
    # Create points for the bottom half of the bounding box
    bottom_points = [
        Point(x1, bottom_y1),
        Point(x2, bottom_y1),
        Point(x2, bottom_y2),
        Point(x1, bottom_y2)
    ]
    
    # Check each court
    for court_idx, court in enumerate(courts):
        # Get the court polygon
        approx = court['approx']
        points = approx.reshape(-1, 2)
        court_polygon = Polygon(points)
        
        # Check if any of the bottom points are inside the court
        for point in bottom_points:
            if court_polygon.contains(point):
                # Person is on this court - now determine if they're on blue (in-bounds) or green (out-bounds)
                x, y = int(point.x), int(point.y)
                
                # Check if the point is on blue area (in-bounds)
                blue_mask = court['blue_mask']
                if y < blue_mask.shape[0] and x < blue_mask.shape[1] and blue_mask[y, x] > 0:
                    return court_idx, 'in_bounds'
                
                # Check if the point is on green area (out-bounds)
                green_mask = court['green_mask']
                if y < green_mask.shape[0] and x < green_mask.shape[1] and green_mask[y, x] > 0:
                    return court_idx, 'out_bounds'
                
                # If not specifically on blue or green, consider it in-bounds if the court has more blue than green
                if court['blue_ratio'] > court['green_ratio']:
                    return court_idx, 'in_bounds'
                else:
                    return court_idx, 'out_bounds'
    
    # If we reached here, the person is not on any court
    return -1, 'off_court'

def process_courts_parallel(blue_contours, blue_mask, green_mask, height, width):
    """Process court contours in parallel using multiprocessing"""
    if not Config.MultiProcessing.ENABLED or len(blue_contours) <= 5:
        # For few contours, sequential processing is faster
        courts = []
        for contour in blue_contours:
            court_info = process_court_contour(contour, blue_mask, green_mask, height, width)
            if court_info:
                courts.append(court_info)
                OutputManager.log(f"Court {len(courts)} accepted: Area={court_info['area']:.1f}, Green nearby pixels={court_info['green_pixels']}", "SUCCESS")
        return courts
    
    # For many contours, use multiprocessing
    OutputManager.log(f"Processing {len(blue_contours)} potential courts with {Config.MultiProcessing.NUM_PROCESSES} processes", "INFO")
    
    # Create a partial function with fixed arguments
    process_func = partial(process_court_contour, blue_mask=blue_mask, green_mask=green_mask, 
                          height=height, width=width)
    
    # Create a pool and process contours in parallel
    with Pool(processes=Config.MultiProcessing.NUM_PROCESSES) as pool:
        results = pool.map(process_func, blue_contours)
    
    # Filter None results and collect valid courts
    courts = [court for court in results if court is not None]
    
    # Log the results
    for i, court in enumerate(courts):
        OutputManager.log(f"Court {i+1} accepted: Area={court['area']:.1f}, Green nearby pixels={court['green_pixels']}", "SUCCESS")
    
    return courts

def analyze_people_positions_parallel(people, courts):
    """Analyze positions of people on courts using multiprocessing"""
    if not Config.MultiProcessing.ENABLED or len(people) <= 5:
        # For few people, sequential processing is faster
        people_locations = []
        for person in people:
            court_idx, area_type = is_person_on_court(person, courts)
            people_locations.append((court_idx, area_type))
        return people_locations
    
    # For many people, use multiprocessing
    OutputManager.log(f"Analyzing positions of {len(people)} people with {Config.MultiProcessing.NUM_PROCESSES} processes", "INFO")
    
    # Create input data for the pool
    input_data = [(person, courts) for person in people]
    
    # Create a pool and process positions in parallel
    with Pool(processes=Config.MultiProcessing.NUM_PROCESSES) as pool:
        people_locations = pool.map(check_person_on_court, input_data)
    
    return people_locations

def detect_tennis_court(image, debug_folder=None):
    """
    Detect tennis courts in an image using color masking and contour analysis.
    Simplified approach: Every blue area next to green is a court.
    Returns list of tennis court contours.
    
    Optimized for Raspberry Pi Zero with multiprocessing support.
    """
    height, width = image.shape[:2]
    
    # Create masks
    OutputManager.status("Creating blue and green masks...")
    blue_mask = create_blue_mask(image)
    green_mask = create_green_mask(image)
    
    # Save raw masks for debugging
    if debug_folder and Config.Output.VERBOSE:
        cv2.imwrite(os.path.join(debug_folder, "blue_mask_raw.png"), blue_mask)
        cv2.imwrite(os.path.join(debug_folder, "green_mask_raw.png"), green_mask)
    
    # Find blue contours (potential courts)
    OutputManager.status("Analyzing potential court shapes...")
    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process contours in parallel
    valid_courts = process_courts_parallel(blue_contours, blue_mask, green_mask, height, width)
    
    # Save debug visualizations
    if debug_folder and Config.Output.VERBOSE:
        # Create visualization of masks
        masks_viz = np.zeros((height, width, 3), dtype=np.uint8)
        masks_viz[blue_mask > 0] = [255, 0, 0]  # Blue
        masks_viz[green_mask > 0] = [0, 255, 0]  # Green
        cv2.imwrite(os.path.join(debug_folder, "color_masks.png"), masks_viz)
        
        # Create visualization of all courts
        if valid_courts:
            courts_viz = image.copy()
            for i, court in enumerate(valid_courts):
                # Draw court outline
                cv2.drawContours(courts_viz, [court['approx']], 0, Config.Visual.COURT_OUTLINE_COLOR, 2)
                
                # Add court number
                x, y, w, h = court['bbox']
                cv2.putText(courts_viz, f"Court {i+1}", (x + w//2 - 40, y + h//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            cv2.imwrite(os.path.join(debug_folder, "courts_detected.png"), courts_viz)
    
    if valid_courts:
        OutputManager.log(f"Found {len(valid_courts)} tennis courts", "SUCCESS")
    else:
        OutputManager.log("No tennis courts detected", "WARNING")
    
    return valid_courts

def is_person_on_court(person, courts):
    """
    Determine if a person is on a tennis court.
    Returns (court_index, area_type) where area_type is 'in_bounds', 'out_bounds', or 'off_court'
    Uses bottom half of bounding box for more accurate placement.
    """
    # Get the bounding box coordinates
    x1, y1, x2, y2 = person['bbox']
    
    # Calculate the bottom half of the bounding box
    bottom_y1 = y1 + (y2 - y1) // 2  # Start from middle of box
    bottom_y2 = y2  # End at bottom of box
    
    # Create points for the bottom half of the bounding box
    bottom_points = [
        Point(x1, bottom_y1),
        Point(x2, bottom_y1),
        Point(x2, bottom_y2),
        Point(x1, bottom_y2)
    ]
    
    # Check each court
    for court_idx, court in enumerate(courts):
        # Get the court polygon
        approx = court['approx']
        points = approx.reshape(-1, 2)
        court_polygon = Polygon(points)
        
        # Check if any of the bottom points are inside the court
        for point in bottom_points:
            if court_polygon.contains(point):
                # Person is on this court - now determine if they're on blue (in-bounds) or green (out-bounds)
                x, y = int(point.x), int(point.y)
                
                # Check if the point is on blue area (in-bounds)
                blue_mask = court['blue_mask']
                if y < blue_mask.shape[0] and x < blue_mask.shape[1] and blue_mask[y, x] > 0:
                    return court_idx, 'in_bounds'
                
                # Check if the point is on green area (out-bounds)
                green_mask = court['green_mask']
                if y < green_mask.shape[0] and x < green_mask.shape[1] and green_mask[y, x] > 0:
                    return court_idx, 'out_bounds'
                
                # If not specifically on blue or green, consider it in-bounds if the court has more blue than green
                if court['blue_ratio'] > court['green_ratio']:
                    return court_idx, 'in_bounds'
                else:
                    return court_idx, 'out_bounds'
    
    # If we reached here, the person is not on any court
    return -1, 'off_court'

def assign_court_numbers(blue_mask_connected):
    """
    Assign court numbers by clustering blue regions
    Returns a labeled mask where each court has a unique number
    """
    # Find all connected components in the blue mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask_connected, connectivity=8)
    
    # The first label (0) is the background, so we start from 1
    courts = []
    
    # Filter out small components (noise)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= Config.Court.MIN_AREA:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            
            # Create a mask for this court
            court_mask = (labels == i).astype(np.uint8) * 255
            
            # Find contours of the court
            contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = contours[0]  # Use the largest contour
                
                # Get convex hull for better shape
                hull = cv2.convexHull(contour)
                
                # Approximate polygon
                perimeter = cv2.arcLength(hull, True)
                approx = cv2.approxPolyDP(hull, 0.02 * perimeter, True)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w//2, y + h//2
                
                courts.append({
                    'id': i,
                    'area': area,
                    'bbox': (x, y, w, h),
                    'centroid': centroids[i],
                    'contour': contour,
                    'approx': approx,
                    'hull': hull,
                    'blue_ratio': 1.0,  # This is a blue region
                    'green_ratio': 0.0,  # Will be updated later
                    'blue_mask': court_mask,
                    'green_mask': np.zeros_like(court_mask),  # Will be updated later
                    'blue_pixels': area,
                    'green_pixels': 0  # Will be updated later
                })
    
    # Sort courts by x-coordinate to assign numbers from left to right
    courts.sort(key=lambda c: c['centroid'][0])
    
    # Create a renumbered mask
    court_mask = np.zeros_like(blue_mask_connected, dtype=np.uint8)
    
    # Assign new court numbers (1, 2, 3, ...) to each court based on sorted order
    for i, court in enumerate(courts):
        court_id = i + 1  # Start numbering from 1
        court['court_number'] = court_id
        # Extract original label mask and assign new number
        court_region = (labels == court['id']).astype(np.uint8) * court_id
        court_mask = cv2.add(court_mask, court_region)
    
    return court_mask, courts

def main():
    """Main function optimized for Raspberry Pi Zero"""
    # Start timer
    start_time = time.time()
    
    # Reset any previously tracked logs
    OutputManager.reset_logs()
    
    # Initialize multiprocessing if enabled
    if Config.MultiProcessing.ENABLED:
        # Set the number of processes if not already set
        if Config.MultiProcessing.NUM_PROCESSES <= 0:
            # Use 3 processes or the number of CPU cores if less than 3
            Config.MultiProcessing.NUM_PROCESSES = min(3, cpu_count())
        
        OutputManager.log(f"Multiprocessing enabled with {Config.MultiProcessing.NUM_PROCESSES} processes", "INFO")
    
    try:
        # Load image
        input_path = Config.Paths.input_path()
        try:
            # First ensure the images directory exists
            images_dir = os.path.dirname(input_path)
            if not os.path.exists(images_dir):
                try:
                    os.makedirs(images_dir, exist_ok=True)
                    OutputManager.log(f"Created images directory at {images_dir}", "INFO")
                except Exception as e:
                    OutputManager.log(f"Cannot create images directory: {str(e)}", "ERROR")
            
            # Load the image
            OutputManager.status("Loading image")
            image = cv2.imread(input_path)
            
            # Check image loaded successfully
            if image is not None:
                OutputManager.log(f"Image loaded successfully: {image.shape[1]}x{image.shape[0]} pixels", "SUCCESS")
                if Config.Output.EXTRA_VERBOSE:
                    OutputManager.log(f"Image type: {image.dtype}, channels: {image.shape[2]}", "INFO")
            else:
                OutputManager.log(f"Unable to open the image at {input_path}", "ERROR")
                # Show final summary with error and exit
                processing_time = time.time() - start_time
                final_summary = OutputManager.create_final_summary(
                    people_count=None, 
                    court_counts={}, 
                    output_path=None,
                    processing_time=processing_time,
                    total_courts=0
                )
                print_error_summary(final_summary)
                return 1
        except Exception as e:
            OutputManager.log(f"Problem loading the image: {str(e)}", "ERROR")
            processing_time = time.time() - start_time
            final_summary = OutputManager.create_final_summary(
                people_count=None, 
                court_counts={}, 
                output_path=None,
                processing_time=processing_time,
                total_courts=0
            )
            print_error_summary(final_summary)
            return 1
        
        # Set up debug folder
        try:
            debug_folder = Config.Paths.debug_dir()
            os.makedirs(debug_folder, exist_ok=True)
            OutputManager.log(f"Debug folder created at {debug_folder}", "DEBUG")
        except Exception as e:
            OutputManager.log(f"Can't create debug folder: {str(e)}", "WARNING")
            debug_folder = None  # Set to None to prevent further debug saves
            # Continue execution even if debug folder can't be created
        
        # Detect tennis courts
        try:
            OutputManager.status("Analyzing court colors")
            blue_mask = create_blue_mask(image)
            green_mask = create_green_mask(image)
            OutputManager.log("Court colors analyzed", "SUCCESS")
            if Config.Output.EXTRA_VERBOSE:
                OutputManager.log(f"Blue mask: {np.count_nonzero(blue_mask)} pixels, Green mask: {np.count_nonzero(green_mask)} pixels", "INFO")
            
            # Process the raw blue mask to avoid connecting unrelated areas like the sky
            blue_mask_raw = blue_mask.copy()
            
            # Create court mask where green overrides blue
            height, width = image.shape[:2]
            court_mask = np.zeros((height, width), dtype=np.uint8)
            court_mask[blue_mask_raw > 0] = 1  # Blue areas
            court_mask[green_mask > 0] = 0     # Green areas override blue
            
            # Filter out blue regions that don't have any green nearby (like sky)
            OutputManager.status("Processing court regions")
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blue_mask_raw, connectivity=8)
            OutputManager.log(f"Found {num_labels-1} connected blue regions", "INFO")
            
            # For each blue region, check if there's green nearby
            filtered_court_mask = np.zeros_like(court_mask)
            valid_regions = 0
            
            for i in range(1, num_labels):
                region = (labels == i).astype(np.uint8)
                area = stats[i, cv2.CC_STAT_AREA]
                
                # Skip very small regions
                if area < Config.Court.MIN_AREA:
                    continue
                
                # Dilate the region to check for nearby green
                kernel = np.ones((15, 15), np.uint8)
                dilated_region = cv2.dilate(region, kernel, iterations=1)
                
                # Check if there's green nearby this blue region
                green_nearby = cv2.bitwise_and(green_mask, dilated_region)
                green_nearby_pixels = cv2.countNonZero(green_nearby)
                
                # Only keep blue regions that have at least some green nearby
                if green_nearby_pixels > 30:  # Reduced from 50 to be more lenient
                    # This is likely a court (not sky) - keep it
                    filtered_court_mask[region > 0] = court_mask[region > 0]
                    valid_regions += 1
                    if Config.Output.EXTRA_VERBOSE:
                        OutputManager.log(f"Region {i}: area={area}, green nearby={green_nearby_pixels} - likely court", "DEBUG")
            
            OutputManager.log(f"Court regions processed: {valid_regions} valid regions found", "SUCCESS")
            
            # Use the filtered court mask for further processing
            court_mask = filtered_court_mask
        except Exception as e:
            OutputManager.log(f"Error processing court colors: {str(e)}", "ERROR")
            # Continue with blank masks as a fallback
            height, width = image.shape[:2]
            blue_mask_raw = np.zeros((height, width), dtype=np.uint8)
            green_mask = np.zeros((height, width), dtype=np.uint8)
            court_mask = np.zeros((height, width), dtype=np.uint8)
        
        # Save raw masks for debugging
        if debug_folder:
            try:
                cv2.imwrite(os.path.join(debug_folder, "blue_mask_raw.png"), blue_mask_raw)
                cv2.imwrite(os.path.join(debug_folder, "green_mask.png"), green_mask)
                cv2.imwrite(os.path.join(debug_folder, "filtered_court_mask.png"), court_mask * 255)
                OutputManager.log("Debug masks saved", "DEBUG")
            except Exception as e:
                OutputManager.log(f"Couldn't save debug masks: {str(e)}", "WARNING")
        
        # Create colored visualization of masks
        try:
            court_mask_viz = np.zeros((height, width, 3), dtype=np.uint8)
            court_mask_viz[blue_mask_raw > 0] = [255, 0, 0]  # Blue for all blue areas
            court_mask_viz[green_mask > 0] = [0, 255, 0]     # Green areas override blue
            
            # Highlight filtered courts in a brighter blue
            filtered_blue = np.zeros_like(court_mask_viz)
            filtered_blue[court_mask > 0] = [255, 127, 0]  # Bright blue for valid courts
            cv2.addWeighted(court_mask_viz, 1, filtered_blue, 0.7, 0, court_mask_viz)
        except Exception as e:
            OutputManager.log(f"Error creating court visualization: {str(e)}", "WARNING")
            court_mask_viz = image.copy()  # Use original image as fallback
        
        # Assign court numbers to each separate blue region
        try:
            OutputManager.status("Identifying courts")
            court_numbers_mask, courts = assign_court_numbers(court_mask)
            
            # Output appropriate message based on court detection
            if len(courts) == 0:
                OutputManager.log("No tennis courts found in the image", "WARNING")
            else:
                OutputManager.log(f"Found {len(courts)} tennis court{'s' if len(courts) > 1 else ''}", "SUCCESS")
                if Config.Output.EXTRA_VERBOSE:
                    # Log details of each court
                    for i, court in enumerate(courts):
                        cx, cy = court['centroid']
                        area = court['area']
                        OutputManager.log(f"Court {i+1}: center=({cx}, {cy}), area={area:.1f} pixels", "DEBUG")
        except Exception as e:
            OutputManager.log(f"Error identifying courts: {str(e)}", "ERROR")
            # Create fallback empty data
            courts = []
            court_numbers_mask = np.zeros_like(court_mask)
        
        # Create a color-coded court mask for visualization
        try:
            court_viz = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Assign different colors to each court
            court_colors = [
                (255, 0, 0),    # Blue
                (0, 0, 255),    # Red
                (255, 0, 255),  # Purple
                (0, 255, 255)   # Yellow
            ]
            
            # Draw each court with a different color
            for court in courts:
                court_id = court['court_number']
                color_idx = (court_id - 1) % len(court_colors)
                court_color = court_colors[color_idx]
                
                # Extract court mask
                court_mask_individual = (court_numbers_mask == court_id).astype(np.uint8) * 255
                # Find contours of the court
                court_contours, _ = cv2.findContours(court_mask_individual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw the court area
                court_area = np.zeros_like(court_viz)
                court_area[court_mask_individual > 0] = court_color
                cv2.addWeighted(court_viz, 1, court_area, 0.7, 0, court_viz)
                
                # Draw court number at center only if enabled in debug visualizations too
                if hasattr(Config.Visual, 'SHOW_COURT_LABELS') and Config.Visual.SHOW_COURT_LABELS:
                    cx, cy = int(court['centroid'][0]), int(court['centroid'][1])
                    cv2.putText(court_viz, f"Court {court_id}", (cx-40, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        except Exception as e:
            OutputManager.log(f"Error creating court visualization: {str(e)}", "WARNING")
            court_viz = image.copy()  # Use original image as fallback
        
        # Save court visualization
        if debug_folder:
            try:
                cv2.imwrite(os.path.join(debug_folder, "courts_numbered.png"), court_viz)
                OutputManager.log("Court visualization saved", "DEBUG")
            except Exception as e:
                OutputManager.log(f"Couldn't save court visualization: {str(e)}", "WARNING")
        
        # Create a semi-transparent overlay of the masks on the original image
        try:
            alpha = 0.5  # Transparency factor
            mask_overlay = image.copy()
            # Apply the colored masks with transparency
            cv2.addWeighted(court_mask_viz, alpha, mask_overlay, 1 - alpha, 0, mask_overlay)
        except Exception as e:
            OutputManager.log(f"Error creating mask overlay: {str(e)}", "WARNING")
            mask_overlay = image.copy()  # Use original image as fallback
        
        # Detect people
        people = []
        try:
            OutputManager.status("Looking for people")
            
            # Check if models directory exists
            models_dir = Config.Paths.MODELS_DIR
            if not os.path.exists(models_dir):
                try:
                    os.makedirs(models_dir, exist_ok=True)
                    OutputManager.log(f"Created models directory at {models_dir}", "INFO")
                except Exception as e:
                    OutputManager.log(f"Cannot create models directory: {str(e)}", "ERROR")
            
            # Get the model name from config and download if needed
            model_name = Config.Model.NAME
            OutputManager.log(f"Using model: {model_name}", "INFO")
            
            # Check if SSL verification should be disabled
            disable_ssl = False
            if hasattr(args, 'disable_ssl_verify') and args.disable_ssl_verify:
                disable_ssl = True
            
            try:
                # Download model if it doesn't exist (new function)
                model_path = download_yolo_model(
                    model_name, 
                    url=Config.Model.get_model_url(model_name),
                    disable_ssl_verify=disable_ssl
                )
            except Exception as e:
                # If YOLOv8 model fails, try YOLOv5s as fallback
                if model_name != "yolov5s":
                    OutputManager.log(f"Falling back to YOLOv5s model after error with {model_name}", "WARNING")
                    try:
                        model_path = download_yolo_model(
                            "yolov5s", 
                            url=Config.Model.get_model_url("yolov5s"),
                            disable_ssl_verify=True  # Force disable SSL for fallback
                        )
                    except Exception as e2:
                        raise Exception(f"Failed to download fallback model: {str(e2)}")
                else:
                    raise e
            
            # Load the YOLO model with better error handling
            try:
                with suppress_stdout_stderr():
                    OutputManager.status(f"Loading {model_name} model")
                    
                    # Determine if this is YOLOv5 or YOLOv8
                    is_yolov8 = model_name.startswith("yolov8")
                    
                    if is_yolov8:
                        try:
                            # Try to import ultralytics
                            from ultralytics import YOLO # type: ignore
                            model = YOLO(model_path)
                            OutputManager.log(f"{model_name} model loaded with Ultralytics API", "SUCCESS")
                        except ImportError:
                            # Fallback to YOLOv5 hub if ultralytics not installed
                            OutputManager.log("Ultralytics not installed, falling back to YOLOv5 hub", "WARNING")
                            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=False)
                    else:
                        # Use YOLOv5 hub for loading
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=False)
                    
                    # Set confidence and IoU thresholds
                    if not is_yolov8 or not isinstance(model, type(lambda: None)):  # Check if not a callable (ultralytics YOLO object)
                        model.conf = Config.Model.CONFIDENCE
                        model.iou = Config.Model.IOU
                        model.classes = Config.Model.CLASSES
                    
                    OutputManager.log(f"{model_name} model loaded successfully", "SUCCESS")
            except Exception as e:
                # Handle SSL certificate errors
                if "ssl" in str(e).lower():
                    OutputManager.log("SSL certificate error, trying with verification disabled", "WARNING")
                    # Try again with SSL verification disabled
                    ssl._create_default_https_context = ssl._create_unverified_context
                    with suppress_stdout_stderr():
                        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=False, force_reload=True)
                        model.conf = Config.Model.CONFIDENCE
                        model.iou = Config.Model.IOU
                        model.classes = Config.Model.CLASSES
                else:
                    raise e
            
            # Run detection
            OutputManager.status("Running person detection with YOLOv5")
            with suppress_stdout_stderr():
                try:
                    # Force CPU device on Raspberry Pi Zero
                    model.cpu()
                    OutputManager.log("Using CPU for inference (optimized for Raspberry Pi)", "INFO")
                    results = model(image)
                except RuntimeError as e:
                    # Check for memory errors
                    if "out of memory" in str(e).lower():
                        OutputManager.log("Memory error, trying with smaller image", "WARNING")
                        # Try scaling the image down
                        scale_factor = 0.5  # Scale to 50%
                        small_img = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
                        results = model(small_img)
                        OutputManager.log(f"Used scaled image ({small_img.shape[1]}x{small_img.shape[0]}) for detection", "INFO")
                    else:
                        raise e
            
            # Process results
            OutputManager.status("Processing detection results")
            people = []
            
            # Handle different model result formats - YOLOv5 vs YOLOv8
            try:
                # Check if this is YOLOv8 model (ultralytics model)
                if model_name.startswith('yolov8'):
                    # YOLOv8 results format
                    results_data = results
                    
                    # Using YOLOv8 format (which is a different structure than YOLOv5)
                    if hasattr(results_data, 'boxes'):
                        # Ultralytics API format
                        for box in results_data.boxes:
                            if (len(Config.Model.CLASSES) == 0 or 
                                int(box.cls.item()) in Config.Model.CLASSES):
                                
                                conf = box.conf.item()
                                if conf >= Config.Model.CONFIDENCE:
                                    xyxy = box.xyxy.cpu().numpy()[0]
                                    x1, y1, x2, y2 = map(int, xyxy)
                                    
                                    # Calculate center point and foot position
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    foot_x = center_x
                                    foot_y = y2  # Bottom of bounding box represents feet
                                    
                                    # Add to people list
                                    people.append({
                                        'position': (center_x, center_y),
                                        'foot_position': (foot_x, foot_y),
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': conf
                                    })
                    elif isinstance(results_data, list) and len(results_data) > 0:
                        # Alternative format sometimes returned by YOLOv8
                        for det in results_data[0]:
                            if len(det) >= 6:  # xyxy, conf, cls
                                x1, y1, x2, y2 = map(int, det[:4])
                                conf = float(det[4])
                                cls_id = int(det[5])
                                
                                if (len(Config.Model.CLASSES) == 0 or 
                                    cls_id in Config.Model.CLASSES) and conf >= Config.Model.CONFIDENCE:
                                    
                                    # Calculate center point and foot position
                                    center_x = (x1 + x2) // 2
                                    center_y = (y1 + y2) // 2
                                    foot_x = center_x
                                    foot_y = y2  # Bottom of bounding box represents feet
                                    
                                    # Add to people list
                                    people.append({
                                        'position': (center_x, center_y),
                                        'foot_position': (foot_x, foot_y),
                                        'bbox': (x1, y1, x2, y2),
                                        'confidence': conf
                                    })
                else:
                    # YOLOv5 results format - using pandas
                    df = results.pandas().xyxy[0]
                    df = df[df['class'] == 0]
                    
                    for _, row in df.iterrows():
                        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                        
                        # Calculate center point and foot position
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        foot_x = center_x
                        foot_y = y2  # Bottom of bounding box represents feet
                        
                        # Add to people list
                        people.append({
                            'position': (center_x, center_y),
                            'foot_position': (foot_x, foot_y),
                            'bbox': (x1, y1, x2, y2),
                            'confidence': row['confidence']
                        })
            except Exception as e:
                error_msg = str(e)
                
                # Add more specific error messages for YOLOv8 vs YOLOv5 issues
                if "'list' object has no attribute 'pandas'" in error_msg:
                    OutputManager.log("YOLOv8 result format error. The script needs to be updated for YOLOv8 format.", "ERROR")
                    OutputManager.log("Try using a YOLOv5 model instead with: --model yolov5s", "INFO")
                elif "no attribute 'numpy'" in error_msg or "tensor" in error_msg.lower():
                    OutputManager.log("YOLOv8 tensor processing error.", "ERROR")
                    OutputManager.log("Try using a YOLOv5 model instead with: --model yolov5s", "INFO")
                else:
                    OutputManager.log(f"Problem detecting people: {error_msg}", "ERROR")
                
                # Continue with empty people list
                people = []
            
            # Report how many people we found
            OutputManager.log(f"Found {len(people)} {'person' if len(people) == 1 else 'people'} in the image", "SUCCESS")
            if Config.Output.EXTRA_VERBOSE and people:
                # Log details of detected people
                for i, person in enumerate(people):
                    x1, y1, x2, y2 = person['bbox']
                    conf = person['confidence']
                    OutputManager.log(f"Person {i+1}: bbox=({x1},{y1},{x2},{y2}), confidence={conf:.2f}", "DEBUG")
        except Exception as e:
            error_msg = str(e)
            
            # Handle different types of errors with specific messages
            if "model" in error_msg.lower() or "yolo" in error_msg.lower() or "no such file" in error_msg.lower():
                OutputManager.log(f"YOLOv5 model not found: {error_msg}", "ERROR")
                OutputManager.log("Model missing - run: mkdir -p models && wget -q https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt -O models/yolov5s.pt", "ERROR")
            elif "cuda" in error_msg.lower() or "gpu" in error_msg.lower():
                OutputManager.log(f"CUDA error: {error_msg}", "ERROR")
                OutputManager.log("CUDA error - run: pip install -r requirements.txt", "ERROR")
            elif "ssl" in error_msg.lower():
                OutputManager.log(f"SSL error: {error_msg}", "ERROR")
                OutputManager.log("SSL error - run: pip install certifi --upgrade", "ERROR")
            elif "memory" in error_msg.lower():
                OutputManager.log(f"Memory error: {error_msg}", "ERROR")
                OutputManager.log("Memory error - try using a smaller image or reducing batch size", "ERROR")
            elif "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                OutputManager.log(f"Network error: {error_msg}", "ERROR")
                OutputManager.log("Network error - check your internet connection and try again", "ERROR")
            else:
                OutputManager.log(f"Problem detecting people: {error_msg}", "ERROR")
            
            # Continue with empty people list
            people = []
        
        # Determine if each person is on a court
        people_locations = []
        try:
            if people and courts:
                OutputManager.status("Analyzing positions using multiprocessing")
                
                # Process in parallel using our optimized function
                people_locations = analyze_people_positions_parallel(people, courts)
                
                OutputManager.log("Positions analyzed successfully", "SUCCESS")
                
                # Count people by location type
                in_bounds_count = sum(1 for _, area_type in people_locations if area_type == 'in_bounds')
                out_bounds_count = sum(1 for _, area_type in people_locations if area_type == 'out_bounds')
                off_court_count = sum(1 for _, area_type in people_locations if area_type == 'off_court')
                
                OutputManager.log(f"Position breakdown: {in_bounds_count} in-bounds, {out_bounds_count} sidelines, {off_court_count} off-court", "INFO")
            else:
                # If no people or no courts, no need to analyze positions
                for _ in range(len(people)):
                    people_locations.append((-1, 'off_court'))
        except Exception as e:
            OutputManager.log(f"Error analyzing positions: {str(e)}", "ERROR")
            # Create fallback position data
            for _ in range(len(people)):
                people_locations.append((-1, 'off_court'))
        
        # Calculate court counts for summary
        court_counts = {}
        for court_idx, area_type in people_locations:
            if court_idx >= 0:
                court_num = court_idx + 1
                if court_num not in court_counts:
                    court_counts[court_num] = 0
                court_counts[court_num] += 1
        
        # Log court counts
        for court_num in sorted(court_counts.keys()):
            OutputManager.log(f"Court {court_num}: {court_counts[court_num]} people", "INFO")
        
        # Create debug visualization showing foot positions on mask
        if debug_folder and people:
            try:
                debug_foot_positions = court_viz.copy()
                for person_idx, person in enumerate(people):
                    if 'foot_position' in person:
                        foot_x, foot_y = person['foot_position']
                        # Draw foot position marker (circle)
                        cv2.circle(debug_foot_positions, (foot_x, foot_y), 10, (255, 255, 255), -1)
                        cv2.circle(debug_foot_positions, (foot_x, foot_y), 10, (0, 0, 0), 2)
                        # Label with person index
                        cv2.putText(debug_foot_positions, f"P{person_idx+1}", (foot_x+15, foot_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                cv2.imwrite(os.path.join(debug_folder, "foot_positions_debug.png"), debug_foot_positions)
                OutputManager.log("Foot positions debug image saved", "DEBUG")
            except Exception as e:
                OutputManager.log(f"Couldn't save foot positions debug image: {str(e)}", "WARNING")
        
        # Create final output image
        try:
            OutputManager.status("Creating output image")
            output_image = image.copy()
            
            # Draw court outlines with different colors
            for court in courts:
                court_id = court['court_number']
                color_idx = (court_id - 1) % len(court_colors)
                court_color = court_colors[color_idx]
                
                # Extract court mask
                court_mask_individual = (court_numbers_mask == court_id).astype(np.uint8) * 255
                # Find contours of the court
                court_contours, _ = cv2.findContours(court_mask_individual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Draw the court outline
                cv2.drawContours(output_image, court_contours, -1, court_color, 2)
                
                # Draw court number at center only if enabled
                if Config.Visual.SHOW_COURT_NUMBER:
                    cx, cy = int(court['centroid'][0]), int(court['centroid'][1])
                    cv2.putText(output_image, f"Court {court_id}", (cx-40, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Draw people and their locations
            for i, person in enumerate(people):
                court_idx, area_type = people_locations[i]
                
                # Draw bounding box and label
                x1, y1, x2, y2 = person['bbox']
                
                # Choose color based on location
                if court_idx >= 0:
                    court_number = court_idx + 1
                    if area_type == 'in_bounds':
                        color = Config.Visual.PERSON_IN_BOUNDS_COLOR
                        label = f"Court {court_number}" if Config.Visual.SHOW_DETAILED_LABELS else ""
                    else:  # out_bounds
                        color = Config.Visual.PERSON_OUT_BOUNDS_COLOR
                        label = f"Court {court_number} • Sideline" if Config.Visual.SHOW_DETAILED_LABELS else ""
                else:
                    color = Config.Visual.PERSON_OFF_COURT_COLOR
                    label = "Not on court" if Config.Visual.SHOW_DETAILED_LABELS else ""
                
                # Draw bounding box
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw foot position marker - smaller and less intrusive
                foot_x, foot_y = person['foot_position']
                cv2.circle(output_image, (foot_x, foot_y), 3, color, -1)
                
                # Only draw text labels if specified
                if Config.Visual.SHOW_DETAILED_LABELS and label:
                    # Draw label with black background for readability
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                               Config.Visual.FONT_SCALE, 
                                               Config.Visual.TEXT_THICKNESS)[0]
                    cv2.rectangle(output_image, (x1, y1 - text_size[1] - 5), 
                                 (x1 + text_size[0], y1), color, -1)
                    cv2.putText(output_image, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               Config.Visual.FONT_SCALE, 
                               Config.Visual.TEXT_COLOR, 
                               Config.Visual.TEXT_THICKNESS)
                    
                    # Add person index number
                    cv2.putText(output_image, f"Person {i+1}", (x1, y2 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, Config.Visual.FONT_SCALE, 
                                color, Config.Visual.TEXT_THICKNESS)
                else:
                    # Just add a small number indicator for simpler display
                    cv2.putText(output_image, f"{i+1}", (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               Config.Visual.FONT_SCALE, 
                               Config.Visual.TEXT_COLOR, 
                               Config.Visual.TEXT_THICKNESS)
            
            OutputManager.log("Output image generated", "SUCCESS")
        except Exception as e:
            OutputManager.log(f"Error creating output image: {str(e)}", "ERROR")
            output_image = image.copy()  # Use original image as fallback
        
        # Save the final output image
        output_path = Config.Paths.output_path()
        try:
            OutputManager.status("Saving output image")
            
            # Ensure output directory exists
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir, exist_ok=True)
                    OutputManager.log(f"Created output directory at {output_dir}", "INFO")
                except Exception as e:
                    OutputManager.log(f"Cannot create output directory: {str(e)}", "ERROR")
            
            cv2.imwrite(output_path, output_image)
            OutputManager.log(f"Output image saved successfully to {output_path}", "SUCCESS")
        except Exception as e:
            OutputManager.log(f"Error saving output image: {str(e)}", "ERROR")
            output_path = None
        
        # Create the adaptive final summary
        processing_time = time.time() - start_time
        OutputManager.log(f"Total processing time: {processing_time:.2f} seconds", "INFO")
        
        final_summary = OutputManager.create_final_summary(
            people_count=len(people),
            court_counts=court_counts,
            output_path=output_path,
            processing_time=None,  # Will be added by fancy_summary
            total_courts=len(courts)  # Pass the total number of courts
        )
        
        # Use the fancy summary method
        OutputManager.fancy_summary(
            "RESULTS SUMMARY", 
            final_summary, 
            processing_time=processing_time
        )
        
        # If there were errors that didn't cause a fatal exit, still indicate an error status
        if OutputManager.errors:
            return 1
        
        return 0
    except Exception as e:
        # This is the main catch-all for any unhandled exceptions in the try block
        OutputManager.log(f"Unhandled error in main function: {str(e)}", "ERROR")
        
        # Create a basic summary with the error
        processing_time = time.time() - start_time
        final_summary = OutputManager.create_final_summary(
            people_count=None, 
            court_counts={}, 
            output_path=None,
            processing_time=processing_time,
            total_courts=0
        )
        print_error_summary(final_summary)
        return 1

# Add a backward compatibility wrapper for the old log function
def log(message, level="INFO"):
    """Wrapper for backward compatibility with the old log function"""
    OutputManager.log(message, level)

def print_error_summary(summary):
    """Print error summary with the fancy box style and troubleshooting information"""
    # First clear any lingering output
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    
    # Add troubleshooting section if errors are present
    if OutputManager.errors:
        # Get the potential fixes
        fixes = OutputManager.get_potential_fixes()
        if fixes:
            if "TROUBLESHOOTING" not in summary:
                summary += "\n\nTROUBLESHOOTING"
                for fix_line in fixes.split('\n'):
                    summary += f"\n{fix_line}"
        
        # Add common checking steps if not already present
        if "requirements.txt" not in summary:
            summary += "\n\nBASIC CHECKS:"
            summary += "\n1. Ensure all dependencies are installed: pip install -r requirements.txt"
            summary += "\n2. Check that the input image exists and is readable"
            summary += "\n3. Verify YOLOv5 model is downloaded in models/ directory"
            summary += "\n4. Check for sufficient disk space and permissions"
            summary += "\n5. Run with --debug flag for more detailed information"
    
    # Use the fancy summary with an error title
    if OutputManager.errors:
        OutputManager.fancy_summary("ERROR SUMMARY", summary)
    else:
        OutputManager.fancy_summary("WARNING SUMMARY", summary)
    
    # Flush to ensure immediate display
    sys.stdout.flush()

def download_yolo_model(model_name, url=None, disable_ssl_verify=False):
    """Download YOLO model if it doesn't exist"""
    # Create models directory if it doesn't exist
    os.makedirs(Config.Paths.MODELS_DIR, exist_ok=True)
    
    model_path = os.path.join(Config.Paths.MODELS_DIR, f"{model_name}.pt")
    
    # Check if model exists
    if os.path.exists(model_path):
        OutputManager.log(f"Model {model_name} found at {model_path}", "SUCCESS")
        return model_path
    
    # Get URL if not provided
    if url is None:
        url = Config.Model.get_model_url(model_name)
    
    OutputManager.log(f"Model {model_name} not found at {model_path}", "WARNING")
    OutputManager.status(f"Downloading {model_name} from {url}")
    
    try:
        # Handle SSL verification
        if disable_ssl_verify:
            # Disable SSL verification entirely
            ssl._create_default_https_context = ssl._create_unverified_context
            OutputManager.log("SSL verification disabled for download", "WARNING")
        elif sys.platform == 'darwin':  # macOS specific fix
            # On macOS, install certificates if needed
            try:
                import certifi # type: ignore
                ssl_context = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                # If certifi is not installed, try to use macOS certificates
                OutputManager.log("Installing SSL certificates for macOS...", "INFO")
                import subprocess
                subprocess.call(['/Applications/Python 3.x/Install Certificates.command'], shell=True)
                # If that fails, we'll continue without it and might encounter errors
        
        # Create a progress reporter for the download
        def report_progress(blocknum, blocksize, totalsize):
            if totalsize > 0:
                percent = min(blocknum * blocksize * 100 / totalsize, 100)
                OutputManager.status(f"Downloading {model_name}: {percent:.1f}% of {totalsize/1024/1024:.1f} MB")
        
        # Download the file
        try:
            urllib.request.urlretrieve(url, model_path, report_progress)
            OutputManager.log(f"Model {model_name} downloaded successfully", "SUCCESS")
            return model_path
        except (urllib.error.URLError, ssl.SSLError) as e:
            OutputManager.log(f"Error downloading model: {str(e)}", "ERROR")
            
            # Special handling for SSL certificate errors
            if isinstance(e, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(e):
                OutputManager.log("SSL certificate verification failed. Try running with --disable-ssl-verify", "WARNING")
                OutputManager.log("Alternatively, you can manually download the model:", "INFO")
                OutputManager.log(f"1. Download from: {url}", "INFO")
                OutputManager.log(f"2. Save it to: {model_path}", "INFO")
                
                # For macOS users, provide specific instructions
                if sys.platform == 'darwin':
                    OutputManager.log("For macOS users, run the following command in Terminal:", "INFO")
                    OutputManager.log("/Applications/Python 3.x/Install Certificates.command", "INFO")
                    OutputManager.log("Replace '3.x' with your Python version (e.g., 3.9, 3.10)", "INFO")
            
            # Try an alternative URL if this is a YOLOv8 model
            if model_name.startswith("yolov8") and "ultralytics/assets" in url:
                alt_url = f"https://github.com/ultralytics/ultralytics/releases/download/v8.0.0/{model_name}.pt"
                OutputManager.log(f"Trying alternative URL for {model_name}: {alt_url}", "INFO")
                try:
                    urllib.request.urlretrieve(alt_url, model_path, report_progress)
                    OutputManager.log(f"Model {model_name} downloaded successfully from alternative URL", "SUCCESS")
                    return model_path
                except Exception as e2:
                    OutputManager.log(f"Alternative download also failed: {str(e2)}", "ERROR")
            
            raise Exception(f"Model file not found and could not be downloaded. Error: {str(e)}")
    
    except Exception as e:
        # Clean up any partially downloaded file
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Log the error and re-raise
        OutputManager.log(f"Failed to download {model_name}: {str(e)}", "ERROR")
        raise

if __name__ == "__main__":
    try:
        # Add command-line arguments for easier use
        parser = argparse.ArgumentParser(description="Tennis Court Detection System")
        parser.add_argument("--input", type=str, help="Path to input image", default=Config.Paths.input_path())
        parser.add_argument("--output", type=str, help="Path for output image", default=Config.Paths.output_path())
        parser.add_argument("--debug", action="store_true", help="Enable debug mode with additional outputs")
        parser.add_argument("--quiet", action="store_true", help="Reduce console output")
        parser.add_argument("--show-labels", action="store_true", help="Show detailed labels on output image")
        parser.add_argument("--show-court-labels", action="store_true", help="Show court numbers on output image")
        parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Device to use for inference", default=None)
        parser.add_argument("--disable-ssl-verify", action="store_true", help="Disable SSL verification for downloads")
        parser.add_argument("--model", type=str, help="YOLO model to use (yolov5s, yolov5m, yolov5l, etc.)", default=Config.Model.NAME)
        parser.add_argument("--no-multiprocessing", action="store_true", help="Disable multiprocessing")
        parser.add_argument("--processes", type=int, help="Number of processes to use for multiprocessing", default=Config.MultiProcessing.NUM_PROCESSES)
        parser.add_argument("--extra-verbose", action="store_true", help="Show extra detailed output for debugging")
        parser.add_argument("--force-macos-cert-install", action="store_true", help="Force macOS certificate installation")
        
        # Parse arguments
        try:
            args = parser.parse_args()
        except Exception as e:
            print(f"\nError parsing command-line arguments: {str(e)}")
            print("Run with --help for usage information")
            sys.exit(1)
        
        # Handle SSL verification setting early
        if args.disable_ssl_verify:
            ssl._create_default_https_context = ssl._create_unverified_context
            print("SSL verification disabled")
        
        # Handle macOS certificate installation
        if args.force_macos_cert_install and sys.platform == 'darwin':
            print("Attempting to install macOS certificates...")
            try:
                import subprocess
                subprocess.call(['/Applications/Python 3.9/Install Certificates.command'], shell=True)
                print("Certificate installation attempted. If it doesn't work, try with Python 3.10 or 3.11 instead.")
            except Exception as e:
                print(f"Certificate installation failed: {str(e)}")
                print("Try manually running the 'Install Certificates.command' script in your Python application folder.")
        
        # Update config based on arguments
        try:
            if args.input != Config.Paths.input_path():
                Config.Paths.INPUT_IMAGE = os.path.basename(args.input)
                Config.Paths.IMAGES_DIR = os.path.dirname(args.input)
            
            if args.output != Config.Paths.output_path():
                Config.Paths.OUTPUT_IMAGE = os.path.basename(args.output)
                Config.Paths.IMAGES_DIR = os.path.dirname(args.output)
            
            Config.DEBUG_MODE = args.debug
            Config.Output.VERBOSE = not args.quiet
            Config.Visual.SHOW_DETAILED_LABELS = args.show_labels
            Config.Visual.SHOW_COURT_LABELS = args.show_court_labels
            
            # Update model name if specified
            if args.model:
                Config.Model.NAME = args.model
                print(f"Using model: {Config.Model.NAME}")
            
            # Update multiprocessing settings
            Config.MultiProcessing.ENABLED = not args.no_multiprocessing
            if args.processes > 0:
                Config.MultiProcessing.NUM_PROCESSES = args.processes
            
            # Update extra verbose setting
            if args.extra_verbose:
                Config.Output.EXTRA_VERBOSE = True
            
            sys.exit(main())
        except Exception as e:
            print(f"\nError setting up configuration: {str(e)}")
            sys.exit(1)
    except ModuleNotFoundError as e:
        # For missing modules, provide direct installation instructions
        module_name = str(e).split("'")[1] if "'" in str(e) else "unknown"
        error_message = f"Missing module: {module_name}"
        install_cmd = ""
        
        # Provide specific installation commands for common modules
        if module_name == "cv2":
            if sys.platform == "linux" and os.uname().machine.startswith('arm'):
                # Raspberry Pi specific OpenCV installation
                install_cmd = "sudo apt update && sudo apt install -y python3-opencv\n\nOR for full requirements:\n\nsudo apt update && sudo apt install -y python3-opencv python3-numpy\npip3 install torch torchvision shapely"
            else:
                install_cmd = "pip install opencv-python"
        elif module_name == "numpy":
            if sys.platform == "linux" and os.uname().machine.startswith('arm'):
                install_cmd = "sudo apt update && sudo apt install -y python3-numpy"
            else:
                install_cmd = "pip install numpy"
        elif module_name == "torch":
            if sys.platform == "linux" and os.uname().machine.startswith('arm'):
                install_cmd = "pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
            else:
                install_cmd = "pip install torch torchvision"
        elif module_name == "shapely":
            if sys.platform == "linux" and os.uname().machine.startswith('arm'):
                install_cmd = "sudo apt update && sudo apt install -y python3-shapely\n\nOR\n\npip3 install shapely"
            else:
                install_cmd = "pip install shapely"
        else:
            install_cmd = f"pip install {module_name}\n\nOr to install all dependencies:\npip install -r requirements.txt"
        
        # Create a simple formatted box to display the error
        print("\n" + "╭" + "─" * 78 + "╮")
        print("│ " + "ERROR: MODULE NOT FOUND".center(78) + " │")
        print("│ " + "─" * 78 + " │")
        print("│ " + error_message.ljust(78) + " │")
        print("│ " + "─" * 78 + " │")
        print("│ " + "To fix this error, run:".ljust(78) + " │")
        for line in install_cmd.split('\n'):
            print("│ " + line.ljust(78) + " │")
            
        # Add a one-line shortcut for all dependencies
        if module_name in ["cv2", "numpy", "torch", "shapely"]:
            if sys.platform == "linux" and os.uname().machine.startswith('arm'):
                print("│ " + "─" * 78 + " │")
                print("│ " + "QUICK FIX FOR RASPBERRY PI:".ljust(78) + " │")
                print("│ " + "sudo apt update && sudo apt install -y python3-opencv python3-numpy".ljust(78) + " │")
                print("│ " + "python3-shapely python3-pip && pip3 install torch torchvision".ljust(78) + " │")
        
        print("╰" + "─" * 78 + "╯")
        sys.exit(1)
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\n\n" + "╭" + "─" * 78 + "╮")
        print("│ " + "PROCESS INTERRUPTED BY USER".center(78) + " │")
        print("╰" + "─" * 78 + "╯")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        # For other unhandled exceptions, provide a generic error message
        error_message = str(e)
        possible_solution = ""
        
        # Try to provide helpful solutions for common errors
        if "permission" in error_message.lower():
            if sys.platform == "win32":
                possible_solution = "Run as Administrator or check file permissions"
            else:
                possible_solution = "Try: chmod +x main.py && sudo ./main.py"
        elif "disk" in error_message.lower() or "space" in error_message.lower():
            possible_solution = "Check available disk space and write permissions"
        elif "network" in error_message.lower() or "connection" in error_message.lower():
            possible_solution = "Check your internet connection"
        elif "import" in error_message.lower():
            possible_solution = "Run: pip install -r requirements.txt"
        elif "memory" in error_message.lower():
            possible_solution = "Try using a smaller image or reduce batch size"
        else:
            possible_solution = "Check requirements with: pip install -r requirements.txt"
        
        print("\n" + "╭" + "─" * 78 + "╮")
        print("│ " + "ERROR: UNHANDLED EXCEPTION".center(78) + " │")
        print("│ " + "─" * 78 + " │")
        
        # Split long error messages
        wrapped_error = []
        for chunk in [error_message[i:i+78] for i in range(0, len(error_message), 78)]:
            wrapped_error.append(chunk)
        
        for line in wrapped_error[:3]:  # Limit to 3 lines to avoid huge error messages
            print("│ " + line.ljust(78) + " │")
        
        if len(wrapped_error) > 3:
            print("│ " + "...".ljust(78) + " │")
        
        print("│ " + "─" * 78 + " │")
        print("│ " + "POSSIBLE SOLUTION:".ljust(78) + " │")
        for sol_line in [possible_solution[i:i+78] for i in range(0, len(possible_solution), 78)]:
            print("│ " + sol_line.ljust(78) + " │")
        print("╰" + "─" * 78 + "╯")
        sys.exit(1)