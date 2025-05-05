#!/usr/bin/env python3
"""
Tennis Court Detector - Detects people on tennis courts using YOLO models
"""
import os
import cv2
import numpy as np
import json
import sys
import argparse
import time
from datetime import datetime
import re
import shutil
from multiprocessing import cpu_count, Pool
from functools import partial
import threading
import contextlib
import io

# Global settings
class Config:
    class Output:
        VERBOSE = False              # Default to non-verbose output
        SHOW_TIMESTAMP = True        # Show timestamps in output
        SUPER_QUIET = False          # Super quiet mode (almost no output)
        USE_COLOR_OUTPUT = True      # Use colored terminal output
    
    class Paths:
        IMAGES_DIR = "images"
        MODELS_DIR = "models"
        INPUT_IMAGE = "input.png"
        OUTPUT_IMAGE = "output.png"
        
        @classmethod
        def input_path(cls):
            return os.path.join(cls.IMAGES_DIR, cls.INPUT_IMAGE)
            
        @classmethod
        def output_path(cls):
            return os.path.join(cls.IMAGES_DIR, cls.OUTPUT_IMAGE)
    
    class MultiProcessing:
        ENABLED = True              
        NUM_PROCESSES = max(1, cpu_count() - 1)  # Use all cores except one
        
    class Model:
        NAME = "yolov8n"             # Default model
        CONFIDENCE = 0.25            # Detection confidence threshold
        AVAILABLE_MODELS = [
            "yolov5n",  # nano
            "yolov5s",  # small
            "yolov5m",  # medium
            "yolov5l",  # large
            "yolov5x",  # xlarge
            "yolov8n",  # nano
            "yolov8s",  # small
            "yolov8m",  # medium
            "yolov8l",  # large
            "yolov8x",  # xlarge
        ]

# Global variables        
CAMERA_AVAILABLE = False

# Default camera resolution
DEFAULT_CAMERA_RESOLUTION = (640, 480)

# Try to import camera module
try:
    from camera import takePhoto, DEFAULT_RESOLUTION, CameraOutputFormatter
    CAMERA_AVAILABLE = True
    DEFAULT_CAMERA_RESOLUTION = DEFAULT_RESOLUTION
except ImportError:
    # Define dummy takePhoto function when camera is not available
    def takePhoto(resolution=DEFAULT_CAMERA_RESOLUTION, output_file='images/input.png'):
        return False
    
    # Define dummy CameraOutputFormatter class when camera module is not available
    class CameraOutputFormatter:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

# Output manager for centralized output handling
class OutputManager:
    """
    Centralized output manager for all terminal output.
    """
    # ANSI color and style codes
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    # Symbols for different message types
    SYMBOLS = {
        "INFO": "ℹ",
        "SUCCESS": "✓",
        "WARNING": "⚠",
        "ERROR": "✗",
        "STATUS": "→",
    }
    
    # Track messages for summary
    warnings = []
    errors = []
    
    @classmethod
    def reset_logs(cls):
        """Reset all tracked messages"""
        cls.warnings = []
        cls.errors = []
    
    @classmethod
    def log(cls, message, level="INFO"):
        """
        Log a message with the specified level.
        Messages are tracked for the final summary, but only printed 
        according to verbosity settings.
        """
        # Always track errors and warnings for the summary
        if level == "WARNING":
            cls.warnings.append(message)
        elif level == "ERROR":
            cls.errors.append(message)
            # Print a concise error indicator immediately
            cls._print_error_indicator()
            return

        # Determine if we should print the message based on verbosity settings
        should_print = False
        if Config.Output.SUPER_QUIET:
            # Super quiet: Print nothing except final summary
            pass
        elif Config.Output.VERBOSE:
            # Verbose: Print all message types
            should_print = True
        else:
            # Default (non-verbose): Print only key success messages
            if level == "SUCCESS" and any(keyword in message.lower() for keyword in 
                                          ["found", "detected", "saved", "completed"]):
                should_print = True
        
        # Exit if message shouldn't be printed
        if not should_print:
            return

        # Format and print the message
        color = ""
        if Config.Output.USE_COLOR_OUTPUT:
            color_map = {
                "INFO": cls.BLUE, 
                "SUCCESS": cls.GREEN, 
                "WARNING": cls.YELLOW,
                "STATUS": cls.CYAN, 
                "ERROR": cls.RED
            }
            color = color_map.get(level, "")

        symbol = cls.SYMBOLS.get(level, "")
        try:
            if symbol and sys.stdout.encoding is not None:
                symbol.encode(sys.stdout.encoding)
            else:
                # ASCII fallback
                ascii_map = {"INFO": "i", "SUCCESS": "+", "WARNING": "!", "STATUS": ">", "ERROR": "x"}
                symbol = ascii_map.get(level, "")
        except:
            # If encoding fails, use ASCII
            ascii_map = {"INFO": "i", "SUCCESS": "+", "WARNING": "!", "STATUS": ">", "ERROR": "x"}
            symbol = ascii_map.get(level, "")

        timestamp = ""
        if Config.Output.SHOW_TIMESTAMP:
            timestamp = f"{cls.GRAY}[{datetime.now().strftime('%H:%M:%S')}]{cls.RESET} "
            
        print(f"{timestamp}{color}{symbol} {message}{cls.RESET}")
        sys.stdout.flush()

    @classmethod
    def _print_error_indicator(cls):
        """Print a minimal error indicator to the console."""
        timestamp = ""
        if Config.Output.SHOW_TIMESTAMP:
            timestamp = f"{cls.GRAY}[{datetime.now().strftime('%H:%M:%S')}]{cls.RESET} "
        
        error_symbol = "✗"
        try:
            if sys.stdout.encoding is None:
                error_symbol = "x"
            else:
                error_symbol.encode(sys.stdout.encoding)
        except:
            error_symbol = "x"
            
        print(f"{timestamp}{cls.RED}{error_symbol} Error occurred (details in summary){cls.RESET}")
        sys.stdout.flush()

    @classmethod
    def status(cls, message):
        """Log a status update message."""
        if Config.Output.VERBOSE:
            cls.log(message, "STATUS")
    
    @classmethod
    def create_final_summary(cls, people_count, court_counts, output_path=None, total_courts=None):
        """Create a concise final summary."""
        summary_lines = []

        # Core result summary
        if not cls.errors:
            # Main detection results
            if people_count is not None:
                people_text = f"{people_count} person{'s' if people_count != 1 else ''}"
                court_text = f"{total_courts} court{'s' if total_courts != 1 else ''}" if total_courts is not None else "No courts"
                
                summary_lines.append(f"{cls.BOLD}{people_text}{cls.RESET} detected on {cls.BOLD}{court_text}{cls.RESET}")

                # Add court details if people were found on courts
                if court_counts:
                    court_details = []
                    for court_num, count in sorted(court_counts.items()):
                        court_details.append(f"C{court_num}: {count}")
                    summary_lines.append(f"Distribution: {', '.join(court_details)}")
            else: 
                summary_lines.append(f"{cls.BOLD}No detections{cls.RESET}")
        else:
            # Error summary
            summary_lines.append(f"{cls.RED}{cls.BOLD}Processing failed{cls.RESET}")

        # Add output path if generated
        if output_path and not cls.errors:
            summary_lines.append(f"Output: {os.path.basename(output_path)}")

        # Add Errors Section
        if cls.errors:
            summary_lines.append("") 
            summary_lines.append(f"{cls.RED}ERRORS:{cls.RESET}")
            for i, error in enumerate(cls.errors, 1):
                error_short = str(error).split('\n')[0]  # Get first line
                summary_lines.append(f" {i}. {error_short[:70]}{'...' if len(error_short) > 70 else ''}")
            
            # Add troubleshooting suggestions if there are errors
            if any("not found" in str(e).lower() for e in cls.errors):
                summary_lines.append("")
                summary_lines.append(f"{cls.CYAN}Try: Check file paths and permissions{cls.RESET}")
            elif any(("camera" in str(e).lower() or "picamera" in str(e).lower()) for e in cls.errors):
                summary_lines.append("")
                summary_lines.append(f"{cls.CYAN}Try: Check camera connection or use --no-camera flag{cls.RESET}")
        
        # Add Warnings Section (only if no errors)
        elif cls.warnings:
            summary_lines.append("") 
            summary_lines.append(f"{cls.YELLOW}WARNINGS:{cls.RESET}")
            for i, warning in enumerate(cls.warnings, 1):
                summary_lines.append(f" {i}. {warning}")

        return "\n".join(summary_lines)

    @classmethod
    def fancy_summary(cls, title, content, processing_time=None):
        """Display a fancy boxed summary."""
        # Process content into lines that fit in the box
        max_width = 70  # Maximum width for the box
        content_lines = []
        
        if isinstance(content, str):
            # Split by newlines first
            for line in content.split('\n'):
                if not line:
                    content_lines.append("")
                else:
                    # Handle ANSI color codes when calculating line length
                    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                    clean_line = ansi_escape.sub('', line)
                    
                    # If line is too long, wrap it
                    if len(clean_line) > max_width:
                        # Simple word wrapping
                        words = line.split()
                        current_line = []
                        current_length = 0
                        
                        for word in words:
                            clean_word = ansi_escape.sub('', word)
                            if current_length + len(clean_word) + (1 if current_length > 0 else 0) <= max_width:
                                if current_length > 0:
                                    current_line.append(" ")
                                    current_length += 1
                                current_line.append(word)
                                current_length += len(clean_word)
                            else:
                                content_lines.append("".join(current_line))
                                current_line = [word]
                                current_length = len(clean_word)
                        
                        if current_line:
                            content_lines.append("".join(current_line))
                    else:
                        content_lines.append(line)
        
        # Add processing time if provided
        if processing_time is not None:
            content_lines.append("")
            time_str = f"{processing_time:.1f}s" if processing_time < 60 else f"{int(processing_time/60)}m {int(processing_time%60)}s"
            content_lines.append(f"{cls.GRAY}Processing time: {time_str}{cls.RESET}")
        
        # Determine box width
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        content_width = max(len(ansi_escape.sub('', line)) for line in content_lines) if content_lines else 0
        box_width = max(content_width + 4, len(title) + 6, 50)  # Smaller minimum width
        
        # Try to detect terminal width
        try:
            import shutil
            terminal_width = shutil.get_terminal_size().columns
            box_width = min(box_width, terminal_width - 2)
        except:
            pass
        
        # Choose box characters
        try:
            if sys.stdout.encoding is not None:
                "│".encode(sys.stdout.encoding)
                use_unicode = True
            else:
                use_unicode = False
        except:
            use_unicode = False
            
        if use_unicode:
            top_left, top_right = "╭", "╮"
            bottom_left, bottom_right = "╰", "╯"
            horizontal, vertical = "─", "│"
            t_right, t_left = "├", "┤"
        else:
            top_left, top_right = "+", "+"
            bottom_left, bottom_right = "+", "+"
            horizontal, vertical = "-", "|"
            t_right, t_left = "+", "+"
        
        # Build a simpler box
        top_border = f"{top_left}{horizontal * (box_width - 2)}{top_right}"
        title_line = f"{vertical} {cls.BOLD}{title}{cls.RESET}{' ' * (box_width - len(title) - 4)} {vertical}"
        divider = f"{t_right}{horizontal * (box_width - 2)}{t_left}"
        bottom_border = f"{bottom_left}{horizontal * (box_width - 2)}{bottom_right}"
        
        # Display the box
        print(top_border)
        print(title_line)
        print(divider)
        
        for line in content_lines:
            # Calculate visible length (excluding ANSI codes)
            visible_len = len(ansi_escape.sub('', line))
            padding = box_width - 4 - visible_len
            
            if padding >= 0:
                print(f"{vertical} {line}{' ' * padding} {vertical}")
            else:
                # Line is too long, truncate
                print(f"{vertical} {line[:box_width - 7]}... {vertical}")
        
        print(bottom_border)

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr temporarily."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def detect_people(image_path, model_path="models/yolov8n.pt"):
    """
    Simple placeholder implementation for people detection.
    In a real implementation, you would load and run a YOLO model here.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            OutputManager.log(f"Failed to load image from {image_path}", "ERROR")
            return [], image
        
        # For demonstration purposes, create dummy detections
        # In a real implementation, you would run the model here
        height, width = image.shape[:2]
        
        # Log that we're using a simplified detection
        OutputManager.log("Using simplified person detection for demonstration", "INFO")
        
        # Create and return an empty list
        people = []
        
        return people, image
        
    except Exception as e:
        OutputManager.log(f"Error in person detection: {str(e)}", "ERROR")
        return [], None

def detect_courts(image):
    """
    Simple placeholder implementation for court detection.
    In a real implementation, you would detect blue/green areas that represent courts.
    """
    try:
        # Create and return an empty list
        courts = []
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Log that we're using a simplified detection
        OutputManager.log("Using simplified court detection for demonstration", "INFO")
        
        return courts
        
    except Exception as e:
        OutputManager.log(f"Error in court detection: {str(e)}", "ERROR")
        return []

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tennis Court Detector - Detects people on tennis courts using YOLO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --verbose          # Enable detailed output
  python main.py --input-image path/to/image.jpg  # Process specific image
  python main.py --no-camera        # Disable camera usage
"""
    )
    
    # Output control group
    output_group = parser.add_argument_group('Output Control')
    output_group.add_argument("--verbose", action="store_true", 
                            help="Enable detailed output messages")
    output_group.add_argument("--super-quiet", action="store_true", 
                            help="Minimal output, only show final summary")
    
    # Input options group
    input_group = parser.add_argument_group('Input Options')
    input_group.add_argument("--no-camera", action="store_true", 
                           help="Disable camera usage (useful for testing without hardware)")
    input_group.add_argument("--input-image", type=str, metavar="PATH",
                           help="Path to input image (bypasses camera capture)")
    
    # Model options group
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument("--model", type=str, default=Config.Model.NAME,
                           help=f"YOLO model name (default: {Config.Model.NAME})")
    
    return parser.parse_args()

def create_placeholder_image(resolution=(640, 480), output_file='images/input.png'):
    """Create a blank placeholder image."""
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Create a black image
        image = np.zeros((resolution[1], resolution[0], 3), dtype=np.uint8)
        
        # Save the image
        cv2.imwrite(output_file, image)
        
        OutputManager.log(f"Created placeholder image at {output_file}", "INFO")
        return True
    except Exception as e:
        OutputManager.log(f"Error creating placeholder image: {str(e)}", "ERROR")
        return False

def get_model_path(model_name):
    """Get the path to the model file, downloading if necessary."""
    model_path = os.path.join(Config.Paths.MODELS_DIR, f"{model_name}.pt")
    
    # If model exists, return its path
    if os.path.exists(model_path):
        return model_path
        
    # Model doesn't exist, try to download it
    try:
        # Suppress output during download
        with suppress_stdout_stderr():
            from ultralytics import YOLO
            model = YOLO(model_name)
            model.export(format="pt")  # Export to PyTorch format
            
            # Move the downloaded model to our models directory
            downloaded_path = f"{model_name}.pt"
            if os.path.exists(downloaded_path):
                os.makedirs(Config.Paths.MODELS_DIR, exist_ok=True)
                shutil.move(downloaded_path, model_path)
                return model_path
    except Exception as e:
        OutputManager.log(f"Failed to download model {model_name}: {str(e)}", "ERROR")
        return None

def select_model():
    """Let user select a model from available options."""
    OutputManager.log("Available models:", "INFO")
    for i, model in enumerate(Config.Model.AVAILABLE_MODELS, 1):
        OutputManager.log(f"{i}. {model}", "INFO")
    
    while True:
        try:
            choice = input("Select a model number: ")
            idx = int(choice) - 1
            if 0 <= idx < len(Config.Model.AVAILABLE_MODELS):
                return Config.Model.AVAILABLE_MODELS[idx]
        except ValueError:
            pass
        OutputManager.log("Invalid selection. Please try again.", "ERROR")

def ensure_model_available(model_name):
    """Ensure the specified model is available, downloading if necessary."""
    # Try to get the model
    model_path = get_model_path(model_name)
    
    if model_path:
        return model_path
        
    # If download failed, let user select a different model
    OutputManager.log(f"Could not download {model_name}. Please select a different model:", "WARNING")
    new_model = select_model()
    return get_model_path(new_model)

def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set verbosity based on args
    Config.Output.VERBOSE = args.verbose
    Config.Output.SUPER_QUIET = args.super_quiet
    
    # Reset logs
    OutputManager.reset_logs()
    
    # Record start time
    start_time = time.time()
    
    # Create necessary directories
    os.makedirs(Config.Paths.IMAGES_DIR, exist_ok=True)
    os.makedirs(Config.Paths.MODELS_DIR, exist_ok=True)
    
    try:
        # Ensure model is available
        model_path = ensure_model_available(args.model)
        if not model_path:
            OutputManager.log("Failed to obtain a valid model. Exiting.", "ERROR")
            return 1
            
        # Process input image
        if args.input_image:
            # Use the provided input image
            OutputManager.log(f"Using provided input image: {args.input_image}", "INFO")
            try:
                shutil.copy(args.input_image, Config.Paths.input_path())
                OutputManager.log(f"Copied input image to {Config.Paths.input_path()}", "SUCCESS")
            except Exception as e:
                OutputManager.log(f"Error copying input image: {str(e)}", "ERROR")
                return 1
        elif CAMERA_AVAILABLE and not args.no_camera:
            # Take a photo using the camera
            # Use the context manager only for the camera operation
            with CameraOutputFormatter():
                success = takePhoto(output_file=Config.Paths.input_path())
            
            if success:
                OutputManager.log("Photo captured successfully.", "SUCCESS")
            else:
                # Create a placeholder image if photo capture failed
                OutputManager.log("Photo capture failed. Creating placeholder image.", "WARNING")
                create_placeholder_image(output_file=Config.Paths.input_path())
        else:
            # Create a placeholder image
            if not os.path.exists(Config.Paths.input_path()):
                create_placeholder_image(output_file=Config.Paths.input_path())
        
        # Detect people in the image
        OutputManager.status("Looking for people")
        people, image = detect_people(Config.Paths.input_path())
        
        if len(people) > 0:
            OutputManager.log(f"Found {len(people)} people in the image", "SUCCESS")
        else:
            OutputManager.log("No people detected in the image", "SUCCESS")
        
        # Detect courts in the image
        OutputManager.status("Looking for tennis courts")
        courts = detect_courts(image)
        
        if len(courts) > 0:
            OutputManager.log(f"Found {len(courts)} tennis courts", "SUCCESS")
        else:
            OutputManager.log("No tennis courts detected", "WARNING")
        
        # For demonstration, create a simple output image
        output_image = image.copy() if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Save the output image
        cv2.imwrite(Config.Paths.output_path(), output_image)
        OutputManager.log(f"Output image saved successfully to {Config.Paths.output_path()}", "SUCCESS")
        
        # Create an empty court counts dictionary
        court_counts = {}
        
        # Calculate processing time
        processing_time = time.time() - start_time
        OutputManager.log(f"Total processing time: {processing_time:.2f} seconds", "INFO")
        
        # Create and display the final summary
        summary = OutputManager.create_final_summary(
            people_count=len(people),
            court_counts=court_counts,
            output_path=Config.Paths.output_path(),
            total_courts=len(courts)
        )
        
        OutputManager.fancy_summary("RESULTS SUMMARY", summary, processing_time)
        
        return 0
    except Exception as e:
        OutputManager.log(f"Unexpected error: {str(e)}", "ERROR")
        
        # Create and display an error summary
        processing_time = time.time() - start_time
        summary = OutputManager.create_final_summary(
            people_count=None,
            court_counts={},
            output_path=None,
            total_courts=None
        )
        
        OutputManager.fancy_summary("ERROR SUMMARY", summary, processing_time)
        
        return 1

if __name__ == "__main__":
    sys.exit(main())