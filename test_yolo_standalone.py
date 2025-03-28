#!/usr/bin/env python3
# This is a simple test script for YOLOv8 models

from ultralytics import YOLO
import cv2
import os
import sys

def test_yolov8_detector(image_path, model_name="yolov8x.pt", confidence=0.15, verbose=True):
    """Test YOLOv8 detection on an image"""
    if verbose:
        print(f"Testing YOLOv8 detection on {image_path} with model {model_name}")
    
    # Load the model
    model = YOLO(model_name)
    
    # Load the image
    if os.path.exists(image_path):
        if verbose:
            print(f"Image found: {image_path}")
        image = cv2.imread(image_path)
        if verbose:
            print(f"Image size: {image.shape}")
    else:
        if verbose:
            print(f"Image not found: {image_path}")
        return []
    
    # Run prediction
    results = model.predict(
        source=image,
        conf=confidence,  # Person class only
        classes=[0],      # Person class only
        verbose=verbose,  # Only show output if verbose
        save=verbose,     # Only save if verbose
        project="test_output",
        name="yolo_test"
    )
    
    # Process results
    people = []
    if len(results) > 0:
        if verbose:
            print(f"\nResults type: {type(results)}")
            print(f"Number of results: {len(results)}")
        
        # Check for boxes
        if hasattr(results[0], 'boxes'):
            boxes = results[0].boxes
            if verbose:
                print(f"\nDetected boxes: {len(boxes)}")
            
            # Process each detection
            for i, box in enumerate(boxes):
                try:
                    cls = int(box.cls.item()) if hasattr(box, 'cls') else -1
                    conf = float(box.conf.item()) if hasattr(box, 'conf') else 0
                    
                    if cls == 0:  # Person class
                        # Get coordinates
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        
                        # Add to people list
                        person = {
                            'position': ((x1 + x2) // 2, (y1 + y2) // 2),
                            'foot_position': ((x1 + x2) // 2, y2),
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf
                        }
                        people.append(person)
                        if verbose:
                            print(f"Person {i+1}: bbox=({x1},{y1},{x2},{y2}), conf={conf:.2f}")
                except Exception as e:
                    if verbose:
                        print(f"Error processing detection {i}: {str(e)}")
            
            if verbose:
                print(f"\nTotal people detected: {len(people)}")
        elif verbose:
            print("No boxes attribute found in results")
    elif verbose:
        print("No results returned from model")
    
    # Return the people list for use in main.py
    return people

if __name__ == "__main__":
    # Get image path from command line or use default
    image_path = "images/input.png"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    
    # Run the test with verbose output
    test_yolov8_detector(image_path, verbose=True) 