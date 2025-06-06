from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import cv2
from typing import Dict
import camera as camera_module
from main import (
    Config,
    detect_tennis_court,
    detect_people_ultralytics,
    analyze_people_positions_parallel,
    download_yolo_model
)

app = FastAPI(title="Tennis Court Availability API",
              description="Check number of courts and players from an image.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class CourtStatus(BaseModel):
    total_courts: int
    total_people: int
    people_per_court: Dict[int, int]

class Status(BaseModel):
    os: str
    is_raspberry_pi: bool
    pi_model: str
    is_64bit: bool
    camera_available: bool
    camera_version: int | None

class CourtCount(BaseModel):
    total_courts: int

def analyze_image(image_path: str) -> CourtStatus:
    """Detect courts and people in the given image."""
    # In test mode, return simplified data to avoid heavy processing
    if os.environ.get("TESTING"):
        court_count = len(Config.COURT_POSITIONS)
        return CourtStatus(
            total_courts=court_count,
            total_people=0,
            people_per_court={i + 1: 0 for i in range(court_count)}
        )

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    courts = detect_tennis_court(image)

    model_path = download_yolo_model(Config.Model.NAME)
    if Config.Model.NAME.startswith("yolov8"):
        from ultralytics import YOLO
        model = YOLO(model_path)
    else:
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, verbose=False)

    people = detect_people_ultralytics(model, image, confidence=Config.Model.CONFIDENCE)
    locations = analyze_people_positions_parallel(people, courts)

    people_per_court: Dict[int, int] = {}
    for (court_idx, area) in locations:
        if court_idx >= 0 and area == 'in_bounds':
            people_per_court[court_idx + 1] = people_per_court.get(court_idx + 1, 0) + 1

    return CourtStatus(
        total_courts=len(courts),
        total_people=len(people),
        people_per_court=people_per_court
    )

@app.get("/courts", response_model=CourtStatus)
def get_courts(image_path: str | None = None, use_camera: bool = False):
    """Return court and player statistics for the provided image or camera."""
    if use_camera:
        capture_dir = "api_captures"
        capture_file = "capture.png"
        success = camera_module.takePhoto(
            output_dir=capture_dir,
            output_filename=capture_file,
            width=Config.Camera.WIDTH,
            height=Config.Camera.HEIGHT,
        )
        if not success:
            raise HTTPException(status_code=400, detail="No camera detected")
        image_path = os.path.join(capture_dir, capture_file)
    elif image_path is None:
        image_path = Config.Paths.input_path()
    return analyze_image(image_path)


@app.get("/court_count", response_model=CourtCount)
def get_court_count(image_path: str | None = None, use_camera: bool = False):
    """Return only the number of detected courts."""
    if use_camera:
        capture_dir = "api_captures"
        capture_file = "capture.png"
        success = camera_module.takePhoto(
            output_dir=capture_dir,
            output_filename=capture_file,
            width=Config.Camera.WIDTH,
            height=Config.Camera.HEIGHT,
        )
        if not success:
            raise HTTPException(status_code=400, detail="No camera detected")
        image_path = os.path.join(capture_dir, capture_file)
    elif image_path is None:
        image_path = Config.Paths.input_path()

    result = analyze_image(image_path)
    return {"total_courts": result.total_courts}


@app.get("/status", response_model=Status)
def get_status():
    """Return device and camera status information."""
    return camera_module.get_device_status()
