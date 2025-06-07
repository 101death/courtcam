# Tennis Court Availability API

This FastAPI server analyzes images of tennis courts and reports how many courts are present and how many people are playing on each court. It powers the `courtcam` project but can also be used as a standalone service.

## Installation

1. Clone the repository and change into the project directory.
2. Create a Python virtual environment and install all dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Alternatively you can run the provided `setup.sh` script which installs the requirements and prepares the `config.json` file interactively.

## Running the API server

Start the development server with Uvicorn:

```bash
uvicorn api:app --reload
```

The API will be available at `http://127.0.0.1:8000`. When running in production you can omit `--reload` and supply `--host`/`--port` as needed.

FastAPI automatically serves an interactive Swagger UI at `http://127.0.0.1:8000/docs` where you can experiment with the endpoints.

## Endpoint reference

### `GET /courts`

Analyze an image and return statistics about the detected courts and players.

#### Query Parameters

- `image_path` *(optional)* – Path to the image file to analyze. If omitted, the
  default input path from `config.json` (`images/input.png` by default) is used.
  The API also stores an annotated result image alongside the original using a
  timestamped filename inside the `images/` directory.
- `use_camera` *(optional)* – Capture a new image using the Raspberry Pi camera
  instead of providing a path. When this flag is enabled the API tries to take a
  photo at the resolution specified under the `Camera` section of `config.json`.
  The captured photo is written to `images/capture_YYYYMMDD_HHMMSS.png` and the
  processed output to `images/result_YYYYMMDD_HHMMSS.png`. If no camera is
  detected the endpoint returns HTTP 400 with `"No camera detected"`.

#### Response JSON

- `total_courts` – Number of tennis courts detected in the image.
- `total_people` – Total number of people detected.
- `people_per_court` – A dictionary mapping court numbers to the number of players on that court.

Example response:

```json
{
  "total_courts": 3,
  "total_people": 4,
  "people_per_court": {"1": 2, "2": 2}
}
```

Example request with an image path:

```bash
curl "http://127.0.0.1:8000/courts?image_path=images/input.png"
```

Example request using the camera:

```bash
curl "http://127.0.0.1:8000/courts?use_camera=true"
```
Each request also appends a record of the input and output file paths to
`api_captures/log.txt` so you can review how the API was used over time.

During automated tests the endpoint returns simplified dummy data to avoid running the heavy computer vision pipeline. This happens when the environment variable `TESTING` is set to `1`.

### `GET /status`

Retrieve basic status information about the host device and camera. This is useful to verify that the Raspberry Pi and its camera are detected correctly.

#### Response JSON

- `os` – Operating system name.
- `is_raspberry_pi` – `true` if running on a Raspberry Pi.
- `pi_model` – Model string reported by the Pi, if available.
- `is_64bit` – `true` if the Python interpreter is 64‑bit.
- `camera_available` – `true` if a supported camera module is detected.
- `camera_version` – `1` for legacy Picamera, `2` for Picamera2, or `null` if no camera libraries are available.

Example request:

```bash
curl "http://127.0.0.1:8000/status"
```

### `GET /court_count`

Return only the number of tennis courts detected in the provided image or from the camera.

#### Query Parameters

- `image_path` *(optional)* – Path to an image file to analyze.
- `use_camera` *(optional)* – Capture a new photo using the Raspberry Pi camera. Returns HTTP 400 if no camera is available.

#### Response JSON

- `total_courts` – Number of detected courts.

Example request specifying an image path:

```bash
curl "http://127.0.0.1:8000/court_count?image_path=images/input.png"
```

Example request using the camera:

```bash
curl "http://127.0.0.1:8000/court_count?use_camera=true"
```

## Usage examples

### Using `curl`

```bash
curl "http://127.0.0.1:8000/courts?image_path=images/input.png"
curl "http://127.0.0.1:8000/courts?use_camera=true"  # captures a new photo
curl "http://127.0.0.1:8000/status"                 # check Pi status
curl "http://127.0.0.1:8000/court_count"            # number of courts only
```

### Using Python `requests`

```python
import requests

resp = requests.get("http://127.0.0.1:8000/courts", params={"image_path": "images/input.png"})
print(resp.json())

resp = requests.get("http://127.0.0.1:8000/courts", params={"use_camera": "true"})
print(resp.json())
# If no camera is connected this call returns:
# {"detail": "No camera detected"}

resp = requests.get("http://127.0.0.1:8000/status")
print(resp.json())

resp = requests.get("http://127.0.0.1:8000/court_count")
print(resp.json())
```

### Calling the analysis function directly

You can use the underlying logic without running the HTTP server:

```python
from api import analyze_image

# Optionally provide an output_path to save an annotated image
result = analyze_image("images/input.png", output_path="images/example_result.png")
print(result)
```

## Customizing input images

Edit `config.json` to change the default input location or any model parameters. The FastAPI server will pick up these settings on the next run.

## Troubleshooting

- Make sure all dependencies listed in `requirements.txt` are installed.
- When running on a system without a camera or GPU you can still analyze static images by providing the `image_path` parameter.

