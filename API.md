# API Documentation

This project includes a small FastAPI application in `api.py` for retrieving tennis court statistics.

codex/create-tennis-courts-api-with-documentation
## Setup

Create a virtual environment and install the required packages (which include
`uvicorn`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running the server

Use Uvicorn to start the API in development mode:

```bash
uvicorn api:app --reload
```

## Endpoints

### `GET /courts`

Returns JSON data describing the number of detected courts and players. Optionally pass an `image_path` query parameter to analyze a specific image.

Example response:

```json
{
  "total_courts": 3,
  "total_people": 4,
  "people_per_court": {"1": 2, "2": 2}
}
```

During automated tests the endpoint returns dummy data without running the heavy detection pipeline.
