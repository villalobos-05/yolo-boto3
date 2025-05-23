# YOLO Image Validation Microservice

This service validates uploaded images with a YOLO object detection model and uploads valid images to Cloudflare R2.

## Features
- REST endpoint `POST /upload-images` accepts multipart/form-data images.
- Validates detections:
  - Requires at least one object of class configured by `TARGET_CLASSES`.
  - Ensures all detections have confidence ≥ `MIN_CONFIDENCE`.
  - Disallows classes specified in `DISALLOWED_CLASSES`.
- Uploads validated images to R2 (or S3) and returns public URLs.

## Configuration
Set the following environment variables:

- `R2_ACCESS_KEY`: Cloudflare R2 access key.
- `R2_SECRET_KEY`: Cloudflare R2 secret key.
- `R2_BUCKET_NAME`: R2 bucket name.
- `R2_ENDPOINT_URL`: Cloudflare R2 endpoint URL (e.g. `https://<account>.r2.cloudflarestorage.com`).
- `YOLO_MODEL_PATH`: Path or name of the YOLO model (default: `yolov8n.pt`).
- `TARGET_CLASSES`: Name of the target object classes (default: `clothing,coat,shirt`).
- `DISALLOWED_CLASSES`: Comma-separated list of disallowed class names (default: `person,animal`).
- `MIN_CONFIDENCE`: Minimum confidence threshold (default: `0.5`).
- `JWT_SECRET`: Jwt secret key for validating the request.
- `ALGORITHM`: Algorithm used for signing the jwt.

## Installation

1. Clone this repository.
2. Create a Python 3.8+ virtual environment and activate it:
  ```bash
  python -m venv .venv
  source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
  ```

  3. Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

  ## Usage

  Start the FastAPI server:
  ```bash
  fastapi dev main.py --port 8000
  ```

  The API will be available at `http://localhost:8000`.

  ## Example Request

  ```bash
  curl -X POST "http://localhost:8000/upload-images" \
    -F "files=@/path/to/image1.jpg" \
    -F "files=@/path/to/image2.jpg"
  ```
