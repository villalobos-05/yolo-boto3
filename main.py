from dotenv import load_dotenv

load_dotenv(override=True)

import os
import logging
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from botocore.client import Config
import io
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY")
R2_BUCKET = os.getenv("R2_BUCKET_NAME")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
DISALLOWED_CLASSES = set(os.getenv("DISALLOWED_CLASSES", "person,animal").split(","))
TARGET_CLASSES = set(os.getenv("TARGET_CLASSES", "clothing,shirt,coat").split(","))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.5))


app = FastAPI()

origins = set(os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)


model = None  # type: YOLO
s3_client = None  # type: boto3.client


def get_s3_client():
    """Dependency: Returns a configured S3 (R2) client."""
    return s3_client


@app.on_event("startup")
def startup_event():
    global model, s3_client
    # Load YOLO model
    model = YOLO(YOLO_MODEL_PATH)
    logger.info(f"Loaded YOLO model from {YOLO_MODEL_PATH}")

    # Initialize S3 client for Cloudflare R2
    session = boto3.session.Session()
    s3_client = session.client(
        "s3",
        region_name="auto",
        endpoint_url=os.getenv("R2_ENDPOINT_URL"),
        aws_access_key_id=R2_ACCESS_KEY,
        aws_secret_access_key=R2_SECRET_KEY,
        config=Config(
            s3={
                "addressing_style": "virtual",
                "request_checksum_calculation": "WHEN_REQUIRED",  # fixes boto3 ≥1.36.0
                "response_checksum_validation": "WHEN_REQUIRED",  # checksum incompatibility
            }
        ),
    )
    logger.info("Initialized Cloudflare R2 client")


def validate_detections(detections: List[Dict[str, Any]], filename: str) -> None:
    """
    Validates YOLO detections against criteria.
    Raises HTTPException if validation fails.
    """
    logger.info(f"Detections: {detections}")
    if not detections:
        raise HTTPException(
            status_code=400,
            detail=[{"filename": filename, "reason": "No objects detected"}],
        )

    has_target = False
    for det in detections:
        cls = det["name"]
        conf = det["confidence"]
        if cls in DISALLOWED_CLASSES:
            raise HTTPException(
                status_code=400,
                detail=[
                    {
                        "filename": filename,
                        "reason": f"Disallowed class detected: {cls}",
                    }
                ],
            )

        if cls in TARGET_CLASSES:
            if conf < MIN_CONFIDENCE:
                raise HTTPException(
                    status_code=400,
                    detail=[
                        {
                            "filename": filename,
                            "reason": f"Low confidence ({conf}) for class {cls}",
                        }
                    ],
                )
            has_target = True

    if not has_target:
        raise HTTPException(
            status_code=400,
            detail=[
                {
                    "filename": filename,
                    "reason": f"No target class '{TARGET_CLASSES}' found",
                }
            ],
        )


@app.post("/upload-images")
async def validate_and_upload(
    files: List[UploadFile] = File(...), s3: boto3.client = Depends(get_s3_client)
):
    """Endpoint to validate images and upload to Cloudflare R2."""
    uploaded = []
    try:
        for file in files:
            contents = await file.read()

            MAX_SIZE_MB = int(os.getenv("MAX_SIZE_MB"))

            if len(contents) > MAX_SIZE_MB * 1024 * 1024:
                raise HTTPException(
                    status_code=400,
                    detail=[
                        {
                            "filename": file.filename,
                            "reason": f"File size exceeds {MAX_SIZE_MB}MB",
                        }
                    ],
                )

            try:
                img = Image.open(io.BytesIO(contents)).convert("RGB")
            except Exception:
                raise HTTPException(
                    status_code=400,
                    detail=[
                        {"filename": file.filename, "reason": "Could not decode image"}
                    ],
                )

            # Run inference
            results = model.predict(source=img, imgsz=640)
            detections = []
            for r in results:
                for *box, conf, cls in r.boxes.data.tolist():
                    name = model.names[int(cls)]
                    detections.append({"name": name.lower(), "confidence": conf})

            # Validate detections
            validate_detections(detections, file.filename)

            # Wrap bytes in a file-like buffer
            buffer = io.BytesIO(contents)
            buffer.seek(0)

            # Upload to R2
            key = f"uploads/{file.filename}"
            try:
                s3.upload_fileobj(
                    Fileobj=buffer,
                    Bucket=R2_BUCKET,
                    Key=key,
                    ExtraArgs={"ACL": "public-read"},
                )
            except (BotoCoreError, ClientError) as e:
                logger.error(f"Upload error for {file.filename}: {e}")
                raise HTTPException(
                    status_code=500, detail="Failed to upload to storage"
                )

            logger.info(f"Uploaded {file.filename} to R2 Bucket at {key}")
            uploaded.append({"filename": file.filename, "path": key})

        return JSONResponse(
            status_code=200, content={"status": "ok", "images": uploaded}
        )

    except HTTPException as e:
        logger.warning(f"Validation failed: {e.detail}")
        return JSONResponse(
            status_code=e.status_code,
            content={
                "status": "error",
                "message": "Image validation failed",
                "details": e.detail,
            },
        )
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload to storage")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
