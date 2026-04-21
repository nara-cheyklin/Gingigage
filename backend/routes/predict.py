from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os

from backend.config.settings import ALLOWED_FILE_TYPES
from backend.services.rosbag_processing import extract_rgb_and_depth_from_rosbag, cv2_to_bytes
from backend.services.inference import run_inference

router = APIRouter()


def save_temp_file(file: UploadFile):
    suffix = os.path.splitext(file.filename)[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        return tmp.name


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type")

    rosbag_path = save_temp_file(file)

    try:
        rgb_frame, depth_frame = extract_rgb_and_depth_from_rosbag(rosbag_path)

        image_bytes = cv2_to_bytes(rgb_frame)

        result = run_inference(
            image_bytes=image_bytes,
            depth_map=depth_frame
        )

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(rosbag_path):
            os.remove(rosbag_path)