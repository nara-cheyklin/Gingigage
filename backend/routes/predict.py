# Its job is to:
# - receive the uploaded image
# - validate the request
# - call preprocessing
# - call quality check
# - call inference
# - return a clean JSON response

from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.services.preprocessing import preprocess_image
from backend.services.quality_check import validate_image_quality
from backend.services.inference import run_inference
from backend.config.settings import ALLOWED_FILE_TYPES

router = APIRouter()

@router.post("/predict")
async def predict(image: UploadFile = File(...)):
    # 1. Validate file type
    if image.content_type not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a JPG or PNG image."
        )

    # 2. Read uploaded file
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        # 3. Preprocess image
        processed_image = preprocess_image(image_bytes)

        # 4. Quality validation
        quality_result = validate_image_quality(processed_image)
        if not quality_result["is_valid"]:
            return {
                "success": False,
                "stage": "quality_check",
                "message": "Image quality is insufficient for reliable analysis.",
                "issues": quality_result["issues"]
            }

        # 5. Run inference
        inference_result = run_inference(processed_image)

        # 6. Return structured response
        return {
            "success": True,
            "stage": "completed",
            "message": "Prediction completed successfully.",
            "data": {
                "kgw_mm": inference_result["kgw_mm"],
                "confidence": inference_result["confidence"],
                "interpretation": inference_result["interpretation"],
                "overlay_url": inference_result.get("overlay_url")
            }
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )