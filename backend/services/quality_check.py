import cv2
import numpy as np
from backend.config.settings import MIN_WIDTH, MIN_HEIGHT,BLUR_THRESHOLD, DARKNESS_THRESHOLD, BRIGHTNESS_THRESHOLD
def validate_image_quality(processed_image: dict) -> dict:
    """
    Validate image quality before inference.

    Checks:
    - minimum resolution
    - blur
    - overly dark image
    - overly bright image

    Returns:
    {
        "is_valid": bool,
        "issues": list[str],
        "metrics": dict
    }
    """

    issues = []

    original_image = processed_image["original_image"]
    original_width, original_height = processed_image["original_size"]

    # Convert PIL image to OpenCV format
    image_np = np.array(original_image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Resolution check
    if original_width < MIN_WIDTH or original_height < MIN_HEIGHT:
        issues.append(
            f"Image resolution is too low ({original_width}x{original_height})."
        )

    # 2. Blur check using Laplacian variance
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        issues.append("Image is too blurry.")

    # 3. Brightness check
    brightness = np.mean(gray)
    if brightness < DARKNESS_THRESHOLD:
        issues.append("Image is too dark.")
    elif brightness > BRIGHTNESS_THRESHOLD:
        issues.append("Image is too bright.")

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "metrics": {
            "blur_score": round(float(blur_score), 2),
            "brightness": round(float(brightness), 2),
            "resolution": f"{original_width}x{original_height}"
        }
    }