from io import BytesIO
from PIL import Image, ImageOps
import numpy as np


TARGET_SIZE = (256, 256)


def preprocess_image(image_bytes: bytes) -> dict:
    """
    Preprocess uploaded image for downstream quality checking and inference.

    Returns a dictionary containing:
    - original PIL image
    - resized PIL image
    - numpy array
    - metadata
    """

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unable to read image: {e}")

    original_width, original_height = image.size

    # Correct orientation from EXIF if present
    image = ImageOps.exif_transpose(image)

    # Keep a copy of original image
    original_image = image.copy()

    # Resize for model input
    resized_image = image.resize(TARGET_SIZE)

    # Convert to numpy array
    image_array = np.array(resized_image)

    return {
        "original_image": original_image,
        "resized_image": resized_image,
        "image_array": image_array,
        "original_size": (original_width, original_height),
        "processed_size": TARGET_SIZE
    }