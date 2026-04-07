# Just a sample, need to later replace DummyModel, calculate_KGW, confidence_logic
# - Loads the model once
# - Handles preprocessing (basic)
# - Runs inference (mock or real)
# - Includes measurement logic placeholder
# - Returns structured output
import torch
import numpy as np
from backend.config.settings import MODEL_PATH, KGW_THRESHOLD_MM
from backend.services.preprocessing import preprocess_image
import logging

logger = logging.getLogger(__name__)

# -------------------------------
# Load model
# -------------------------------

model = None #Define Model Later

try:
    model = YourRealModel()  # replace this
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully.")

except Exception as e:
    model = None
    logger.warning(f"Model loading failed: {e}")

# -------------------------------
# Measurement (placeholder)
# -------------------------------

def calculate_kgw(mask):
    pixel_count = np.sum(mask > 0.5)
    kgw_mm = pixel_count / 1000.0
    return round(kgw_mm, 2)

# -------------------------------
# Main inference
# -------------------------------

def run_inference(image_bytes):

    if model is None:
        raise RuntimeError("Model is not loaded")

    input_tensor = preprocess_image(image_bytes)

    with torch.no_grad():
        output = model(input_tensor)

    mask = output.squeeze().cpu().numpy()
    binary_mask = (mask > 0.5).astype(np.uint8)

    kgw_mm = calculate_kgw(binary_mask)

    confidence = float(np.mean(mask))  # placeholder

    interpretation = (
        "Adequate keratinized gingiva width"
        if kgw_mm >= KGW_THRESHOLD_MM
        else "Inadequate keratinized gingiva width"
    )

    return {
        "kgw_mm": kgw_mm,
        "confidence": round(confidence, 2),
        "interpretation": interpretation
    }