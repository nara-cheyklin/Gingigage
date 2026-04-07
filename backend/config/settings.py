import os

# -----------------------------
# Project Info
# -----------------------------
PROJECT_NAME = "gingigage"
API_PREFIX = "/api"

# -----------------------------
# File Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pth")

# -----------------------------
# Image Settings
# -----------------------------
IMAGE_SIZE = (256, 256)

ALLOWED_FILE_TYPES = [
    "image/jpeg",
    "image/png",
    "image/jpg"
]

MAX_IMAGE_SIZE_MB = 5

# -----------------------------
# Quality Check Thresholds
# -----------------------------
MIN_WIDTH = 300
MIN_HEIGHT = 300

BLUR_THRESHOLD = 80.0
DARKNESS_THRESHOLD = 40
BRIGHTNESS_THRESHOLD = 220

# -----------------------------
# Inference Settings
# -----------------------------
CONFIDENCE_THRESHOLD = 0.5

# KGW interpretation threshold (example)
KGW_THRESHOLD_MM = 2.0

# -----------------------------
# Debug / Environment
# -----------------------------
DEBUG = True