import os
import torch

# Base Directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# File & Storage Paths
INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Ensure required directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# YOLO Model Configuration
# Primary pre-trained license plate detection model file
DEFAULT_MODEL_FILENAME = "license_plate_detector.pt"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, DEFAULT_MODEL_FILENAME)

# Primary & Fallback download URLs for pre-trained License Plate YOLO weights
LP_WEIGHTS_URLS = [
    "https://huggingface.co/nickmcdonald/yolov8n-license-plate/resolve/main/best.pt",
    "https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/best.pt",
    "https://github.com/marnixgaudriaan/License-Plate-Detection/raw/main/license_plate_detector.pt"
]

# Detection & Recognition Thresholds
MIN_DETECTION_CONF = 0.35   # Minimum confidence for YOLO license plate bounding box
MIN_OCR_CONF = 0.80         # Minimum confidence for PaddleOCR text reading
MIN_PLATE_LENGTH = 4        # Minimum character length for valid license plate
MAX_PLATE_LENGTH = 14       # Maximum character length for valid license plate

# Early Stopping Settings (Stop video processing once a plate is accurately recognized)
EARLY_STOP_ENABLE = True
EARLY_STOP_THRESHOLD = 0.90  # 90% confidence threshold to trigger early stop

# Performance Settings
FRAME_SKIP = 5              # Process every Nth frame (1 = every frame, 2 = every 2nd frame)

# Device Configuration (CPU / CUDA)
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
