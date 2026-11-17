# Models Directory

This directory stores the pre-trained YOLO model weights for License Plate Detection.

## Default Model
The default model filename expected by `config.py` is:
`license_plate_detector.pt`

## Automatic Downloading
If `license_plate_detector.pt` is not present when starting the application, `plate_detector/detector.py` will attempt to automatically download pre-trained License Plate YOLOv8 weights from Hugging Face / GitHub.

## Custom Models
If you have a custom trained YOLO `.pt` model for license plate detection:
1. Place your `.pt` file in this directory (e.g. `models/my_custom_model.pt`).
2. Update `YOLO_MODEL_PATH` in `config.py` to point to your custom file.
