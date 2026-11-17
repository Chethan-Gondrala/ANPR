import os
import requests
from ultralytics import YOLO
import config

class LicensePlateDetector:
    """
    YOLO-based License Plate / Vehicle Region Detector.
    Responsible for 'Where is the license plate?' in a frame.
    """
    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or config.YOLO_MODEL_PATH
        self.device = device or config.DEVICE
        self.model = None
        self.is_custom_lp_model = False
        self._ensure_model_exists()
        self._load_model()

    def _ensure_model_exists(self):
        """Check for model weights or auto-download default YOLO model."""
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path) > 1000000:
            return

        print(f"[PlateDetector] Model file not found at '{self.model_path}'. Using standard YOLOv8 detector.")
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        # Standard YOLOv8 fallback path
        fallback_path = os.path.join(config.MODELS_DIR, "yolov8n.pt")
        if not os.path.exists(fallback_path):
            print("[PlateDetector] Downloading standard YOLOv8 weights...")
            try:
                # Ultralytics will auto-download yolov8n.pt
                temp_m = YOLO("yolov8n.pt")
                temp_m.save(fallback_path)
            except Exception:
                pass
        self.model_path = fallback_path

    def _load_model(self):
        """Load YOLO model into memory on configured device."""
        try:
            print(f"[PlateDetector] Loading YOLO model from '{self.model_path}' on device '{self.device}'...")
            self.model = YOLO(self.model_path)

            # Check if model has custom license plate classes
            if hasattr(self.model, "names") and self.model.names:
                names = [str(n).lower() for n in self.model.names.values()]
                if any("plate" in n or "license" in n for n in names) or len(names) == 1:
                    self.is_custom_lp_model = True

            print(f"[PlateDetector] Model loaded successfully (is_custom_lp_model={self.is_custom_lp_model}).")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {str(e)}")

    def detect(self, frame, min_conf=None):
        """
        Detect license plates / vehicle regions in an OpenCV image frame.

        Args:
            frame (np.ndarray): BGR image frame from OpenCV.
            min_conf (float, optional): Minimum confidence threshold.

        Returns:
            list[dict]: List of detection results containing 'bbox' (x1, y1, x2, y2) and 'confidence'.
        """
        if self.model is None or frame is None:
            return []

        conf_threshold = min_conf if min_conf is not None else config.MIN_DETECTION_CONF

        try:
            results = self.model(frame, verbose=False, device=self.device, conf=conf_threshold)
        except Exception as e:
            if self.device == "cuda":
                print(f"[PlateDetector] CUDA error during inference ({e}). Retrying on CPU...")
                self.device = "cpu"
                results = self.model(frame, verbose=False, device="cpu", conf=conf_threshold)
            else:
                raise e

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())

                if self.is_custom_lp_model:
                    # Specialized LP model directly detects license plate bounding box
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "confidence": confidence
                    })
                else:
                    # COCO classes: 2=car, 3=motorcycle, 5=bus, 7=truck
                    if cls_id in [2, 3, 5, 7]:
                        w = x2 - x1
                        h = y2 - y1

                        # Primary candidate: Full vehicle / plate area
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": confidence
                        })

                        # Secondary candidate: Lower central vehicle region where plate is mounted
                        plate_y1 = y1 + int(h * 0.45)
                        plate_y2 = y2
                        plate_x1 = x1 + int(w * 0.15)
                        plate_x2 = x2 - int(w * 0.15)

                        if plate_x2 > plate_x1 and plate_y2 > plate_y1:
                            detections.append({
                                "bbox": (plate_x1, plate_y1, plate_x2, plate_y2),
                                "confidence": confidence
                            })
                    else:
                        # Other object classes
                        detections.append({
                            "bbox": (x1, y1, x2, y2),
                            "confidence": confidence
                        })

        return detections
