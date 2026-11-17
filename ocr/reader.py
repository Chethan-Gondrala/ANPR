import os
# Disable mkldnn / oneDNN instruction optimization to prevent Windows PaddlePaddle execution crash
os.environ["FLAGS_use_mkldnn"] = "0"

import re
import cv2
import numpy as np
import config

class PlateOCRReader:
    """
    License Plate Text Recognizer.
    Supports RapidOCR (ONNX-based, fast & DLL-safe), EasyOCR, and PaddleOCR engines with automatic fallback.
    """
    def __init__(self, lang='en'):
        self.ocr = None
        self.engine_type = None
        self._init_ocr(lang)

    def _init_ocr(self, lang):
        """Initialize OCR engine safely with fallbacks."""
        # 1. Try RapidOCR (ONNX Runtime - Fast, lightweight, avoids pandas DLL blocks & PaddleX PDX bugs)
        try:
            from rapidocr_onnxruntime import RapidOCR
            print("[PlateOCRReader] Initializing RapidOCR (ONNX engine)...")
            self.ocr = RapidOCR()
            self.engine_type = "rapidocr"
            print("[PlateOCRReader] RapidOCR initialized successfully.")
            return
        except Exception as e:
            print(f"[PlateOCRReader] RapidOCR initialization skipped/failed: {e}")

        # 2. Try EasyOCR (PyTorch based)
        try:
            import easyocr
            print(f"[PlateOCRReader] Initializing EasyOCR (lang='{lang}')...")
            self.ocr = easyocr.Reader([lang], gpu=False)
            self.engine_type = "easyocr"
            print("[PlateOCRReader] EasyOCR initialized successfully.")
            return
        except Exception as e:
            print(f"[PlateOCRReader] EasyOCR initialization skipped/failed: {e}")

        # 3. Fallback to PaddleOCR
        try:
            print(f"[PlateOCRReader] Initializing PaddleOCR (lang='{lang}')...")
            from paddleocr import PaddleOCR
            try:
                self.ocr = PaddleOCR(
                    use_textline_orientation=True,
                    lang=lang,
                    enable_mkldnn=False
                )
            except Exception:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=lang,
                    enable_mkldnn=False
                )
            self.engine_type = "paddleocr"
            print("[PlateOCRReader] PaddleOCR initialized successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize any OCR engine (RapidOCR / EasyOCR / PaddleOCR): {str(e)}")

    def clean_text(self, text):
        """
        Normalize and clean extracted OCR text for license plates (especially Indian formats).
        - Converts to uppercase
        - Removes spaces, hyphens, and non-alphanumeric noise characters
        """
        if not text:
            return ""

        text = str(text).upper()
        clean = re.sub(r'[^A-Z0-9]', '', text)
        return clean

    def read_plate(self, plate_crop):
        """
        Extract text and confidence score from a cropped license plate image.

        Args:
            plate_crop (np.ndarray): Cropped BGR/Grayscale license plate image.

        Returns:
            dict: {
                "text": clean_plate_string,
                "raw_text": raw_ocr_string,
                "confidence": confidence_float
            } or None if no valid text recognized.
        """
        if self.ocr is None or plate_crop is None or plate_crop.size == 0:
            return None

        candidates = []

        try:
            if self.engine_type == "rapidocr":
                results, _ = self.ocr(plate_crop)
                if results:
                    for item in results:
                        if len(item) >= 3:
                            raw_text = item[1]
                            conf = float(item[2])
                            candidates.append((str(raw_text), conf))

            elif self.engine_type == "easyocr":
                results = self.ocr.readtext(plate_crop)
                if results:
                    for bbox, raw_text, conf in results:
                        candidates.append((str(raw_text), float(conf)))

            elif self.engine_type == "paddleocr":
                results = self.ocr.ocr(plate_crop)
                if results:
                    for res in results:
                        if not res:
                            continue
                        if isinstance(res, dict):
                            rec_texts = res.get("rec_texts", [])
                            rec_scores = res.get("rec_scores", [])
                            for raw_text, conf in zip(rec_texts, rec_scores):
                                candidates.append((str(raw_text), float(conf)))
                        elif isinstance(res, list):
                            for line in res:
                                if not line or len(line) < 2:
                                    continue
                                if isinstance(line[1], (tuple, list)):
                                    raw_text, conf = line[1][0], float(line[1][1])
                                    candidates.append((str(raw_text), conf))
        except Exception as e:
            print(f"[PlateOCRReader] OCR processing error ({self.engine_type}): {e}")
            return None

        best_text = ""
        best_raw = ""
        best_conf = 0.0

        for raw_text, confidence in candidates:
            clean = self.clean_text(raw_text)

            # Check plate length constraints
            if config.MIN_PLATE_LENGTH <= len(clean) <= config.MAX_PLATE_LENGTH:
                if confidence > best_conf:
                    best_conf = confidence
                    best_text = clean
                    best_raw = raw_text

        if best_text and best_conf >= config.MIN_OCR_CONF:
            return {
                "text": best_text,
                "raw_text": best_raw,
                "confidence": best_conf
            }

        return None

