import cv2
import numpy as np

def crop_image(frame, bbox, padding=4):
    """
    Safely crops bounding box region from an image frame with optional padding.

    Args:
        frame (np.ndarray): Original image frame.
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
        padding (int): Pixel padding surrounding the bounding box.

    Returns:
        np.ndarray: Cropped image region.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # Apply padding safely within boundaries
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2].copy()

def preprocess_plate_image(cropped_img):
    """
    Enhances cropped license plate image quality for better OCR text recognition.

    Args:
        cropped_img (np.ndarray): Cropped BGR license plate image.

    Returns:
        list[np.ndarray]: List of candidate preprocessed image variants for OCR evaluation.
    """
    if cropped_img is None or cropped_img.size == 0:
        return []

    variants = []

    # Variant 1: Original BGR Crop
    variants.append(cropped_img)

    # Convert to Grayscale
    if len(cropped_img.shape) == 3:
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cropped_img.copy()

    # Resize if too small (PaddleOCR prefers crops at least 150px wide)
    h, w = gray.shape[:2]
    if w < 150:
        scale = 150.0 / w
        new_w = 150
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Variant 2: Scaled Grayscale
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    variants.append(gray_bgr)

    # Variant 3: CLAHE Contrast Enhanced
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    variants.append(enhanced_bgr)

    return variants

def draw_detection(frame, bbox, plate_text=None, confidence=None):
    """
    Draws styled bounding box and text label overlay on an OpenCV frame.

    Args:
        frame (np.ndarray): OpenCV image frame to draw on.
        bbox (tuple): Coordinates (x1, y1, x2, y2).
        plate_text (str, optional): License plate text to display.
        confidence (float, optional): Confidence score.

    Returns:
        np.ndarray: Annotated frame.
    """
    if frame is None:
        return frame

    x1, y1, x2, y2 = bbox

    # Colors (BGR)
    box_color = (0, 255, 128)      # Spring Green
    banner_color = (0, 200, 100)   # Dark Green background
    text_color = (255, 255, 255)   # White

    # Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Text Overlay
    if plate_text:
        conf_str = f" {int(confidence * 100)}%" if confidence else ""
        label = f"{plate_text}{conf_str}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Position text box above bounding box if space allows, otherwise inside
        text_y1 = max(0, y1 - text_height - 10)
        text_y2 = y1

        cv2.rectangle(frame, (x1, text_y1), (x1 + text_width + 12, text_y2), banner_color, -1)
        cv2.putText(frame, label, (x1 + 6, text_y2 - 6), font, font_scale, text_color, thickness, cv2.LINE_AA)

    return frame
