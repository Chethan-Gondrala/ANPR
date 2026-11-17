import os
import cv2
import numpy as np
import config
from utils.image_utils import crop_image, preprocess_plate_image, draw_detection

class VideoProcessor:
    """
    Handles frame-by-frame video processing, ANPR pipeline execution,
    result aggregation/deduplication, and early-stopping on high-confidence detections.
    """
    def __init__(self, detector, reader, frame_skip=None, early_stop=None, early_stop_thresh=None):
        self.detector = detector
        self.reader = reader
        self.frame_skip = frame_skip if frame_skip is not None else getattr(config, 'FRAME_SKIP', 5)
        self.early_stop = early_stop if early_stop is not None else getattr(config, 'EARLY_STOP_ENABLE', True)
        self.early_stop_thresh = early_stop_thresh if early_stop_thresh is not None else getattr(config, 'EARLY_STOP_THRESHOLD', 0.90)

    def process_video(self, input_path, output_path=None, progress_callback=None):
        """
        Process uploaded video through the ANPR pipeline.

        Args:
            input_path (str): Path to input video file.
            output_path (str, optional): Target path for processed output MP4.
            progress_callback (callable, optional): Callback fn(current_frame, total_frames, status_msg).

        Returns:
            dict: {
                "output_video_path": str,
                "summary": list[dict],
                "total_frames_processed": int,
                "total_video_frames": int,
                "stopped_early": bool,
                "stop_reason": str
            }
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not open video file: {input_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if output_path is None:
            filename = os.path.basename(input_path)
            output_path = os.path.join(config.OUTPUT_DIR, f"processed_{filename}")

        # Setup VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        aggregated_plates = {}
        cached_current_detections = []
        frame_idx = 0
        stopped_early = False
        stop_reason = ""

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Frame sampling: Run detection & OCR only on sampled frames
                if (frame_idx - 1) % self.frame_skip == 0:
                    cached_current_detections = []
                    detections = self.detector.detect(frame)

                    for det in detections:
                        bbox = det["bbox"]
                        yolo_conf = det["confidence"]

                        crop = crop_image(frame, bbox)
                        if crop is None:
                            continue

                        # Generate preprocessed candidate crops for OCR
                        crop_variants = preprocess_plate_image(crop)

                        ocr_result = None
                        for variant in crop_variants:
                            ocr_res = self.reader.read_plate(variant)
                            if ocr_res:
                                ocr_result = ocr_res
                                break

                        if ocr_result:
                            plate_text = ocr_result["text"]
                            ocr_conf = ocr_result["confidence"]

                            # Deduplication & aggregation logic
                            if plate_text in aggregated_plates:
                                aggregated_plates[plate_text]["count"] += 1
                                aggregated_plates[plate_text]["last_frame"] = frame_idx

                                if ocr_conf > aggregated_plates[plate_text]["best_confidence"]:
                                    aggregated_plates[plate_text]["best_confidence"] = ocr_conf
                                    aggregated_plates[plate_text]["yolo_conf"] = yolo_conf
                                    aggregated_plates[plate_text]["best_bbox"] = bbox
                            else:
                                aggregated_plates[plate_text] = {
                                    "plate_text": plate_text,
                                    "best_confidence": ocr_conf,
                                    "yolo_conf": yolo_conf,
                                    "count": 1,
                                    "first_frame": frame_idx,
                                    "last_frame": frame_idx,
                                    "best_bbox": bbox
                                }

                            cached_current_detections.append({
                                "bbox": bbox,
                                "text": plate_text,
                                "confidence": ocr_conf
                            })

                            # Check for Early Stopping Condition
                            if self.early_stop and ocr_conf >= self.early_stop_thresh:
                                stopped_early = True
                                stop_reason = f"High accuracy detection ({plate_text}: {int(ocr_conf*100)}%) reached at frame {frame_idx}."
                                break

                        else:
                            cached_current_detections.append({
                                "bbox": bbox,
                                "text": "PLATE",
                                "confidence": yolo_conf
                            })

                # Annotate current frame with bounding box & label overlays
                annotated_frame = frame.copy()
                for det in cached_current_detections:
                    draw_detection(
                        annotated_frame,
                        det["bbox"],
                        det["text"],
                        det["confidence"]
                    )

                out.write(annotated_frame)

                if progress_callback and total_frames > 0:
                    progress_callback(frame_idx, total_frames, stop_reason if stopped_early else "")

                if stopped_early:
                    print(f"[VideoProcessor] {stop_reason} Terminating video processing early.")
                    break

        finally:
            cap.release()
            out.release()

        # Build final aggregated detection summary
        summary = sorted(
            aggregated_plates.values(),
            key=lambda x: (x["best_confidence"], x["count"]),
            reverse=True
        )

        return {
            "output_video_path": output_path,
            "summary": summary,
            "total_frames_processed": frame_idx,
            "total_video_frames": total_frames,
            "detected_plates_count": len(summary),
            "stopped_early": stopped_early,
            "stop_reason": stop_reason
        }
