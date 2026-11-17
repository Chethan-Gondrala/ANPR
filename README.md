# Automatic Number Plate Recognition (ANPR) - Phase 1 Prototype

An end-to-end Python prototype for Automatic Number Plate Recognition (ANPR) designed for parking management systems. The application accepts a vehicle/parking video file, detects license plates using a pre-trained **YOLOv8** model, crops and preprocesses plate regions, extracts text characters using **PaddleOCR**, deduplicates frame-by-frame readings, and displays interactive results in a **Streamlit** web interface.

---

## 1. What is ANPR?

**Automatic Number Plate Recognition (ANPR)** is a computer vision technology used to read vehicle registration plates automatically. It converts optical images of vehicle license plates into digital alphanumeric text string data.

In parking management systems, ANPR automates:
- Entry and exit logging
- Ticketless parking
- Automated boom barrier operation
- Parking fee calculations

---

## 2. How This Project Works

This Phase 1 prototype processes uploaded vehicle videos frame-by-frame:

1. **Video Ingestion**: Video files (`.mp4`, `.avi`, `.mov`) are opened via OpenCV.
2. **Frame Sampling**: Frames are processed at a configurable rate (`FRAME_SKIP`) for high performance.
3. **Plate Detection**: YOLO locates bounding box coordinates `(x1, y1, x2, y2)` for license plates.
4. **Cropping & Preprocessing**: The license plate region is safely cropped and enhanced (contrast adjustment / grayscale) for optimal OCR quality.
5. **Character Recognition**: PaddleOCR reads the text and returns character strings with confidence scores.
6. **Aggregation & Deduplication**: Detections across consecutive frames are aggregated, selecting the highest-confidence reading per vehicle while preventing duplicate listings.
7. **Visual Overlay**: Processed video frames are rendered with bounding boxes and plate labels for preview and download.

---

## 3. System Architecture & Pipeline

```text
Car Video File (MP4/AVI/MOV)
         │
         ▼
┌──────────────────┐
│  OpenCV Reader   │
└────────┬─────────┘
         │ (Frames)
         ▼
┌──────────────────┐
│    YOLO Model    │  ──►  "Where is the license plate?" (Bounding Box)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Crop & Enhance  │  ──►  Image Preprocessing (Grayscale / CLAHE)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│    PaddleOCR     │  ──►  "What characters are written on the plate?"
└────────┬─────────┘
         │ (Cleaned Alphanumeric String + Confidence)
         ▼
┌──────────────────┐
│Aggregation Engine│ ──►  Group repeated readings & deduplicate per plate
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   Streamlit UI   │  ──►  Processed Video Player + Results Summary Table
└──────────────────┘
```

---

## 4. Prerequisites

- **OS**: Windows 10 / 11 (or Linux/macOS)
- **Python**: Python 3.10 or 3.11
- **Hardware**: CPU (Minimum 4 cores) or NVIDIA GPU with CUDA support

---

## 5. Installation & Setup Instructions

Follow these exact steps to set up and run the application locally on Windows.

### Step 1: Clone or Open the Workspace

Open PowerShell or Command Prompt in your workspace directory:
```powershell
cd "d:\Desktop\anpr project"
```

### Step 2: Create a Virtual Environment

Create a clean Python virtual environment named `venv`:
```powershell
python -m venv venv
```

### Step 3: Activate Virtual Environment

```powershell
venv\Scripts\activate
```

### Step 4: Install Project Dependencies

Install all required Python packages:
```powershell
pip install -r requirements.txt
```

---

## 6. YOLO Model Setup & Configuration

### Automatic Weight Download
The application is pre-configured to automatically download pre-trained License Plate YOLO weights (`license_plate_detector.pt`) into the `models/` directory on first launch.

### Custom Model Configuration
To use your own pre-trained YOLO license plate model (`.pt` file):

1. Place your model file in `models/`:
   ```text
   models/my_license_plate_model.pt
   ```
2. Open `config.py` and update `DEFAULT_MODEL_FILENAME`:
   ```python
   DEFAULT_MODEL_FILENAME = "my_license_plate_model.pt"
   ```

---

## 7. Running the Application

Ensure your virtual environment is active (`venv\Scripts\activate`), then start the Streamlit web server:

```powershell
streamlit run app.py
```

The application will open in your web browser automatically at:
`http://localhost:8501`

---

## 8. How to Use the UI

1. Click **"Browse files"** or drag-and-drop your vehicle video (`parking_video.mp4`).
2. Preview the original uploaded video on the left card.
3. Click the **"🚀 Process Video"** button.
4. Monitor processing progress on the real-time progress bar.
5. View the processed video output with license plate bounding box overlays.
6. Inspect the **Detected License Plates** summary table showing plate numbers and confidence percentages.
7. Optionally click **"Download Processed Video"** to save the output video file.

---

## 9. How YOLO & PaddleOCR Work Together

- **YOLO (You Only Look Once)**: A real-time object detector responsible strictly for spatial localization (`x1, y1, x2, y2`). It does **not** read characters.
- **PaddleOCR**: An Optical Character Recognition engine trained on text datasets. It accepts cropped license plate image patches and extracts text characters (`KA01AB1234`) with confidence scores.

Combining localized bounding box cropping with focused OCR yields significantly higher recognition accuracy than running raw OCR over full HD video frames.

---

## 10. CPU / GPU Acceleration

- **CPU Mode**: Runs out of the box on standard computer hardware using PyTorch CPU and PaddlePaddle CPU builds.
- **CUDA / GPU Mode**: Automatically detected if CUDA-enabled PyTorch and GPU drivers are installed. In CUDA mode, both YOLO inference and PaddleOCR execute on GPU for multi-fold speed gains.
- The UI header displays a badge indicating current execution mode (`🖥️ Processing Device: CPU` or `⚡ Processing Device: CUDA`).

---

## 11. Project Directory Structure

```text
anpr_phase1/
│
├── app.py                   # Streamlit Web UI application
├── config.py                # Configuration parameters & threshold settings
├── requirements.txt         # Project dependencies
├── README.md                # Documentation
│
├── models/                  # Storage for YOLO license plate model weights (.pt)
│   └── README.md
│
├── input/                   # Storage for uploaded video files
├── output/                  # Storage for processed output video files
│
├── plate_detector/          # YOLO License Plate Detection module
│   ├── __init__.py
│   └── detector.py
│
├── ocr/                     # PaddleOCR Text Extraction module
│   ├── __init__.py
│   └── reader.py
│
├── video/                   # Video processing & frame aggregation engine
│   ├── __init__.py
│   └── processor.py
│
└── utils/                   # Image helper functions (crop, preprocess, draw overlay)
    ├── __init__.py
    └── image_utils.py
```

---

## 12. Troubleshooting & Common Errors

| Error | Cause | Solution |
| :--- | :--- | :--- |
| `No license plate was detected` | Low contrast, extreme angles, or tiny plate dimensions | Adjust `MIN_DETECTION_CONF` in `config.py` or test with a clearer video sample. |
| `Model file missing` | Failed weight download or incorrect file path | Verify internet connectivity for initial weight download or manually place `.pt` file in `models/`. |
| `OpenCV VideoCapture failed` | Corrupted video file or unsupported codec | Convert input video to standard H.264 MP4 format. |
| `PaddleOCR init error` | Missing dependencies | Ensure `paddlepaddle` and `paddleocr` are installed via `pip install -r requirements.txt`. |

---

## 13. Phase 1 Scope & Limitations

This prototype focuses **exclusively** on the core vision pipeline (`Video -> YOLO -> Crop -> PaddleOCR -> Aggregated Results`).

### Excluded Features in Phase 1:
- Database storage & persistence
- Payment processing & parking tickets
- RTSP / IP camera live streaming
- Boom barrier control hardware
- User authentication & role management
- WhatsApp / SMS notifications
