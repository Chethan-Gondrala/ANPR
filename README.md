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

1. Choose your video source:
   - Select an existing test clip from the dropdown menu (pre-loaded in `input/`), OR
   - Upload a custom vehicle or parking video (`.mp4`, `.avi`, `.mov`).
2. Preview the original source video on the left panel.
3. Configure settings in the sidebar if needed (Frame Sampling Interval, Early Stop Confidence Threshold).
4. Click the **"Run Detection Pipeline"** button.
5. Monitor real-time progress on the frame counter and status indicator.
6. Review the **Primary Detected Plate** high-contrast card and key summary metrics (Confidence, Detection Count, Scanned Frames).
7. Inspect the side-by-side annotated video with bounding box overlays and download the processed output MP4.
8. View or export the structured detections registry table as a CSV file.

---

## 9. How YOLO & PaddleOCR Work Together

- **YOLO (You Only Look Once)**: A real-time object detector responsible strictly for spatial localization (`x1, y1, x2, y2`). It does not read characters.
- **PaddleOCR**: An Optical Character Recognition engine trained on text datasets. It accepts cropped license plate image patches and extracts text characters (`KA01AB1234`) with confidence scores.

Combining localized bounding box cropping with focused OCR yields significantly higher recognition accuracy than running raw OCR over full HD video frames.

---

## 10. CPU / GPU Acceleration

- **CPU Mode**: Runs out of the box on standard computer hardware using PyTorch CPU and PaddlePaddle CPU builds.
- **CUDA / GPU Mode**: Automatically detected if CUDA-enabled PyTorch and GPU drivers are installed. In CUDA mode, both YOLO inference and PaddleOCR execute on GPU for multi-fold speed gains.
- The UI header displays a status badge indicating current execution mode (`Compute: CPU` or `Compute: CUDA`).

---

## 11. Project Directory Structure

```text
anpr_phase1/
│
├── app.py                   # Streamlit Web UI application
├── config.py                # Configuration parameters & threshold settings
├── requirements.txt         # Project dependencies
├── packages.txt             # Linux system dependencies for cloud deployment
├── README.md                # Documentation
│
├── models/                  # Storage for YOLO license plate model weights (.pt)
│   └── README.md
│
├── input/                   # Storage for sample & uploaded video files
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

## 13. Deployment Guide

### Can I deploy this directly on GitHub?
**No, not directly on GitHub Pages.**
- **GitHub** is a Git source code hosting service, not a server runtime.
- **GitHub Pages** only serves static HTML, CSS, and client-side JavaScript. It **cannot** execute a Python backend, PyTorch neural networks, YOLO weights, or OpenCV video processing.
- However, you **use GitHub to host the repository**, which then integrates directly with cloud platforms to run the live application.

### Recommended Free & Cloud Deployment Options

#### Option A: Streamlit Community Cloud (Free & Recommended for Demos)
Streamlit provides free cloud hosting for Streamlit applications connected to GitHub:
1. Push your code to a public or private GitHub repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git push -u origin main
   ```
2. Navigate to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub account.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the main file path to `app.py`.
5. Streamlit Cloud automatically detects `requirements.txt` and `packages.txt` (which installs required Linux OpenCV libraries `libgl1` and `libglib2.0-0`).
6. Click **Deploy**. The application will be assigned a live public URL.

#### Option B: Hugging Face Spaces (Free CPU / Paid GPU)
Hugging Face Spaces is designed specifically for machine learning demos:
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK.
3. Clone the Space repo or push your files (including `requirements.txt` and `packages.txt`).
4. Hugging Face builds the environment with 2 vCPUs and 16 GB RAM for free, with optional GPU upgrades.

#### Option C: Cloud Virtual Machines (AWS EC2 / DigitalOcean / Azure / GCP)
For production workloads with dedicated GPU acceleration (NVIDIA CUDA):
1. Provision an Ubuntu VM instance (e.g., AWS EC2 `g4dn.xlarge` with NVIDIA T4 GPU).
2. Clone your GitHub repository onto the server.
3. Install NVIDIA drivers, CUDA toolkit, and Python.
4. Run Streamlit behind a reverse proxy (Nginx with SSL via Let's Encrypt) or via Docker:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

---

## 14. Phase 1 Scope & Limitations

This prototype focuses **exclusively** on the core vision pipeline (`Video -> YOLO -> Crop -> OCR -> Aggregated Results`).

### Excluded Features in Phase 1:
- Database storage & persistence
- Payment processing & parking tickets
- RTSP / IP camera live streaming
- Boom barrier control hardware
- User authentication & role management
- WhatsApp / SMS notifications
