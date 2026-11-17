import os
import tempfile
import streamlit as st
import config
from plate_detector import LicensePlateDetector
from ocr import PlateOCRReader
from video import VideoProcessor

# Streamlit Page Setup
st.set_page_config(
    page_title="ANPR - Automatic Number Plate Recognition",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 0.5rem;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .badge-cpu {
        background-color: #F1F5F9;
        color: #334155;
        border: 1px solid #CBD5E1;
    }
    .badge-cuda {
        background-color: #DCFCE7;
        color: #166534;
        border: 1px solid #86EFAC;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 0.75rem;
        padding: 1.25rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .early-stop-banner {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.75rem;
        border-radius: 0.5rem;
        border-left: 5px solid #F59E0B;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing YOLO and PaddleOCR models...")
def load_anpr_pipeline():
    """Load detector and OCR reader models once per Streamlit session."""
    detector = LicensePlateDetector(device=config.DEVICE)
    reader = PlateOCRReader()
    return detector, reader


def main():
    # Header Section
    st.markdown('<div class="main-header">ANPR - Automatic Number Plate Recognition</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload a vehicle video to detect and recognize license plates.</div>', unsafe_allow_html=True)

    # Device Badge Indicator
    device_name = config.DEVICE.upper()
    badge_class = "badge-cuda" if config.DEVICE == "cuda" else "badge-cpu"
    st.markdown(f'<div class="status-badge {badge_class}">Processing Device: {device_name}</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.header("Settings & Performance Controls")
    
    default_early_stop = getattr(config, 'EARLY_STOP_ENABLE', True)
    default_thresh = getattr(config, 'EARLY_STOP_THRESHOLD', 0.90)

    early_stop_toggle = st.sidebar.checkbox(
        "Stop Early on High Accuracy",
        value=default_early_stop,
        help="Stop processing immediately once a license plate is recognized with high confidence."
    )
    
    early_stop_thresh_pct = st.sidebar.slider(
        "High Confidence Threshold (%)",
        min_value=70,
        max_value=98,
        value=int(default_thresh * 100),
        step=1,
        help="Threshold to trigger early stopping"
    )
    early_stop_thresh = early_stop_thresh_pct / 100.0

    st.sidebar.markdown(f"**YOLO Model:** `{config.DEFAULT_MODEL_FILENAME}`")
    st.sidebar.markdown(f"**Frame Skip:** Every `{config.FRAME_SKIP}` frames")
    st.sidebar.markdown(f"**Min OCR Conf:** `{config.MIN_OCR_CONF * 100:.0f}%`")
    st.sidebar.markdown("---")
    st.sidebar.caption("ANPR Phase 1 Prototype | OpenCV + YOLO + PaddleOCR")

    # Video Upload Section
    st.subheader("Video Upload")
    uploaded_file = st.file_uploader(
        "Choose a vehicle or parking video file:",
        type=["mp4", "avi", "mov"],
        help="Supported formats: MP4, AVI, MOV"
    )

    if uploaded_file is not None:
        input_filename = uploaded_file.name
        input_path = os.path.join(config.INPUT_DIR, input_filename)

        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Selected video: `{input_filename}` ({uploaded_file.size / (1024*1024):.2f} MB)")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Video Preview")
            st.video(input_path)

        with col2:
            st.markdown("### Pipeline Execution")
            st.info(f"Early Stop is {'**ENABLED** (' + str(early_stop_thresh_pct) + '%)' if early_stop_toggle else '**DISABLED**'}. Click 'Process Video' below.")

            process_button = st.button("Process Video", type="primary", use_container_width=True)

        if process_button:
            try:
                # Load models
                detector, reader = load_anpr_pipeline()
                processor = VideoProcessor(
                    detector=detector,
                    reader=reader,
                    frame_skip=config.FRAME_SKIP,
                    early_stop=early_stop_toggle,
                    early_stop_thresh=early_stop_thresh
                )

                st.markdown("---")
                st.subheader("Processing Video...")
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total, msg=""):
                    pct = int((current / total) * 100) if total > 0 else 0
                    progress_bar.progress(pct)
                    if msg:
                        status_text.text(f"⚡ {msg}")
                    else:
                        status_text.text(f"Processing frame {current} of {total} ({pct}%)...")

                output_path = os.path.join(config.OUTPUT_DIR, f"processed_{input_filename}")

                # Execute pipeline
                results = processor.process_video(
                    input_path=input_path,
                    output_path=output_path,
                    progress_callback=update_progress
                )

                if results["stopped_early"]:
                    st.markdown(
                        f'<div class="early-stop-banner"> <b>Early Stop Triggered:</b> {results["stop_reason"]} Processed {results["total_frames_processed"]} of {results["total_video_frames"]} frames.</div>',
                        unsafe_allow_html=True
                    )
                    progress_bar.progress(100)
                else:
                    status_text.text("Processing complete for all frames!")
                    progress_bar.progress(100)

                st.toast("ANPR processing completed!!")

                # Results Display Section
                st.markdown("---")
                st.subheader("ANPR Detection Results")

                summary = results["summary"]

                if not summary:
                    st.warning("⚠️ No license plate was detected in this video or OCR could not confidently read the characters.")
                else:
                    res_col1, res_col2 = st.columns([1, 1])

                    with res_col1:
                        st.markdown("### Processed Video Output")
                        if os.path.exists(results["output_video_path"]):
                            st.video(results["output_video_path"])
                            with open(results["output_video_path"], "rb") as vid_file:
                                st.download_button(
                                    label="Download Processed Video",
                                    data=vid_file,
                                    file_name=f"processed_{input_filename}",
                                    mime="video/mp4"
                                )

                    with res_col2:
                        st.markdown("### Recognized License Plate")

                        # Top Detection Card
                        top_plate = summary[0]
                        st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="margin:0; color:#475569;">Detected Plate Number</h4>
                                <h1 style="margin:0.2rem 0; color:#0F172A; font-family:monospace; font-size: 2.5rem;">{top_plate['plate_text']}</h1>
                                <p style="margin:0; color:#166534; font-weight:600; font-size: 1.1rem;">Confidence: {int(top_plate['best_confidence'] * 100)}%</p>
                            </div>
                        """, unsafe_allow_html=True)

                        st.write("")
                        st.markdown("#### All Detections Summary")
                        table_data = []
                        for idx, item in enumerate(summary, start=1):
                            table_data.append({
                                "#": idx,
                                "Plate Number": item["plate_text"],
                                "Confidence": f"{int(item['best_confidence'] * 100)}%",
                                "Detections Count": item["count"]
                            })

                        st.dataframe(
                            table_data,
                            use_container_width=True,
                            hide_index=True
                        )

            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
                st.exception(e)
    else:
        st.info("Please upload an MP4, AVI, or MOV video file to get started.")


if __name__ == "__main__":
    main()
