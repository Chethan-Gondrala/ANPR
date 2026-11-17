import os
import io
import pandas as pd
import streamlit as st
import config
from plate_detector import LicensePlateDetector
from ocr import PlateOCRReader
from video import VideoProcessor

# Streamlit Page Setup
st.set_page_config(
    page_title="ANPR System - Vehicle Plate Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise Theme & Styling (Human Designed, No Emojis/Icons)
st.markdown("""
<style>
    /* Global Typography & Reset */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Top Navigation / App Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(140, 149, 159, 0.25);
    }
    .header-text-group {
        display: flex;
        flex-direction: column;
    }
    .app-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.25;
        color: inherit;
    }
    .app-description {
        font-size: 0.875rem;
        color: rgba(140, 149, 159, 0.9);
        margin-top: 0.25rem;
    }
    .header-badges {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .system-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.3rem 0.65rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        border: 1px solid rgba(140, 149, 159, 0.3);
        background: rgba(140, 149, 159, 0.08);
        color: inherit;
    }
    .status-indicator-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
    }
    .status-indicator-dot.cpu {
        background-color: #64748b;
    }

    /* Professional License Plate Presentation Box */
    .plate-card-container {
        border: 1px solid rgba(140, 149, 159, 0.3);
        background: rgba(140, 149, 159, 0.04);
        border-radius: 8px;
        padding: 1.25rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
    }
    .plate-badge-inner {
        background: #0f172a;
        border: 2px solid #334155;
        border-radius: 6px;
        padding: 0.65rem 2rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
        margin: 0.5rem 0;
    }
    .plate-number-value {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        color: #f8fafc;
        text-transform: uppercase;
        margin: 0;
    }
    .plate-card-subtext {
        font-size: 0.8rem;
        color: rgba(140, 149, 159, 0.85);
        margin-top: 0.25rem;
    }

    /* Metric Stat Card */
    .stat-metric-card {
        border: 1px solid rgba(140, 149, 159, 0.2);
        background: rgba(140, 149, 159, 0.04);
        border-radius: 6px;
        padding: 1rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: rgba(140, 149, 159, 0.9);
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        line-height: 1.2;
    }

    /* Notice & Alert Strips */
    .status-banner {
        border-radius: 6px;
        padding: 0.75rem 1rem;
        font-size: 0.85rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(59, 130, 246, 0.35);
        background: rgba(59, 130, 246, 0.07);
        color: inherit;
    }
    .status-banner-success {
        border-color: rgba(16, 185, 129, 0.35);
        background: rgba(16, 185, 129, 0.08);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading YOLO detector and OCR models...")
def load_anpr_pipeline():
    """Load detector and OCR reader models once per Streamlit session."""
    detector = LicensePlateDetector(device=config.DEVICE)
    reader = PlateOCRReader()
    return detector, reader


def get_available_sample_videos():
    """Scan input directory for available sample video files."""
    valid_exts = (".mp4", ".avi", ".mov")
    if not os.path.exists(config.INPUT_DIR):
        return []
    files = [
        f for f in os.listdir(config.INPUT_DIR)
        if f.lower().endswith(valid_exts) and not f.startswith(".")
    ]
    return sorted(files)


def main():
    # Application Header
    hardware_name = config.DEVICE.upper()
    dot_class = "cpu" if config.DEVICE == "cpu" else ""

    st.markdown(f"""
        <div class="app-header">
            <div class="header-text-group">
                <div class="app-title">Automatic Number Plate Recognition (ANPR)</div>
                <div class="app-description">High-performance video inference pipeline using YOLOv8 spatial localization and OCR text recognition.</div>
            </div>
            <div class="header-badges">
                <div class="system-pill">
                    <span class="status-indicator-dot {dot_class}"></span>
                    <span>Compute: {hardware_name}</span>
                </div>
                <div class="system-pill">
                    <span>Engine: YOLOv8 + OCR</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar: System Configuration & Performance Parameters
    st.sidebar.markdown("### Pipeline Configuration")

    default_early_stop = getattr(config, 'EARLY_STOP_ENABLE', True)
    default_thresh = getattr(config, 'EARLY_STOP_THRESHOLD', 0.90)
    default_frame_skip = getattr(config, 'FRAME_SKIP', 5)

    early_stop_toggle = st.sidebar.checkbox(
        "Early Stopping Optimization",
        value=default_early_stop,
        help="Halts processing as soon as a license plate is recognized with high confidence."
    )

    early_stop_thresh_pct = st.sidebar.slider(
        "Confidence Threshold (%)",
        min_value=70,
        max_value=98,
        value=int(default_thresh * 100),
        step=1,
        help="Target OCR confidence percentage to trigger early termination."
    )
    early_stop_thresh = early_stop_thresh_pct / 100.0

    frame_skip_val = st.sidebar.slider(
        "Frame Sampling Interval",
        min_value=1,
        max_value=15,
        value=default_frame_skip,
        step=1,
        help="Inference executed on every Nth frame. Higher values increase processing speed."
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Specifications")
    st.sidebar.caption(f"Detector Model: `{config.DEFAULT_MODEL_FILENAME}`")
    st.sidebar.caption(f"Min OCR Cutoff: `{int(config.MIN_OCR_CONF * 100)}%`")
    st.sidebar.caption(f"Device Acceleration: `{config.DEVICE.upper()}`")

    # Video Source Selection
    st.markdown("#### Video Source")
    input_mode = st.radio(
        "Select video source:",
        options=["Select from Sample Videos", "Upload Local Video File"],
        horizontal=True,
        label_visibility="collapsed"
    )

    selected_video_path = None
    selected_video_name = None

    sample_videos = get_available_sample_videos()

    if input_mode == "Select from Sample Videos":
        if sample_videos:
            selected_video_name = st.selectbox(
                "Choose a sample video from the repository:",
                options=sample_videos,
                index=0
            )
            if selected_video_name:
                selected_video_path = os.path.join(config.INPUT_DIR, selected_video_name)
        else:
            st.info("No sample videos found in the input folder. Please upload a file below.")

    else:
        uploaded_file = st.file_uploader(
            "Upload vehicle video file (MP4, AVI, MOV):",
            type=["mp4", "avi", "mov"],
            help="Select a video file to evaluate."
        )
        if uploaded_file is not None:
            selected_video_name = uploaded_file.name
            selected_video_path = os.path.join(config.INPUT_DIR, selected_video_name)
            with open(selected_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    # Main Processing Section
    if selected_video_path and os.path.exists(selected_video_path):
        file_size_mb = os.path.getsize(selected_video_path) / (1024 * 1024)

        st.markdown(f"""
            <div class="status-banner">
                Active Source: <b>{selected_video_name}</b> &nbsp;|&nbsp; File Size: <b>{file_size_mb:.2f} MB</b> &nbsp;|&nbsp; Early Stopping: <b>{'Enabled (' + str(early_stop_thresh_pct) + '%)' if early_stop_toggle else 'Disabled'}</b>
            </div>
        """, unsafe_allow_html=True)

        col_video_src, col_video_action = st.columns([1, 1], gap="medium")

        with col_video_src:
            st.markdown("##### Source Video")
            st.video(selected_video_path)

        with col_video_action:
            st.markdown("##### Pipeline Execution")
            st.write("Click below to run plate localization, cropping, enhancement, and OCR recognition.")
            run_button = st.button("Run Detection Pipeline", type="primary", use_container_width=True)

            if not run_button:
                st.caption("Awaiting user trigger to start inference.")

        if run_button:
            try:
                detector, reader = load_anpr_pipeline()
                processor = VideoProcessor(
                    detector=detector,
                    reader=reader,
                    frame_skip=frame_skip_val,
                    early_stop=early_stop_toggle,
                    early_stop_thresh=early_stop_thresh
                )

                st.markdown("---")
                st.markdown("##### Inference Progress")
                progress_bar = st.progress(0)
                status_box = st.empty()

                def update_progress(current, total, msg=""):
                    pct = int((current / total) * 100) if total > 0 else 0
                    progress_bar.progress(pct)
                    if msg:
                        status_box.markdown(f"Status: **{msg}**")
                    else:
                        status_box.markdown(f"Status: Processing frame **{current}** of **{total}** ({pct}%)")

                output_path = os.path.join(config.OUTPUT_DIR, f"processed_{selected_video_name}")

                results = processor.process_video(
                    input_path=selected_video_path,
                    output_path=output_path,
                    progress_callback=update_progress
                )

                progress_bar.progress(100)

                if results["stopped_early"]:
                    st.markdown(f"""
                        <div class="status-banner status-banner-success">
                            <b>Early Stop Triggered:</b> {results['stop_reason']} Processed {results['total_frames_processed']} of {results['total_video_frames']} frames.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    status_box.markdown(f"Status: **Completed analysis across all {results['total_frames_processed']} frames.**")

                st.toast("Processing complete.")

                # Results Presentation
                st.markdown("---")
                st.markdown("#### Recognition Summary")

                summary = results.get("summary", [])

                if not summary:
                    st.warning("No license plate was detected in this video or OCR confidence did not meet the required threshold.")
                else:
                    top_plate = summary[0]
                    plate_str = top_plate["plate_text"]
                    conf_pct = int(top_plate["best_confidence"] * 100)
                    count_val = top_plate["count"]
                    total_scanned = results["total_frames_processed"]

                    # Display Top Detected Plate in High-Contrast License Plate Card
                    st.markdown(f"""
                        <div class="plate-card-container">
                            <span style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(140, 149, 159, 0.9); font-weight: 600;">Primary Detected Plate</span>
                            <div class="plate-badge-inner">
                                <span class="plate-number-value">{plate_str}</span>
                            </div>
                            <div class="plate-card-subtext">Confidence Score: <strong style="color: #10b981;">{conf_pct}%</strong> &nbsp;|&nbsp; Verified Occurrences: <strong>{count_val}</strong> frames</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Key Metrics Row
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.markdown(f"""
                            <div class="stat-metric-card">
                                <div class="stat-label">Recognized Plate</div>
                                <div class="stat-value">{plate_str}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                            <div class="stat-metric-card">
                                <div class="stat-label">Confidence</div>
                                <div class="stat-value">{conf_pct}%</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m3:
                        st.markdown(f"""
                            <div class="stat-metric-card">
                                <div class="stat-label">Detections Count</div>
                                <div class="stat-value">{count_val}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m4:
                        st.markdown(f"""
                            <div class="stat-metric-card">
                                <div class="stat-label">Frames Evaluated</div>
                                <div class="stat-value">{total_scanned}</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # Video Output & Data Table Row
                    res_col_vid, res_col_tbl = st.columns([1, 1], gap="medium")

                    with res_col_vid:
                        st.markdown("##### Annotated Video Output")
                        if os.path.exists(results["output_video_path"]):
                            st.video(results["output_video_path"])
                            with open(results["output_video_path"], "rb") as vid_file:
                                st.download_button(
                                    label="Download Processed Video",
                                    data=vid_file,
                                    file_name=f"processed_{selected_video_name}",
                                    mime="video/mp4",
                                    use_container_width=True
                                )

                    with res_col_tbl:
                        st.markdown("##### Detection Registry")
                        table_records = []
                        for idx, item in enumerate(summary, start=1):
                            table_records.append({
                                "Index": idx,
                                "Plate Number": item["plate_text"],
                                "Confidence (%)": f"{int(item['best_confidence'] * 100)}%",
                                "Frame Count": item["count"],
                                "First Frame": item.get("first_frame", "-"),
                                "Last Frame": item.get("last_frame", "-")
                            })

                        df_results = pd.DataFrame(table_records)
                        st.dataframe(df_results, use_container_width=True, hide_index=True)

                        # CSV Export
                        csv_buffer = io.StringIO()
                        df_results.to_csv(csv_buffer, index=False)
                        st.download_button(
                            label="Export Detections to CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"anpr_detections_{selected_video_name}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

            except Exception as e:
                st.error(f"Inference error: {str(e)}")

    else:
        st.info("Select a sample video from above or upload a video file to proceed.")


if __name__ == "__main__":
    main()
