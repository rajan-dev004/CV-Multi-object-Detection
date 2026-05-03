import streamlit as st
import cv2
import tempfile
import os
import time
from detector import Detector
from tracker import ObjectTracker
from annotate import annotate_frame
from utils import ensure_dir

def run_app():
    st.set_page_config(page_title="Multi-Object Tracking UI", layout="wide")
    
    st.title("🏃 Multi-Object Detection & Persistent Tracking")
    st.markdown("""
    Upload a sports or event video to detect and track subjects with persistent IDs.
    """)

    # Sidebar for configuration
    st.sidebar.header("Tracking Configuration")
    conf_thresh = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    tracker_type = st.sidebar.selectbox("Tracker Type", ["bytetrack.yaml", "botsort.yaml"])
    
    # Class Selection
    st.sidebar.subheader("Detection Categories")
    track_all = st.sidebar.checkbox("Track All Possible Objects", value=False)
    
    class_options = {
        "Person": 0, "Sports Ball": 32, "Bicycle": 1, "Car": 2, 
        "Motorcycle": 3, "Dog": 16, "Horse": 17, "Backpack": 24, 
        "Frisbee": 29, "Skis": 30, "Snowboard": 31, "Skateboard": 36, 
        "Surfboard": 37, "Tennis Racket": 38, "Baseball Bat": 34,
        "Baseball Glove": 35, "Kite": 33
    }
    
    if track_all:
        class_ids = None # None tells the AI to track all 80+ COCO classes
        st.sidebar.info("Tracking all 80+ object categories.")
    else:
        selected_classes = st.sidebar.multiselect(
            "Select specific objects", 
            options=list(class_options.keys()), 
            default=["Person", "Sports Ball"]
        )
        class_ids = [class_options[c] for c in selected_classes]
    
    show_paths = st.sidebar.checkbox("Show Trajectory Paths", value=True)
    frame_skip = st.sidebar.number_input("Frame Skip", min_value=0, max_value=10, value=0)

    uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        input_path = tfile.name

        # Setup output paths
        ensure_dir("outputs")
        output_path = os.path.join("outputs", "st_output_video.mp4")

        # Initialize Detector and Tracker
        detector = Detector(model_weight="yolov8n.pt", conf_threshold=conf_thresh)
        tracker = ObjectTracker(detector, tracker_type=tracker_type)

        if st.button("Start Tracking"):
            cap = cv2.VideoCapture(input_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Video Writer
            # Using 'avc1' (H.264) for better browser compatibility
            fourcc = cv2.VideoWriter_fourcc(*'avc1') 
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not out.isOpened():
                # Fallback to mp4v if avc1 is not available
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_preview = st.empty()

            frame_count = 0
            start_time = time.time()
            unique_ids = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    continue

                # Run tracking
                result, track_history = tracker.update(frame, classes=class_ids)
                
                # Stats
                if result.boxes is not None and result.boxes.id is not None:
                    unique_ids.update(result.boxes.id.int().cpu().tolist())

                # Annotate
                annotated_frame = annotate_frame(
                    frame=frame,
                    result=result,
                    track_history=track_history,
                    show_paths=show_paths
                )

                # Write to output
                out.write(annotated_frame)

                # Update UI
                if frame_count % 5 == 0:
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing frame {frame_count}/{total_frames}...")
                    # Show a small preview
                    preview_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_preview.image(preview_frame, channels="RGB", width='stretch')

            cap.release()
            out.release()
            
            end_time = time.time()
            duration = end_time - start_time

            st.success(f"Processing Complete! Total unique IDs tracked: {len(unique_ids)}")
            st.info(f"Total time: {duration:.2f}s | Avg FPS: {frame_count/duration:.2f}")

            # Display the output video
            if os.path.exists(output_path):
                st.subheader("Final Tracked Video")
                # For Streamlit to play it, we might need to convert it to H.264 if mp4v fails
                # But let's try the direct file first.
                st.video(output_path)

if __name__ == "__main__":
    run_app()
