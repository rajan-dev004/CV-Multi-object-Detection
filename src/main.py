import cv2
import time
import os
from utils import parse_args, setup_logger, ensure_dir, check_file_exists
from detector import Detector
from tracker import ObjectTracker
from annotate import annotate_frame

def main():
    args = parse_args()
    logger = setup_logger()

    logger.info("Initializing Multi-Object Detection and Tracking pipeline...")

    # Validate inputs
    try:
        check_file_exists(args.input)
    except FileNotFoundError as e:
        logger.error(e)
        return

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir:
        ensure_dir(output_dir)

    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        logger.error(f"Error opening video stream or file: {args.input}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video Info: {width}x{height} @ {fps}fps, {total_frames} frames")

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Initialize Detector and Tracker
    logger.info(f"Loading YOLO model with confidence threshold: {args.confidence}")
    detector = Detector(model_weight="yolov8n.pt", conf_threshold=args.confidence)
    
    logger.info(f"Initializing tracker: {args.tracker}")
    tracker = ObjectTracker(detector, tracker_type=args.tracker)

    frame_count = 0
    start_time = time.time()
    
    # Tracking statistics
    unique_ids_seen = set()

    logger.info("Starting processing...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Frame skipping
        if args.frame_skip > 0 and frame_count % (args.frame_skip + 1) != 0:
            continue

        # Tracking
        t1 = time.time()
        result, track_history = tracker.update(frame, classes=args.classes)
        t2 = time.time()
        
        # Compute processing FPS
        processing_fps = 1.0 / (t2 - t1) if (t2 - t1) > 0 else 0

        # Update statistics
        if result.boxes is not None and result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
            unique_ids_seen.update(track_ids)

        # Annotate frame
        annotated_frame = annotate_frame(
            frame=frame,
            result=result,
            track_history=track_history,
            fps=processing_fps,
            show_paths=args.show_paths
        )

        # Write out
        out.write(annotated_frame)

        # Logging progress
        if frame_count % 30 == 0:
            logger.info(f"Processed frame {frame_count}/{total_frames} ({processing_fps:.1f} fps)")

    # Cleanup
    cap.release()
    out.release()
    total_time = time.time() - start_time
    
    logger.info("Processing complete!")
    logger.info("====== Tracking Statistics ======")
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Average processing FPS: {frame_count / total_time:.2f}")
    logger.info(f"Total unique IDs tracked: {len(unique_ids_seen)}")
    logger.info(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
