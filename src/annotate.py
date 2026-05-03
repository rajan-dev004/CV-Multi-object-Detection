import cv2
import numpy as np

def get_color(idx):
    """
    Generate a distinct, pseudo-random color based on the track ID.
    """
    idx = int(idx)
    # Simple hash to generate a RGB tuple
    return ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

def annotate_frame(frame, result, track_history, fps=None, show_paths=False):
    """
    Annotate the frame with bounding boxes, tracking IDs, trajectory paths, and FPS.
    
    Args:
        frame (np.ndarray): Original image frame.
        result: Ultralytics YOLO result object for the frame.
        track_history (dict): Dictionary mapping track_id to a list of (x, y) coordinates.
        fps (float, optional): Frames per second to display.
        show_paths (bool): Whether to draw trailing paths.
        
    Returns:
        np.ndarray: The annotated frame.
    """
    # Create a copy to draw on
    annotated_frame = frame.copy()
    
    # Extract boxes, IDs, classes, and class names
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        track_ids = boxes.id.int().cpu().tolist()
        xyxy_boxes = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.int().cpu().tolist()
        names = result.names

        for box, track_id, cls in zip(xyxy_boxes, track_ids, classes):
            x1, y1, x2, y2 = map(int, box)
            color = get_color(track_id)
            
            # Draw Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw Label Background and Text
            label = f"{names[cls]} ID:{track_id}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Ensure we don't draw outside the top of the frame
            y_label_box = max(0, y1 - h - 10)
            cv2.rectangle(annotated_frame, (x1, y_label_box), (x1 + w, y_label_box + h + 10), color, -1)
            cv2.putText(annotated_frame, label, (x1, y_label_box + h + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw Trajectory Paths
            if show_paths and track_id in track_history:
                history = track_history[track_id]
                if len(history) > 1:
                    # Convert to numpy array of integer coordinates
                    points = np.hstack(history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=color, thickness=2)

    # Draw FPS
    if fps is not None:
        # A nice background for FPS
        cv2.rectangle(annotated_frame, (10, 10), (160, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
    return annotated_frame
