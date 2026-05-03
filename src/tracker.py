from collections import defaultdict
import numpy as np

class ObjectTracker:
    """
    Manages object tracking and trajectory history.
    """
    def __init__(self, detector, tracker_type="bytetrack.yaml", max_history=30):
        """
        Args:
            detector (Detector): An instance of the Detector class.
            tracker_type (str): Type of tracker, e.g. 'bytetrack.yaml' or 'botsort.yaml'.
            max_history (int): Maximum number of frames to store for trajectory plotting.
        """
        self.detector = detector
        self.tracker_type = tracker_type
        self.max_history = max_history
        # Store tracking history: track_id -> list of (x, y) coordinates
        self.track_history = defaultdict(lambda: [])
        # To avoid memory leaks in long videos, we can track missed frames
        self.missed_frames = defaultdict(int)
        self.max_missed = 60 # Remove history if unseen for 60 frames

    def update(self, frame, classes=None):
        """
        Runs object tracking on a single frame.
        
        Args:
            frame (np.ndarray): The image frame.
            classes (list, optional): List of class indices to track (e.g. [0, 32]).
            
        Returns:
            result: The raw Ultralytics result object.
            track_history: Dictionary of current trajectory paths.
        """
        results = self.detector.model.track(
            frame, 
            persist=True, 
            tracker=self.tracker_type, 
            conf=self.detector.conf_threshold, 
            verbose=False,
            classes=classes
        )
        
        result = results[0]
        boxes = result.boxes
        
        current_ids = set()
        
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            xywhs = boxes.xywh.cpu().numpy()
            
            for track_id, xywh in zip(track_ids, xywhs):
                current_ids.add(track_id)
                self.missed_frames[track_id] = 0 # Reset missed count
                
                # Get center coordinate
                x, y, w, h = xywh
                center = (float(x), float(y))
                self.track_history[track_id].append(center)
                
                # Limit history length
                if len(self.track_history[track_id]) > self.max_history:
                    self.track_history[track_id].pop(0)
                    
        # Cleanup old trajectories
        for track_id in list(self.track_history.keys()):
            if track_id not in current_ids:
                self.missed_frames[track_id] += 1
                if self.missed_frames[track_id] > self.max_missed:
                    del self.track_history[track_id]
                    del self.missed_frames[track_id]
                    
        return result, self.track_history
