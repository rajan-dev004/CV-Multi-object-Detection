from ultralytics import YOLO

class Detector:
    """
    Wrapper for the YOLOv8 Object Detection model.
    """
    def __init__(self, model_weight="yolov8n.pt", conf_threshold=0.5):
        """
        Initializes the YOLO model.
        Args:
            model_weight (str): Path to the model weights or model name (e.g., 'yolov8n.pt').
            conf_threshold (float): Confidence threshold for detections.
        """
        self.model = YOLO(model_weight)
        self.conf_threshold = conf_threshold

    def get_model(self):
        """
        Returns the underlying YOLO model instance.
        """
        return self.model
