import argparse
import logging
import os
import sys

def setup_logger(name="Tracker"):
    """
    Setup a basic logger with formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
    return logger

def parse_args():
    """
    Parse command line arguments for the multi-object tracker.
    """
    parser = argparse.ArgumentParser(description="Multi-Object Detection and Persistent ID Tracking")
    parser.add_argument("--input", type=str, required=True, help="Path to the input video.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video.")
    parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold for detection.")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker type (bytetrack.yaml or botsort.yaml).")
    parser.add_argument("--frame_skip", type=int, default=0, help="Number of frames to skip.")
    parser.add_argument("--classes", type=int, nargs='+', default=None, help="List of class IDs to track. Default is None (track all).")
    parser.add_argument("--show_paths", action="store_true", help="Draw trajectory paths behind objects.")
    return parser.parse_args()

def check_file_exists(filepath):
    """
    Check if a file exists.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

def ensure_dir(dir_path):
    """
    Ensure the directory exists.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
