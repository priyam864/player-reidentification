"""
Configuration file for player re-identification system.
"""

# Detection Configuration
DETECTION_CONFIG = {
    "model_path": "models/best.pt",
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "input_size": (640, 640),
    "max_detections": 50,
    "model_type": "yolov11n"  # YOLOv11 nano as fallback
}

# Tracking Configuration
TRACKING_CONFIG = {
    "max_disappeared": 40,  # Max frames a player can be missing before being considered lost
    "max_distance": 100,    # Max distance for matching detections to existing tracks
    "feature_similarity_threshold": 0.7,  # Threshold for feature similarity matching
    "position_weight": 0.2,  # Weight for position in similarity calculation
    "appearance_weight": 0.8  # Weight for appearance in similarity calculation
}

# Re-identification Configuration
REID_CONFIG = {
    "feature_dim": 512,     # Dimension of feature vectors
    "similarity_threshold": 0.58,  # Threshold for re-identification
    "memory_size": 150,     # Number of feature vectors to store per player
    "update_rate": 0.1      # Rate of feature vector updates
}

# Video Processing Configuration
VIDEO_CONFIG = {
    "fps": 30,
    "frame_skip": 1,        # Process every nth frame
    "resize_height": 720,   # Resize video height for processing
    "output_codec": "mp4v"
}

# Visualization Configuration
VIS_CONFIG = {
    "bbox_thickness": 2,
    "text_size": 0.8,
    "text_thickness": 2,
    "colors": [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (255, 192, 203), # Pink
        (0, 128, 0),    # Dark Green
    ]
}
