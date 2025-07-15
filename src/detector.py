"""
Player detection module using YOLO model.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from ultralytics import YOLO
import torch

from .config import DETECTION_CONFIG


class PlayerDetector:
    """
    Player detection using YOLO model.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the player detector.
        
        Args:
            model_path: Path to the YOLO model file
        """
        self.model_path = model_path or DETECTION_CONFIG["model_path"]
        self.confidence_threshold = DETECTION_CONFIG["confidence_threshold"]
        self.iou_threshold = DETECTION_CONFIG["iou_threshold"]
        self.input_size = DETECTION_CONFIG["input_size"]
        self.max_detections = DETECTION_CONFIG["max_detections"]
        
        # Load YOLO model
        self.model = self._load_model()
        
    def _load_model(self) -> YOLO:
        """
        Load the YOLO model.
        
        Returns:
            Loaded YOLO model
        """
        try:
            model = YOLO(self.model_path)
            print(f"Successfully loaded custom model from {self.model_path}")
            return model
        except Exception as e:
            print(f"Error loading custom model: {e}")
            print("Using default YOLOv11 model")
            return YOLO('yolo11n.pt')  # Fallback to YOLOv11 nano model
    
    def detect_players(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect players in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detections as (x1, y1, x2, y2, confidence)
        """
        detections = []
        
        try:
            # Run inference
            results = self.model(frame, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates and confidence
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter for person class (class_id 0 in COCO)
                        # If using custom model, adjust class filtering as needed
                        if class_id == 0:  # Person class
                            detections.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))
            
            # Sort by confidence and limit number of detections
            detections = sorted(detections, key=lambda x: x[4], reverse=True)[:self.max_detections]
            
        except Exception as e:
            print(f"Error in player detection: {e}")
            
        return detections
    
    def extract_player_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract player crop from frame using bounding box.
        
        Args:
            frame: Input frame
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Cropped player image
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        return crop
    
    def get_bbox_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Get center point of bounding box.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Center point as (cx, cy)
        """
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return cx, cy
    
    def get_bbox_area(self, bbox: Tuple[int, int, int, int]) -> int:
        """
        Calculate area of bounding box.
        
        Args:
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Area of bounding box
        """
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
