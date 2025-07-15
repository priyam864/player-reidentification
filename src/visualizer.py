"""
Visualization module for drawing tracking results.
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List

from .config import VIS_CONFIG


class Visualizer:
    """
    Visualize tracking results on video frames.
    """
    
    def __init__(self):
        """
        Initialize the visualizer.
        """
        self.colors = VIS_CONFIG["colors"]
        self.bbox_thickness = VIS_CONFIG["bbox_thickness"]
        self.text_size = VIS_CONFIG["text_size"]
        self.text_thickness = VIS_CONFIG["text_thickness"]
        
    def get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """
        Get consistent color for a track ID.
        
        Args:
            track_id: Track identifier
            
        Returns:
            RGB color tuple
        """
        color_idx = track_id % len(self.colors)
        return self.colors[color_idx]
    
    def draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                  track_id: int, confidence: float = None) -> np.ndarray:
        """
        Draw bounding box with track ID.
        
        Args:
            frame: Input frame
            bbox: Bounding box as (x1, y1, x2, y2)
            track_id: Track identifier
            confidence: Detection confidence (optional)
            
        Returns:
            Frame with drawn bounding box
        """
        x1, y1, x2, y2 = bbox
        color = self.get_color_for_id(track_id)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
        
        # Prepare label
        label = f"ID: {track_id}"
        if confidence is not None:
            label += f" ({confidence:.2f})"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, self.text_size, self.text_thickness
        )
        
        # Draw label background
        cv2.rectangle(frame, 
                     (x1, y1 - text_height - baseline - 5), 
                     (x1 + text_width, y1), 
                     color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, self.text_size, 
                   (255, 255, 255), self.text_thickness)
        
        return frame
    
    def draw_tracks(self, frame: np.ndarray, 
                   tracks: Dict[int, Tuple[int, int, int, int]]) -> np.ndarray:
        """
        Draw all tracks on frame.
        
        Args:
            frame: Input frame
            tracks: Dictionary mapping track IDs to bounding boxes
            
        Returns:
            Frame with drawn tracks
        """
        result_frame = frame.copy()
        
        for track_id, bbox in tracks.items():
            result_frame = self.draw_bbox(result_frame, bbox, track_id)
        
        return result_frame
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Tuple[int, int, int, int, float]]) -> np.ndarray:
        """
        Draw raw detections on frame.
        
        Args:
            frame: Input frame
            detections: List of detections as (x1, y1, x2, y2, confidence)
            
        Returns:
            Frame with drawn detections
        """
        result_frame = frame.copy()
        
        for i, detection in enumerate(detections):
            x1, y1, x2, y2, confidence = detection
            color = (0, 255, 0)  # Green for detections
            
            # Draw bounding box
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, self.bbox_thickness)
            
            # Draw confidence
            label = f"Det: {confidence:.2f}"
            cv2.putText(result_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, self.text_size, 
                       color, self.text_thickness)
        
        return result_frame
    
    def draw_info(self, frame: np.ndarray, frame_number: int, 
                 track_count: int, fps: float = None) -> np.ndarray:
        """
        Draw information overlay on frame.
        
        Args:
            frame: Input frame
            frame_number: Current frame number
            track_count: Number of active tracks
            fps: Processing FPS (optional)
            
        Returns:
            Frame with information overlay
        """
        result_frame = frame.copy()
        
        # Prepare info text
        info_lines = [
            f"Frame: {frame_number}",
            f"Active Tracks: {track_count}"
        ]
        
        if fps is not None:
            info_lines.append(f"FPS: {fps:.1f}")
        
        # Draw info background
        line_height = 30
        text_height = len(info_lines) * line_height
        cv2.rectangle(result_frame, (10, 10), (250, 20 + text_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(result_frame, (10, 10), (250, 20 + text_height), 
                     (255, 255, 255), 2)
        
        # Draw info text
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * line_height
            cv2.putText(result_frame, line, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 255, 255), 2)
        
        return result_frame
    
    def create_comparison_frame(self, original: np.ndarray, 
                              processed: np.ndarray) -> np.ndarray:
        """
        Create side-by-side comparison frame.
        
        Args:
            original: Original frame
            processed: Processed frame with tracking
            
        Returns:
            Combined comparison frame
        """
        # Resize frames to same height
        h = min(original.shape[0], processed.shape[0])
        original_resized = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h))
        processed_resized = cv2.resize(processed, (int(processed.shape[1] * h / processed.shape[0]), h))
        
        # Add labels
        cv2.putText(original_resized, "Original", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(processed_resized, "Tracked", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Combine horizontally
        combined = np.hstack([original_resized, processed_resized])
        
        return combined
