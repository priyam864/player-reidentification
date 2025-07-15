"""
Player tracking module for maintaining player identities across frames.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque
import cv2

from .config import TRACKING_CONFIG, REID_CONFIG
from .feature_extractor import FeatureExtractor


class PlayerTrack:
    """
    Represents a single player track.
    """
    
    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], 
                 features: np.ndarray, frame_id: int):
        """
        Initialize a player track.
        
        Args:
            track_id: Unique track identifier
            bbox: Bounding box as (x1, y1, x2, y2)
            features: Feature vector
            frame_id: Frame number when track was created
        """
        self.track_id = track_id
        self.bbox = bbox
        self.center = self._get_center(bbox)
        self.features = features
        self.feature_history = deque([features], maxlen=REID_CONFIG["memory_size"])
        self.last_seen_frame = frame_id
        self.disappeared_count = 0
        self.is_active = True
        
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def update(self, bbox: Tuple[int, int, int, int], features: np.ndarray, frame_id: int):
        """
        Update track with new detection.
        
        Args:
            bbox: New bounding box
            features: New feature vector
            frame_id: Current frame number
        """
        self.bbox = bbox
        self.center = self._get_center(bbox)
        self.last_seen_frame = frame_id
        self.disappeared_count = 0
        
        # Update features with moving average
        if len(self.feature_history) > 0:
            alpha = REID_CONFIG["update_rate"]
            self.features = alpha * features + (1 - alpha) * self.features
        else:
            self.features = features
        
        # Add to feature history
        self.feature_history.append(features)
    
    def increment_disappeared(self):
        """Increment disappeared counter."""
        self.disappeared_count += 1
        if self.disappeared_count >= TRACKING_CONFIG["max_disappeared"]:
            self.is_active = False
    
    def get_average_features(self) -> np.ndarray:
        """Get average features from history."""
        if len(self.feature_history) == 0:
            return self.features
        
        return np.mean(list(self.feature_history), axis=0)


class PlayerTracker:
    """
    Multi-object tracker for players with re-identification capabilities.
    """
    
    def __init__(self):
        """
        Initialize the player tracker.
        """
        self.tracks: Dict[int, PlayerTrack] = {}
        self.next_track_id = 1
        self.frame_count = 0
        self.max_disappeared = TRACKING_CONFIG["max_disappeared"]
        self.max_distance = TRACKING_CONFIG["max_distance"]
        self.feature_similarity_threshold = TRACKING_CONFIG["feature_similarity_threshold"]
        self.position_weight = TRACKING_CONFIG["position_weight"]
        self.appearance_weight = TRACKING_CONFIG["appearance_weight"]
        
        # Feature extractor for re-identification
        self.feature_extractor = FeatureExtractor()
        
        # Store inactive tracks for re-identification
        self.inactive_tracks: List[PlayerTrack] = []
        
        # Track re-identification events
        self.recent_reidentifications = []
        
        # Track recent re-identifications for analysis
        self.recent_reidentifications = []
        
    def _compute_distance(self, center1: Tuple[int, int], center2: Tuple[int, int]) -> float:
        """
        Compute Euclidean distance between two points.
        
        Args:
            center1: First point
            center2: Second point
            
        Returns:
            Distance between points
        """
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def _compute_combined_similarity(self, track: PlayerTrack, detection_bbox: Tuple[int, int, int, int],
                                   detection_features: np.ndarray) -> float:
        """
        Compute combined similarity based on position and appearance.
        
        Args:
            track: Existing track
            detection_bbox: Detection bounding box
            detection_features: Detection features
            
        Returns:
            Combined similarity score
        """
        # Position similarity
        detection_center = self._get_center(detection_bbox)
        distance = self._compute_distance(track.center, detection_center)
        position_similarity = 1.0 / (1.0 + distance / self.max_distance)
        
        # Appearance similarity
        appearance_similarity = self.feature_extractor.compute_similarity(
            track.get_average_features(), detection_features
        )
        
        # Combined similarity
        combined_similarity = (self.position_weight * position_similarity + 
                             self.appearance_weight * appearance_similarity)
        
        return combined_similarity
    
    def _get_center(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def _attempt_reidentification(self, detection_bbox: Tuple[int, int, int, int],
                                detection_features: np.ndarray) -> Optional[int]:
        """
        Attempt to re-identify a detection with inactive tracks.
        
        Args:
            detection_bbox: Detection bounding box
            detection_features: Detection features
            
        Returns:
            Track ID if re-identification successful, None otherwise
        """
        if not self.inactive_tracks:
            return None
            
        best_similarity = 0.0
        best_track_id = None
        best_track = None
        
        print(f"    Attempting re-identification with {len(self.inactive_tracks)} inactive tracks")
        
        for inactive_track in self.inactive_tracks:
            # Use average features for primary comparison
            avg_similarity = self.feature_extractor.compute_similarity(
                inactive_track.get_average_features(), detection_features
            )
            
            # Also check against individual feature vectors for better matching
            max_individual_similarity = 0.0
            for historical_features in inactive_track.feature_history:
                individual_sim = self.feature_extractor.compute_similarity(
                    historical_features, detection_features
                )
                max_individual_similarity = max(max_individual_similarity, individual_sim)
            
            # Use the maximum of average and individual similarities
            final_similarity = max(avg_similarity, max_individual_similarity * 0.9)  # Slight penalty for individual matches
            
            print(f"      Track {inactive_track.track_id}: avg={avg_similarity:.3f}, max_ind={max_individual_similarity:.3f}, final={final_similarity:.3f}")
            
            if final_similarity > best_similarity and final_similarity > REID_CONFIG["similarity_threshold"]:
                best_similarity = final_similarity
                best_track_id = inactive_track.track_id
                best_track = inactive_track
        
        if best_track_id is not None:
            # Remove from inactive tracks
            self.inactive_tracks = [t for t in self.inactive_tracks if t.track_id != best_track_id]
            
            # Create new active track with the same ID, preserving feature history
            new_track = PlayerTrack(best_track_id, detection_bbox, detection_features, self.frame_count)
            
            # Preserve feature history from the old track
            new_track.feature_history = best_track.feature_history.copy()
            new_track.feature_history.append(detection_features)
            
            # Update features with weighted combination of old and new
            alpha = 0.3  # Weight for new features
            new_track.features = alpha * detection_features + (1 - alpha) * best_track.features
            
            self.tracks[best_track_id] = new_track
            
            # Record re-identification event
            self.recent_reidentifications.append({
                'track_id': best_track_id,
                'bbox': detection_bbox,
                'confidence': best_similarity,
                'frame': self.frame_count
            })
            
            print(f"Player {best_track_id} re-identified with confidence {best_similarity:.3f}")
            
            return best_track_id
        else:
            print(f"    No re-identification found (threshold: {REID_CONFIG['similarity_threshold']:.3f})")
        
        return None
    
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
               frame: np.ndarray) -> Dict[int, Tuple[int, int, int, int]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections as (x1, y1, x2, y2, confidence)
            frame: Current frame
            
        Returns:
            Dictionary mapping track IDs to bounding boxes
        """
        self.frame_count += 1
        
        # Clear recent re-identifications from previous frame
        self.recent_reidentifications.clear()
        
        # Extract features for all detections
        detection_features = []
        detection_bboxes = []
        
        for detection in detections:
            bbox = detection[:4]
            crop = self._extract_crop(frame, bbox)
            features = self.feature_extractor.extract_features(crop)
            detection_features.append(features)
            detection_bboxes.append(bbox)
        
        # If no existing tracks, create new ones
        if not self.tracks:
            for i, (bbox, features) in enumerate(zip(detection_bboxes, detection_features)):
                track = PlayerTrack(self.next_track_id, bbox, features, self.frame_count)
                self.tracks[self.next_track_id] = track
                print(f"Player {self.next_track_id} initial assignment")
                self.next_track_id += 1
        else:
            # Compute similarity matrix
            similarity_matrix = np.zeros((len(self.tracks), len(detection_bboxes)))
            track_ids = list(self.tracks.keys())
            
            for i, track_id in enumerate(track_ids):
                track = self.tracks[track_id]
                for j, (bbox, features) in enumerate(zip(detection_bboxes, detection_features)):
                    similarity = self._compute_combined_similarity(track, bbox, features)
                    similarity_matrix[i, j] = similarity
            
            # Hungarian algorithm for assignment (simplified greedy approach)
            used_detections = set()
            used_tracks = set()
            
            # Sort by similarity
            matches = []
            for i in range(len(track_ids)):
                for j in range(len(detection_bboxes)):
                    if i not in used_tracks and j not in used_detections:
                        similarity = similarity_matrix[i, j]
                        if similarity > self.feature_similarity_threshold:
                            matches.append((i, j, similarity))
            
            # Sort matches by similarity
            matches.sort(key=lambda x: x[2], reverse=True)
            
            # Apply matches
            for track_idx, detection_idx, similarity in matches:
                if track_idx not in used_tracks and detection_idx not in used_detections:
                    track_id = track_ids[track_idx]
                    bbox = detection_bboxes[detection_idx]
                    features = detection_features[detection_idx]
                    
                    self.tracks[track_id].update(bbox, features, self.frame_count)
                    used_tracks.add(track_idx)
                    used_detections.add(detection_idx)
            
            # Handle unmatched detections
            for j in range(len(detection_bboxes)):
                if j not in used_detections:
                    bbox = detection_bboxes[j]
                    features = detection_features[j]
                    
                    # Try re-identification
                    reid_track_id = self._attempt_reidentification(bbox, features)
                    
                    if reid_track_id is None:
                        # Create new track
                        track = PlayerTrack(self.next_track_id, bbox, features, self.frame_count)
                        self.tracks[self.next_track_id] = track
                        print(f"Player {self.next_track_id} new assignment")
                        self.next_track_id += 1
            
            # Handle unmatched tracks
            for i in range(len(track_ids)):
                if i not in used_tracks:
                    track_id = track_ids[i]
                    self.tracks[track_id].increment_disappeared()
                    
                    # Move to inactive if disappeared too long
                    if not self.tracks[track_id].is_active:
                        print(f"Player {track_id} disappeared (moved to inactive)")
                        self.inactive_tracks.append(self.tracks[track_id])
                        del self.tracks[track_id]
        
        # Return current active tracks
        result = {}
        for track_id, track in self.tracks.items():
            result[track_id] = track.bbox
        
        return result
    
    def _extract_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract crop from frame using bounding box.
        
        Args:
            frame: Input frame
            bbox: Bounding box as (x1, y1, x2, y2)
            
        Returns:
            Cropped image
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
    
    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)
    
    def get_all_track_ids(self) -> List[int]:
        """Get all active track IDs."""
        return list(self.tracks.keys())
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.inactive_tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
