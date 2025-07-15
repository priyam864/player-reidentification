"""
Feature extraction module for player re-identification.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .config import REID_CONFIG


class FeatureExtractor:
    """
    Extract features from player crops for re-identification.
    """
    
    def __init__(self):
        """
        Initialize the feature extractor.
        """
        self.feature_dim = REID_CONFIG["feature_dim"]
        self.transform = self._get_transform()
        
    def _get_transform(self):
        """
        Get image transformation pipeline.
        
        Returns:
            Transform pipeline
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),  # Standard ReID input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_color_histogram(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract color histogram features from player crop.
        
        Args:
            crop: Player crop image
            
        Returns:
            Color histogram features
        """
        if crop.size == 0:
            return np.zeros(64)
        
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Calculate histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        
        # Concatenate histograms
        features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        
        # Normalize
        features = features / (np.sum(features) + 1e-7)
        
        return features.astype(np.float32)
    
    def extract_texture_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract texture features using Local Binary Patterns.
        
        Args:
            crop: Player crop image
            
        Returns:
            Texture features
        """
        if crop.size == 0:
            return np.zeros(26)
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Simple texture features using gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate texture statistics
        features = []
        for grad in [grad_x, grad_y]:
            features.extend([
                np.mean(grad),
                np.std(grad),
                np.mean(np.abs(grad)),
                np.std(np.abs(grad)),
                np.percentile(grad, 25),
                np.percentile(grad, 75),
                np.percentile(grad, 90),
                np.percentile(grad, 10),
                np.max(grad),
                np.min(grad),
                np.median(grad),
                np.var(grad),
                np.mean(grad ** 2)
            ])
        
        features = np.array(features)
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features.astype(np.float32)
    
    def extract_spatial_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract spatial features from player crop.
        
        Args:
            crop: Player crop image
            
        Returns:
            Spatial features
        """
        if crop.size == 0:
            return np.zeros(8)
        
        h, w = crop.shape[:2]
        
        # Basic spatial features
        features = [
            h / w,  # Aspect ratio
            h,      # Height
            w,      # Width
            h * w,  # Area
            np.mean(crop),  # Mean intensity
            np.std(crop),   # Std intensity
            np.min(crop),   # Min intensity
            np.max(crop)    # Max intensity
        ]
        
        features = np.array(features)
        
        # Normalize
        features = features / (np.linalg.norm(features) + 1e-7)
        
        return features.astype(np.float32)
    
    def extract_features(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive features from player crop.
        
        Args:
            crop: Player crop image
            
        Returns:
            Combined feature vector
        """
        # Extract different types of features
        color_features = self.extract_color_histogram(crop)
        texture_features = self.extract_texture_features(crop)
        spatial_features = self.extract_spatial_features(crop)
        
        # Combine all features
        features = np.concatenate([color_features, texture_features, spatial_features])
        
        # Final normalization
        features = normalize([features])[0]
        
        return features.astype(np.float32)
    
    def compute_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        """
        Compute similarity between two feature vectors.
        
        Args:
            features1: First feature vector
            features2: Second feature vector
            
        Returns:
            Similarity score (0-1)
        """
        if features1.size == 0 or features2.size == 0:
            return 0.0
        
        # Reshape for cosine similarity
        features1 = features1.reshape(1, -1)
        features2 = features2.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(features1, features2)[0, 0]
        
        # Convert to 0-1 range
        similarity = (similarity + 1) / 2
        
        return float(similarity)
    
    def compute_batch_similarity(self, query_features: np.ndarray, 
                                gallery_features: List[np.ndarray]) -> List[float]:
        """
        Compute similarity between query features and a gallery of features.
        
        Args:
            query_features: Query feature vector
            gallery_features: List of gallery feature vectors
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for gallery_feat in gallery_features:
            sim = self.compute_similarity(query_features, gallery_feat)
            similarities.append(sim)
        
        return similarities
