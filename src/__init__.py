"""
Player Re-identification Package

This package provides tools for player re-identification in sports footage.
"""

from .main import PlayerReIDPipeline
from .detector import PlayerDetector
from .tracker import PlayerTracker
from .feature_extractor import FeatureExtractor
from .visualizer import Visualizer

__version__ = "1.0.0"
__author__ = "AI Intern"
__email__ = "arshdeep@liat.ai"

__all__ = [
    "PlayerReIDPipeline",
    "PlayerDetector", 
    "PlayerTracker",
    "FeatureExtractor",
    "Visualizer"
]
