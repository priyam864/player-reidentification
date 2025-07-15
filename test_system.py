"""
Simple test script for the player re-identification system.
"""

import cv2
import numpy as np
from src.main import PlayerReIDPipeline
import os


def test_pipeline():
    """
    Test the player re-identification pipeline.
    """
    print("Testing Player Re-identification Pipeline...")
    
    # Create pipeline
    pipeline = PlayerReIDPipeline()
    
    # Test with a simple frame
    test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128  # Gray frame
    
    # Process frame
    try:
        result_frame, tracks = pipeline.process_frame(test_frame)
        print(f"✓ Frame processing successful")
        print(f"  - Result frame shape: {result_frame.shape}")
        print(f"  - Number of tracks: {len(tracks)}")
        
        # Get statistics
        stats = pipeline.get_statistics()
        print(f"  - Statistics: {stats}")
        
    except Exception as e:
        print(f"✗ Frame processing failed: {e}")
        return False
    
    print("✓ Pipeline test completed successfully!")
    return True


def test_components():
    """
    Test individual components.
    """
    print("\nTesting individual components...")
    
    # Test detector
    try:
        from src.detector import PlayerDetector
        detector = PlayerDetector()
        print("✓ PlayerDetector initialized successfully")
        
        # Test detection on dummy frame
        test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        detections = detector.detect_players(test_frame)
        print(f"  - Detections: {len(detections)}")
        
    except Exception as e:
        print(f"✗ PlayerDetector test failed: {e}")
    
    # Test feature extractor
    try:
        from src.feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        print("✓ FeatureExtractor initialized successfully")
        
        # Test feature extraction
        test_crop = np.ones((100, 50, 3), dtype=np.uint8) * 128
        features = extractor.extract_features(test_crop)
        print(f"  - Feature dimension: {features.shape}")
        
    except Exception as e:
        print(f"✗ FeatureExtractor test failed: {e}")
    
    # Test tracker
    try:
        from src.tracker import PlayerTracker
        tracker = PlayerTracker()
        print("✓ PlayerTracker initialized successfully")
        
        # Test tracking
        test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        test_detections = [(100, 100, 200, 300, 0.8)]
        tracks = tracker.update(test_detections, test_frame)
        print(f"  - Tracks: {len(tracks)}")
        
    except Exception as e:
        print(f"✗ PlayerTracker test failed: {e}")
    
    # Test visualizer
    try:
        from src.visualizer import Visualizer
        visualizer = Visualizer()
        print("✓ Visualizer initialized successfully")
        
        # Test visualization
        test_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128
        test_tracks = {1: (100, 100, 200, 300)}
        result = visualizer.draw_tracks(test_frame, test_tracks)
        print(f"  - Visualization result shape: {result.shape}")
        
    except Exception as e:
        print(f"✗ Visualizer test failed: {e}")


def main():
    """
    Main test function.
    """
    print("=" * 50)
    print("Player Re-identification System Test")
    print("=" * 50)
    
    # Check if required directories exist
    required_dirs = ["models", "data", "output"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"Creating directory: {dir_name}")
            os.makedirs(dir_name)
    
    # Test components
    test_components()
    
    # Test pipeline
    test_pipeline()
    
    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
