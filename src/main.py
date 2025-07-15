"""
Main processing pipeline for player re-identification.
"""

import cv2
import numpy as np
import time
import os
from typing import Dict, List, Tuple, Optional
import argparse

from .detector import PlayerDetector
from .tracker import PlayerTracker
from .visualizer import Visualizer
from .config import VIDEO_CONFIG


class PlayerReIDPipeline:
    """
    Main pipeline for player re-identification in sports footage.
    """
    
    def __init__(self, model_path: str = None, output_dir: str = "output"):
        """
        Initialize the pipeline.
        
        Args:
            model_path: Path to YOLO model file
            output_dir: Directory to save output videos
        """
        self.detector = PlayerDetector(model_path)
        self.tracker = PlayerTracker()
        self.visualizer = Visualizer()
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Performance tracking
        self.frame_times = []
        self.total_frames = 0
        
    def process_video(self, input_path: str, output_path: str = None, 
                     show_detections: bool = False, save_comparison: bool = False) -> bool:
        """
        Process a video file for player re-identification.
        
        Args:
            input_path: Path to input video file
            output_path: Path to output video file (optional)
            show_detections: Whether to show raw detections
            save_comparison: Whether to save side-by-side comparison
            
        Returns:
            True if processing successful, False otherwise
        """
        print(f"Processing video: {input_path}")
        
        # Open video file
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {input_path}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(self.output_dir, f"{base_name}_tracked.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CONFIG["output_codec"])
        
        if save_comparison:
            out_width = width * 2
            comparison_path = os.path.join(self.output_dir, f"{base_name}_comparison.mp4")
            comparison_writer = cv2.VideoWriter(comparison_path, fourcc, fps, (out_width, height))
        
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out_writer.isOpened():
            print(f"Error: Could not create output video {output_path}")
            return False
        
        # Reset tracker
        self.tracker.reset()
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        print("Starting video processing...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start_time = time.time()
            
            # Resize frame if needed
            if VIDEO_CONFIG["resize_height"] and height > VIDEO_CONFIG["resize_height"]:
                scale = VIDEO_CONFIG["resize_height"] / height
                new_width = int(width * scale)
                new_height = VIDEO_CONFIG["resize_height"]
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Skip frames if configured
            if frame_count % VIDEO_CONFIG["frame_skip"] != 0:
                frame_count += 1
                continue
            
            # Detect players
            detections = self.detector.detect_players(frame)
            
            # Update tracker
            tracks = self.tracker.update(detections, frame)
            
            # Create visualization
            result_frame = frame.copy()
            
            if show_detections:
                result_frame = self.visualizer.draw_detections(result_frame, detections)
            
            result_frame = self.visualizer.draw_tracks(result_frame, tracks)
            
            # Calculate processing FPS
            frame_time = time.time() - frame_start_time
            self.frame_times.append(frame_time)
            
            if len(self.frame_times) > 30:  # Keep last 30 frames for FPS calculation
                self.frame_times.pop(0)
            
            processing_fps = 1.0 / np.mean(self.frame_times) if self.frame_times else 0
            
            # Add info overlay
            result_frame = self.visualizer.draw_info(result_frame, frame_count, 
                                                   self.tracker.get_track_count(), 
                                                   processing_fps)
            
            # Write output frame
            out_writer.write(result_frame)
            
            # Write comparison frame if requested
            if save_comparison:
                comparison_frame = self.visualizer.create_comparison_frame(frame, result_frame)
                comparison_writer.write(comparison_frame)
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            frame_count += 1
        
        # Cleanup
        cap.release()
        out_writer.release()
        if save_comparison:
            comparison_writer.release()
        
        # Calculate statistics
        end_time = time.time()
        processing_time = end_time - start_time
        avg_fps = frame_count / processing_time if processing_time > 0 else 0
        
        print(f"Processing complete!")
        print(f"Total frames processed: {frame_count}")
        print(f"Processing time: {processing_time:.2f} seconds")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Output saved to: {output_path}")
        
        if save_comparison:
            print(f"Comparison video saved to: {comparison_path}")
        
        return True
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, int]]]:
        """
        Process a single frame for real-time applications.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, tracks_dict)
        """
        # Detect players
        detections = self.detector.detect_players(frame)
        
        # Update tracker
        tracks = self.tracker.update(detections, frame)
        
        # Create visualization
        result_frame = self.visualizer.draw_tracks(frame.copy(), tracks)
        
        return result_frame, tracks
    
    def get_statistics(self) -> Dict:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
        """
        return {
            "total_frames": self.total_frames,
            "avg_frame_time": np.mean(self.frame_times) if self.frame_times else 0,
            "avg_fps": 1.0 / np.mean(self.frame_times) if self.frame_times else 0,
            "active_tracks": self.tracker.get_track_count(),
            "total_tracks_created": self.tracker.next_track_id - 1
        }


def main():
    """
    Main function for command-line usage.
    """
    parser = argparse.ArgumentParser(description="Player Re-identification in Sports Footage")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("--output", help="Output video file path")
    parser.add_argument("--model", help="Path to YOLO model file")
    parser.add_argument("--show-detections", action="store_true", 
                       help="Show raw detections in output")
    parser.add_argument("--save-comparison", action="store_true",
                       help="Save side-by-side comparison video")
    parser.add_argument("--output-dir", default="output",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} does not exist")
        return
    
    # Create pipeline
    pipeline = PlayerReIDPipeline(model_path=args.model, output_dir=args.output_dir)
    
    # Process video
    success = pipeline.process_video(
        input_path=args.input,
        output_path=args.output,
        show_detections=args.show_detections,
        save_comparison=args.save_comparison
    )
    
    if success:
        # Print statistics
        stats = pipeline.get_statistics()
        print("\nProcessing Statistics:")
        print(f"Total tracks created: {stats['total_tracks_created']}")
        print(f"Average processing FPS: {stats['avg_fps']:.2f}")
        print(f"Average frame time: {stats['avg_frame_time']:.4f}s")
    else:
        print("Processing failed!")


if __name__ == "__main__":
    main()
