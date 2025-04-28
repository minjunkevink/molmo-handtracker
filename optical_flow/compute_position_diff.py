#!/usr/bin/env python3
"""
Position Difference Computation Script

Automatically processes all MP4 files from libero_10_highres_mp4 and libero_90_highres_mp4 directories.
Groups videos by task and processes all demos for each task:
1. Tracks a point at (104, 24) using CoTracker for each demo
2. Computes frame-to-frame position differences
3. Saves all demos for each task in a single NPZ file in dedicated output directories
"""

import os
import h5py
import numpy as np
import glob
import torch
import imageio.v3 as iio
import re
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# Configuration
MP4_DIRS = [
    '/scr/shared/datasets/kimkj/libero_10_highres_mp4',
    '/scr/shared/datasets/kimkj/libero_90_highres_mp4'
]

OUTPUT_DIRS = [
    '/scr/shared/datasets/kimkj/libero_10_npz',
    '/scr/shared/datasets/kimkj/libero_90_npz'
]

# Default tracking point coordinates
DEFAULT_TRACK_POINT = (104, 24)


def track_point_with_cotracker(video_path, start_point=DEFAULT_TRACK_POINT, device=None):
    """
    Track a point through a video using CoTracker.
    
    Args:
        video_path: Path to MP4 video file
        start_point: (x, y) coordinates of the point to track, starting in the first frame
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        NumPy array of shape [num_frames, 2] containing (x,y) coordinates for each frame
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading video: {video_path}")
    try:
        # Load frames from video
        frames = iio.imread(video_path, plugin="FFMPEG")
        print(f"Loaded video with {len(frames)} frames, shape: {frames.shape}")
    except Exception as e:
        print(f"Error loading video {video_path}: {str(e)}")
        raise
    
    # Prepare video tensor
    video_tensor = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)
    
    # Define initial point
    x, y = start_point
    initial_frame = 0
    initial_point = torch.tensor([[[initial_frame, x, y]]], dtype=torch.float32).to(device)
    
    # Load CoTracker model
    print("Loading CoTracker model...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    
    # Run tracking
    print(f"Tracking point ({x}, {y}) through {len(frames)} frames...")
    with torch.no_grad():
        tracks, visibility = cotracker(video_tensor, queries=initial_point)
    
    # Extract coordinates
    tracks_np = tracks[0, :, 0, :].cpu().numpy()  # Shape: [num_frames, 2]
    visibility_np = visibility[0, :, 0].cpu().numpy()  # Shape: [num_frames]
    
    # Apply visibility mask (replace low visibility points with previous known position)
    for i in range(1, len(tracks_np)):
        if visibility_np[i] < 0.5:
            tracks_np[i] = tracks_np[i-1]
    
    return tracks_np


def compute_position_diffs_from_tracks(tracks):
    """
    Compute frame-to-frame differences from a sequence of tracked positions.
    
    Args:
        tracks: NumPy array of shape [num_frames, 2] containing (x,y) coordinates
        
    Returns:
        NumPy array of shape [num_frames, 2] containing (x,y) position differences
    """
    num_frames = tracks.shape[0]
    diffs = np.zeros_like(tracks, dtype=np.float32)
    
    # First frame has no displacement
    diffs[0] = [0, 0]
    
    # Calculate displacement for subsequent frames
    for i in range(1, num_frames):
        diffs[i] = tracks[i] - tracks[i-1]
    
    return diffs


def parse_video_filename(mp4_path):
    """
    Parse the video filename to extract task name and demo number.
    
    Args:
        mp4_path: Path to MP4 file
        
    Returns:
        Tuple of (task_name, demo_number)
    """
    # Extract filename without extension
    filename = os.path.splitext(os.path.basename(mp4_path))[0]
    
    # Extract demo number and task name using regex
    match = re.match(r'VID-DEMO_(\d+)_(.+)_demo', filename)
    if match:
        demo_number = int(match.group(1))
        task_name = match.group(2)
    else:
        # Fallback if the pattern doesn't match
        parts = filename.split('_')
        if len(parts) >= 3 and parts[0] == 'VID-DEMO' and parts[1].isdigit():
            demo_number = int(parts[1])
            task_name = '_'.join(parts[2:-1]) if parts[-1] == 'demo' else '_'.join(parts[2:])
        else:
            # If we can't extract the demo number, just use the filename as the task
            demo_number = 0
            task_name = filename
    
    return task_name, demo_number


def group_videos_by_task(mp4_dir):
    """
    Group videos in a directory by task name.
    
    Args:
        mp4_dir: Directory containing MP4 files
        
    Returns:
        Dictionary mapping task names to lists of (video_path, demo_number) tuples
    """
    # Get all MP4 files in the directory
    mp4_files = glob.glob(os.path.join(mp4_dir, "**/*.mp4"), recursive=True)
    
    # Group by task
    task_videos = defaultdict(list)
    for mp4_file in mp4_files:
        task_name, demo_number = parse_video_filename(mp4_file)
        task_videos[task_name].append((mp4_file, demo_number))
    
    # Sort videos in each task by demo number
    for task_name in task_videos:
        task_videos[task_name].sort(key=lambda x: x[1])
    
    return task_videos


def process_task(task_name, video_tuples, output_dir, device=None):
    """
    Process all videos for a given task and save position differences.
    
    Args:
        task_name: Task name
        video_tuples: List of (video_path, demo_number) tuples
        output_dir: Directory to save the output NPZ file
        device: Device to run inference on ('cuda' or 'cpu')
        
    Returns:
        Path to the saved NPZ file or None if processing failed
    """
    # Create output filename
    output_path = os.path.join(output_dir, f"{task_name}_diff.npz")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip if the output file already exists
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists, skipping.")
        return output_path
    
    # Dictionary to store position differences for each demo
    position_diff_data = {}
    
    # Process each video
    successful = 0
    for video_path, demo_number in tqdm(video_tuples, desc=f"Processing demos for task '{task_name}'"):
        try:
            # Track point through the video
            tracks = track_point_with_cotracker(video_path, device=device)
            print(f"Successfully tracked point through {tracks.shape[0]} frames in demo {demo_number}")
            
            # Compute position differences from tracks
            pos_diffs = compute_position_diffs_from_tracks(tracks)
            
            # Store in dictionary with demo_N key
            position_diff_data[f"demo_{demo_number}"] = pos_diffs.astype(np.float32)
            successful += 1
            
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    
    # Save results if any demos were processed successfully
    if successful > 0:
        np.savez_compressed(output_path, **position_diff_data)
        print(f"Saved {successful} demos' position differences to {output_path}")
        return output_path
    
    print(f"Failed to process any demos for task '{task_name}'")
    return None


def process_directory(mp4_dir, output_dir, device=None):
    """
    Process all tasks in a directory.
    
    Args:
        mp4_dir: Directory containing MP4 files
        output_dir: Directory to save the output NPZ files
        device: Device to run inference on ('cuda' or 'cpu')
    """
    # Check if directories exist
    if not os.path.exists(mp4_dir):
        print(f"MP4 directory {mp4_dir} doesn't exist!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Group videos by task
    task_videos = group_videos_by_task(mp4_dir)
    
    if not task_videos:
        print(f"No MP4 files found in {mp4_dir}")
        return
    
    print(f"Found {len(task_videos)} tasks in {mp4_dir}")
    
    # Process each task
    successful_tasks = 0
    for task_name, video_tuples in task_videos.items():
        print(f"\nProcessing task: {task_name} ({len(video_tuples)} demos)")
        if process_task(task_name, video_tuples, output_dir, device=device):
            successful_tasks += 1
    
    print(f"Successfully processed {successful_tasks}/{len(task_videos)} tasks from {mp4_dir}")


def main():
    """
    Main function to process all tasks in the configured directories.
    """
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Tracking point at coordinates: {DEFAULT_TRACK_POINT}")
    
    # Process each MP4 directory
    for mp4_dir, output_dir in zip(MP4_DIRS, OUTPUT_DIRS):
        print(f"\nProcessing directory: {mp4_dir}")
        print(f"Saving results to: {output_dir}")
        process_directory(mp4_dir, output_dir, device=device)
    
    print("\nProcessing complete!")


if __name__ == "__main__":
    main() 