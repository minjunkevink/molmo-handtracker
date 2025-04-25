#!/usr/bin/env python3
"""
Gripper Tracking Script

Tracks a single point (gripper midpoint) through a video and stores its
relative displacement between frames in HDF5 format.
"""

import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import argparse
from pathlib import Path
from tqdm import tqdm
from cotracker.utils.visualizer import Visualizer
import glob

def track_gripper(
    video_path,
    output_dir,
    gripper_point=(104, 24),  
    device=None,
    disable_visualization=False,
    output_to_custom_path=False,
    custom_output_path=None
):
    """
    Track a single gripper point through the video and record its relative movement.
    
    Args:
        video_path: Path to video file
        output_dir: Directory to save results
        gripper_point: (x, y) coordinates of the gripper in the first frame
        device: Device to run inference on ('cuda' or 'cpu')
        disable_visualization: If True, skips generating visualizations
        output_to_custom_path: If True, saves HDF5 file to custom_output_path
        custom_output_path: Custom path to save HDF5 file (if output_to_custom_path is True)
    
    Returns:
        Tuple containing:
        - Path to the saved HDF5 file
        - NumPy array of shape [num_frames, 2] containing (x,y) coordinates for each frame
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the CoTracker model
    print("Loading CoTracker model...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    
    # Load video
    print(f"Loading video: {video_path}")
    reader = imageio.get_reader(video_path)
    frames = [np.array(im) for im in reader]
    
    # Create video-specific output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Get video dimensions
    height, width = frames[0].shape[0:2]
    num_frames = len(frames)
    print(f"Video dimensions: {width}x{height}, {num_frames} frames")
    
    # Define gripper point
    gripper_x, gripper_y = gripper_point
    
    # Make sure point is within frame bounds
    if gripper_x < 0 or gripper_x >= width or gripper_y < 0 or gripper_y >= height:
        print(f"Warning: Gripper point {gripper_point} is outside frame bounds. Adjusting...")
        gripper_x = min(max(0, gripper_x), width - 1)
        gripper_y = min(max(0, gripper_y), height - 1)
    
    # Visualize the gripper point on the first frame
    if not disable_visualization:
        plt.figure(figsize=(10, 10))
        plt.imshow(frames[0])
        plt.scatter(gripper_x, gripper_y, c='red', s=100, marker='x')
        plt.text(gripper_x+10, gripper_y+10, f"Gripper ({gripper_x}, {gripper_y})", 
                color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        
        # Add zoomed inset
        from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
        
        # Zoom factor depends on image size
        zoom_factor = min(8, min(width, height) / 40)
        
        # Create zoomed inset
        axins = zoomed_inset_axes(plt.gca(), zoom=zoom_factor, loc=2)  # Upper left
        axins.imshow(frames[0])
        axins.scatter(gripper_x, gripper_y, c='red', s=100, marker='x')
        
        # Set limits for zoom region
        zoom_radius = min(20, min(width, height) / 8)
        axins.set_xlim(gripper_x - zoom_radius, gripper_x + zoom_radius)
        axins.set_ylim(gripper_y + zoom_radius, gripper_y - zoom_radius)  # Reversed y-axis
        axins.set_xticks([])
        axins.set_yticks([])
        mark_inset(plt.gca(), axins, loc1=1, loc2=3, fc="none", ec="red")
        
        plt.title("Gripper Point Verification")
        plt.savefig(os.path.join(video_output_dir, "gripper_point.png"))
        plt.close()
    
    # Initialize tracking point
    initial_frame = 0
    initial_point = torch.tensor([[[initial_frame, gripper_x, gripper_y]]], dtype=torch.float32).to(device)
    print(f"Initial point: ({gripper_x}, {gripper_y})")
    
    # Prepare video tensor
    frames_array = np.stack(frames)
    video_tensor = torch.tensor(frames_array).permute(0, 3, 1, 2)[None].float().to(device)
    print(f"Video tensor shape: {video_tensor.shape}")
    
    # Perform bidirectional tracking
    print("Running forward tracking...")
    with torch.no_grad():
        # Forward tracking
        tracks_fwd, visibility_fwd = cotracker(video_tensor, queries=initial_point, backward_tracking=False)
        
        # Reverse video for backward tracking
        video_reversed = torch.flip(video_tensor, dims=[1])
        
        print("Running backward tracking...")
        # Get position at last frame from forward tracking
        last_frame_position = tracks_fwd[:, -1, :, :]
    
    # Convert to numpy arrays
    tracks_np = tracks_fwd[0, :, 0, :].cpu().numpy()  # Shape: [num_frames, 2]
    visibility_np = visibility_fwd[0, :, 0].cpu().numpy()  # Shape: [num_frames]
    
    # Calculate relative displacements between consecutive frames
    displacements = np.zeros_like(tracks_np)  # Initialize with zeros
    
    # First frame has no displacement (relative to itself)
    displacements[0] = [0, 0]
    
    # Calculate displacement for each subsequent frame
    for i in range(1, num_frames):
        if visibility_np[i] > 0.5 and visibility_np[i-1] > 0.5:
            # If both current and previous frames have visible points,
            # calculate the displacement
            displacements[i] = tracks_np[i] - tracks_np[i-1]
        else:
            # If either frame has low visibility, set displacement to 0
            displacements[i] = [0, 0]
    
    # Normalize coordinates
    tracks_normalized = tracks_np / np.array([width, height])
    
    # Calculate cumulative displacement (from first frame)
    cumulative_displacements = np.zeros_like(tracks_np)
    for i in range(1, num_frames):
        if visibility_np[i] > 0.5:
            cumulative_displacements[i] = tracks_np[i] - tracks_np[0]
        else:
            # If current frame has low visibility, use previous cumulative displacement
            cumulative_displacements[i] = cumulative_displacements[i-1]
    
    # Prepare data for HDF5 storage
    if output_to_custom_path and custom_output_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(custom_output_path), exist_ok=True)
        # Use custom filename
        hdf5_path = os.path.join(custom_output_path, f"{video_name}_2D.hdf5")
    else:
        hdf5_path = os.path.join(video_output_dir, f"{video_name}_gripper_tracking.hdf5")
    
    with h5py.File(hdf5_path, 'w') as f:
        # Create metadata group
        metadata = f.create_group('metadata')
        metadata.attrs['filename'] = video_path
        metadata.attrs['width'] = width
        metadata.attrs['height'] = height
        metadata.attrs['num_frames'] = num_frames
        metadata.attrs['gripper_x'] = gripper_x
        metadata.attrs['gripper_y'] = gripper_y
        
        # Create tracking group
        tracking = f.create_group('tracking')
        
        # Store raw tracks
        tracking.create_dataset('positions', data=tracks_np, compression='gzip')
        tracking.create_dataset('visibility', data=visibility_np, compression='gzip')
        
        # Store normalized coordinates
        tracking.create_dataset('positions_normalized', data=tracks_normalized, compression='gzip')
        
        # Store frame-to-frame displacements (key data)
        tracking.create_dataset('displacements', data=displacements, compression='gzip')
        
        # Store cumulative displacements
        tracking.create_dataset('cumulative_displacements', data=cumulative_displacements, compression='gzip')
        
        # Add point_tracking_results compatible format (similar to compute_flow_features)
        flow_features = f.create_group('flow_features')
        
        # Create a mask (all ones since we're tracking a single point reliably)
        points_mask = np.ones_like(visibility_np)
        
        # Store in the format used by compute_flow_features
        flow_features.create_dataset('points', data=tracks_np.reshape(1, num_frames, 2), compression='gzip')
        flow_features.create_dataset('points_visibility', data=visibility_np.reshape(1, num_frames), compression='gzip')
        flow_features.create_dataset('points_mask', data=points_mask.reshape(1, num_frames), compression='gzip')
        flow_features.create_dataset('points_normalized', data=tracks_normalized.reshape(1, num_frames, 2), compression='gzip')
        
        # Add frame-to-frame displacements in compute_flow_features format
        flow_features.create_dataset('points_displacements', data=displacements.reshape(1, num_frames, 2), compression='gzip')
    
    print(f"Tracking data saved to: {hdf5_path}")
    
    # Visualize the tracking results if not disabled
    if not disable_visualization:
        # 1. Create MP4 visualization using CoTracker visualizer
        vis = Visualizer(save_dir=video_output_dir, pad_value=120, linewidth=3)
        vis.visualize(
            video=video_tensor,
            tracks=tracks_fwd,
            visibility=visibility_fwd,
            filename=f"{video_name}_tracking",
            save_video=True,
            opacity=1.0
        )
        
        # 2. Create displacement plot
        plt.figure(figsize=(12, 8))
        frames = np.arange(num_frames)
        
        # Plot X displacements
        plt.subplot(2, 1, 1)
        plt.plot(frames, displacements[:, 0], 'r-', label='X Displacement')
        plt.title('Gripper X Displacement Between Frames')
        plt.xlabel('Frame Number')
        plt.ylabel('X Displacement (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Plot Y displacements
        plt.subplot(2, 1, 2)
        plt.plot(frames, displacements[:, 1], 'b-', label='Y Displacement')
        plt.title('Gripper Y Displacement Between Frames')
        plt.xlabel('Frame Number')
        plt.ylabel('Y Displacement (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(video_output_dir, f"{video_name}_displacements.png"))
        plt.close()
        
        # 3. Create trajectory plot
        plt.figure(figsize=(10, 10))
        
        # Plot trajectory
        visible_mask = visibility_np > 0.5
        plt.scatter(tracks_np[0, 0], tracks_np[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(tracks_np[-1, 0], tracks_np[-1, 1], c='red', s=100, marker='s', label='End')
        plt.plot(tracks_np[visible_mask, 0], tracks_np[visible_mask, 1], 'b-', alpha=0.7, label='Trajectory')
        
        # Add arrows to show direction
        arrow_frames = list(range(0, num_frames, max(1, num_frames // 20)))
        for i in arrow_frames:
            if i > 0 and visibility_np[i] > 0.5 and visibility_np[i-1] > 0.5:
                dx = tracks_np[i, 0] - tracks_np[i-1, 0]
                dy = tracks_np[i, 1] - tracks_np[i-1, 1]
                
                # Only draw arrow if there's significant movement
                if dx**2 + dy**2 > 1.0:
                    plt.arrow(tracks_np[i-1, 0], tracks_np[i-1, 1], dx, dy, 
                             head_width=3, head_length=5, fc='red', ec='red', alpha=0.7)
        
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Reversed y-axis for image coordinates
        plt.title('Gripper Trajectory')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(video_output_dir, f"{video_name}_trajectory.png"))
        plt.close()
        
        print(f"Visualizations saved to: {video_output_dir}")
    else:
        print("Visualizations disabled.")
    
    # Return both the HDF5 path and the coordinates array
    return hdf5_path, tracks_np

def process_folder(
    input_dir,
    output_dir,
    gripper_point=(64, 64),
    device=None,
    disable_visualization=False,
    output_to_custom_path=False,
    custom_output_path=None
):
    """
    Process all MP4 files in a directory through track_gripper.
    
    Args:
        input_dir: Directory containing MP4 files to process
        output_dir: Base directory to save results
        gripper_point: (x, y) coordinates of the gripper in the first frame
        device: Device to run inference on ('cuda' or 'cpu')
        disable_visualization: If True, skips generating visualizations
        output_to_custom_path: If True, saves HDF5 files to custom_output_path
        custom_output_path: Custom path to save HDF5 files (if output_to_custom_path is True)
    
    Returns:
        Dictionary mapping video filenames to tuples of:
        - Path to the saved HDF5 file
        - NumPy array of shape [num_frames, 2] containing (x,y) coordinates for each frame
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if output_to_custom_path and custom_output_path:
        os.makedirs(custom_output_path, exist_ok=True)
    
    # Get all MP4 files in the input directory
    mp4_files = glob.glob(os.path.join(input_dir, "*.mp4"))
    
    if not mp4_files:
        print(f"No MP4 files found in {input_dir}")
        return {}
    
    print(f"Found {len(mp4_files)} MP4 files to process")
    
    # Process each file
    results = {}
    for mp4_file in tqdm(mp4_files, desc="Processing videos"):
        try:
            # Extract the video name
            video_name = os.path.splitext(os.path.basename(mp4_file))[0]
            print(f"\nProcessing {video_name}...")
            
            # Run gripper tracking
            result_file, tracks_np = track_gripper(
                video_path=mp4_file,
                output_dir=output_dir,
                gripper_point=gripper_point,
                device=device,
                disable_visualization=disable_visualization,
                output_to_custom_path=output_to_custom_path,
                custom_output_path=custom_output_path
            )
            
            # Store both the HDF5 path and the coordinates array
            results[video_name] = (result_file, tracks_np)
            print(f"Completed processing {video_name}")
            
        except Exception as e:
            print(f"Error processing {mp4_file}: {str(e)}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Track a gripper point through video(s) and record displacements')
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', help='Path to a single video file')
    group.add_argument('--input_dir', help='Directory containing multiple MP4 files to process')
    
    # Output options
    parser.add_argument('--output', default='gripper_tracking_results', 
                        help='Output directory for tracking results')
    parser.add_argument('--output_to_custom_path', action='store_true', 
                        help='Save HDF5 files to a custom path instead of output subdirectories')
    parser.add_argument('--custom_output_path', 
                        help='Custom path to save HDF5 files when using --output_to_custom_path')
    
    # Gripper point coordinates
    parser.add_argument('--x', type=float, default=64, 
                        help='X coordinate of gripper point in first frame')
    parser.add_argument('--y', type=float, default=64, 
                        help='Y coordinate of gripper point in first frame')
    
    # Processing options
    parser.add_argument('--device', choices=['cuda', 'cpu'], 
                        help='Device to run inference on')
    parser.add_argument('--disable_visualization', action='store_true', 
                        help='Disable rendering of visualizations for faster processing')
    
    args = parser.parse_args()
    
    # Validation
    if args.output_to_custom_path and not args.custom_output_path:
        print("Error: --custom_output_path must be specified when using --output_to_custom_path")
        return 1
    
    # Process single file or directory
    if args.input:
        if not os.path.isfile(args.input):
            print(f"Error: Input file '{args.input}' not found")
            return 1
        
        # Run gripper tracking on single file
        hdf5_path, coordinates = track_gripper(
            video_path=args.input,
            output_dir=args.output,
            gripper_point=(args.x, args.y),
            device=args.device,
            disable_visualization=args.disable_visualization,
            output_to_custom_path=args.output_to_custom_path,
            custom_output_path=args.custom_output_path
        )
        print(f"Saved tracking data to: {hdf5_path}")
        print(f"Tracked coordinates shape: {coordinates.shape}")
    else:
        if not os.path.isdir(args.input_dir):
            print(f"Error: Input directory '{args.input_dir}' not found")
            return 1
        
        # Process all files in directory
        results = process_folder(
            input_dir=args.input_dir,
            output_dir=args.output,
            gripper_point=(args.x, args.y),
            device=args.device,
            disable_visualization=args.disable_visualization,
            output_to_custom_path=args.output_to_custom_path,
            custom_output_path=args.custom_output_path
        )
        
        print(f"Processed {len(results)} videos successfully")
        for video_name, (hdf5_path, coordinates) in results.items():
            print(f"Video: {video_name}, Coordinates shape: {coordinates.shape}")
    
    print("Processing complete!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 