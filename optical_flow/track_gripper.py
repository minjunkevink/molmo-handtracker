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

def track_gripper(
    video_path,
    output_dir,
    gripper_point=(64, 64),  # Default to center of frame if not specified
    device=None
):
    """
    Track a single gripper point through the video and record its relative movement.
    e
    Args:
        video_path: Path to video fil
        output_dir: Directory to save results
        gripper_point: (x, y) coordinates of the gripper in the first frame
        device: Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        Path to the saved HDF5 file
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
        
        # # Create query for backward tracking
        # backward_initial_point = torch.cat([
        #     torch.zeros_like(last_frame_position[:, :, :1]),
        #     last_frame_position
        # ], dim=2)
        
        # # Run backward tracking
        # tracks_bwd, visibility_bwd = cotracker(video_reversed, queries=backward_initial_point, backward_tracking=True)
        
        # # Flip backward tracks to align with forward timeline
        # tracks_bwd = torch.flip(tracks_bwd, dims=[1])
        # visibility_bwd = torch.flip(visibility_bwd, dims=[1])
    
    # Combine tracks using weighted averaging
    # print("Combining forward and backward tracks...")
    # T = video_tensor.shape[1]
    # frame_indices = torch.arange(T, device=device).view(1, T, 1, 1).float()
    
    # # Weights decay with distance from reference frame
    # weights_fwd = 1.0 - frame_indices / (T - 1)
    # weights_bwd = frame_indices / (T - 1)
    
    # # Apply weights
    # tracks_combined = (weights_fwd * tracks_fwd + weights_bwd * tracks_bwd) / (weights_fwd + weights_bwd)
    
    # For visibility, use a minimum threshold and take the maximum
    # min_visibility = 0.3
    # visibility_combined = torch.maximum(
    #     torch.minimum(visibility_fwd, visibility_bwd),
    #     torch.ones_like(visibility_fwd) * min_visibility
    # )
    
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
    hdf5_path = os.path.join('/scr/shared/datasets/LIBERO/libero_10_2D', f"{video_name}_2D.hdf5")
    
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
    
    # Visualize the tracking results
    
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
    
    return hdf5_path

def main():
    parser = argparse.ArgumentParser(description='Track a gripper point through video and record displacements')
    parser.add_argument('--input', required=True, help='Path to video file')
    parser.add_argument('--output', default='gripper_tracking_results', help='Output directory')
    parser.add_argument('--x', type=float, default=64, help='X coordinate of gripper point in first frame')
    parser.add_argument('--y', type=float, default=64, help='Y coordinate of gripper point in first frame')
    parser.add_argument('--device', choices=['cuda', 'cpu'], help='Device to run inference on')
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return 1
    
    # Run gripper tracking
    track_gripper(
        video_path=args.input,
        output_dir=args.output,
        gripper_point=(args.x, args.y),
        device=args.device
    )
    
    print("Processing complete!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 