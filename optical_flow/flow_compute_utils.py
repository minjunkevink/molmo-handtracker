#!/usr/bin/env python3
import torch
import imageio
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import glob
from tqdm import tqdm
from cotracker.utils.visualizer import Visualizer


def process_video(video_path, output_dir, point1=None, point2=None):
    """
    Process a single video with bidirectional tracking.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save the tracking results
        point1 (tuple, optional): (x, y) coordinates for the first tracking point
        point2 (tuple, optional): (x, y) coordinates for the second tracking point
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the CoTracker model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    
    # Load video
    print(f"Loading video: {video_path}")
    reader = imageio.get_reader(video_path)
    frames = [np.array(im) for im in reader]
    
    # Create a video-specific subdirectory for results
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    
    # Check frame shape
    print(f"Frame shape: {frames[0].shape}")
    height, width = frames[0].shape[0:2]
    
    # Define tracking points
    # If no points are provided, use default points or ask for user input
    if point1 is None:
        point1_x, point1_y = 55.0, 22.0
    else:
        point1_x, point1_y = point1
        
    if point2 is None:
        point2_x, point2_y = 70.0, 22.0
    else:
        point2_x, point2_y = point2
    
    initial_frame = 0
    initial_points = torch.tensor([[[initial_frame, point1_x, point1_y], 
                                   [initial_frame, point2_x, point2_y]]], dtype=torch.float32).to(device)
    print(f"Initial points shape: {initial_points.shape}, values: {initial_points.cpu().numpy()}")
    
    # Visualize the initial points on the first frame
    plt.figure(figsize=(10, 10))
    plt.imshow(frames[0])
    
    # Plot point 1
    plt.scatter(point1_x, point1_y, c='red', s=100, marker='x')
    plt.text(point1_x+10, point1_y+10, f"Point 1 ({point1_x}, {point1_y})", 
             color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
    
    # Plot point 2
    plt.scatter(point2_x, point2_y, c='blue', s=100, marker='x')
    plt.text(point2_x+10, point2_y+10, f"Point 2 ({point2_x}, {point2_y})", 
             color='white', fontsize=12, bbox=dict(facecolor='blue', alpha=0.5))
    
    # Add zoomed insets for both points
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
    
    # Zoomed inset for point 1
    axins1 = zoomed_inset_axes(plt.gca(), zoom=4, loc=2)  # Upper left
    axins1.imshow(frames[0])
    axins1.scatter(point1_x, point1_y, c='red', s=100, marker='x')
    axins1.set_xlim(point1_x-20, point1_x+20)
    axins1.set_ylim(point1_y+20, point1_y-20)  # Reversed y-axis
    axins1.set_xticks([])
    axins1.set_yticks([])
    mark_inset(plt.gca(), axins1, loc1=1, loc2=3, fc="none", ec="red")
    
    # Zoomed inset for point 2
    axins2 = zoomed_inset_axes(plt.gca(), zoom=4, loc=3)  # Lower left
    axins2.imshow(frames[0])
    axins2.scatter(point2_x, point2_y, c='blue', s=100, marker='x')
    axins2.set_xlim(point2_x-20, point2_x+20)
    axins2.set_ylim(point2_y+20, point2_y-20)  # Reversed y-axis
    axins2.set_xticks([])
    axins2.set_yticks([])
    mark_inset(plt.gca(), axins2, loc1=2, loc2=4, fc="none", ec="blue")
    
    plt.title("Initial Points Verification")
    plt.savefig(os.path.join(video_output_dir, "initial_points.png"))
    plt.close()
    
    # Prepare the video tensor
    frames_array = np.stack(frames)
    video = torch.tensor(frames_array).permute(0, 3, 1, 2)[None].float().to(device)
    
    print(f"Video tensor shape: {video.shape}")
    
    # Implement bidirectional tracking
    print("Running forward tracking...")
    with torch.no_grad():
        # Forward tracking
        pred_tracks_fwd, pred_visibility_fwd = cotracker(video, queries=initial_points)
        
        # Reverse the video for backward tracking
        video_reversed = torch.flip(video, dims=[1])  # Flip along the temporal dimension
        
        print("Running backward tracking...")
        last_frame_idx = video.shape[1] - 1
        
        # Get the predicted positions at the last frame from forward tracking
        last_frame_positions = pred_tracks_fwd[:, -1, :, :]  # Shape: [B, N, 2]
        
        # Create new query points for backward tracking starting from the last frame
        backward_initial_points = torch.cat([
            torch.zeros_like(last_frame_positions[:, :, :1]) + initial_frame,  # t=0 for first frame in reversed video
            last_frame_positions
        ], dim=2)
        
        # Run backward tracking
        pred_tracks_bwd, pred_visibility_bwd = cotracker(video_reversed, queries=backward_initial_points)
        
        # Flip the backward tracks to align with forward video timeline
        pred_tracks_bwd = torch.flip(pred_tracks_bwd, dims=[1])
        pred_visibility_bwd = torch.flip(pred_visibility_bwd, dims=[1])
    
    # Combine forward and backward tracks with different strategies
    print("Combining forward and backward tracks...")
    
    # 1. Average combination - simple but effective
    combined_tracks_avg = (pred_tracks_fwd + pred_tracks_bwd) / 2.0
    
    # Use maximum of visibilities - need to ensure tensors for both arguments
    combined_visibility_avg = torch.maximum(pred_visibility_fwd, pred_visibility_bwd)  # Use maximum visibility
    
    # 2. Weighted combination based on distance to initial frame
    # Points closer to the initial frame get higher weight from forward tracking
    T = video.shape[1]
    frame_indices = torch.arange(T, device=device).view(1, T, 1, 1).float()
    weights_fwd = 1.0 - frame_indices / (T - 1)  # From 1.0 at first frame to 0.0 at last frame
    weights_bwd = frame_indices / (T - 1)        # From 0.0 at first frame to 1.0 at last frame
    
    # Apply weights to get weighted combination
    combined_tracks_weighted = (weights_fwd * pred_tracks_fwd + weights_bwd * pred_tracks_bwd) / (weights_fwd + weights_bwd)
    
    # For visibility, create a tensor with the minimum value we want to maintain
    min_visibility = torch.ones_like(pred_visibility_fwd) * 0.3
    
    # For visibility, we use a more conservative approach with tensor operations
    combined_visibility_weighted = torch.minimum(
        torch.maximum(pred_visibility_fwd, min_visibility),  # Ensure some minimum visibility
        torch.maximum(pred_visibility_bwd, min_visibility)   # Ensure some minimum visibility
    )
    
    # Let's use the weighted combination for visualization
    final_tracks = combined_tracks_weighted
    final_visibility = combined_visibility_weighted
    
    print(f"Forward tracks shape: {pred_tracks_fwd.shape}")
    print(f"Backward tracks shape: {pred_tracks_bwd.shape}")
    print(f"Combined tracks shape: {final_tracks.shape}")
    
    # Save the tracking data for later use
    torch.save({
        'forward_tracks': pred_tracks_fwd.cpu(),
        'forward_visibility': pred_visibility_fwd.cpu(),
        'backward_tracks': pred_tracks_bwd.cpu(),
        'backward_visibility': pred_visibility_bwd.cpu(),
        'combined_tracks': final_tracks.cpu(),
        'combined_visibility': final_visibility.cpu(),
        'initial_points': initial_points.cpu(),
        'video_shape': video.shape
    }, os.path.join(video_output_dir, f"{video_name}_tracking_data.pth"))
    
    # Visualize the tracking results - forward, backward, and combined
    # Visualize forward tracking
    vis_fwd = Visualizer(save_dir=video_output_dir, pad_value=120, linewidth=3)
    vis_fwd.visualize(
        video=video, 
        tracks=pred_tracks_fwd, 
        visibility=pred_visibility_fwd, 
        filename=f"{video_name}_forward",
        save_video=True,
        opacity=1.0
    )
    
    # Visualize backward tracking
    vis_bwd = Visualizer(save_dir=video_output_dir, pad_value=120, linewidth=3)
    vis_bwd.visualize(
        video=video, 
        tracks=pred_tracks_bwd, 
        visibility=pred_visibility_bwd, 
        filename=f"{video_name}_backward",
        save_video=True,
        opacity=1.0
    )
    
    # Visualize combined tracking
    vis_combined = Visualizer(save_dir=video_output_dir, pad_value=120, linewidth=3)
    vis_combined.visualize(
        video=video, 
        tracks=final_tracks, 
        visibility=final_visibility, 
        filename=f"{video_name}_bidirectional",
        save_video=True,
        opacity=1.0
    )
    
    print(f"Tracking visualizations saved to {video_output_dir}/:")
    print(f" - Forward tracking: {video_name}_forward.mp4")
    print(f" - Backward tracking: {video_name}_backward.mp4")
    print(f" - Bidirectional (combined) tracking: {video_name}_bidirectional.mp4")
    
    # Return tracking results for future use if needed
    return {
        'forward_tracks': pred_tracks_fwd,
        'forward_visibility': pred_visibility_fwd,
        'backward_tracks': pred_tracks_bwd,
        'backward_visibility': pred_visibility_bwd,
        'combined_tracks': final_tracks,
        'combined_visibility': final_visibility
    }


def main():
    parser = argparse.ArgumentParser(description='Compute bidirectional optical flow on videos using CoTracker')
    parser.add_argument('--input', required=True, help='Path to a video file or directory containing videos')
    parser.add_argument('--output', default='tracking_results', help='Directory to save tracking results')
    parser.add_argument('--point1x', type=float, default=55.0, help='X coordinate of the first tracking point')
    parser.add_argument('--point1y', type=float, default=22.0, help='Y coordinate of the first tracking point')
    parser.add_argument('--point2x', type=float, default=70.0, help='X coordinate of the second tracking point')
    parser.add_argument('--point2y', type=float, default=22.0, help='Y coordinate of the second tracking point')
    args = parser.parse_args()
    
    # Define tracking points from arguments
    point1 = (args.point1x, args.point1y)
    point2 = (args.point2x, args.point2y)
    
    # Determine if input is a file or directory
    if os.path.isfile(args.input):
        if args.input.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            print(f"Processing single video: {args.input}")
            process_video(args.input, args.output, point1, point2)
        else:
            print(f"Error: {args.input} is not a supported video file format.")
            print("Supported formats: .mp4, .avi, .mov, .mkv")
            return
    elif os.path.isdir(args.input):
        print(f"Processing all videos in directory: {args.input}")
        # Find all video files in the directory
        video_files = []
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            video_files.extend(glob.glob(os.path.join(args.input, f'*{ext}')))
        
        if not video_files:
            print(f"No video files found in {args.input}")
            return
        
        print(f"Found {len(video_files)} videos to process:")
        for video in video_files:
            print(f" - {os.path.basename(video)}")
            
        # Process each video
        for video_path in tqdm(video_files, desc="Processing videos"):
            try:
                process_video(video_path, args.output, point1, point2)
            except Exception as e:
                print(f"Error processing {video_path}: {str(e)}")
    else:
        print(f"Error: {args.input} is not a valid file or directory.")
        return
    
    print("All processing complete!")


if __name__ == "__main__":
    main()