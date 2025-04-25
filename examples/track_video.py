#!/usr/bin/env python3
"""
Example script showing how to use track_gripper from another project.

Usage:
    python track_video.py --input /path/to/video.mp4 --x 104 --y 24
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from optical_flow.flow_compute_utils import track_gripper

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Track a point in video and display coordinates')
    parser.add_argument('--input', required=True, help='Path to input video file')
    parser.add_argument('--output', default='tracking_output', help='Output directory')
    parser.add_argument('--x', type=float, default=104, help='X coordinate of point to track')
    parser.add_argument('--y', type=float, default=24, help='Y coordinate of point to track')
    parser.add_argument('--disable_visualization', action='store_true', 
                      help='Disable visualization generation')
    args = parser.parse_args()
    
    print(f"Tracking point ({args.x}, {args.y}) in video: {args.input}")
    
    # Run the tracking function
    hdf5_path, coordinates = track_gripper(
        video_path=args.input,
        output_dir=args.output,
        gripper_point=(args.x, args.y),
        disable_visualization=args.disable_visualization
    )
    
    # Print information about the tracking results
    print(f"\nTracking Results:")
    print(f"Coordinates shape: {coordinates.shape}")
    print(f"First 5 coordinates: {coordinates[:5]}")
    print(f"Last 5 coordinates: {coordinates[-5:]}")
    
    # Calculate total displacement
    total_displacement = np.linalg.norm(coordinates[-1] - coordinates[0])
    print(f"\nTotal displacement: {total_displacement:.2f} pixels")
    
    # Plot the trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(coordinates[:, 0], coordinates[:, 1], 'b-', linewidth=1)
    plt.scatter(coordinates[0, 0], coordinates[0, 1], c='green', s=100, marker='o', label='Start')
    plt.scatter(coordinates[-1, 0], coordinates[-1, 1], c='red', s=100, marker='s', label='End')
    
    # Add arrows to show direction every N frames
    N = max(1, len(coordinates) // 20)
    for i in range(0, len(coordinates) - 1, N):
        dx = coordinates[i+1, 0] - coordinates[i, 0]
        dy = coordinates[i+1, 1] - coordinates[i, 1]
        
        # Only draw arrow if there's significant movement
        if dx**2 + dy**2 > 1.0:
            plt.arrow(coordinates[i, 0], coordinates[i, 1], dx, dy, 
                     head_width=3, head_length=5, fc='red', ec='red', alpha=0.7)
    
    plt.title('Tracked Point Trajectory')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save the plot
    plt.savefig(f"{args.output}/trajectory_plot.png")
    plt.close()
    
    print(f"\nResults saved to: {args.output}")
    print(f"HDF5 file: {hdf5_path}")
    print(f"Trajectory plot: {args.output}/trajectory_plot.png")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 