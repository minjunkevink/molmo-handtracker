#!/usr/bin/env python3
import h5py
import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
import cv2

def analyze_tracking(hdf5_path, output_dir=None, show_plots=False):
    """
    Analyze and visualize gripper tracking data from an HDF5 file.
    
    Args:
        hdf5_path: Path to the HDF5 file
        output_dir: Directory to save visualizations (default: same as HDF5 file directory)
        show_plots: Whether to display plots interactively
    """
    # Set output directory
    if output_dir is None:
        output_dir = str(Path(hdf5_path).parent)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # Open the HDF5 file
    with h5py.File(hdf5_path, 'r') as f:
        # Get metadata
        metadata = f['metadata']
        print("=== Metadata ===")
        for key in metadata.attrs:
            print(f"{key}: {metadata.attrs[key]}")
        
        # Get tracking data
        tracking = f['tracking']
        positions = tracking['positions'][:]
        visibility = tracking['visibility'][:]
        displacements = tracking['displacements'][:]
        cumulative_displacements = tracking['cumulative_displacements'][:]
        
        print("\n=== Tracking Data Shapes ===")
        for dataset_name in tracking.keys():
            print(f"{dataset_name} shape: {tracking[dataset_name].shape}")
        
        # Get video dimensions and number of frames
        width = metadata.attrs['width']
        height = metadata.attrs['height']
        num_frames = metadata.attrs['num_frames']
        
        # Calculate statistics
        valid_mask = visibility > 0.5
        valid_displacements = displacements[valid_mask]
        
        print("\n=== Displacement Statistics ===")
        if len(valid_displacements) > 0:
            avg_displacement = np.mean(np.sqrt(np.sum(valid_displacements**2, axis=1)))
            max_displacement = np.max(np.sqrt(np.sum(valid_displacements**2, axis=1)))
            total_distance = np.sum(np.sqrt(np.sum(valid_displacements**2, axis=1)))
            
            print(f"Average displacement magnitude: {avg_displacement:.2f} pixels per frame")
            print(f"Maximum displacement magnitude: {max_displacement:.2f} pixels")
            print(f"Total distance traveled: {total_distance:.2f} pixels")
            
            # Get final displacement (straight-line distance from start to end)
            if valid_mask[-1]:
                final_displacement = np.sqrt(np.sum(cumulative_displacements[-1]**2))
                print(f"Final displacement (straight-line): {final_displacement:.2f} pixels")
                
                # Calculate path efficiency (ratio of straight-line distance to total path length)
                path_efficiency = final_displacement / total_distance if total_distance > 0 else 0
                print(f"Path efficiency: {path_efficiency:.2%}")
        else:
            print("No valid displacements found")
        
        # Calculate visibility statistics
        print(f"\nAverage visibility score: {np.mean(visibility):.2%}")
        print(f"Frames with high visibility (>0.5): {np.sum(visibility > 0.5)} out of {num_frames} ({np.mean(visibility > 0.5):.2%})")
        
        # Plot 1: Trajectory
        plt.figure(figsize=(10, 10))
        plt.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start')
        plt.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='s', label='End')
        
        # Color trajectory by visibility
        plt.scatter(positions[:, 0], positions[:, 1], c=visibility, cmap='viridis', 
                   s=10, alpha=0.7, label='Trajectory')
        plt.colorbar(label='Visibility')
        
        # Draw trajectory as line
        plt.plot(positions[valid_mask, 0], positions[valid_mask, 1], 'b-', alpha=0.3)
        
        # Add directional arrows
        arrow_frames = list(range(0, num_frames, max(1, num_frames // 15)))
        for i in arrow_frames:
            if i > 0 and visibility[i] > 0.5 and visibility[i-1] > 0.5:
                dx = positions[i, 0] - positions[i-1, 0]
                dy = positions[i, 1] - positions[i-1, 1]
                
                # Only draw arrow if there's significant movement
                if dx**2 + dy**2 > 1.0:
                    plt.arrow(positions[i-1, 0], positions[i-1, 1], dx, dy, 
                             head_width=3, head_length=5, fc='red', ec='red', alpha=0.5)
        
        plt.xlim(0, width)
        plt.ylim(height, 0)  # Reversed y-axis for image coordinates
        plt.title('Gripper Trajectory with Visibility')
        plt.xlabel('X (pixels)')
        plt.ylabel('Y (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        trajectory_path = str(Path(output_dir) / "trajectory_with_visibility.png")
        plt.savefig(trajectory_path)
        print(f"\nTrajectory plot saved to: {trajectory_path}")
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot 2: Displacement Over Time
        plt.figure(figsize=(14, 8))
        
        frames = np.arange(num_frames)
        displacement_magnitude = np.sqrt(np.sum(displacements**2, axis=1))
        
        plt.subplot(2, 1, 1)
        plt.plot(frames, displacement_magnitude, 'b-', label='Magnitude')
        plt.axhline(y=avg_displacement, color='r', linestyle='--', label=f'Avg: {avg_displacement:.2f}')
        plt.fill_between(frames, displacement_magnitude, alpha=0.3, color='blue')
        plt.title('Gripper Displacement Magnitude Between Frames')
        plt.xlabel('Frame Number')
        plt.ylabel('Displacement (pixels)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(frames, visibility, 'g-', label='Visibility Score')
        plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')
        plt.fill_between(frames, visibility, alpha=0.3, color='green')
        plt.title('Tracking Visibility Score')
        plt.xlabel('Frame Number')
        plt.ylabel('Visibility Score')
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        displacements_path = str(Path(output_dir) / "displacement_analysis.png")
        plt.savefig(displacements_path)
        print(f"Displacement analysis plot saved to: {displacements_path}")
        if show_plots:
            plt.show()
        plt.close()
        
        # Plot 3: Motion Direction Analysis
        plt.figure(figsize=(10, 10))
        
        # Calculate angles of motion
        angles = np.arctan2(displacements[1:, 1], displacements[1:, 0]) * 180 / np.pi
        valid_angles = angles[valid_mask[1:]]
        
        if len(valid_angles) > 0:
            # Histogram of motion directions
            bins = np.linspace(-180, 180, 37)  # 20-degree bins
            hist, bin_edges = np.histogram(valid_angles, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Convert to polar plot
            ax = plt.subplot(111, projection='polar')
            
            # Convert angles to radians and shift by pi/2 to make 0 degrees point upward
            theta = np.radians(bin_centers)
            
            # Plot the histogram as a polar bar chart
            width = np.radians(bins[1] - bins[0])
            bars = ax.bar(theta, hist, width=width, bottom=0.0, alpha=0.7)
            
            # Set the direction of increasing theta to be clockwise
            ax.set_theta_direction(-1)
            
            # Set the 0 degrees on top
            ax.set_theta_zero_location("N")
            
            plt.title('Gripper Movement Direction Distribution')
            
            direction_path = str(Path(output_dir) / "movement_direction.png")
            plt.savefig(direction_path)
            print(f"Movement direction plot saved to: {direction_path}")
            if show_plots:
                plt.show()
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Analyze gripper tracking data from HDF5 file')
    parser.add_argument('--input', required=True, help='Path to the tracking HDF5 file')
    parser.add_argument('--output', help='Directory to save visualizations')
    parser.add_argument('--show', action='store_true', help='Display plots interactively')
    
    args = parser.parse_args()
    
    analyze_tracking(
        hdf5_path=args.input,
        output_dir=args.output,
        show_plots=args.show
    )
    
    print("\nAnalysis complete!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 