#!/usr/bin/env python3
"""
Trajectory Visualization Script

Visualizes the trajectories from position difference data:
1. Loads NPZ files containing position differences
2. Reconstructs original trajectories by cumulative summing
3. Creates 2D plots and GIFs showing movement for each demo
4. Uploads visualizations to Weights & Biases
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import wandb
import glob
import argparse
from tqdm import tqdm
import tempfile
from PIL import Image


def load_position_diff_data(npz_file):
    """
    Load position difference data from an NPZ file.
    
    Args:
        npz_file: Path to NPZ file
        
    Returns:
        Dictionary containing position differences for each demo
    """
    try:
        data = np.load(npz_file)
        return {key: data[key] for key in data.keys()}
    except Exception as e:
        print(f"Error loading {npz_file}: {str(e)}")
        return {}


def reconstruct_trajectory(position_diffs):
    """
    Reconstruct the original trajectory from position differences.
    
    Args:
        position_diffs: NumPy array of shape [num_frames, 2] with (x,y) differences
        
    Returns:
        NumPy array of shape [num_frames, 2] with (x,y) positions
    """
    # Start with zero position
    positions = np.zeros_like(position_diffs)
    
    # Cumulatively sum the differences to get positions
    for i in range(1, len(position_diffs)):
        positions[i] = positions[i-1] + position_diffs[i]
    
    return positions


def create_trajectory_gif(trajectory, output_path, demo_key, task_name, fps=10, tail_length=30):
    """
    Create a GIF showing the progression of a trajectory.
    
    Args:
        trajectory: NumPy array of shape [num_frames, 2] with (x,y) positions
        output_path: Path to save the GIF
        demo_key: Demo identifier
        task_name: Task name
        fps: Frames per second for the GIF
        tail_length: Number of previous positions to show as a tail
        
    Returns:
        Path to the saved GIF
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine axis limits with some padding
    all_x = trajectory[:, 0]
    all_y = trajectory[:, 1]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    # Add 10% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Set up plot elements
    point, = ax.plot([], [], 'ro', markersize=10)  # Current position
    tail, = ax.plot([], [], 'r-', alpha=0.7)       # Tail showing recent path
    path, = ax.plot([], [], 'b-', alpha=0.3)       # Full path so far
    
    # Add title and labels
    ax.set_title(f"Task: {task_name}, {demo_key.capitalize()} - Trajectory")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(True, alpha=0.3)
    
    # Mark start and end points
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, marker='o', label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, marker='x', label='End')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Initialization function
    def init():
        point.set_data([], [])
        tail.set_data([], [])
        path.set_data([], [])
        return point, tail, path
    
    # Animation function
    def animate(i):
        # Current position
        x = trajectory[i, 0]
        y = trajectory[i, 1]
        point.set_data([x], [y])
        
        # Tail (last N positions)
        start_idx = max(0, i - tail_length)
        tail_x = trajectory[start_idx:i+1, 0]
        tail_y = trajectory[start_idx:i+1, 1]
        tail.set_data(tail_x, tail_y)
        
        # Full path so far
        path_x = trajectory[:i+1, 0]
        path_y = trajectory[:i+1, 1]
        path.set_data(path_x, path_y)
        
        # Add a percentage indicator
        ax.set_title(f"Task: {task_name}, {demo_key.capitalize()} - Frame {i}/{len(trajectory)-1} ({i/(len(trajectory)-1)*100:.1f}%)")
        
        return point, tail, path
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=len(trajectory),
                                 init_func=init, blit=True, repeat=True,
                                 interval=1000/fps)
    
    # Save as GIF
    temp_dir = tempfile.mkdtemp()
    temp_frames = []
    
    # Render each frame to a separate file
    for i in range(len(trajectory)):
        # Update the animation
        animate(i)
        
        # Save the frame
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_path)
        temp_frames.append(frame_path)
    
    # Create the GIF from the frames
    frames = [Image.open(frame) for frame in temp_frames]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=int(1000/fps),
        loop=0
    )
    
    # Clean up
    plt.close(fig)
    for frame in temp_frames:
        os.remove(frame)
    
    print(f"Created GIF: {output_path}")
    return output_path


def visualize_task(npz_file, output_dir=None, upload_to_wandb=True, project_name="trajectory-visualization"):
    """
    Visualize trajectories for all demos in a task.
    
    Args:
        npz_file: Path to NPZ file
        output_dir: Directory to save plots (if None, uses the same directory as the NPZ file)
        upload_to_wandb: Whether to upload plots to wandb
        project_name: Name of the wandb project to log to
        
    Returns:
        List of paths to saved plot files and GIFs
    """
    # Get task name from filename
    task_name = os.path.splitext(os.path.basename(npz_file))[0].replace('_diff', '')
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(npz_file)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load position difference data
    diff_data = load_position_diff_data(npz_file)
    if not diff_data:
        print(f"No data found in {npz_file}")
        return []
    
    print(f"Visualizing task: {task_name} ({len(diff_data)} demos)")
    
    # Prepare plot for all demos
    plt.figure(figsize=(12, 10))
    
    # Sort demo keys by demo number
    demo_keys = sorted(diff_data.keys(), key=lambda x: int(x.split('_')[-1]))
    
    # Reconstruct and plot each demo
    output_files = []
    colors = plt.cm.tab10.colors  # Use tab10 color cycle
    
    # First create an aggregate plot with all demos
    for i, demo_key in enumerate(demo_keys):
        # Reconstruct trajectory
        trajectory = reconstruct_trajectory(diff_data[demo_key])
        
        # Plot trajectory
        color = colors[i % len(colors)]
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, 
                 label=f"{demo_key} ({len(trajectory)} frames)", linewidth=2, alpha=0.7)
        
        # Mark start and end points
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color=color, s=100, marker='o')
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color=color, s=100, marker='x')
    
    # Add labels and legend
    plt.title(f"All Trajectories for Task: {task_name}")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Save figure
    all_demos_file = os.path.join(output_dir, f"{task_name}_all_demos.png")
    plt.tight_layout()
    plt.savefig(all_demos_file, dpi=150)
    output_files.append(all_demos_file)
    plt.close()
    
    # Then create individual plots and GIFs for each demo
    for i, demo_key in enumerate(demo_keys):
        # Reconstruct trajectory
        trajectory = reconstruct_trajectory(diff_data[demo_key])
        
        # Create new figure for this demo
        plt.figure(figsize=(10, 8))
        
        # Plot trajectory
        color = colors[i % len(colors)]
        
        # Create a colormap to show progression through time
        points = plt.scatter(trajectory[:, 0], trajectory[:, 1], c=np.arange(len(trajectory)), 
                            cmap='viridis', s=10, zorder=2)
        plt.colorbar(points, label='Frame Number')
        
        # Plot the line connecting points
        plt.plot(trajectory[:, 0], trajectory[:, 1], '-', color=color, 
                alpha=0.5, linewidth=1, zorder=1)
        
        # Mark start and end points
        plt.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, marker='o', label='Start', zorder=3)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=100, marker='x', label='End', zorder=3)
        
        # Add trajectory segments with arrows to show direction
        step = max(1, len(trajectory) // 20)  # Add arrows at 20 points
        for j in range(step, len(trajectory), step):
            plt.annotate('', 
                        xy=(trajectory[j, 0], trajectory[j, 1]),
                        xytext=(trajectory[j-step, 0], trajectory[j-step, 1]),
                        arrowprops=dict(facecolor=color, edgecolor=color,
                                        width=1, headwidth=8, alpha=0.7),
                        zorder=4)
        
        # Add labels and legend
        plt.title(f"Task: {task_name}, {demo_key.capitalize()}")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Save figure
        demo_file = os.path.join(output_dir, f"{task_name}_{demo_key}.png")
        plt.tight_layout()
        plt.savefig(demo_file, dpi=150)
        output_files.append(demo_file)
        plt.close()
        
        # Create a GIF for this demo
        gif_path = os.path.join(output_dir, f"{task_name}_{demo_key}.gif")
        gif_file = create_trajectory_gif(trajectory, gif_path, demo_key, task_name)
        output_files.append(gif_file)
    
    # Upload to wandb if requested
    if upload_to_wandb:
        wandb_log_plots(task_name, output_files, demo_keys, diff_data, project_name)
    
    return output_files


def wandb_log_plots(task_name, plot_files, demo_keys, diff_data, project_name="trajectory-visualization"):
    """
    Log plots to wandb.
    
    Args:
        task_name: Task name
        plot_files: List of plot file paths
        demo_keys: List of demo keys
        diff_data: Dictionary containing position differences
        project_name: Name of the wandb project to log to
    """
    # Initialize wandb
    wandb.init(project=project_name, name=f"task-{task_name}", 
              config={"task_name": task_name, "num_demos": len(demo_keys)})
    
    # Log static plots
    for plot_file in plot_files:
        plot_name = os.path.splitext(os.path.basename(plot_file))[0]
        file_ext = os.path.splitext(plot_file)[1].lower()
        
        if file_ext == '.png':
            wandb.log({f"static/{plot_name}": wandb.Image(plot_file)})
        elif file_ext == '.gif':
            wandb.log({f"animations/{plot_name}": wandb.Video(plot_file, fps=10, format="gif")})
    
    # Create a unified, interactive 2D scatter plot for wandb
    data = []
    for i, demo_key in enumerate(demo_keys):
        trajectory = reconstruct_trajectory(diff_data[demo_key])
        for j, (x, y) in enumerate(trajectory):
            data.append([float(x), float(y), i, j, demo_key])
    
    # Convert to table
    columns = ["x", "y", "demo_idx", "frame", "demo_id"]
    wandb_table = wandb.Table(data=data, columns=columns)
    
    # Log interactive scatter plot
    wandb.log({"interactive/trajectory_plot": wandb.plot.scatter(
        wandb_table, "x", "y", 
        title=f"Interactive Trajectory Plot for {task_name}")
    })
    
    # Finish wandb run
    wandb.finish()


def process_directory(input_dir, output_dir=None, upload_to_wandb=True, project_name="trajectory-visualization"):
    """
    Process all NPZ files in a directory.
    
    Args:
        input_dir: Directory containing NPZ files
        output_dir: Directory to save plots (if None, uses input_dir)
        upload_to_wandb: Whether to upload plots to wandb
        project_name: Name of the wandb project to log to
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all NPZ files in the directory
    npz_files = glob.glob(os.path.join(input_dir, "*_diff.npz"))
    
    if not npz_files:
        print(f"No NPZ files found in {input_dir}")
        return
    
    print(f"Found {len(npz_files)} NPZ files in {input_dir}")
    
    # Process each file
    for npz_file in tqdm(npz_files, desc="Visualizing tasks"):
        visualize_task(npz_file, output_dir, upload_to_wandb, project_name)


def main():
    """
    Main function to process command-line arguments and visualize trajectories.
    """
    parser = argparse.ArgumentParser(description='Visualize trajectories from position difference data')
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', help='Path to a single NPZ file')
    group.add_argument('--input_dir', help='Directory containing NPZ files')
    
    # Output options
    parser.add_argument('--output_dir', help='Directory to save plots')
    
    # Wandb options
    parser.add_argument('--no_wandb', action='store_true', 
                        help='Disable uploading to wandb')
    parser.add_argument('--project', default="trajectory-visualization",
                        help='Wandb project name')
    
    args = parser.parse_args()
    
    # Process single file or directory
    if args.input:
        visualize_task(args.input, args.output_dir, not args.no_wandb, args.project)
    else:
        process_directory(args.input_dir, args.output_dir, not args.no_wandb, args.project)
    
    print("Visualization complete!")


if __name__ == "__main__":
    main() 