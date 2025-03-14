import os
import cv2
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import MolmoForVisionText
from PIL import Image

from clam.utils.logger import log


def get_hand_points_from_molmo(image_path, model):
    """Use MOLMO to detect hand points in an image."""
    image = Image.open(image_path)
    # Prompt MOLMO to identify hand points
    prompt = "Point to all visible parts of the hand in this image."
    outputs = model.generate_points(image=image, prompt=prompt)
    # Extract points from MOLMO output
    points = outputs.points  # This will be a numpy array of shape (N, 2)
    return points

def process_video(video_path, output_dir):
    # Load MOLMO model
    model = MolmoForVisionText.from_pretrained("allenai/Molmo-7B-D-0924")
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load CoTracker
    from clam.scripts.optical_flow.compute_2d_flow import (
        generate_point_tracks,
        load_cotracker,
    )
    cotracker = load_cotracker({"max_tracks": 50})  # Adjust parameters as needed
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames = np.array(frames)
    
    # Get first frame and detect hand points
    first_frame = frames[0]
    first_frame_path = os.path.join(output_dir, "first_frame.jpg")
    cv2.imwrite(first_frame_path, first_frame)
    initial_points = get_hand_points_from_molmo(first_frame_path, model)
    
    # Track points through video
    queries = np.concatenate([np.zeros((len(initial_points), 1)), initial_points], axis=1)
    points, visibility = generate_point_tracks(
        {"max_tracks": 50},  # Adjust parameters as needed
        cotracker,
        frames,
        queries=queries
    )
    
    # Visualize and save results
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plt.tight_layout()
    ax.axis("off")
    
    def animate(i):
        ax.clear()
        ax.axis("off")
        ax.imshow(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB))
        # Plot points at current timestep
        visible_points = points[0][i][visibility[0][i] > 0.5]
        ax.scatter(visible_points[:, 0], visible_points[:, 1], color="orange", s=30)
        # Plot flow vectors
        if i < len(frames) - 1:
            next_visible_points = points[0][i + 1][visibility[0][i + 1] > 0.5]
            current_visible_points = points[0][i][visibility[0][i] > 0.5]
            if len(next_visible_points) == len(current_visible_points):
                flow = next_visible_points - current_visible_points
                ax.quiver(
                    current_visible_points[:, 0],
                    current_visible_points[:, 1],
                    flow[:, 0],
                    flow[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=0.15,
                    color="green",
                )
        ax.set_title(f"Frame {i}")
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=100, blit=False
    )
    
    # Save animation
    writer = animation.FFMpegWriter(fps=10)
    output_path = os.path.join(output_dir, "hand_tracking.mp4")
    anim.save(output_path, writer=writer)
    plt.close()
    
    return {
        "points": points[0],
        "visibility": visibility[0],
        "output_video": output_path
    }

if __name__ == "__main__":
    video_path = "path/to/your/video.mp4"  # Replace with your video path
    output_dir = "hand_tracking_output"
    results = process_video(video_path, output_dir)
    print(f"Results saved to {output_dir}")
