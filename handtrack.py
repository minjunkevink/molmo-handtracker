import os
import torch
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import imageio.v3 as iio
from cotracker.utils.visualizer import Visualizer

def verify_video_format(video_path):
    """Verify video format and convert to MP4 if necessary."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Get video info using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # If not MP4 or wrong codec, convert it
    if not video_path.lower().endswith('.mp4'):
        print(f"Converting video to MP4 format...")
        output_path = os.path.splitext(video_path)[0] + '_converted.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        cap.release()
        return output_path
    
    cap.release()
    return video_path

def get_hand_point_from_mediapipe(image):
    """Get hand point coordinates from first frame using MediaPipe."""
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    
    # Convert PIL Image to cv2 format
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Process the image
    results = hands.process(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
    
    # Get hand landmarks
    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]  # Get first hand
        # Get wrist point (point 0) as anchor point
        h, w = image.size[1], image.size[0]
        x = hand.landmark[0].x * w
        y = hand.landmark[0].y * h
        print(f"Found hand at coordinates: x={x}, y={y}")
        return np.array([[x, y]])
    
    print("No hand detected in the frame")
    return None

def track_hand(video_path, save_dir="tracking_results"):
    """Main function to track hand in video."""
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Verify and convert video format if necessary
    video_path = verify_video_format(video_path)
    
    # Read video using imageio
    print("Reading video...")
    frames = iio.imread(video_path, plugin="FFMPEG")
    
    # Convert to tensor and prepare for model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Ensure frames are in float32 format and normalized to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    
    # Convert to tensor with shape [B, T, C, H, W]
    # frames shape is [T, H, W, C], we need [B, T, C, H, W]
    video = torch.from_numpy(frames).permute(3, 0, 1, 2)[None]  # [1, C, T, H, W]
    video = video.permute(0, 2, 1, 3, 4)  # [1, T, C, H, W]
    video = video.float().to(device)
    print(f"Video shape: {video.shape}")
    
    # Get first frame
    first_frame = Image.fromarray((frames[0] * 255).astype(np.uint8))
    
    # Get hand point from first frame using MediaPipe
    print("Getting hand point from MediaPipe...")
    hand_point = get_hand_point_from_mediapipe(first_frame)
    if hand_point is None:
        raise ValueError("Could not detect hand in the first frame")
    
    # Initialize CoTracker3 (offline mode for better accuracy)
    print("Initializing CoTracker3...")
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    
    # Track the point
    print("Tracking points...")
    queries = torch.from_numpy(hand_point).float().to(device)
    pred_tracks, pred_visibility = cotracker(
        video,
        queries=queries[None]  # Add batch dimension
    )
    
    # Visualize results
    print("Generating visualization...")
    vis = Visualizer(
        save_dir=save_dir,
        pad_value=10,
        linewidth=2,
        mode="points",
        tracks_leave_trace=-1
    )
    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename="hand_tracking.mp4"
    )
    
    return pred_tracks, pred_visibility

if __name__ == "__main__":
    video_path = "demo.mp4"  # Using demo.mp4 as the video path
    tracks, visibility = track_hand(video_path)
    print("Tracking complete! Results saved in tracking_results/hand_tracking.mp4")