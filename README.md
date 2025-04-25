# MOLMO CoTrack

A toolkit for tracking hand movements in videos using advanced computer vision techniques, including CoTracker for optical flow tracking and Segment Anything Model (SAM) for object segmentation.

## Features

- **Point Tracking**: Track specific points (like hand/gripper positions) through video sequences
- **Bidirectional Tracking**: Uses CoTracker to perform both forward and backward point tracking with automatic reconciliation
- **HDF5 to MP4 Conversion**: Extract RGB frames from HDF5 datasets and convert them to MP4 videos
- **Visualization Tools**: Generate visualization videos showing tracking results
- **Coordinate Export**: Extract (x,y) coordinates for every tracked frame

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/minjunkevink/molmo_cotrack.git
cd molmo_cotrack
```

### 2. Install CoTracker dependency:
```bash
pip install git+https://github.com/facebookresearch/co-tracker.git
```

### 3. Install this package in development mode:
```bash
pip install -e .
```

## Usage

### Track a single video:

```python
from optical_flow.flow_compute_utils import track_gripper

# Track a point in a video and get the tracked coordinates
file_path = "path/to/your/video.mp4"
hdf5_path, coordinates = track_gripper(
    video_path=file_path,
    output_dir="tracking_results",
    gripper_point=(104, 24)  # Specify the (x,y) coordinates to track
)

# The coordinates variable is a numpy array with shape [num_frames, 2]
# containing the (x,y) coordinates for each frame
print(f"Coordinates shape: {coordinates.shape}")
print(f"First 5 coordinates: {coordinates[:5]}")
```

### Process all videos in a directory:

```python
from optical_flow.flow_compute_utils import process_folder

# Process all MP4 files in a directory
results = process_folder(
    input_dir="path/to/videos",
    output_dir="tracking_results",
    gripper_point=(104, 24)
)

# Results is a dictionary mapping video names to tuples of (hdf5_path, coordinates)
for video_name, (hdf5_path, coordinates) in results.items():
    print(f"Video: {video_name}, Coordinates shape: {coordinates.shape}")
```

### Command Line Usage:

```bash
# Track a single video
python -m optical_flow.flow_compute_utils --input path/to/video.mp4 --x 104 --y 24

# Process all videos in a directory
python -m optical_flow.flow_compute_utils --input_dir path/to/videos --x 104 --y 24

# Run the example script
python examples/track_video.py --input path/to/video.mp4 --x 104 --y 24
```

## Requirements

- Python 3.10+
- PyTorch
- CUDA-capable GPU (recommended)
- Dependencies listed in setup.py
