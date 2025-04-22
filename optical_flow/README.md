# Bidirectional Optical Flow Tracking

This tool implements bidirectional tracking using CoTracker, combining forward and backward optical flow tracking for more robust and accurate results.

## Features

- Track multiple points through video sequences
- Bidirectional tracking (forward + backward) for improved robustness
- Process individual videos or entire directories
- Custom tracking point selection
- Generates visualizations for forward, backward, and combined tracking
- Saves tracking data for further analysis

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install torch imageio matplotlib tqdm
# CoTracker installation
pip install git+https://github.com/facebookresearch/co-tracker.git
```

## Usage

### Basic Usage

```bash
# Process a single video
python compute_flow.py --input /path/to/video.mp4 --output tracking_results

# Process all videos in a directory
python compute_flow.py --input /path/to/video_directory --output tracking_results
```

### Advanced Options

You can specify custom tracking points using the following arguments:

```bash
python compute_flow.py --input /path/to/video.mp4 --output tracking_results \
    --point1x 55.0 --point1y 22.0 --point2x 70.0 --point2y 22.0
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Path to a video file or directory containing videos | (Required) |
| `--output` | Directory to save tracking results | `tracking_results` |
| `--point1x` | X coordinate of the first tracking point | 55.0 |
| `--point1y` | Y coordinate of the first tracking point | 22.0 |
| `--point2x` | X coordinate of the second tracking point | 70.0 |
| `--point2y` | Y coordinate of the second tracking point | 22.0 |

## Output

For each processed video, the tool creates a directory with the following outputs:

1. `initial_points.png` - Visualization of the initial tracking points
2. `{video_name}_forward.mp4` - Video with forward tracking visualization
3. `{video_name}_backward.mp4` - Video with backward tracking visualization 
4. `{video_name}_bidirectional.mp4` - Video with combined bidirectional tracking
5. `{video_name}_tracking_data.pth` - PyTorch file containing tracking data for further analysis

## How It Works

### Bidirectional Tracking

This tool implements a bidirectional tracking approach that combines the advantages of forward and backward tracking:

1. **Forward Tracking**: Tracks points from the first frame to the last frame
2. **Backward Tracking**: Tracks points from the last frame back to the first frame
3. **Combined Tracking**: Fuses both tracking results using a weighted combination

The bidirectional approach is especially useful for handling occlusions, motion blur, and other challenging tracking scenarios. It reduces drift and provides more robust tracking results.

### Implementation Details

- **Forward Tracking**: Standard CoTracker implementation from first frame to last
- **Backward Tracking**: Reverse the video and track from what was the last frame
- **Weighting Strategy**: Points are weighted based on their temporal distance to each endpoint

## Limitations

- Tracking accuracy depends on the quality and resolution of the input video
- Tracking can fail in cases of severe occlusion or complex motion
- Processing large videos may require significant memory and computational resources

## Further Customization

The code can be modified for:
- Tracking more than two points
- Using different weighting strategies for combining forward and backward tracks
- Integrating with other tracking algorithms

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This tool uses [CoTracker](https://github.com/facebookresearch/co-tracker) for optical flow tracking. 