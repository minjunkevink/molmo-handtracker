# MolMo Hand Tracker

A computer vision toolkit for tracking hand movements in demonstration videos using optical flow and bidirectional tracking techniques.

## Overview

This project provides tools for:
1. Converting HDF5 demonstration datasets to MP4 videos
2. Applying state-of-the-art optical flow tracking to follow hand movements in videos
3. Implementing bidirectional tracking for improved point tracking reliability

## Features

- **HDF5 to MP4 Conversion**: Extract RGB frames from HDF5 datasets and convert them to MP4 videos
- **Bidirectional Tracking**: Uses CoTracker to perform both forward and backward point tracking with automatic reconciliation
- **Visualization Tools**: Generate visualization videos showing tracking results
- **Jupyter Notebook Support**: Example notebooks for testing and experimenting with the trackers

## Installation

### Environment Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate molmo

# Install additional dependencies
pip install torch torchvision
pip install h5py opencv-python matplotlib tqdm
pip install imageio imageio-ffmpeg
```

### Module Installation
```bash
cd modules && bash install_submodules.sh
```

### CoTracker Installation

This project uses [CoTracker](https://github.com/facebookresearch/co-tracker) for optical flow tracking. It will be automatically downloaded via PyTorch Hub when used.

## Usage

### Converting HDF5 Files to MP4

```bash
python preprocessing/hdf5_to_mp4.py --input_dir /path/to/hdf5/files --output_dir /path/to/output
```

### Computing Optical Flow Tracking

```bash
python optical_flow/compute_flow.py --input /path/to/video.mp4 --output tracking_results
```

Optional arguments:
- `--point1x` and `--point1y`: First point coordinates (default: 55.0, 22.0)
- `--point2x` and `--point2y`: Second point coordinates (default: 70.0, 22.0)

## Requirements

- Python 3.10+
- PyTorch
- CUDA-capable GPU (recommended)
- Dependencies listed in environment.yml
