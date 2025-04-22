#!/usr/bin/env python3
import h5py
import numpy as np
import cv2
import os
import argparse
import glob
from tqdm import tqdm


def hdf5_to_mp4(hdf5_file_path, output_dir):
    """
    Convert an HDF5 file containing demonstration data to MP4 videos.
    
    Args:
        hdf5_file_path (str): Path to the HDF5 file
        output_dir (str): Output directory for the MP4 files
    """
    try:
        # Get the base name of the file without extension
        file_name = os.path.basename(hdf5_file_path).split('.')[0]
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Open the HDF5 file
        with h5py.File(hdf5_file_path, "r") as f:
            # Check if the expected structure exists
            if 'data' not in f:
                print(f"No 'data' group found in {hdf5_file_path}")
                return
            
            data_group = f['data']
            demo_keys = list(data_group.keys())
            
            if not demo_keys:
                print(f"No demonstrations found in {hdf5_file_path}")
                return
            
            # Process each demonstration
            for demo_idx, demo_key in enumerate(demo_keys):
                # Extract demo number from the key (assuming format 'demo_X')
                demo_num = demo_key.split('_')[-1]
                
                demo_group = data_group[demo_key]
                
                # Check if the demo contains observation data
                if 'obs' not in demo_group:
                    print(f"No observation data found in {demo_key} of {hdf5_file_path}")
                    continue
                
                obs_group = demo_group['obs']
                
                # Only process the main camera view
                camera_view = 'agentview_rgb'
                if camera_view not in obs_group:
                    print(f"No {camera_view} data found in {demo_key} of {hdf5_file_path}")
                    continue
                
                # Get the frames
                frames = obs_group[camera_view][:]
                
                # Check if frames are valid
                if len(frames.shape) != 4 or frames.shape[3] != 3:
                    print(f"Invalid frame shape {frames.shape} for {camera_view} in {demo_key} of {hdf5_file_path}")
                    continue
                
                # Construct output file path with new naming convention
                output_file = os.path.join(output_dir, f"VID-DEMO_{demo_num}_{file_name}.mp4")
                
                # Get video dimensions
                height, width = frames.shape[1:3]
                
                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_file, fourcc, 30.0, (width, height))
                
                # Write frames to video
                for frame in tqdm(frames, desc=f"Converting DEMO_{demo_num}_{file_name}"):
                    # OpenCV uses BGR format, so convert RGB to BGR
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    # Flip the frame vertically
                    frame_bgr = cv2.flip(frame_bgr, 0)
                    
                    out.write(frame_bgr)
                
                # Release the VideoWriter
                out.release()
                print(f"Successfully created {output_file}")
        
    except Exception as e:
        print(f"Error processing {hdf5_file_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 files to MP4 videos')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing HDF5 files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save MP4 files')
    args = parser.parse_args()
    
    # Get all HDF5 files in the input directory
    hdf5_files = glob.glob(os.path.join(args.input_dir, "**", "*.hdf5"), recursive=True)
    
    if not hdf5_files:
        print(f"No HDF5 files found in {args.input_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    
    # Process each HDF5 file
    for hdf5_file in hdf5_files:
        # Determine the correct output directory based on the file path
        if "libero_10" in hdf5_file:
            output_subdir = "libero_10_mp4"
        elif "libero_90" in hdf5_file:
            output_subdir = "libero_90_mp4"
        else:
            rel_path = os.path.relpath(os.path.dirname(hdf5_file), args.input_dir)
            output_subdir = os.path.join(rel_path, "mp4")
        
        # Combine with the base output directory
        output_dir = os.path.join(args.output_dir, output_subdir)
        
        # Convert the file
        hdf5_to_mp4(hdf5_file, output_dir)
    
    print("Conversion complete")


if __name__ == "__main__":
    main()
