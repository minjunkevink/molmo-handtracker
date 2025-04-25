"""
MOLMO CoTrack - Optical flow tracking utilities for hand tracking.

This package provides utilities for tracking hand and gripper movements
in videos using CoTracker and other computer vision techniques.
"""

from .flow_compute_utils import track_gripper, process_folder

__all__ = ['track_gripper', 'process_folder'] 