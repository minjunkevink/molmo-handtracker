"""
Given a video, we use GroundedSAM2 to generate
object masks for the first frame of the video.

The object masks are used to filter query points for CoTracker.
We apply CoTracker to track the query points across the video
to generate the 2D "object/scene" flow.

The output is a set of points (N, 2) for each video where N is the desired number
of points to track. We also get the visibility of each point across the video,
note not all the points are visible at each frame because of occlusions.
"""

import os
import sys
import time
from base64 import b64encode
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv
import torch
import yaml
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from IPython.display import HTML
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

# Add the grounded sam2 folder to path to import the necessary modules
sys.path.append("/scr/aliang80/Grounded-SAM-2")
import matplotlib.pyplot as plt
from cotracker.predictor import CoTrackerPredictor
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from scripts.utils import load_config_as_namespace
from clam.utils.logger import log

# ===============================================
# HELPER FUNCTIONS
# ===============================================


def load_sam_model(cfg):
    log("Building SAM2 model")
    start = time.time()

    sam_ckpt = os.path.join(cfg.sam_path, cfg.sam2_checkpoint)
    sam2_image_model = build_sam2(cfg.model_cfg, sam_ckpt)
    image_predictor = SAM2ImagePredictor(sam2_image_model)
    log(f"Built SAM2 model in {time.time() - start:.2f}s")

    return sam2_image_model, image_predictor


def load_cotracker(cfg):
    log("Initializing CoTracker model")
    cotracker_ckpt = os.path.join(cfg.base_path, cfg.cotracker_ckpt)
    model = CoTrackerPredictor(checkpoint=cotracker_ckpt)
    model = model.to(cfg.device)
    return model


def get_seg_mask(
    cfg, sam_image_model, image_predictor, video=None, image=None, text=None
):
    """
    Step 1: Environment settings and model initialization
    """
    # init grounding dino model from huggingface
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        cfg.model_id
    ).to(cfg.device)

    if video is not None:
        # image is first frame of the video
        image = Image.fromarray(video[0])
    else:
        image = Image.fromarray(image)

    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
    """

    # run Grounding DINO on the image
    text = cfg.text if text is None else text
    log(f"Running Grounding DINO on the image, with text: {text}")
    start = time.time()
    inputs = processor(images=image, text=text, return_tensors="pt").to(cfg.device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)
    log(f"Finished running Grounding DINO, took {time.time() - start:.2f}s")

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]
    log(f"Inferred boxes: {input_boxes}")
    log(f"Detected objects: {OBJECTS}")

    # prompt SAM 2 image predictor to get the mask for the object
    log("Running SAM2 image predictor to get the mask for the object")
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)
    log(f"Got masks of shape: {masks.shape}")
    log(f"Object scores: {scores}")

    # Combine all the masks to generate one mask of relevant objects
    combined_mask = np.zeros_like(masks[0])
    for mask in masks:
        combined_mask += mask

    log(f"Mask min: {combined_mask.min()}, max: {combined_mask.max()}")

    # create 2 subplots and visualize mask and masked image
    ax = plt.subplot(1, 2, 1)
    ax.imshow(combined_mask)

    ax = plt.subplot(1, 2, 2)
    ax.imshow((combined_mask[..., None] * image / 255.0))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.base_path, "seg_mask_vis.png"))

    # also just save mask separately
    np.save(os.path.join(cfg.base_path, cfg.seg_mask_file), combined_mask)

    return combined_mask


def generate_point_tracks(cfg, cotracker, video, segm_mask=None, queries=None):
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    video = video.to(cfg.device)

    log("Running CoTracker on the video")
    start = time.time()

    if cfg.seg_mask_file:
        seg_mask_file = os.path.join(cfg.base_path, cfg.seg_mask_file)
        segm_mask = np.load(seg_mask_file)

    if queries is not None:
        log(f"Using provided queries: {queries}")
        queries = torch.from_numpy(queries).float().to(cfg.device)

    if queries is not None:
        pred_tracks, pred_visibility = cotracker(
            video,
            segm_mask=torch.from_numpy(segm_mask)[None, None],
            queries=queries[None],
        )
    else:
        pred_tracks, pred_visibility = cotracker(
            video,
            grid_size=cfg.grid_size,
            segm_mask=torch.from_numpy(segm_mask)[None, None],
        )
    log(
        f"Predicted tracks shape: {pred_tracks.shape}, visibility shape: {pred_visibility.shape}"
    )
    log(f"Finished running CoTracker, took {time.time() - start:.2f}s")

    # generate visualization of the tracking results
    if cfg.save_tracking_video:
        save_dir = os.path.join(cfg.base_path, cfg.tracking_dir)
        vis = Visualizer(
            save_dir=save_dir,
            pad_value=10,
            linewidth=2,
            mode="optical_flow",
            tracks_leave_trace=-1,
        )
        vis.visualize(
            video=video,
            tracks=pred_tracks,
            visibility=pred_visibility,
            filename=cfg.tracking_filename,
        )

    return pred_tracks, pred_visibility


def main():
    cfg = load_config_as_namespace("compute_flow.yaml")

    video_file = os.path.join(cfg.base_path, cfg.video_file)
    log(f"Reading video from path {video_file}")
    video = read_video_from_path(video_file)
    log(f"Video shape: {video.shape}")

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam, image_predictor = load_sam_model(cfg)

    cotracker = load_cotracker(cfg)

    # Step 1: Get object segmentation mask for the first frame.
    segm_mask = get_seg_mask(cfg, sam, image_predictor, video=video)

    # Step 2: Track the query points across the video.
    # Return a list of tracked points and their visibility.
    points, visibility = generate_point_tracks(cfg, cotracker, video, segm_mask)

    # Save tracked points to file
    tracking_results = {"points": points, "visibility": visibility}
    np.save(
        os.path.join(cfg.base_path, cfg.tracking_dir, "points.npy"), tracking_results
    )


if __name__ == "__main__":
    main()