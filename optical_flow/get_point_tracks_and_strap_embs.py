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

import pickle as pkl
from glob import glob
from pathlib import Path

import blosc
import hydra
import numpy as np
import torch
import tqdm
from clam.scripts.optical_flow.compute_2d_flow import (
    generate_point_tracks,
    load_cotracker,
)
from clam.utils.logger import log
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from calvin_env.envs.play_table_env import PlayTableSimEnv
from hand_demos.utils.general_utils import to_numpy
from hand_demos.utils.retrieval_utils import (
    get_dino_features,
)


@hydra.main(version_base=None, config_name="convert_to_tfds", config_path="../../cfg")
def main(cfg):
    log("Loading CoTracker", "green")
    cotracker = load_cotracker(cfg.flow)

    GlobalHydra.instance().clear()
    log("Initializing CALVIN environment", "green")
    with initialize(config_path="../../../calvin_env/conf/"):
        env_cfg = compose(
            config_name="config_data_collection.yaml",
            overrides=["cameras=static_and_gripper"],
        )
        env_cfg.env["use_egl"] = False
        env_cfg.env["show_gui"] = False
        env_cfg.env["use_vr"] = False
        env_cfg.env["use_scene_info"] = True
        print(env_cfg.env)

    env_cfg.cameras.static.width = 200
    env_cfg.cameras.static.height = 200
    env_cfg = {**env_cfg.env}

    env_cfg.pop("_target_", None)
    env_cfg.pop("_recursive_", None)
    env = PlayTableSimEnv(**env_cfg)

    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # demo_data_dir = "/scr/shared/clam/datasets/calvin/training/C+0/"
    demo_data_dir = "/scr/shared/clam/datasets/calvin/train_tasks/D+0"
    data_dirs = glob(str(demo_data_dir) + "/*")

    files = []
    for task in data_dirs:
        files.extend(list(sorted(Path(task).glob("*.dat"))))

    # list all files in the directory
    files = sorted(files)
    log(f"Processing {len(files)} files", "green")

    for indx, file in tqdm.tqdm(
        enumerate(files), total=len(files), desc="Processing files"
    ):
        with open(file, "rb") as f:
            compressed_data = f.read()  # Reads all lines into a list
            decompressed_data = blosc.decompress(compressed_data)
            data = pkl.loads(decompressed_data)

        video = data[1]
        state = data[5]
        eef = state[:, :3]

        # figure out where the eef is in the image space
        eef_hom = np.concatenate([eef, np.ones((eef.shape[0], 1))], axis=1)
        xy = env.cameras[0].project(eef_hom[0])
        query = np.array([0, xy[0], xy[1] - 20])
        queries = np.array([query])

        # Track the query points across the video.
        # Return a list of tracked points and their visibility.
        points, visibility = generate_point_tracks(
            cfg.flow, cotracker, video, queries=queries
        )

        # Save tracked points to file
        tracked_points = {
            "points": to_numpy(points[0]),
            "visibility": to_numpy(visibility[0]),
        }

        # point tracks
        data.append(tracked_points)

        # STRAP dinov2 features
        data.append(get_dino_features(data[1]))

        with open(file, "wb") as f:
            compressed_data = blosc.compress(pkl.dumps(data))
            f.write(compressed_data)

        log(f"overwriting {file}", "yellow")

if __name__ == "__main__":
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    main()