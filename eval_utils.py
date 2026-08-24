from copy import deepcopy

import numpy as np
import torch


def extract_init_goal(dataset, episodes_idx, start_steps, goal_offset):
    ep_idx_arr = np.array(episodes_idx)
    start_arr = np.array(start_steps)
    try:
        data = dataset.load_chunk(ep_idx_arr, start_arr, start_arr + goal_offset + 1)
    except OSError as err:
        raise RuntimeError(
            "Failed to read dataset chunks. If this is an HDF5/Blosc error, "
            "make sure hdf5plugin is installed and importable in this environment."
        ) from err

    init_lists = {}
    goal_lists = {}
    dataset_videos = []

    for ep in data:
        for col in dataset.column_names:
            if col.startswith("goal"):
                continue
            if col.startswith("pixels"):
                ep[col] = ep[col].permute(0, 2, 3, 1)
            val = ep[col]
            if not isinstance(val, (torch.Tensor, np.ndarray)):
                continue
            arr = val.numpy() if isinstance(val, torch.Tensor) else val
            init_lists.setdefault(col, []).append(arr[0])
            goal_lists.setdefault(col, []).append(arr[-1])
            if col == "pixels":
                dataset_videos.append(arr)

    init_state = {k: np.stack(v) for k, v in init_lists.items()}
    goal_state = {}
    for k, v in goal_lists.items():
        goal_state["goal" if k == "pixels" else f"goal_{k}"] = np.stack(v)

    return init_state, goal_state, dataset_videos


def apply_callables(env, callables, init_state):
    for spec in callables or []:
        method = spec["method"]
        if not hasattr(env, method):
            continue
        prepared = {}
        for name, data in spec.get("args", {}).items():
            if data.get("in_dataset", True):
                key = data.get("value")
                if key in init_state:
                    prepared[name] = deepcopy(init_state[key])
            else:
                prepared[name] = data.get("value")
        getattr(env, method)(**prepared)


def copy_dataset_infos(world, init_state, goal_state):
    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for key, value in src.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()
