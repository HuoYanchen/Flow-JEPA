import os

os.environ["MUJOCO_GL"] = "egl"

import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

from eval_utils import apply_callables, copy_dataset_infos, extract_init_goal
from perturbations import build_gaussian_perturbation


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = (
        "episode_idx"
        if "episode_idx" in dataset.column_names
        else "ep_idx"
    )
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array(
        [np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes]
    )


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.get("cache_dir") or swm.data.utils.get_cache_dir())
    if not hasattr(swm.data, "HDF5Dataset"):
        raise RuntimeError(
            "stable_worldmodel.data.HDF5Dataset is unavailable. "
            "Install the HDF5 extras, including hdf5plugin/Blosc support, "
            f"before loading '{dataset_name}' from {dataset_path / 'datasets'}."
        )
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def evaluate_with_gaussian_noise(
    world,
    dataset,
    episodes_idx,
    start_steps,
    goal_offset,
    eval_budget,
    callables,
    video,
    cfg,
):
    """Evaluate with one fixed Gaussian patch on current and goal images."""
    n = len(episodes_idx)
    assert n == world.num_envs

    init_state, goal_state, dataset_videos = extract_init_goal(
        dataset, episodes_idx, start_steps, goal_offset
    )
    world.reset(seed=init_state.get("seed"))

    merged = {**init_state, **goal_state}
    for index in range(n):
        env_init = {key: value[index] for key, value in merged.items()}
        apply_callables(
            world.envs.envs[index].unwrapped,
            callables,
            env_init,
        )

    copy_dataset_infos(world, init_state, goal_state)
    perturbation = build_gaussian_perturbation(
        cfg.perturbation,
        cfg.world.env_name,
        seed=cfg.seed,
    )
    perturbation.setup(world, init_state, goal_state)

    copy_dataset_infos(world, init_state, goal_state)
    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}
    successes = np.zeros(n, dtype=bool)
    frames = defaultdict(list) if video else None
    alive = np.ones(n, dtype=bool)

    for _ in range(eval_budget):
        policy_info = perturbation.apply_to_policy_info(world.infos)
        actions = world.policy.get_action(policy_info)
        mask = alive if not alive.all() else None
        _, world.rewards, world.terminateds, world.truncateds, world.infos = (
            world.envs.step(actions, mask=mask)
        )
        world.infos.update(deepcopy(goal_snapshot))
        successes |= world.terminateds

        if frames is not None:
            frame_info = perturbation.apply_to_policy_info(world.infos)
            for index in range(n):
                frame = frame_info["pixels"][index]
                frame = frame[-1] if frame.ndim > 3 else frame
                frames[index].append(np.asarray(frame).copy())

        done = alive & (world.terminateds | world.truncateds)
        alive[done] = False
        if not alive.any():
            break

    if frames:
        from stable_worldmodel.plot import save_panel_videos

        save_panel_videos(
            Path(video),
            {
                "agent": frames,
                "dataset": dataset_videos,
                "goal": goal_state["goal"],
            },
            fps=15,
        )

    return {
        "success_rate": float(successes.mean() * 100.0),
        "episode_successes": successes,
        "seeds": init_state.get("seed"),
    }


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Evaluate a checkpoint with MPC, optionally under Gaussian image noise."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block
        <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    col_name = (
        "episode_idx"
        if "episode_idx" in dataset.column_names
        else "ep_idx"
    )
    ep_indices = np.unique(dataset.get_col_data(col_name))

    process = {}
    for column in cfg.dataset.keys_to_cache:
        if column == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        values = dataset.get_col_data(column)
        values = values[~np.isnan(values).any(axis=1)]
        processor.fit(values)
        process[column] = processor
        if column != "action":
            process[f"goal_{column}"] = processor

    if cfg.policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to(cfg.solver.device).eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        model.set_flow_seed(cfg.flow_seed)
        plan_config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=plan_config,
            process=process,
            transform=transform,
        )
    else:
        policy = swm.policy.RandomPolicy(seed=cfg.seed)

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_by_episode = {
        ep_id: max_start_idx[index]
        for index, ep_id in enumerate(ep_indices)
    }
    max_start_per_row = np.array(
        [
            max_start_by_episode[ep_id]
            for ep_id in dataset.get_col_data(col_name)
        ]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    generator = np.random.default_rng(cfg.seed)
    selected = generator.choice(
        len(valid_indices) - 1,
        size=cfg.eval.num_eval,
        replace=False,
    )
    selected = np.sort(valid_indices[selected])
    print(selected)

    selected_rows = dataset.get_row_data(selected)
    eval_episodes = selected_rows[col_name]
    eval_start_idx = selected_rows["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    if cfg.output.get("dir"):
        results_path = Path(cfg.output.dir)
    results_path.mkdir(parents=True, exist_ok=True)
    video_path = results_path if cfg.output.get("save_video", True) else None

    start_time = time.time()
    callables = OmegaConf.to_container(
        cfg.eval.get("callables"),
        resolve=True,
    )
    if cfg.perturbation.enabled:
        metrics = evaluate_with_gaussian_noise(
            world=world,
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=callables,
            video=video_path,
            cfg=cfg,
        )
    else:
        metrics = world.evaluate(
            dataset=dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=callables,
            video=video_path,
        )
    evaluation_time = time.time() - start_time
    print(metrics)

    results_file = results_path / cfg.output.filename
    with results_file.open("a") as stream:
        stream.write("\n==== CONFIG ====\n")
        stream.write(OmegaConf.to_yaml(cfg))
        stream.write("\n==== RESULTS ====\n")
        stream.write(f"metrics: {metrics}\n")
        stream.write(f"evaluation_time: {evaluation_time} seconds\n")


if __name__ == "__main__":
    run()
