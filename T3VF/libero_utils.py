import os
import math
import time
import json
import imageio

import numpy as np
from torchvision.transforms import v2
from torchvision.transforms.functional import to_tensor

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def _make_transform(size):
    return v2.Compose([v2.Resize(size), v2.CenterCrop(size)])


def get_libero_env(task, model_family, resolution=512):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]

    wrist_img = obs['robot0_eye_in_hand_image']
    wrist_img = wrist_img[::-1, ::-1]

    img_tensor = to_tensor(np.ascontiguousarray(img))
    wrist_img_tensor = to_tensor(np.ascontiguousarray(wrist_img))

    primary_image_transform = _make_transform(256)
    wrist_image_transform = _make_transform(256)

    img_tensor = primary_image_transform(img_tensor)
    wrist_img_tensor = wrist_image_transform(wrist_img_tensor)

    return [img_tensor, wrist_img_tensor], [img, wrist_img]


def save_rollout_video_v2(
    rollout_images,
    episode_idx,
    success,
    task_description,
    episode_time,
    task_suite_name,
    perturbation_category=None,
    actual_steps=None,
    log_file=None,
):
    """Saves an MP4 replay and episode metadata to JSON (base model evaluation)."""
    if perturbation_category:
        folder_name = f"{task_suite_name}_{perturbation_category.replace(' ', '_')}"
    else:
        folder_name = task_suite_name

    rollout_dir = f"./experiments/rollouts/{folder_name}"
    os.makedirs(rollout_dir, exist_ok=True)

    mp4_path = f"{rollout_dir}/{episode_idx}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()

    metadata_path = f"{rollout_dir}/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {"episodes": []}

    episode_info = {
        "episode": int(episode_idx),
        "success": bool(success),
        "task": str(task_description),
        "time": float(round(episode_time, 2)),
        "steps": int(actual_steps) if actual_steps is not None else None,
    }

    existing_idx = next((i for i, ep in enumerate(metadata["episodes"]) if ep["episode"] == episode_idx), None)
    if existing_idx is not None:
        metadata["episodes"][existing_idx] = episode_info
    else:
        metadata["episodes"].append(episode_info)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved rollout MP4: {mp4_path} (time: {episode_time:.2f}s)")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4: {mp4_path} (time: {episode_time:.2f}s)\n")

    return mp4_path


def _build_ttt_folder_name(task_suite_name, perturbation_category, ttt_lr, ttt_gap, ttt_batch_size, additional_suffix):
    """Build folder name for TTT experiment results."""
    if perturbation_category:
        base_name = f"{task_suite_name}_{perturbation_category.replace(' ', '_')}"
    else:
        base_name = task_suite_name

    lr_str = f"{ttt_lr:.0e}" if ttt_lr < 0.01 else f"{ttt_lr}"
    ttt_suffix = f"metaquery_embed_{lr_str}_{ttt_gap}_{ttt_batch_size}{additional_suffix}"
    return f"{base_name}_{ttt_suffix}"


def _save_rollout_video_ttt_impl(
    rollout_images,
    episode_idx,
    success,
    task_description,
    episode_time,
    task_suite_name,
    rollout_base_dir,
    perturbation_category=None,
    ttt_lr=1e-4,
    ttt_gap=4,
    ttt_batch_size=4,
    actual_steps=None,
    timing_stats=None,
    log_file=None,
    additional_suffix="",
):
    """Internal implementation for saving TTT rollout video and metadata."""
    folder_name = _build_ttt_folder_name(
        task_suite_name, perturbation_category, ttt_lr, ttt_gap, ttt_batch_size, additional_suffix
    )
    rollout_dir = f"./experiments/{rollout_base_dir}/{folder_name}"
    os.makedirs(rollout_dir, exist_ok=True)

    mp4_path = f"{rollout_dir}/{episode_idx}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()

    metadata_path = f"{rollout_dir}/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "ttt_settings": {
                "lr": float(ttt_lr),
                "gap": int(ttt_gap),
                "batch_size": int(ttt_batch_size),
            },
            "episodes": []
        }

    episode_info = {
        "episode": int(episode_idx),
        "success": bool(success),
        "task": str(task_description),
        "time": float(round(episode_time, 2)),
        "steps": int(actual_steps) if actual_steps is not None else None,
    }
    if timing_stats is not None:
        episode_info["timing"] = timing_stats

    existing_idx = next((i for i, ep in enumerate(metadata["episodes"]) if ep["episode"] == episode_idx), None)
    if existing_idx is not None:
        metadata["episodes"][existing_idx] = episode_info
    else:
        metadata["episodes"].append(episode_info)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved rollout MP4: {mp4_path} (time: {episode_time:.2f}s)")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4: {mp4_path} (time: {episode_time:.2f}s)\n")

    return mp4_path


def save_rollout_video_ttt(
    rollout_images, episode_idx, success, task_description, episode_time,
    task_suite_name, perturbation_category=None, ttt_lr=1e-4, ttt_gap=4,
    ttt_batch_size=4, actual_steps=None, timing_stats=None, log_file=None,
    additional_suffix="",
):
    """Saves TTT rollout video (w/ Perturbed Train setting) to experiments/rollouts/."""
    return _save_rollout_video_ttt_impl(
        rollout_images, episode_idx, success, task_description, episode_time,
        task_suite_name, "rollouts", perturbation_category, ttt_lr, ttt_gap,
        ttt_batch_size, actual_steps, timing_stats, log_file, additional_suffix,
    )


def save_rollout_video_ttt_ood(
    rollout_images, episode_idx, success, task_description, episode_time,
    task_suite_name, perturbation_category=None, ttt_lr=1e-4, ttt_gap=4,
    ttt_batch_size=4, actual_steps=None, timing_stats=None, log_file=None,
    additional_suffix="",
):
    """Saves TTT rollout video (w/o Perturbed Train setting) to experiments/rollouts_ood/."""
    return _save_rollout_video_ttt_impl(
        rollout_images, episode_idx, success, task_description, episode_time,
        task_suite_name, "rollouts_ood", perturbation_category, ttt_lr, ttt_gap,
        ttt_batch_size, actual_steps, timing_stats, log_file, additional_suffix,
    )


def get_completed_episodes(rollout_dir: str) -> set:
    """Get set of completed episode indices from metadata.json."""
    metadata_path = f"{rollout_dir}/metadata.json"
    if not os.path.exists(metadata_path):
        return set()

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return {ep["episode"] for ep in metadata.get("episodes", [])}
    except (json.JSONDecodeError, KeyError):
        return set()


def quat2axisangle(quat):
    """Converts quaternion to axis-angle format."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
