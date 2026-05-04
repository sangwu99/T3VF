import os
import sys
import tqdm
import time
import json
import torch
import random
import draccus
import gc

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from dataclasses import dataclass
from typing import Optional
from mantis_vla_utils import MantisVLA, MantisVLATTT

LIBERO_PLUS_PATH = os.environ.get("LIBERO_PLUS_PATH", "")
if LIBERO_PLUS_PATH:
    sys.path.append(LIBERO_PLUS_PATH)
from libero.libero import benchmark

from libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video_v2,
    save_rollout_video_ttt,
    get_completed_episodes,
)


@dataclass
class GenerateConfig:
    model_family: str = "mantis"
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    run_id_note: Optional[str] = None
    local_log_dir: str = "experiments/libero_eval_logs"
    norm_file_path: str = "config/norm_stats.json"

    seed: int = 7
    model_id: str = None
    checkpoints_dir: str = None
    action_dim: int = 7
    future_action_window_size: int = 4
    perturbation_category: Optional[str] = None

    # TTT settings
    ttt_enabled: bool = False
    ttt_lr: float = 1e-4
    ttt_gap: int = 4
    ttt_batch_size: int = 4
    ttt_time_profile: bool = False
    additional_suffix: str = ""

    # Action variance filtering
    action_variance_enabled: bool = False
    action_variance_n_samples: int = 5
    action_variance_warmup: int = 10
    action_variance_percentile: float = 0.7


MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def set_seed_everywhere(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)
    cfg.unnorm_key = "libero_plus"

    # Load model
    if cfg.ttt_enabled:
        mantis_vla = MantisVLATTT(
            cfg.model_id,
            cfg.checkpoints_dir,
            cfg.norm_file_path,
            ttt_lr=cfg.ttt_lr,
            ttt_time_profile=cfg.ttt_time_profile,
            target_image_size=256,
            action_variance_enabled=cfg.action_variance_enabled,
            action_variance_n_samples=cfg.action_variance_n_samples,
            action_variance_warmup=cfg.action_variance_warmup,
            action_variance_percentile=cfg.action_variance_percentile,
        )
    else:
        mantis_vla = MantisVLA(
            cfg.model_id,
            cfg.checkpoints_dir,
            cfg.norm_file_path,
            target_image_size=256,
        )

    # Local logging
    DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
    run_id = f"EVAL-{cfg.task_suite_name}-{cfg.model_family}-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to: {local_log_filepath}")

    # Initialize task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    print(f"Task suite: {cfg.task_suite_name}")
    log_file.write(f"Task suite: {cfg.task_suite_name}\n")

    resize_size = [256, 256]
    total_episodes, total_successes = 0, 0

    # Resume support: check completed episodes
    if cfg.perturbation_category:
        base_name = f"{cfg.task_suite_name}_{cfg.perturbation_category.replace(' ', '_')}"
    else:
        base_name = cfg.task_suite_name

    if cfg.ttt_enabled:
        lr_str = f"{cfg.ttt_lr:.0e}" if cfg.ttt_lr < 0.01 else f"{cfg.ttt_lr}"
        ttt_suffix = f"metaquery_embed_{lr_str}_{cfg.ttt_gap}_{cfg.ttt_batch_size}{cfg.additional_suffix}"
        folder_name = f"{base_name}_{ttt_suffix}"
    else:
        folder_name = base_name

    rollout_dir = f"./experiments/rollouts/{folder_name}"
    completed_episodes = get_completed_episodes(rollout_dir)
    if completed_episodes:
        print(f"Resuming: Found {len(completed_episodes)} completed episodes")

    # Filter tasks by perturbation category
    classification_path = os.path.join(LIBERO_PLUS_PATH, "libero/libero/benchmark/task_classification.json")
    with open(classification_path, "r") as f:
        task_classification = json.load(f)
    if cfg.perturbation_category is not None:
        task_ids = [t["id"] - 1 for t in task_classification[cfg.task_suite_name]
                    if t["category"] == cfg.perturbation_category]
        print(f"Perturbation: {cfg.perturbation_category} -> {len(task_ids)} tasks")
    else:
        task_ids = list(range(num_tasks_in_suite))

    for task_id in tqdm.tqdm(task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        gc.collect()
        torch.cuda.empty_cache()
        env, task_description = get_libero_env(task, cfg.model_family, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.num_trials_per_task)):
            if (total_episodes + 1) in completed_episodes:
                task_episodes += 1
                total_episodes += 1
                continue

            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            episode_start_time = time.time()

            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            replay_images = []
            max_steps = MAX_STEPS[cfg.task_suite_name]

            if cfg.ttt_enabled:
                mantis_vla.reset_ttt_state()
                ttt_cache_buffer = {}
                ttt_update_buffer = []

            all_time_actions = np.zeros(
                (max_steps + cfg.num_steps_wait, max_steps + cfg.num_steps_wait + cfg.future_action_window_size, cfg.action_dim),
                dtype=np.float64
            )

            t = 0
            while t < max_steps + cfg.num_steps_wait:
                if t < cfg.num_steps_wait:
                    obs, reward, done, info = env.step(get_libero_dummy_action(cfg.model_family))
                    t += 1
                    continue

                img, save_img = get_libero_image(obs, resize_size)
                replay_images.append(save_img[0])
                observation = {
                    "full_image": img,
                    "state": np.concatenate(
                        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                    ),
                }

                adjusted_t = t - cfg.num_steps_wait

                # TTT: collect predicted-attained pairs for batch update
                if cfg.ttt_enabled and (adjusted_t - cfg.ttt_gap) in ttt_cache_buffer:
                    ttt_cache = ttt_cache_buffer.pop(adjusted_t - cfg.ttt_gap)

                    variance = ttt_cache.get("action_variance", None)
                    if not mantis_vla.should_skip_by_variance(variance):
                        ttt_update_buffer.append((ttt_cache, img[0]))

                    if len(ttt_update_buffer) >= cfg.ttt_batch_size:
                        batch_to_update = ttt_update_buffer[:cfg.ttt_batch_size]
                        ttt_update_buffer = ttt_update_buffer[cfg.ttt_batch_size:]
                        mantis_vla.ttt_update_batch(batch_to_update)

                # Model inference
                if cfg.ttt_enabled:
                    actions, ttt_cache = mantis_vla.inference(
                        observation["full_image"],
                        task_description,
                        unnorm_key=cfg.unnorm_key,
                        gap=cfg.ttt_gap,
                    )
                    ttt_cache_buffer[adjusted_t] = ttt_cache
                else:
                    actions = mantis_vla.inference(
                        observation["full_image"],
                        task_description,
                        unnorm_key=cfg.unnorm_key,
                    )
                all_time_actions[t, t:t + cfg.future_action_window_size] = actions

                # Temporal ensemble
                actions_for_curr_step = all_time_actions[:, t]
                mask = np.any(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[mask]
                if len(actions_for_curr_step) == 1:
                    raw_action = actions_for_curr_step.squeeze()
                else:
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights /= exp_weights.sum()
                    exp_weights = exp_weights[:, None]
                    raw_action = (actions_for_curr_step * exp_weights).sum(axis=0)

                obs, reward, done, info = env.step(raw_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    break
                t += 1

            # Process remaining TTT buffer
            if cfg.ttt_enabled:
                while len(ttt_update_buffer) >= cfg.ttt_batch_size:
                    batch_to_update = ttt_update_buffer[:cfg.ttt_batch_size]
                    ttt_update_buffer = ttt_update_buffer[cfg.ttt_batch_size:]
                    mantis_vla.ttt_update_batch(batch_to_update)
                ttt_update_buffer = []

            task_episodes += 1
            total_episodes += 1
            episode_time = time.time() - episode_start_time
            actual_steps = t - cfg.num_steps_wait

            # Save replay video
            if cfg.ttt_enabled:
                timing_stats = mantis_vla.get_timing_stats() if cfg.ttt_time_profile else None
                save_rollout_video_ttt(
                    rollout_images=replay_images,
                    episode_idx=total_episodes,
                    success=done,
                    task_description=task_description,
                    episode_time=episode_time,
                    task_suite_name=cfg.task_suite_name,
                    perturbation_category=cfg.perturbation_category,
                    ttt_lr=cfg.ttt_lr,
                    ttt_gap=cfg.ttt_gap,
                    ttt_batch_size=cfg.ttt_batch_size,
                    actual_steps=actual_steps,
                    timing_stats=timing_stats,
                    log_file=log_file,
                    additional_suffix=cfg.additional_suffix,
                )
            else:
                save_rollout_video_v2(
                    rollout_images=replay_images,
                    episode_idx=total_episodes,
                    success=done,
                    task_description=task_description,
                    episode_time=episode_time,
                    task_suite_name=cfg.task_suite_name,
                    perturbation_category=cfg.perturbation_category,
                    actual_steps=actual_steps,
                    log_file=log_file,
                )

            # Log results
            print(f"Success: {done}, Time: {episode_time:.2f}s")
            print(f"Episodes: {total_episodes}, Successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"Episodes: {total_episodes}, Successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        print(f"Task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    eval_libero()
