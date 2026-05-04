import os
import sys
import json
import time
import torch

import numpy as np
from typing import Optional, Dict, Any

from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mantis import Mantis


class MantisVLA:
    """Base Mantis VLA wrapper for inference-only evaluation."""

    def __init__(
        self,
        model_id: str,
        checkpoints_dir: str,
        norm_file_path: str,
        target_image_size: int = 512,
        vae_downsample_f: int = 32,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        input_size = target_image_size // vae_downsample_f

        self.model = Mantis.from_pretrained(model_id, input_size=input_size, ignore_mismatched_sizes=True)
        if checkpoints_dir:
            state_dict = torch.load(f"{checkpoints_dir}/model.pt", map_location='cpu')
            self.model.load_state_dict(state_dict)

        # Remove unused components for inference-only
        if hasattr(self.model.model, 'transformer'):
            del self.model.model.transformer
        if hasattr(self.model.model, 'connector'):
            del self.model.model.connector
        if hasattr(self.model.model, 'vae'):
            del self.model.model.vae

        self.model.to("cuda")
        self.model.eval()

        with open(norm_file_path, "r") as f:
            self.norm_stats = json.load(f)

    @torch.inference_mode()
    def inference(self, image: list, instruction: str, unnorm_key: str = None) -> np.ndarray:
        model = self.model.module if hasattr(self.model, "module") else self.model
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output_action, _, _ = model.sample_actions(
                caption=instruction,
                input_images=[image],
                num_images_per_prompt=1,
                gap=4,
            )

        action_norm_stats = self.get_action_stats(unnorm_key)
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        unnorm_actions = 0.5 * (output_action + 1) * (action_high - action_low) + action_low
        unnorm_actions[..., -1] = np.where(unnorm_actions[..., -1] >= 0, 1.0, -1.0)
        return unnorm_actions

    def get_action_stats(self, unnorm_key=None):
        if unnorm_key is None:
            unnorm_key = next(iter(self.norm_stats.keys()))
        return self.norm_stats[unnorm_key]["action"]


class MantisVLATTT:
    """
    TTT-enabled Mantis VLA wrapper.
    Updates metaquery token embeddings at test time using predicted-attained image pairs.
    Supports adaptive update filtering via action variance.
    """

    def __init__(
        self,
        model_id: str,
        checkpoints_dir: str,
        norm_file_path: str,
        ttt_lr: float = 1e-4,
        ttt_time_profile: bool = False,
        target_image_size: int = 256,
        vae_downsample_f: int = 32,
        action_variance_enabled: bool = False,
        action_variance_n_samples: int = 5,
        action_variance_warmup: int = 10,
        action_variance_percentile: float = 0.7,
    ):
        self.device = "cuda"
        self.ttt_time_profile = ttt_time_profile

        # Action variance settings
        self.action_variance_enabled = action_variance_enabled
        self.action_variance_n_samples = action_variance_n_samples
        self.action_variance_warmup = action_variance_warmup
        self.action_variance_percentile = action_variance_percentile
        self.action_variance_history = []
        self.action_variance_threshold = None

        input_size = target_image_size // vae_downsample_f

        # Load model (keep connector, transformer, vae for TTT)
        self.model = Mantis.from_pretrained(model_id, input_size=input_size, ignore_mismatched_sizes=True)
        state_dict = torch.load(f"{checkpoints_dir}/model.pt", map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        self._enable_gradient_checkpointing()

        with open(norm_file_path, "r") as f:
            self.norm_stats = json.load(f)

        # Freeze all parameters, then selectively unfreeze metaquery embeddings
        for param in self.model.parameters():
            param.requires_grad = False

        self._init_ttt(ttt_lr)

        self.ttt_update_count = 0
        self.time_forward = 0.0
        self.time_backward = 0.0

    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory-efficient backward pass."""
        self.model.model.mllm_backbone.gradient_checkpointing_enable()
        if hasattr(self.model.model.connector[0], 'gradient_checkpointing_enable'):
            self.model.model.connector[0].gradient_checkpointing_enable()
        self.model.model.transformer.enable_gradient_checkpointing()

    def _init_ttt(self, lr: float):
        """Initialize TTT: make metaquery token embeddings trainable."""
        tokenizer = self.model.model.tokenizer.tokenizer
        self.metaquery_start_id = tokenizer.convert_tokens_to_ids("<begin_of_img>")
        self.metaquery_end_id = tokenizer.convert_tokens_to_ids("<end_of_img>")
        metaquery_ids = [tokenizer.convert_tokens_to_ids(f"<img{i}>")
                        for i in range(self.model.model.tokenizer.num_metaqueries)]
        self.metaquery_token_ids = [self.metaquery_start_id, self.metaquery_end_id] + metaquery_ids

        embed_tokens = self.model.get_input_embeddings()
        self.original_metaquery_embeds = embed_tokens.weight.data[self.metaquery_token_ids].clone()

        embed_tokens.weight.requires_grad = True
        self.optimizer = torch.optim.AdamW([embed_tokens.weight], lr=lr)

        print(f"TTT initialized: metaquery_embed, lr={lr}, tokens={len(self.metaquery_token_ids)}")

    def reset_ttt_state(self):
        """Reset metaquery embeddings to original checkpoint values."""
        embed_tokens = self.model.get_input_embeddings()
        embed_tokens.weight.data[self.metaquery_token_ids] = self.original_metaquery_embeds.clone()

        self.optimizer.zero_grad()
        self.optimizer.state.clear()
        self.ttt_update_count = 0
        self.time_forward = 0.0
        self.time_backward = 0.0
        self.action_variance_history = []
        self.action_variance_threshold = None

    def inference(self, image: list, instruction: str, unnorm_key: str, gap: int = 4) -> tuple:
        """Action inference. Optionally computes action variance for adaptive filtering."""
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.action_variance_enabled:
                    output_action, action_variance, _ = self._sample_actions_with_variance(
                        image, instruction, gap
                    )
                else:
                    output_action, _, _ = self.model.sample_actions(
                        caption=instruction,
                        input_images=[image],
                        num_images_per_prompt=1,
                        gap=gap,
                    )
                    action_variance = None

        action_norm_stats = self.norm_stats[unnorm_key]["action"]
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        unnorm_actions = 0.5 * (output_action + 1) * (action_high - action_low) + action_low
        unnorm_actions[..., -1] = np.where(unnorm_actions[..., -1] >= 0, 1.0, -1.0)

        ttt_cache = {
            "input_image": image,
            "instruction": instruction,
            "gap": gap,
            "action_variance": action_variance,
        }
        return unnorm_actions, ttt_cache

    @torch.no_grad()
    def _sample_actions_with_variance(self, image: list, instruction: str, gap: int) -> tuple:
        """
        Run MLLM once, then sample N action rollouts with different noise through DiT DDIM.
        Returns: (mean_action, mean_variance_scalar, None)
        """
        device = next(self.model.parameters()).device
        N = self.action_variance_n_samples

        # Step 1: MLLM forward (single pass)
        tokenize_func = self.model.get_tokenize_fn()
        tokenizer = self.model.get_tokenizer()

        input_ids, attention_mask, pixel_values, image_sizes, _ = tokenize_func(
            tokenizer, [instruction], [gap], [[img for img in image]], training_mode="action"
        )
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        pixel_values = pixel_values.to(device) if pixel_values is not None else None
        image_sizes = image_sizes.to(device) if image_sizes is not None else None

        mllm_output = self.model.model.mllm_backbone(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_sizes,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        action_prompt_embeds = self.model.model.encode_condition_action(
            input_ids=input_ids,
            mllm_output=mllm_output,
        )
        cognition_features = action_prompt_embeds

        # Step 2: Batch N noise samples through DiT DDIM
        model_dtype = next(self.model.model.policy_head.net.parameters()).dtype
        cognition_features = cognition_features.to(model_dtype)
        cognition_batched = cognition_features.repeat(N, 1, 1)

        noise_batched = torch.randn(
            N,
            self.model.model.policy_head.future_action_window_size,
            self.model.model.policy_head.in_channels,
            device=device
        ).to(model_dtype)

        model_kwargs = dict(z=cognition_batched)
        sample_fn = self.model.model.policy_head.net.forward

        if self.model.model.policy_head.ddim_diffusion is None:
            self.model.model.policy_head.create_ddim(ddim_step=10)

        samples = self.model.model.policy_head.ddim_diffusion.ddim_sample_loop(
            sample_fn,
            noise_batched.shape,
            noise_batched,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
            eta=0.0
        )

        # Step 3: Compute variance and mean action
        actions_np = samples.cpu().numpy()
        mean_action = actions_np.mean(axis=0)
        action_variance = actions_np.var(axis=0).mean()

        return mean_action, float(action_variance), None

    def should_skip_by_variance(self, action_variance: float) -> bool:
        """
        Determine whether to skip TTT update based on action variance.
        Uses adaptive percentile threshold over running history (variance buffer).
        """
        if not self.action_variance_enabled or action_variance is None:
            return False

        self.action_variance_history.append(action_variance)

        # Warmup phase: collect statistics, never skip
        if len(self.action_variance_history) < self.action_variance_warmup:
            return False

        # Compute threshold from entire history
        sorted_variances = sorted(self.action_variance_history)
        idx = min(int(self.action_variance_percentile * len(sorted_variances)), len(sorted_variances) - 1)
        self.action_variance_threshold = sorted_variances[idx]

        return action_variance > self.action_variance_threshold

    def ttt_update_batch(self, ttt_batch: list) -> float:
        """
        Compute image prediction loss and update metaquery embeddings.

        Args:
            ttt_batch: List of (ttt_cache, target_image_tensor) tuples
        Returns:
            loss value (averaged over batch)
        """
        if len(ttt_batch) == 0:
            return 0.0

        input_images = [item[0]["input_image"] for item in ttt_batch]
        instructions = [item[0]["instruction"] for item in ttt_batch]
        gaps = [item[0]["gap"] for item in ttt_batch]
        target_tensors = [item[1] for item in ttt_batch]

        # Prepare targets for VAE (expects [-1, 1])
        targets = torch.stack([(t * 2.0 - 1.0) for t in target_tensors]).to(self.device)
        bsz = targets.shape[0]

        if self.ttt_time_profile:
            torch.cuda.synchronize()
            forward_start = time.time()

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # VAE encode target latents (no grad)
            with torch.no_grad():
                latents = self.model.vae.encode(targets).latent
                if "shift_factor" in self.model.vae.config and self.model.vae.config.shift_factor is not None:
                    latents = latents - self.model.vae.config.shift_factor
                latents = latents * self.model.vae.config.scaling_factor

            # Tokenize batch
            tokenize_func = self.model.get_tokenize_fn()
            tokenizer = self.model.get_tokenizer()
            input_ids, attention_mask, pixel_values, image_sizes, _ = tokenize_func(
                tokenizer, instructions, gaps, input_images, training_mode="image"
            )
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            pixel_values = pixel_values.to(self.device) if pixel_values is not None else None
            image_sizes = image_sizes.to(self.device) if image_sizes is not None else None

            # MLLM forward (gradient flows through metaquery embeddings)
            mllm_output = self.model.model.mllm_backbone(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_sizes,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

            # Connector: encode condition for image prediction
            prompt_embeds, attn_mask = self.model.model.encode_condition(
                input_ids=input_ids,
                attention_mask=attention_mask,
                mllm_output=mllm_output,
            )

            # Flow matching loss
            noise = torch.randn_like(latents)
            u = compute_density_for_timestep_sampling("uniform", batch_size=bsz, logit_mean=0.0, logit_std=1.0, mode_scale=1.29)
            indices = (u * self.model.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.model.noise_scheduler.timesteps[indices].to(self.device)
            sigmas = self.model.get_sigmas(timesteps, self.device, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise

            # SanaTransformer forward
            model_pred = self.model.model(
                x=noisy_latents,
                timestep=timesteps,
                prompt_embeds=prompt_embeds,
                attention_mask=attn_mask,
            )

            target_for_loss = noise - latents
            weighting = compute_loss_weighting_for_sd3("uniform", sigmas=sigmas)
            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target_for_loss.float()) ** 2).reshape(bsz, -1),
                dim=1,
            ).mean()

        if self.ttt_time_profile:
            torch.cuda.synchronize()
            self.time_forward += time.time() - forward_start

        loss_value = loss.item()

        if self.ttt_time_profile:
            torch.cuda.synchronize()
            backward_start = time.time()

        # Backward and update (only metaquery embeddings receive gradients)
        loss.backward()

        embed_tokens = self.model.get_input_embeddings()
        mask = torch.ones(embed_tokens.weight.shape[0], dtype=torch.bool, device=embed_tokens.weight.device)
        mask[self.metaquery_token_ids] = False
        embed_tokens.weight.grad[mask] = 0

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.ttt_update_count += 1

        if self.ttt_time_profile:
            torch.cuda.synchronize()
            self.time_backward += time.time() - backward_start

        torch.cuda.empty_cache()
        return loss_value

    def get_timing_stats(self) -> dict:
        stats = {
            "time_forward": round(self.time_forward, 3),
            "time_backward": round(self.time_backward, 3),
            "update_count": self.ttt_update_count,
        }
        if self.action_variance_enabled:
            stats["action_variance_threshold"] = self.action_variance_threshold
            stats["action_variance_history_len"] = len(self.action_variance_history)
            if self.action_variance_history:
                stats["action_variance_mean"] = round(float(np.mean(self.action_variance_history)), 6)
        return stats


class MantisVLATTTOOD(MantisVLATTT):
    """
    OOD variant of MantisVLATTT.
    - checkpoints_dir is optional (OOD models use pretrained weights directly)
    - Gripper action post-processing is inverted (>= 0.5 → -1.0, else 1.0)
    """

    def __init__(
        self,
        model_id: str,
        checkpoints_dir: str,
        norm_file_path: str,
        ttt_lr: float = 1e-4,
        ttt_time_profile: bool = False,
        target_image_size: int = 256,
        vae_downsample_f: int = 32,
        action_variance_enabled: bool = False,
        action_variance_n_samples: int = 5,
        action_variance_warmup: int = 10,
        action_variance_percentile: float = 0.7,
    ):
        self.device = "cuda"
        self.ttt_time_profile = ttt_time_profile

        self.action_variance_enabled = action_variance_enabled
        self.action_variance_n_samples = action_variance_n_samples
        self.action_variance_warmup = action_variance_warmup
        self.action_variance_percentile = action_variance_percentile
        self.action_variance_history = []
        self.action_variance_threshold = None

        input_size = target_image_size // vae_downsample_f

        # OOD: checkpoints_dir is optional
        self.model = Mantis.from_pretrained(model_id, input_size=input_size, ignore_mismatched_sizes=True)
        if checkpoints_dir:
            state_dict = torch.load(f"{checkpoints_dir}/model.pt", map_location='cpu')
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)

        self._enable_gradient_checkpointing()

        with open(norm_file_path, "r") as f:
            self.norm_stats = json.load(f)

        for param in self.model.parameters():
            param.requires_grad = False
        self._init_ttt(ttt_lr)

        self.ttt_update_count = 0
        self.time_forward = 0.0
        self.time_backward = 0.0

    def inference(self, image: list, instruction: str, unnorm_key: str, gap: int = 4) -> tuple:
        """Action inference with OOD gripper post-processing."""
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if self.action_variance_enabled:
                    output_action, action_variance, _ = self._sample_actions_with_variance(
                        image, instruction, gap
                    )
                else:
                    output_action, _, _ = self.model.sample_actions(
                        caption=instruction,
                        input_images=[image],
                        num_images_per_prompt=1,
                        gap=gap,
                    )
                    action_variance = None

        # OOD gripper post-processing: >= 0.5 → -1.0, else 1.0
        action_norm_stats = self.norm_stats[unnorm_key]["action"]
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        unnorm_actions = 0.5 * (output_action + 1) * (action_high - action_low) + action_low
        unnorm_actions[..., -1] = np.where(unnorm_actions[..., -1] >= 0.5, -1.0, 1.0)

        ttt_cache = {
            "input_image": image,
            "instruction": instruction,
            "gap": gap,
            "action_variance": action_variance,
        }
        return unnorm_actions, ttt_cache
