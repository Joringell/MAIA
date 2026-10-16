import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from accelerate import DistributedType
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from einops import rearrange
from tqdm import tqdm

from runner.ge_trainer_base import BaseGETrainer, logger
from utils import save_video
from utils.data_utils import (
    apply_color_jitter_to_video,
    gen_noise_from_condition_frame_latent,
    get_latents,
    get_text_conditions,
)
from utils.memory_utils import free_memory, get_memory_statistics
from utils.model_utils import forward_pass, unwrap_model
from utils.optimizer_utils import get_optimizer


def _masked_mean(loss_per_sample: torch.Tensor, sample_mask: torch.Tensor) -> torch.Tensor:
    sample_mask = sample_mask.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype).reshape(-1)
    if sample_mask.shape[0] != loss_per_sample.shape[0]:
        raise ValueError(
            f"Mask batch size {sample_mask.shape[0]} does not match loss batch size {loss_per_sample.shape[0]}"
        )
    return (loss_per_sample * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)


def _interaction_language_relational_loss(
    interaction_semantic: torch.Tensor,
    language_semantic: torch.Tensor,
    valid_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Weakly match pairwise interaction similarities to frozen language similarities."""
    interaction_semantic = interaction_semantic.float()
    language_semantic = language_semantic.float().detach()
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=interaction_semantic.device, dtype=torch.bool).reshape(-1)
        interaction_semantic = interaction_semantic[valid_mask]
        language_semantic = language_semantic[valid_mask]
    if interaction_semantic.shape[0] < 2:
        return torch.zeros((), device=interaction_semantic.device, dtype=interaction_semantic.dtype)

    z_int = F.normalize(interaction_semantic, dim=-1)
    z_lang = F.normalize(language_semantic, dim=-1)
    sim_int = z_int @ z_int.transpose(0, 1)
    sim_lang = z_lang @ z_lang.transpose(0, 1)
    off_diag = ~torch.eye(sim_int.shape[0], device=sim_int.device, dtype=torch.bool)
    return F.mse_loss(sim_int[off_diag], sim_lang[off_diag])


def _slice_batch(batch, start: int, end: int):
    sliced = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            sliced[key] = value[start:end]
        elif isinstance(value, list):
            sliced[key] = value[start:end]
        elif isinstance(value, tuple):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _resolve_force_layout(args):
    diffusion_cfg = args.diffusion_model["config"]
    force_dim = int(args.data["train"].get("force_dim", diffusion_cfg.get("force_out_channels", 0) or 0))
    action_dim = args.data["train"].get("action_dim")
    if action_dim is None:
        action_dim = diffusion_cfg.get("action_condition_dim", diffusion_cfg.get("action_dim"))
    if action_dim is None:
        raise ValueError("action_dim or action_condition_dim must be provided.")
    if int(action_dim) <= 0 or force_dim <= 0:
        raise ValueError(f"action and force dimensions must be positive, got {action_dim} and {force_dim}")
    return int(action_dim), int(force_dim)


class Trainer(BaseGETrainer):
    """Trainer for LTX video backbone + action/force adapter.

    Expected dataset: CustomLeRobotChunkTraverseDataset.
    Batch keys used:
      video: [B, C, V, T_mem+T_video, H, W]
      actions: [B, T_mem+T_action, action_dim]
      force_targets: [B, T_action, force_dim] with force_mode="last"
      history_force_state: [B, Hf, force_dim]
      label: scalar window label where 1 means normal/success.
    """

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        super().__init__(config_file=config_file, to_log=to_log, output_dir=output_dir)
        self.action_condition_dim, self.force_dim = _resolve_force_layout(self.args)
        if not getattr(self.args, "return_force", False):
            raise ValueError("Action-force trainer requires return_force=True.")
        if getattr(self.args, "return_action", False):
            raise ValueError("Action-force trainer uses actions as conditions; it does not predict actions.")

        diffusion_cfg = self.args.diffusion_model["config"]
        diffusion_cfg["use_action_force_adapter"] = True
        diffusion_cfg["action_dim"] = self.action_condition_dim
        diffusion_cfg["action_condition_dim"] = self.action_condition_dim
        diffusion_cfg["force_dim"] = self.force_dim
        diffusion_cfg["force_in_channels"] = self.force_dim
        diffusion_cfg.setdefault("force_out_channels", self.force_dim)
        diffusion_cfg.setdefault("out_channels", diffusion_cfg.get("in_channels", 128))
        if int(diffusion_cfg.get("force_out_channels", self.force_dim)) != self.force_dim:
            logger.warning(
                "override force_out_channels: "
                f"{diffusion_cfg.get('force_out_channels')} -> {self.force_dim}"
            )
            diffusion_cfg["force_out_channels"] = self.force_dim

    def prepare_trainable_parameters(self):
        super().prepare_trainable_parameters()
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)

    def _prepare_action_force_batch(self, batch, accelerator, weight_dtype):
        video = batch["video"].to(accelerator.device, dtype=weight_dtype, non_blocking=True).contiguous()
        batch_size, _, n_view, _, _, _ = video.shape
        video = rearrange(video, "b c v t h w -> (b v) c t h w")
        if self.args.use_color_jitter:
            video = apply_color_jitter_to_video(video)

        mem_size = int(self.args.data["train"]["n_previous"])
        action_chunk = int(self.args.data["train"].get("action_chunk", self.args.data["train"].get("future_steps")))
        mem = video[:, :, :mem_size]
        future_video = video[:, :, mem_size:]

        future_actions = batch["actions"][:, -action_chunk:].to(
            accelerator.device,
            dtype=weight_dtype,
            non_blocking=True,
        ).contiguous()
        if future_actions.shape[-1] != self.action_condition_dim:
            raise ValueError(f"Expected action dim {self.action_condition_dim}, got {future_actions.shape[-1]}")

        force_targets = batch["force_targets"].to(
            accelerator.device,
            dtype=weight_dtype,
            non_blocking=True,
        ).contiguous()
        if force_targets.shape[-1] != self.force_dim:
            raise ValueError(f"Expected force dim {self.force_dim}, got {force_targets.shape[-1]}")
        if force_targets.shape[1] != future_actions.shape[1]:
            raise ValueError(f"Action/force lengths differ: {future_actions.shape[1]} vs {force_targets.shape[1]}")

        history_force_states = batch.get("history_force_state")
        if history_force_states is None:
            raise ValueError("history_force_state is required for action-force WM training.")
        history_force_states = history_force_states.to(
            accelerator.device,
            dtype=weight_dtype,
            non_blocking=True,
        ).contiguous()
        if history_force_states.shape[-1] != self.force_dim:
            raise ValueError(
                f"Expected history force dim {self.force_dim}, got {history_force_states.shape[-1]}"
            )

        if self.label_filter == "only_normal":
            if "label" not in batch:
                raise KeyError("label_filter='only_normal' requires batch['label']")
            normal_mask = batch["label"].to(accelerator.device, dtype=torch.float32, non_blocking=True).reshape(-1)
        elif self.label_filter == "all":
            normal_mask = torch.ones(batch_size, device=accelerator.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported label_filter={self.label_filter}; use all or only_normal")

        return (
            video,
            mem,
            future_video,
            future_actions,
            force_targets,
            history_force_states,
            normal_mask,
            batch_size,
            n_view,
            mem_size,
        )

    def prepare_val_dataset(self) -> None:
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        if not hasattr(self.args, "val_data_class_path"):
            self.args.val_data_class_path = self.args.train_data_class_path
        return super().prepare_val_dataset()

    def prepare_optimizer(self):
        logger.info("Initializing action-force WM optimizer and lr scheduler")

        train_mode = self.args.train_mode
        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps

        if self.args.mixed_precision == "fp16":
            cast_training_params([self.diffusion_model], dtype=torch.float32)

        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        adapter_prefixes = (
            "action_condition_encoder",
            "force_history_encoder",
            "force_context_encoder",
            "action_force_context",
            "af_adapters",
            "interaction_readout",
            "force_decoder",
            "interaction_semantic_proj",
        )
        diffusion_model_trainable_params = []
        for name, param in self.diffusion_model.named_parameters():
            if train_mode == "all":
                param.requires_grad = True
            elif train_mode == "adapter_only":
                param.requires_grad = name.startswith(adapter_prefixes)
            elif train_mode == "video_only":
                param.requires_grad = not name.startswith(adapter_prefixes)
            else:
                raise NotImplementedError(
                    f"unsupported train_mode={train_mode}; use all, adapter_only, or video_only"
                )
            if param.requires_grad:
                diffusion_model_trainable_params.append(param)

        self.state.num_trainable_parameters = sum(p.numel() for p in diffusion_model_trainable_params)
        logger.info(f"Total trainable parameters: {self.state.num_trainable_parameters}")

        self.optimizer = get_optimizer(
            params_to_optimize=[{"params": diffusion_model_trainable_params, "lr": self.state.learning_rate}],
            optimizer_name=self.args.optimizer,
            learning_rate=self.args.lr,
            beta1=self.args.beta1,
            beta2=self.args.beta2,
            beta3=self.args.beta3,
            epsilon=self.args.epsilon,
            weight_decay=self.args.weight_decay,
            use_8bit=self.args.optimizer_8bit,
            use_torchao=self.args.optimizer_torchao,
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.args.gradient_accumulation_steps)
        if self.state.train_steps is None:
            self.state.train_steps = self.state.train_epochs * num_update_steps_per_epoch
            self.state.overwrote_max_train_steps = True

        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

    def _make_val_pipe(self):
        pipe = self.pipeline_class(
            self.scheduler,
            self.vae,
            self.text_encoder,
            self.tokenizer,
            unwrap_model(self.state.accelerator, self.diffusion_model),
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe

    @staticmethod
    def _save_force_curve(save_path, pred_force, gt_force, title):
        pred_force = pred_force.detach().cpu().float()
        gt_force = gt_force.detach().cpu().float()
        force_dim = pred_force.shape[-1]
        n_rows = min(3, force_dim)
        n_cols = int(math.ceil(force_dim / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 2.2 * n_rows), squeeze=False)
        steps = torch.arange(pred_force.shape[0]).numpy()
        for dim in range(force_dim):
            ax = axes[dim // n_cols][dim % n_cols]
            ax.plot(steps, gt_force[:, dim].numpy(), label="gt", linewidth=1.5)
            ax.plot(steps, pred_force[:, dim].numpy(), label="pred", linewidth=1.2)
            ax.set_title(f"force {dim}")
            ax.grid(True, alpha=0.25)
        for dim in range(force_dim, n_rows * n_cols):
            axes[dim // n_cols][dim % n_cols].axis("off")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right")
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    def validate(self, accelerator, output_dir, global_step):
        if self.val_dataloader is None:
            logger.warning("Skipping validation because val_dataloader is None")
            return
        os.makedirs(output_dir, exist_ok=True)
        self.diffusion_model.eval()
        pipe = self._make_val_pipe()

        val_num_samples = max(1, int(getattr(self.args, "val_num_samples", 4)))
        val_num_inference_steps = int(
            getattr(self.args, "val_num_inference_steps", getattr(self.args, "num_inference_step", 5))
        )
        val_guidance_scale = float(getattr(self.args, "val_guidance_scale", 1.0))
        val_decode_timestep = float(getattr(self.args, "val_decode_timestep", 0.03))
        val_decode_noise_scale = float(getattr(self.args, "val_decode_noise_scale", 0.025))
        val_noise_seed = int(getattr(self.args, "val_noise_seed", getattr(self.args, "seed", 42) or 42))
        n_previous = int(self.args.data["train"]["n_previous"])
        action_chunk = int(self.args.data["train"].get("action_chunk", self.args.data["train"]["future_steps"]))
        latent_chunk = (int(self.args.data["train"]["chunk"]) - 1) // int(self.TEMPORAL_DOWN_RATIO) + 1
        fps = int(
            getattr(self.args, "basic_fps", 30)
            / max(1, int(self.args.data["train"]["action_chunk"]) // int(self.args.data["train"]["chunk"]))
        )

        saved = 0
        force_mse_values = []
        force_delta_mse_values = []
        for batch in self.val_dataloader:
            if saved >= val_num_samples:
                break
            batch_size = batch["video"].shape[0]
            for sample_idx in range(batch_size):
                if saved >= val_num_samples:
                    break
                sample = _slice_batch(batch, sample_idx, sample_idx + 1)
                video = sample["video"].to(accelerator.device, dtype=self.state.weight_dtype, non_blocking=True).contiguous()
                _, _, n_view, _, height, width = video.shape
                image = video[:, :, :, :n_previous]
                image = rearrange(image, "b c v t h w -> (b v) c t h w")
                future_actions = sample["actions"][:, -action_chunk:].to(
                    accelerator.device,
                    dtype=self.state.weight_dtype,
                    non_blocking=True,
                ).contiguous()
                history_force_states = sample["history_force_state"].to(
                    accelerator.device,
                    dtype=self.state.weight_dtype,
                    non_blocking=True,
                ).contiguous()

                if hasattr(pipe.scheduler, "_step_index"):
                    pipe.scheduler._step_index = None
                with torch.no_grad():
                    preds = pipe.infer(
                        image=image,
                        prompt=sample["caption"],
                        negative_prompt="",
                        num_inference_steps=val_num_inference_steps,
                        decode_timestep=val_decode_timestep,
                        decode_noise_scale=val_decode_noise_scale,
                        guidance_scale=val_guidance_scale,
                        height=height,
                        width=width,
                        n_view=n_view,
                        return_force=True,
                        n_prev=n_previous,
                        chunk=latent_chunk,
                        return_video=True,
                        return_init=False,
                        noise_seed=val_noise_seed + saved,
                        action_chunk=action_chunk,
                        history_force_state=history_force_states,
                        pixel_wise_timestep=bool(getattr(self.args, "pixel_wise_timestep", True)),
                        n_chunk=1,
                        force_dim=self.force_dim,
                        future_actions=future_actions,
                        show_progress=False,
                    )[0]

                sample_dir = os.path.join(output_dir, f"sample_{saved:03d}")
                os.makedirs(sample_dir, exist_ok=True)
                if "video" in preds and preds["video"] is not None:
                    pred_video = preds["video"].detach().cpu()
                    save_video(
                        rearrange(pred_video, "(b v) c t h w -> b c t h (v w)", v=n_view)[0],
                        os.path.join(sample_dir, "pred.mp4"),
                        fps=fps,
                    )
                    gt_future = sample["video"][:, :, :, n_previous:].detach().cpu()
                    save_video(
                        rearrange(gt_future[0], "c v t h w -> c t h (v w)", v=n_view),
                        os.path.join(sample_dir, "gt_future.mp4"),
                        fps=fps,
                    )

                pred_force = preds["force"].detach().float()[:, :action_chunk]
                gt_force = sample["force_targets"].to(pred_force.device, dtype=pred_force.dtype)[:, -action_chunk:]
                valid_len = min(pred_force.shape[1], gt_force.shape[1])
                pred_force = pred_force[:, :valid_len]
                gt_force = gt_force[:, :valid_len]
                force_mse = (pred_force - gt_force).pow(2).mean()
                force_mse_values.append(force_mse.detach().cpu())
                if valid_len > 1:
                    force_delta_mse = (
                        (pred_force[:, 1:] - pred_force[:, :-1])
                        - (gt_force[:, 1:] - gt_force[:, :-1])
                    ).pow(2).mean()
                    force_delta_mse_values.append(force_delta_mse.detach().cpu())
                self._save_force_curve(
                    os.path.join(sample_dir, "force.png"),
                    pred_force[0],
                    gt_force[0],
                    title=f"step {global_step} sample {saved}",
                )
                saved += 1

        if force_mse_values:
            val_force_mse = torch.stack(force_mse_values).mean().item()
            accelerator.log({"val_force_mse": val_force_mse}, step=global_step)
            if self.writer is not None:
                self.writer.add_scalar("Validation/force_mse", val_force_mse, global_step)
        if force_delta_mse_values:
            val_force_delta_mse = torch.stack(force_delta_mse_values).mean().item()
            accelerator.log({"val_force_delta_mse": val_force_delta_mse}, step=global_step)
            if self.writer is not None:
                self.writer.add_scalar("Validation/force_delta_mse", val_force_delta_mse, global_step)

        del pipe
        self.diffusion_model.train()

    def train(self):
        logger.info("Starting action-force WM training")
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory before training start: {json.dumps(memory_statistics, indent=4)}")

        self.state.train_batch_size = (
            self.args.batch_size * self.state.accelerator.num_processes * self.args.gradient_accumulation_steps
        )
        logger.info(
            "Training configuration: "
            + json.dumps(
                {
                    "trainable parameters": self.state.num_trainable_parameters,
                    "total samples": len(self.train_dataset),
                    "train epochs": self.state.train_epochs,
                    "train steps": self.state.train_steps,
                    "batches per device": self.args.batch_size,
                    "total batches observed per epoch": len(self.train_dataloader),
                    "train batch size": self.state.train_batch_size,
                    "gradient accumulation steps": self.args.gradient_accumulation_steps,
                },
                indent=4,
            )
        )

        global_step = int(getattr(self.args, "start_step", 0) or 0)
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=global_step,
            desc="Training steps",
            disable=not self.state.accelerator.is_local_main_process,
        )
        if global_step > 0:
            logger.info(f"Resuming from start_step={global_step}")
            try:
                self.lr_scheduler.step(global_step)
            except TypeError:
                for _ in range(global_step):
                    self.lr_scheduler.step()

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)

        for epoch in range(self.state.train_epochs):
            if global_step >= self.state.train_steps:
                break

            self.diffusion_model.train()
            running_loss = 0.0

            for step, batch in enumerate(self.train_dataloader):
                del step
                with accelerator.accumulate([self.diffusion_model]):
                    (
                        video,
                        mem,
                        future_video,
                        future_actions,
                        force_targets,
                        history_force_states,
                        normal_mask,
                        batch_size,
                        n_view,
                        mem_size,
                    ) = self._prepare_action_force_batch(batch, accelerator, weight_dtype)

                    _, _, raw_frames, raw_height, raw_width = future_video.shape
                    latent_frames = raw_frames // self.TEMPORAL_DOWN_RATIO + 1 + mem_size
                    latent_height = raw_height // self.SPATIAL_DOWN_RATIO
                    latent_width = raw_width // self.SPATIAL_DOWN_RATIO

                    dropout_factor = torch.rand(batch_size, device=accelerator.device, dtype=weight_dtype)
                    dropout_mask_prompt = (dropout_factor < self.args.caption_dropout_p).unsqueeze(1).unsqueeze(2)

                    mem_latents, future_video_latents = get_latents(self.vae, mem, future_video)
                    mem_latents = rearrange(
                        mem_latents,
                        "(b v m) (h w) c -> (b v) c m h w",
                        b=batch_size,
                        m=mem_size,
                        h=latent_height,
                    )
                    future_video_latents = rearrange(
                        future_video_latents,
                        "(b v) (f h w) c -> (b v) c f h w",
                        b=batch_size,
                        h=latent_height,
                        w=latent_width,
                    )
                    latents = torch.cat((mem_latents, future_video_latents), dim=2)
                    latents = rearrange(latents, "bv c f h w -> bv (f h w) c")

                    text_conds = get_text_conditions(self.tokenizer, self.text_encoder, batch["caption"])
                    prompt_embeds = text_conds["prompt_embeds"]
                    prompt_attention_mask = text_conds["prompt_attention_mask"]
                    prompt_embeds = (
                        self.uncond_prompt_embeds.repeat(batch_size, 1, 1) * dropout_mask_prompt
                        + prompt_embeds * ~dropout_mask_prompt
                    )

                    density = compute_density_for_timestep_sampling(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        batch_size=batch_size,
                        logit_mean=self.args.flow_logit_mean,
                        logit_std=self.args.flow_logit_std,
                        mode_scale=self.args.flow_mode_scale,
                    )

                    if self.args.noisy_video:
                        video_density = torch.zeros_like(density).unsqueeze(1).repeat(1, n_view)
                    else:
                        video_density = density.unsqueeze(1).repeat(1, n_view)
                    indices = (rearrange(video_density, "b v -> (b v)") * self.scheduler.config.num_train_timesteps).long()
                    sigmas = scheduler_sigmas[indices]
                    timesteps = (sigmas * 1000.0).long()

                    noise, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                        mem_latents,
                        latent_frames,
                        latent_height,
                        latent_width,
                        noise_to_condition_frames=self.args.noise_to_first_frame,
                    )
                    if self.args.pixel_wise_timestep:
                        timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask)
                    else:
                        timesteps = timesteps.unsqueeze(-1) * (1 - cond_indicator)

                    ss = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
                    if self.args.noisy_video:
                        ss = torch.full_like(ss, 1.0)
                    noisy_latents = (1.0 - ss) * latents + ss * noise
                    video_weights = compute_loss_weighting_for_sd3(
                        weighting_scheme=self.args.flow_weighting_scheme,
                        sigmas=sigmas,
                    ).reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))

                    pred_all = forward_pass(
                        model=self.diffusion_model,
                        timesteps=timesteps,
                        noisy_latents=noisy_latents,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        num_frames=latent_frames,
                        height=latent_height,
                        width=latent_width,
                        n_view=n_view,
                        future_action_states=future_actions,
                        history_force_states=history_force_states,
                        return_video=self.args.return_video,
                        return_force=self.args.return_force,
                        return_interaction=(
                            getattr(self.args, "return_interaction", False)
                            or float(getattr(self.args, "interaction_semantic_loss_scale", 0.0)) > 0.0
                        ),
                        video_attention_mask=None,
                    )["latents"]

                    if self.args.return_video:
                        pred_video = pred_all["video"]
                        target_video = noise - latents
                        loss_video = video_weights.float() * (pred_video.float() - target_video.float()).pow(2)
                        loss_video = loss_video * (
                            1 - conditioning_mask.unsqueeze(-1).repeat(1, 1, loss_video.size(-1))
                        )
                        loss_video = loss_video.mean(dim=(1, 2))
                        video_sample_mask = normal_mask.repeat_interleave(n_view)
                        loss_video = _masked_mean(loss_video, video_sample_mask)
                    else:
                        loss_video = torch.zeros((), device=accelerator.device, dtype=torch.float32)

                    pred_force = pred_all["force"].float()
                    loss_force = (pred_force - force_targets.float()).pow(2).mean(dim=(1, 2))
                    loss_force = _masked_mean(loss_force, normal_mask)

                    if pred_force.shape[1] > 1:
                        pred_delta = pred_force[:, 1:] - pred_force[:, :-1]
                        gt_delta = force_targets.float()[:, 1:] - force_targets.float()[:, :-1]
                        loss_force_delta = (pred_delta - gt_delta).pow(2).mean(dim=(1, 2))
                        loss_force_delta = _masked_mean(loss_force_delta, normal_mask)
                    else:
                        loss_force_delta = torch.zeros((), device=accelerator.device, dtype=torch.float32)

                    if pred_force.shape[1] > 2:
                        pred_second_delta = pred_force[:, 2:] - 2 * pred_force[:, 1:-1] + pred_force[:, :-2]
                        loss_force_smooth = pred_second_delta.pow(2).mean(dim=(1, 2))
                        loss_force_smooth = _masked_mean(loss_force_smooth, normal_mask)
                    else:
                        loss_force_smooth = torch.zeros((), device=accelerator.device, dtype=torch.float32)

                    interaction_semantic_loss_scale = float(
                        getattr(self.args, "interaction_semantic_loss_scale", 0.0)
                    )
                    if interaction_semantic_loss_scale > 0.0:
                        if "interaction_semantic" not in pred_all or "language_semantic" not in pred_all:
                            raise KeyError(
                                "interaction_semantic_loss_scale > 0 requires model outputs "
                                "'interaction_semantic' and 'language_semantic'."
                            )
                        semantic_valid_mask = (
                            ~dropout_mask_prompt.reshape(batch_size)
                        ) & (normal_mask.reshape(batch_size) > 0)
                        loss_interaction_semantic = _interaction_language_relational_loss(
                            pred_all["interaction_semantic"],
                            pred_all["language_semantic"],
                            valid_mask=semantic_valid_mask,
                        )
                    else:
                        loss_interaction_semantic = torch.zeros((), device=accelerator.device, dtype=torch.float32)

                    force_loss_scale = float(getattr(self.args, "force_loss_scale", 1.0))
                    video_loss_scale = float(getattr(self.args, "video_loss_scale", 1.0))
                    force_delta_loss_scale = float(getattr(self.args, "force_delta_loss_scale", 0.0))
                    force_smooth_loss_scale = float(getattr(self.args, "force_smooth_loss_scale", 0.0))
                    loss = (
                        video_loss_scale * loss_video
                        + force_loss_scale * loss_force
                        + force_delta_loss_scale * loss_force_delta
                        + force_smooth_loss_scale * loss_force_smooth
                        + interaction_semantic_loss_scale * loss_interaction_semantic
                    )

                    assert not torch.isnan(loss), "NaN loss detected"
                    grad_norm_value = None
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm_value = accelerator.clip_grad_norm_(
                            self.diffusion_model.parameters(), self.args.max_grad_norm
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                loss = accelerator.reduce(loss.detach(), reduction="mean")
                loss_video = accelerator.reduce(loss_video.detach(), reduction="mean")
                loss_force = accelerator.reduce(loss_force.detach(), reduction="mean")
                loss_force_delta = accelerator.reduce(loss_force_delta.detach(), reduction="mean")
                loss_force_smooth = accelerator.reduce(loss_force_smooth.detach(), reduction="mean")
                loss_interaction_semantic = accelerator.reduce(loss_interaction_semantic.detach(), reduction="mean")
                normal_ratio = accelerator.reduce(normal_mask.detach().mean(), reduction="mean")

                running_loss += loss.item()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {
                    "loss": loss.item(),
                    "loss_video": loss_video.item(),
                    "loss_force": loss_force.item(),
                    "normal_ratio": normal_ratio.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                if force_delta_loss_scale > 0:
                    logs["loss_force_delta"] = loss_force_delta.item()
                if force_smooth_loss_scale > 0:
                    logs["loss_force_smooth"] = loss_force_smooth.item()
                if interaction_semantic_loss_scale > 0:
                    logs["loss_interaction_semantic"] = loss_interaction_semantic.item()
                if grad_norm_value is not None:
                    logs["grad_norm"] = grad_norm_value.item() if hasattr(grad_norm_value, "item") else grad_norm_value
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step % self.args.steps_to_log == 0 and accelerator.is_main_process and self.writer is not None:
                    self.writer.add_scalar("Training Loss", loss.item(), global_step)
                    self.writer.add_scalar("Video loss", loss_video.item(), global_step)
                    self.writer.add_scalar("Force loss", loss_force.item(), global_step)
                    if interaction_semantic_loss_scale > 0:
                        self.writer.add_scalar(
                            "Interaction semantic loss", loss_interaction_semantic.item(), global_step
                        )
                    self.writer.add_scalar("Normal ratio", normal_ratio.item(), global_step)

                if global_step % self.args.steps_to_save == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_to_save = unwrap_model(accelerator, self.diffusion_model)
                        model_save_dir = os.path.join(self.save_folder, f"step_{global_step}")
                        model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                        del model_to_save
                if (
                    getattr(self.args, "steps_to_val", None) is not None
                    and int(self.args.steps_to_val) > 0
                    and global_step % int(self.args.steps_to_val) == 0
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        val_dir = os.path.join(self.save_folder, f"step_{global_step}", "validation")
                        self.validate(accelerator, val_dir, global_step)
                    accelerator.wait_for_everyone()

                if global_step >= self.state.train_steps:
                    logger.info(">>> max train step reached")
                    break

            if accelerator.is_main_process and self.writer is not None:
                avg_loss = running_loss / max(1, len(self.train_dataloader))
                self.writer.add_scalar("Average Training Loss", avg_loss, epoch)

        progress_bar.close()
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
            model_save_dir = os.path.join(self.save_folder, f"step_{global_step}")
            self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)

        del self.diffusion_model, self.scheduler
        free_memory()
        memory_statistics = get_memory_statistics()
        logger.info(f"Memory after training end: {json.dumps(memory_statistics, indent=4)}")
        accelerator.end_training()
