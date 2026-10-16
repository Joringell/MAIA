import json
import math
import os

import torch
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


def _masked_mean(loss_per_sample: torch.Tensor, sample_mask: torch.Tensor | None) -> torch.Tensor:
    if sample_mask is None:
        return loss_per_sample.mean()
    sample_mask = sample_mask.to(device=loss_per_sample.device, dtype=loss_per_sample.dtype).reshape(-1)
    if sample_mask.shape[0] != loss_per_sample.shape[0]:
        raise ValueError(
            f"Mask batch size {sample_mask.shape[0]} does not match loss batch size {loss_per_sample.shape[0]}"
        )
    return (loss_per_sample * sample_mask).sum() / sample_mask.sum().clamp_min(1.0)


def _resolve_action_dim(args) -> int:
    diffusion_cfg = args.diffusion_model["config"]
    action_dim = args.data["train"].get("action_dim")
    if action_dim is None:
        action_dim = diffusion_cfg.get("action_condition_dim", diffusion_cfg.get("action_dim"))
    if action_dim is None or int(action_dim) <= 0:
        raise ValueError("action_dim or action_condition_dim must be provided and positive.")
    return int(action_dim)


class Trainer(BaseGETrainer):
    """Trainer for action-conditioned LTX video world modeling.

    Required batch keys:
      video: [B, C, V, T_mem+T_video, H, W]
      actions: [B, T_mem+T_action, action_dim]
      caption: list[str]

    Force is intentionally not read from the dataset and not passed to the model.
    """

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        super().__init__(config_file=config_file, to_log=to_log, output_dir=output_dir)
        if getattr(self.args, "return_force", False):
            raise ValueError("Action-only trainer must not set return_force=True.")
        if getattr(self.args, "return_action", False):
            raise ValueError("Action-conditioned GE uses actions as conditions; it does not predict actions.")
        self.action_condition_dim = _resolve_action_dim(self.args)
        diffusion_cfg = self.args.diffusion_model["config"]
        diffusion_cfg["use_action_adapter"] = True
        diffusion_cfg["action_dim"] = self.action_condition_dim
        diffusion_cfg["action_condition_dim"] = self.action_condition_dim
        diffusion_cfg.setdefault("out_channels", diffusion_cfg.get("in_channels", 128))

    def prepare_trainable_parameters(self):
        super().prepare_trainable_parameters()
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)

    def prepare_optimizer(self):
        logger.info("Initializing action-conditioned GE optimizer and lr scheduler")
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
            "action_context",
            "action_adapters",
            "interaction_readout",
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
                raise NotImplementedError(f"unsupported train_mode={train_mode}; use all, adapter_only, or video_only")
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

    def _prepare_action_batch(self, batch, accelerator, weight_dtype):
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
            accelerator.device, dtype=weight_dtype, non_blocking=True
        ).contiguous()
        if future_actions.shape[-1] != self.action_condition_dim:
            raise ValueError(f"Expected action dim {self.action_condition_dim}, got {future_actions.shape[-1]}")

        if self.label_filter == "only_normal":
            if "label" not in batch:
                raise KeyError("label_filter='only_normal' requires batch['label']")
            sample_mask = (batch["label"].to(accelerator.device, non_blocking=True).reshape(-1) == 1).float()
        else:
            sample_mask = None

        return video, mem, future_video, future_actions, sample_mask, batch_size, n_view, mem_size

    def train(self):
        logger.info("Starting action-conditioned GE training")
        logger.info(f"Memory before training start: {json.dumps(get_memory_statistics(), indent=4)}")

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

            for batch in self.train_dataloader:
                with accelerator.accumulate([self.diffusion_model]):
                    (
                        _video,
                        mem,
                        future_video,
                        future_actions,
                        sample_mask,
                        batch_size,
                        n_view,
                        mem_size,
                    ) = self._prepare_action_batch(batch, accelerator, weight_dtype)

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
                        return_video=True,
                        return_interaction=getattr(self.args, "return_interaction", False),
                        video_attention_mask=None,
                    )["latents"]

                    pred_video = pred_all["video"]
                    target_video = noise - latents
                    loss_video = video_weights.float() * (pred_video.float() - target_video.float()).pow(2)
                    loss_video = loss_video * (
                        1 - conditioning_mask.unsqueeze(-1).repeat(1, 1, loss_video.size(-1))
                    )
                    loss_video = loss_video.mean(dim=(1, 2))
                    video_sample_mask = sample_mask.repeat_interleave(n_view) if sample_mask is not None else None
                    loss_video = _masked_mean(loss_video, video_sample_mask)

                    video_loss_scale = float(getattr(self.args, "video_loss_scale", 1.0))
                    loss = video_loss_scale * loss_video
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
                running_loss += loss.item()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {
                    "loss": loss.item(),
                    "loss_video": loss_video.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                if grad_norm_value is not None:
                    logs["grad_norm"] = grad_norm_value.item() if hasattr(grad_norm_value, "item") else grad_norm_value
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(logs)

                if global_step >= self.state.train_steps:
                    break

                if global_step % self.args.steps_to_log == 0 and accelerator.is_main_process and self.writer is not None:
                    self.writer.add_scalar("Training Loss", loss.item(), global_step)
                    self.writer.add_scalar("Video loss", loss_video.item(), global_step)

                if global_step % self.args.steps_to_val == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_save_dir = os.path.join(self.save_folder, f"Validation_step_{global_step}")
                        self.validate(accelerator, model_save_dir, global_step, n_view=n_view, n_chunk=1)

                if global_step % self.args.steps_to_save == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        model_to_save = unwrap_model(accelerator, self.diffusion_model)
                        model_save_dir = os.path.join(self.save_folder, f"step_{global_step}")
                        model_to_save.save_pretrained(model_save_dir, safe_serialization=True)
                        del model_to_save

            if accelerator.is_main_process and self.writer is not None:
                self.writer.add_scalar("Average Training Loss", running_loss / max(1, len(self.train_dataloader)), epoch)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            self.diffusion_model = unwrap_model(accelerator, self.diffusion_model)
            model_save_dir = os.path.join(self.save_folder, f"step_{global_step}")
            self.diffusion_model.save_pretrained(model_save_dir, safe_serialization=True)

        del self.diffusion_model, self.scheduler
        free_memory()
        logger.info(f"Memory after training end: {json.dumps(get_memory_statistics(), indent=4)}")
        accelerator.end_training()

    def validate(self, accelerator, model_save_dir, global_step, n_view=1, n_chunk=1, to_log=True, **kwargs):
        del kwargs
        os.makedirs(model_save_dir, exist_ok=True)
        pipe = self.pipeline_class(
            self.scheduler,
            self.vae,
            self.text_encoder,
            self.tokenizer,
            unwrap_model(accelerator, self.diffusion_model) if accelerator is not None else self.diffusion_model,
        )
        pipe.set_progress_bar_config(disable=True)

        batch = next(iter(self.val_dataloader))
        image = batch["video"][:, :, :, : self.args.data["train"]["n_previous"]].clone()
        prompt = batch["caption"]
        gt_video = batch["video"]
        b, _, v, _, h, w = image.shape
        batch_size = min(int(getattr(self.args, "val_num_samples", 1)), b)
        action_chunk = int(self.args.data["train"].get("action_chunk", self.args.data["train"].get("future_steps")))
        future_actions = batch["actions"][:batch_size, -action_chunk:].contiguous()

        image = image[:batch_size]
        image = rearrange(image, "b c v t h w -> (b v) c t h w")
        preds = pipe.infer(
            image=image,
            prompt=prompt[:batch_size],
            negative_prompt="",
            num_inference_steps=int(getattr(self.args, "val_num_inference_steps", self.args.num_inference_step)),
            decode_timestep=float(getattr(self.args, "val_decode_timestep", 0.03)),
            decode_noise_scale=float(getattr(self.args, "val_decode_noise_scale", 0.025)),
            guidance_scale=float(getattr(self.args, "val_guidance_scale", 1.0)),
            height=h,
            width=w,
            n_view=v,
            n_prev=self.args.data["train"]["n_previous"],
            chunk=(self.args.data["train"]["chunk"] - 1) // self.TEMPORAL_DOWN_RATIO + 1,
            return_video=True,
            noise_seed=getattr(self.args, "val_noise_seed", 42),
            action_chunk=action_chunk,
            future_actions=future_actions,
            pixel_wise_timestep=self.args.pixel_wise_timestep,
            n_chunk=n_chunk,
        )[0]

        fps = int(getattr(self.args, "basic_fps", 30) / max(1, self.args.data["train"]["action_chunk"] // self.args.data["train"]["chunk"]))
        save_video(rearrange(gt_video[0].data.cpu(), "c v t h w -> c t h (v w)", v=n_view), os.path.join(model_save_dir, "Validation_gt.mp4"), fps=fps)
        video = preds["video"].data.cpu()
        save_video(rearrange(video, "(b v) c t h w -> b c t h (v w)", v=n_view)[0], os.path.join(model_save_dir, "Validation.mp4"), fps=fps)

        pred_video = preds["video"]
        gt_video_eval = gt_video[:batch_size, :, :, -pred_video.shape[2] :, :, :]
        gt_video_eval = rearrange(gt_video_eval, "b c v t h w -> (b v) c t h w").to(pred_video.device)
        val_video_mse = torch.mean((pred_video.float() - gt_video_eval.float()) ** 2).item()
        accelerator.log({"val_video_mse": val_video_mse}, step=global_step)
        if to_log and self.writer is not None:
            self.writer.add_scalar("Validation/video_mse", val_video_mse, global_step)
            self.writer.add_text(f"step_{global_step}/Validation prompt:", prompt[0], global_step)
