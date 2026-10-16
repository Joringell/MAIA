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


def _masked_batch_mean(values, mask):
    if mask is None:
        return values.mean()
    mask = mask.to(device=values.device, dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


class Trainer(BaseGETrainer):
    """Pure video GE trainer.

    This trainer owns only video world-model training:
      P(future_video | history_video, text)

    Action-conditioned and action+force variants live in their own trainers.
    """

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        super().__init__(config_file=config_file, to_log=to_log, output_dir=output_dir)
        if getattr(self.args, "return_action", False):
            raise ValueError("ge_trainer.py is video-only. Use runner/ge_trainer_action.py for action-conditioned GE.")
        if getattr(self.args, "return_force", False):
            raise ValueError("ge_trainer.py is video-only. Use runner/ge_trainer_action_force.py for action+force GE.")

    def prepare_trainable_parameters(self):
        super().prepare_trainable_parameters()
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)

    def prepare_optimizer(self):
        logger.info("Initializing video GE optimizer and lr scheduler")
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

        params = []
        for _name, param in self.diffusion_model.named_parameters():
            param.requires_grad = True
            params.append(param)
        self.state.num_trainable_parameters = sum(p.numel() for p in params)
        logger.info(f"Total trainable parameters: {self.state.num_trainable_parameters}")

        self.optimizer = get_optimizer(
            params_to_optimize=[{"params": params, "lr": self.state.learning_rate}],
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

    def _prepare_video_batch(self, batch, accelerator, weight_dtype):
        video = batch["video"].to(accelerator.device, dtype=weight_dtype, non_blocking=True).contiguous()
        batch_size, _, n_view, _, _, _ = video.shape
        sample_loss_mask = None
        if self.label_filter == "only_normal":
            if "label" not in batch:
                raise KeyError("label_filter='only_normal' requires batch['label']")
            sample_loss_mask = batch["label"].to(accelerator.device, non_blocking=True).view(batch_size) == 1
        video = rearrange(video, "b c v t h w -> (b v) c t h w")
        if self.args.use_color_jitter:
            video = apply_color_jitter_to_video(video)
        mem_size = int(self.args.data["train"]["n_previous"])
        mem = video[:, :, :mem_size]
        future_video = video[:, :, mem_size:]
        return video, mem, future_video, sample_loss_mask, batch_size, n_view, mem_size

    def train(self):
        logger.info("Starting video GE training")
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

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        scheduler_sigmas = self.scheduler.sigmas.clone().to(device=accelerator.device, dtype=weight_dtype)
        global_step = int(getattr(self.args, "start_step", 0) or 0)
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=global_step,
            desc="Training video GE",
            disable=not accelerator.is_local_main_process,
        )
        if global_step > 0:
            logger.info(f"Resuming from start_step={global_step}")
            try:
                self.lr_scheduler.step(global_step)
            except TypeError:
                for _ in range(global_step):
                    self.lr_scheduler.step()

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
                        sample_loss_mask,
                        batch_size,
                        n_view,
                        mem_size,
                    ) = self._prepare_video_batch(batch, accelerator, weight_dtype)

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
                    weights_for_t = density.unsqueeze(1).repeat(1, n_view)
                    indices = (
                        rearrange(weights_for_t, "b v -> (b v)") * self.scheduler.config.num_train_timesteps
                    ).long()
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
                    noisy_latents = (1.0 - ss) * latents + ss * noise
                    loss_weights = compute_loss_weighting_for_sd3(
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
                        return_video=True,
                        video_attention_mask=None,
                        condition_mask=conditioning_mask,
                    )["latents"]

                    target = noise - latents
                    loss_video = loss_weights.float() * (pred_all["video"].float() - target.float()).pow(2)
                    loss_video = loss_video * (
                        1 - conditioning_mask.unsqueeze(-1).repeat(1, 1, loss_video.size(-1))
                    )
                    loss_video = loss_video.mean(dim=(1, 2))
                    video_loss_mask = (
                        sample_loss_mask.unsqueeze(1).repeat(1, n_view).reshape(-1)
                        if sample_loss_mask is not None
                        else None
                    )
                    loss = _masked_batch_mean(loss_video, video_loss_mask)

                    assert not torch.isnan(loss), "NaN loss detected"
                    accelerator.backward(loss)
                    if accelerator.sync_gradients and accelerator.distributed_type != DistributedType.DEEPSPEED:
                        grad_norm = accelerator.clip_grad_norm_(self.diffusion_model.parameters(), self.args.max_grad_norm)
                    else:
                        grad_norm = None
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                loss = accelerator.reduce(loss.detach(), reduction="mean")
                running_loss += loss.item()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.item(), "loss_video": loss.item(), "lr": self.lr_scheduler.get_last_lr()[0]}
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.state.train_steps:
                    break
                if global_step % self.args.steps_to_log == 0 and accelerator.is_main_process and self.writer is not None:
                    self.writer.add_scalar("Training Loss", loss.item(), global_step)
                    self.writer.add_scalar("Video loss", loss.item(), global_step)
                if global_step % self.args.steps_to_val == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        self.validate(accelerator, os.path.join(self.save_folder, f"Validation_step_{global_step}"), global_step, n_view=n_view)
                    accelerator.wait_for_everyone()
                if global_step % self.args.steps_to_save == 0:
                    self.save_diffusion_model(global_step)

            if accelerator.is_main_process and self.writer is not None:
                self.writer.add_scalar("Average Training Loss", running_loss / max(1, len(self.train_dataloader)), epoch)

        progress_bar.close()
        self.save_diffusion_model(global_step)
        accelerator.wait_for_everyone()
        self.cleanup_training()
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
        batch_size = 1
        _, _, v, _, h, w = image.shape
        image = rearrange(image[:batch_size], "b c v t h w -> (b v) c t h w")

        preds = pipe.infer(
            image=image,
            prompt=prompt[:batch_size],
            negative_prompt="",
            num_inference_steps=int(getattr(self.args, "num_inference_step", 5)),
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=1.0,
            height=h,
            width=w,
            n_view=v,
            n_prev=self.args.data["train"]["n_previous"],
            chunk=(self.args.data["train"]["chunk"] - 1) // self.TEMPORAL_DOWN_RATIO + 1,
            return_video=True,
            noise_seed=42,
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
        val_loss = torch.mean((pred_video.float() - gt_video_eval.float()) ** 2).item()
        accelerator.log({"val_loss": val_loss}, step=global_step)
        if to_log and self.writer is not None:
            self.writer.add_scalar("Validation/video_mse", val_loss, global_step)
            self.writer.add_text(f"step_{global_step}/Validation prompt:", prompt[0], global_step)
