import json
import math
import os

import numpy as np
import torch
from diffusers.optimization import get_scheduler
from einops import rearrange
from tqdm import tqdm

from models.pipeline import custom_pipeline_v as base_pipeline
from models.pipeline.custom_pipeline_v import calculate_shift, retrieve_latents, retrieve_timesteps
from runner.ge_trainer_video_action_anomaly import Trainer as AnomalyTrainer
from runner.ge_trainer_video_action_anomaly import logger
from utils.data_utils import gen_noise_from_condition_frame_latent, get_text_conditions
from utils.memory_utils import free_memory, get_memory_statistics
from utils.model_utils import unwrap_model
from utils.optimizer_utils import get_optimizer


class Trainer(AnomalyTrainer):
    """Joint trainer for action adapter and action anomaly head.

    The video backbone stays frozen. The last denoising step keeps gradients for
    the action adapter/readout and anomaly head. Force is intentionally unused.
    """

    def _use_feature_cache(self):
        return False

    def prepare_trainable_parameters(self):
        logger.info("Freezing video WM backbone; training action adapter and anomaly head")
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()

        self.diffusion_model.requires_grad_(False)
        adapter_prefixes = (
            "action_condition_encoder",
            "action_context",
            "action_adapters",
            "interaction_readout",
        )
        if bool(getattr(self.args, "joint_train_caption_projection", False)):
            adapter_prefixes = adapter_prefixes + ("caption_projection",)
        for name, param in self.diffusion_model.named_parameters():
            param.requires_grad = name.startswith(adapter_prefixes)
        self.diffusion_model.train()

        if bool(getattr(self.args, "anomaly_train_classifier_only", False)):
            logger.info("Freezing anomaly head feature encoders; training classifier head only")
            self.anomaly_head.requires_grad_(False)
            if hasattr(self.anomaly_head, "step_head"):
                self.anomaly_head.step_head.requires_grad_(True)
            if hasattr(self.anomaly_head, "window_head"):
                self.anomaly_head.window_head.requires_grad_(True)
        else:
            self.anomaly_head.requires_grad_(True)
        self.anomaly_head.train()

        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_optimizer(self):
        logger.info("Initializing joint action-adapter/anomaly-head optimizer and lr scheduler")
        self.state.train_epochs = self.args.train_epochs
        self.state.train_steps = self.args.train_steps
        self.state.learning_rate = self.args.lr
        if self.args.scale_lr:
            self.state.learning_rate = (
                self.state.learning_rate
                * self.args.gradient_accumulation_steps
                * self.args.batch_size
                * self.state.accelerator.num_processes
            )

        params = [
            p
            for module in (self.diffusion_model, self.anomaly_head)
            for p in module.parameters()
            if p.requires_grad
        ]
        self.state.num_trainable_parameters = sum(p.numel() for p in params)
        logger.info(f"Total joint trainable parameters: {self.state.num_trainable_parameters}")

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

    def prepare_for_training(self):
        (
            self.diffusion_model,
            self.anomaly_head,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        ) = self.state.accelerator.prepare(
            self.diffusion_model,
            self.anomaly_head,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )

    def _make_pipe(self):
        pipe = self.pipeline_class(
            self.scheduler,
            self.vae,
            self.text_encoder,
            self.tokenizer,
            unwrap_model(self.state.accelerator, self.diffusion_model),
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _run_sampling_with_final_step_grad(
        self,
        pipe,
        mem,
        batch,
        future_actions,
        batch_size,
        n_view,
        mem_size,
        height,
        width,
    ):
        guidance_scale = float(getattr(self.args, "anomaly_guidance_scale", 1.0))
        if guidance_scale != 1.0:
            raise ValueError("Joint action anomaly training currently requires anomaly_guidance_scale=1.0")

        if hasattr(pipe.scheduler, "_step_index"):
            pipe.scheduler._step_index = None
        text_conds = get_text_conditions(self.tokenizer, self.text_encoder, batch["caption"])
        prompt_embeds = text_conds["prompt_embeds"]
        prompt_attention_mask = text_conds["prompt_attention_mask"]

        chunk = int(self.args.data["train"]["chunk"])
        latent_num_frames = mem_size + chunk
        latent_height = height // pipe.vae_spatial_compression_ratio
        latent_width = width // pipe.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width
        device = self.state.accelerator.device
        hidden_dtype = prompt_embeds.dtype

        image = mem.to(device=device, dtype=hidden_dtype)
        image_for_vae = rearrange(image, "bv c t h w -> (bv t) c h w").unsqueeze(2)
        with torch.no_grad():
            init_latents = retrieve_latents(pipe.vae.encode(image_for_vae), None, sample_mode="sample")
            init_latents = pipe._normalize_latents(init_latents, pipe.vae.latents_mean, pipe.vae.latents_std)
            if mem_size == 1 and mem_size < latent_num_frames:
                init_latents = init_latents.repeat(1, 1, mem_size, 1, 1)
            else:
                init_latents = rearrange(init_latents, "(b t) c f h w -> b c (t f) h w", t=mem_size)

            num_steps = int(getattr(self.args, "num_inference_step", 5))
            sigmas = np.linspace(1.0, 1 / num_steps, num_steps)
            mu = calculate_shift(
                video_sequence_length,
                pipe.scheduler.config.base_image_seq_len,
                pipe.scheduler.config.max_image_seq_len,
                pipe.scheduler.config.base_shift,
                pipe.scheduler.config.max_shift,
            )
            timesteps, _ = retrieve_timesteps(pipe.scheduler, num_steps, device, None, sigmas=sigmas, mu=mu)
            if getattr(pipe.scheduler.config, "final_sigmas_type", None) == "sigma_min":
                pipe.scheduler.sigmas[-1] = pipe.scheduler.sigmas[-2]

            latents, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
                init_latents,
                latent_num_frames,
                latent_height,
                latent_width,
                None,
                noise_to_condition_frames=0,
            )

        latent_frame_rate = 30 / pipe.vae_temporal_compression_ratio
        rope_interpolation_scale = (
            1 / latent_frame_rate,
            pipe.vae_spatial_compression_ratio,
            pipe.vae_spatial_compression_ratio,
        )
        future_actions = future_actions.to(device=device, dtype=hidden_dtype)

        interaction_tokens = None
        pred_latents = None
        for i, t in enumerate(timesteps):
            is_last_step = i == len(timesteps) - 1
            grad_context = torch.enable_grad() if is_last_step else torch.no_grad()
            with grad_context:
                latent_model_input = latents.clone().to(hidden_dtype)
                timestep = t.expand(latent_model_input.shape[0])
                if bool(getattr(self.args, "pixel_wise_timestep", True)):
                    timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)
                else:
                    timestep = timestep.unsqueeze(-1) * (1 - cond_indicator)

                noise_pred = self.diffusion_model(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    return_dict=False,
                    future_action_states=future_actions,
                    return_video=True,
                    return_interaction=is_last_step,
                    n_view=n_view,
                    video_attention_mask=None,
                )[0]

                if is_last_step:
                    interaction_tokens = noise_pred["interaction_tokens"].float()

                video_noise_pred = noise_pred["video"].float()
                video_noise_pred = pipe._unpack_latents(
                    video_noise_pred,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    pipe.transformer_spatial_patch_size,
                    pipe.transformer_temporal_patch_size,
                )
                latents_unpacked = pipe._unpack_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    pipe.transformer_spatial_patch_size,
                    pipe.transformer_temporal_patch_size,
                )
                video_noise_pred = video_noise_pred[:, :, mem_size:]
                noise_latents = latents_unpacked[:, :, mem_size:]
                pred_latents = pipe.scheduler.step(video_noise_pred, t, noise_latents, return_dict=False)[0]
                latents_unpacked = torch.cat([latents_unpacked[:, :, :mem_size], pred_latents.detach()], dim=2)
                latents = pipe._pack_latents(
                    latents_unpacked,
                    pipe.transformer_spatial_patch_size,
                    pipe.transformer_temporal_patch_size,
                ).detach()

            if base_pipeline.XLA_AVAILABLE:
                base_pipeline.xm.mark_step()

        return {
            "interaction_tokens": interaction_tokens,
            "pred_latents": pred_latents.detach(),
        }

    def train(self):
        logger.info("Starting joint action adapter/anomaly-head training")
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
                    "train batch size": self.state.train_batch_size,
                    "gradient accumulation steps": self.args.gradient_accumulation_steps,
                    "anomaly target": self.anomaly_target,
                    "caption projection trainable": bool(getattr(self.args, "joint_train_caption_projection", False)),
                },
                indent=4,
            )
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        pipe = self._make_pipe()
        global_step = int(getattr(self.args, "start_step", 0) or 0)
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=global_step,
            desc="Joint action anomaly training",
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(self.state.train_epochs):
            if global_step >= self.state.train_steps:
                break
            self.diffusion_model.train()
            self.anomaly_head.train()
            if self.vae is not None:
                self.vae.eval()
            if self.text_encoder is not None:
                self.text_encoder.eval()
            running_loss = 0.0

            for batch in self.train_dataloader:
                with accelerator.accumulate([self.diffusion_model, self.anomaly_head]):
                    mem, future_actions, batch_size, n_view, mem_size, height, width = self._prepare_video_action_batch(
                        batch,
                        accelerator,
                        weight_dtype,
                    )
                    target = self._target_from_batch(batch, accelerator.device)

                    preds = self._run_sampling_with_final_step_grad(
                        pipe,
                        mem,
                        batch,
                        future_actions,
                        batch_size,
                        n_view,
                        mem_size,
                        height,
                        width,
                    )
                    pred_latents = preds["pred_latents"].reshape(
                        batch_size,
                        n_view,
                        preds["pred_latents"].shape[1],
                        preds["pred_latents"].shape[2],
                        preds["pred_latents"].shape[3],
                        preds["pred_latents"].shape[4],
                    )
                    head_out = self.anomaly_head(
                        interaction_tokens=preds["interaction_tokens"],
                        pred_latents=pred_latents,
                    )
                    loss, loss_window, loss_risk, loss_step = self._compute_anomaly_loss(head_out, target, batch)

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params = [
                            p
                            for module in (self.diffusion_model, self.anomaly_head)
                            for p in module.parameters()
                            if p.requires_grad
                        ]
                        grad_norm = accelerator.clip_grad_norm_(params, self.args.max_grad_norm)
                    else:
                        grad_norm = None
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                loss = accelerator.reduce(loss.detach(), reduction="mean")
                loss_window = accelerator.reduce(loss_window, reduction="mean")
                loss_risk = accelerator.reduce(loss_risk, reduction="mean")
                loss_step = accelerator.reduce(loss_step, reduction="mean")
                probs = torch.sigmoid(head_out["window_logit"].detach())
                pred = (probs >= 0.5).to(target.dtype)
                acc = accelerator.reduce((pred == target).float().mean(), reduction="mean")
                target_mean = accelerator.reduce(target.detach().float().mean(), reduction="mean")
                running_loss += loss.item()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {
                    "loss": loss.item(),
                    "loss_window": loss_window.item(),
                    "loss_risk": loss_risk.item(),
                    "loss_step": loss_step.item(),
                    "acc": acc.item(),
                    "target_pos_ratio": target_mean.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)

                if global_step % self.args.steps_to_log == 0 and accelerator.is_main_process and self.writer is not None:
                    for key, value in logs.items():
                        if key != "lr":
                            self.writer.add_scalar(f"ActionJoint/{key}", value, global_step)

                if global_step % self.args.steps_to_save == 0:
                    self._save_joint(global_step)

                if (
                    getattr(self.args, "steps_to_val", None) is not None
                    and int(self.args.steps_to_val) > 0
                    and global_step % int(self.args.steps_to_val) == 0
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        self.validate(pipe, accelerator, global_step)
                    accelerator.wait_for_everyone()

                if global_step >= self.state.train_steps:
                    break

            if accelerator.is_main_process and self.writer is not None:
                self.writer.add_scalar("ActionJoint/epoch_loss", running_loss / max(1, len(self.train_dataloader)), epoch)

        progress_bar.close()
        self._save_joint(global_step)
        accelerator.wait_for_everyone()
        del pipe, self.diffusion_model, self.scheduler
        free_memory()
        logger.info(f"Memory after training end: {json.dumps(get_memory_statistics(), indent=4)}")
        accelerator.end_training()

    def _save_joint(self, global_step):
        self.state.accelerator.wait_for_everyone()
        if not self.state.accelerator.is_main_process:
            return
        save_dir = os.path.join(self.save_folder, f"step_{global_step}")
        os.makedirs(save_dir, exist_ok=True)

        model_to_save = unwrap_model(self.state.accelerator, self.diffusion_model)
        model_to_save.save_pretrained(save_dir, safe_serialization=True)

        head_to_save = self.state.accelerator.unwrap_model(self.anomaly_head)
        torch.save(
            {
                "state_dict": head_to_save.state_dict(),
                "anomaly_head": getattr(self.args, "anomaly_head", {}),
                "anomaly_head_class": getattr(self, "anomaly_head_class", "VideoActionAnomalyHead"),
                "anomaly_head_class_path": getattr(
                    self,
                    "anomaly_head_class_path",
                    "models/anomaly_head/action_anomaly.py",
                ),
                "anomaly_target": self.anomaly_target,
                "anomaly_positive_is": self.anomaly_positive_is,
                "baseline": "action_joint",
            },
            os.path.join(save_dir, "anomaly_head.pt"),
        )
