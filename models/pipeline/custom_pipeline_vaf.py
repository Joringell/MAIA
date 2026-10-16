from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import time
from diffusers.image_processor import PipelineImageInput
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

from models.pipeline import custom_pipeline_v as base_pipeline
from models.pipeline.custom_pipeline_v import (
    CustomPipeline as BaseCustomPipeline,
    CustomPipelineOutput,
    calculate_shift,
    retrieve_latents,
    retrieve_timesteps,
)
from utils.data_utils import gen_noise_from_condition_frame_latent


class CustomPipeline(BaseCustomPipeline):
    """Inference pipeline for LTX video backbone + action/force adapter.

    Video is denoised iteratively. Force is predicted from the final denoising
    transformer step and is not denoised as a separate diffusion variable.
    """

    @torch.no_grad()
    def infer(
        self,
        image: PipelineImageInput = None,
        n_prev: int = 4,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 192,
        width: int = 256,
        chunk: int = 8,
        frame_rate: int = 30,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 3,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        init_latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        return_dict: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 128,
        return_init: bool = True,
        n_view: int = 3,
        return_force: bool = True,
        action_chunk: int = 50,
        force_dim: int = 6,
        future_actions: torch.Tensor = None,
        history_force_state: torch.Tensor = None,
        return_video: bool = True,
        return_interaction: bool = False,
        return_video_features: bool = False,
        return_latents: bool = False,
        noise_seed: int = None,
        pixel_wise_timestep: bool = True,
        n_chunk: int = 1,
        show_progress: bool = False,
        profile_timings: bool = False,
        **kwargs,
    ):
        del latents, kwargs
        if not return_force:
            raise ValueError("Action-force pipeline expects return_force=True.")
        if future_actions is None:
            raise ValueError("future_actions must be provided.")
        if history_force_state is None:
            raise ValueError("history_force_state must be provided.")
        if n_chunk != 1:
            raise ValueError("Action-force pipeline currently supports n_chunk=1.")

        self.transformer.eval()
        latent_num_frames = n_prev + chunk
        latent_height = height // self.vae_spatial_compression_ratio
        latent_width = width // self.vae_spatial_compression_ratio
        video_sequence_length = latent_num_frames * latent_height * latent_width

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        device = self._execution_device
        profile = {} if profile_timings else None

        def _sync_for_profile():
            if profile is not None and torch.cuda.is_available() and torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)

        def _start_profile():
            _sync_for_profile()
            return time.perf_counter() * 1000.0

        def _end_profile(name: str, start_ms: float) -> None:
            if profile is None:
                return
            _sync_for_profile()
            profile[name] = profile.get(name, 0.0) + (time.perf_counter() * 1000.0 - start_ms)

        total_profile_start = _start_profile() if profile is not None else None

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        preds = {}
        if prompt_embeds is None:
            section_start = _start_profile()
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                max_sequence_length=max_sequence_length,
                device=device,
            )
            _end_profile("text_encode_ms", section_start)
            preds["text_embedding"] = {
                "prompt_embeds": prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_prompt_attention_mask": negative_prompt_attention_mask,
            }

        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        sample_mode = "argmax" if noise_seed else "sample"

        if init_latents is None:
            if image is None:
                raise ValueError("image must be provided when init_latents is None.")
            if len(image.shape) == 4:
                image = image.unsqueeze(2)
            section_start = _start_profile()
            image = image.to(device=device, dtype=prompt_embeds.dtype)
            _end_profile("image_to_device_ms", section_start)

            video_list = image.clone() if return_init else None
            mem_size = image.shape[2]

            section_start = _start_profile()
            image_for_vae = rearrange(image, "bv c t h w -> (bv t) c h w").unsqueeze(2)
            init_latents = retrieve_latents(self.vae.encode(image_for_vae), generator, sample_mode=sample_mode)
            init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
            if mem_size == 1 and n_prev > 1:
                init_latents = init_latents.repeat(1, 1, n_prev, 1, 1)
            else:
                init_latents = rearrange(init_latents, "(b t) c f h w -> b c (t f) h w", t=mem_size)
            _end_profile("vae_encode_ms", section_start)
        else:
            video_list = None
            section_start = _start_profile()
            init_latents = init_latents.to(device=device, dtype=prompt_embeds.dtype)
            _end_profile("init_latents_to_device_ms", section_start)

        section_start = _start_profile()
        if future_actions.shape != (batch_size, action_chunk, self.transformer.action_condition_dim):
            raise ValueError(
                "future_actions must have shape "
                f"{(batch_size, action_chunk, self.transformer.action_condition_dim)}, "
                f"got {tuple(future_actions.shape)}"
            )
        if force_dim != self.transformer.force_in_channels:
            raise ValueError(
                f"force_dim must match transformer force dim {self.transformer.force_in_channels}, got {force_dim}"
            )
        if history_force_state.ndim != 3 or history_force_state.shape[-1] != force_dim:
            raise ValueError(
                f"history_force_state must have shape [B,H,{force_dim}], got {tuple(history_force_state.shape)}"
            )
        if history_force_state.shape[0] != batch_size:
            raise ValueError(
                f"history_force_state batch {history_force_state.shape[0]} does not match prompt batch {batch_size}"
            )

        future_actions = future_actions.to(device=device, dtype=prompt_embeds.dtype)
        history_force_state = history_force_state.to(device=device, dtype=prompt_embeds.dtype)
        _end_profile("action_force_prepare_ms", section_start)

        section_start = _start_profile()
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        mu = calculate_shift(
            video_sequence_length,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas=sigmas,
            mu=mu,
        )
        if getattr(self.scheduler.config, "final_sigmas_type", None) == "sigma_min":
            self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]
        self._num_timesteps = len(timesteps)
        _end_profile("scheduler_prepare_ms", section_start)

        latent_frame_rate = frame_rate / self.vae_temporal_compression_ratio
        rope_interpolation_scale = (
            1 / latent_frame_rate,
            self.vae_spatial_compression_ratio,
            self.vae_spatial_compression_ratio,
        )

        section_start = _start_profile()
        video_generator = torch.Generator(device=device).manual_seed(noise_seed) if noise_seed else generator
        latents, conditioning_mask, cond_indicator = gen_noise_from_condition_frame_latent(
            init_latents,
            latent_num_frames,
            latent_height,
            latent_width,
            video_generator,
            noise_to_condition_frames=0,
        )
        if self.do_classifier_free_guidance:
            conditioning_mask = torch.cat([conditioning_mask, conditioning_mask], dim=0)
        _end_profile("noise_prepare_ms", section_start)

        force_pred = None
        interaction_tokens = None
        video_features = None
        pred_latents = None
        if not show_progress:
            self.set_progress_bar_config(disable=True)
        section_start = _start_profile()
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                is_last_step = i == len(timesteps) - 1
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents.clone()
                latent_model_input = latent_model_input.to(prompt_embeds.dtype)

                if self.do_classifier_free_guidance:
                    future_actions_in = torch.cat([future_actions, future_actions], dim=0)
                    history_force_in = torch.cat([history_force_state, history_force_state], dim=0)
                else:
                    future_actions_in = future_actions
                    history_force_in = history_force_state

                timestep = t.expand(latent_model_input.shape[0])
                if pixel_wise_timestep:
                    timestep = timestep.unsqueeze(-1) * (1 - conditioning_mask)
                else:
                    timestep = timestep.unsqueeze(-1) * (1 - cond_indicator)

                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    encoder_attention_mask=prompt_attention_mask,
                    num_frames=latent_num_frames,
                    height=latent_height,
                    width=latent_width,
                    rope_interpolation_scale=rope_interpolation_scale,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    future_action_states=future_actions_in,
                    history_force_states=history_force_in,
                    return_video=True,
                    return_force=is_last_step,
                    return_interaction=is_last_step and return_interaction,
                    return_video_features=is_last_step and return_video_features,
                    n_view=n_view,
                    video_attention_mask=None,
                )[0]

                if is_last_step:
                    force_noise_pred = noise_pred["force"].float()
                    if self.do_classifier_free_guidance:
                        force_uncond, force_text = force_noise_pred.chunk(2)
                        force_noise_pred = force_uncond + self.guidance_scale * (force_text - force_uncond)
                    force_pred = force_noise_pred

                    if return_interaction:
                        interaction_tokens = noise_pred["interaction_tokens"].float()
                        if self.do_classifier_free_guidance:
                            _, interaction_tokens = interaction_tokens.chunk(2)
                    if return_video_features:
                        video_features = noise_pred["video_features"].float()
                        if self.do_classifier_free_guidance:
                            _, video_features = video_features.chunk(2)

                video_noise_pred = noise_pred["video"].float()
                if self.do_classifier_free_guidance:
                    video_uncond, video_text = video_noise_pred.chunk(2)
                    video_noise_pred = video_uncond + self.guidance_scale * (video_text - video_uncond)

                video_noise_pred = self._unpack_latents(
                    video_noise_pred,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )
                latents_unpacked = self._unpack_latents(
                    latents,
                    latent_num_frames,
                    latent_height,
                    latent_width,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )
                video_noise_pred = video_noise_pred[:, :, n_prev:]
                noise_latents = latents_unpacked[:, :, n_prev:]
                pred_latents = self.scheduler.step(video_noise_pred, t, noise_latents, return_dict=False)[0]
                latents_unpacked = torch.cat([latents_unpacked[:, :, :n_prev], pred_latents], dim=2)
                latents = self._pack_latents(
                    latents_unpacked,
                    self.transformer_spatial_patch_size,
                    self.transformer_temporal_patch_size,
                )

                progress_bar.update()
                if base_pipeline.XLA_AVAILABLE:
                    base_pipeline.xm.mark_step()
        _end_profile("denoise_ms", section_start)

        self.maybe_free_model_hooks()

        if return_video:
            section_start = _start_profile()
            clean_frames = self.decode_to_frames(
                pred_latents,
                dtype=prompt_embeds.dtype,
                generator=generator,
                device=device,
                batch_size=batch_size * n_view,
                decode_timestep=decode_timestep,
                decode_noise_scale=decode_noise_scale,
            )
            video_list = clean_frames if video_list is None else torch.cat((video_list, clean_frames), dim=2)
            preds["video"] = video_list
            _end_profile("decode_ms", section_start)

        preds["force"] = force_pred
        if return_interaction:
            preds["interaction_tokens"] = interaction_tokens
        if return_video_features:
            preds["video_features"] = video_features
        if return_latents:
            preds["pred_latents"] = pred_latents
        if profile is not None:
            _end_profile("total_profile_ms", total_profile_start)
            preds["_timings"] = profile
        self.transformer.train()

        if not return_dict:
            return (preds,)
        return CustomPipelineOutput(frames=preds)
