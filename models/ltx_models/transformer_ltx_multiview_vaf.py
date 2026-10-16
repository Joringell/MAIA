# Copyright 2024 The Genmo team and The HuggingFace Team.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from einops import rearrange

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.utils import is_torch_version, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph

from models.ft_encoder.ft_encoder_torch import FTSequenceTokenEncoder
from models.ltx_models.ltx_attention_processor import Attention
from models.ltx_models.transformer_ltx_multiview_v import (
    LTXVideoAttentionProcessor2_0,
    LTXVideoRotaryPosEmbed,
    LTXVideoTransformerBlock,
)


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SinusoidalSequenceEmbedding(nn.Module):
    def __init__(self, dim: int, base_seq_length: int = 54) -> None:
        super().__init__()
        self.dim = dim
        self.base_seq_length = base_seq_length

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[1]
        positions = torch.arange(seq_length, device=hidden_states.device, dtype=torch.float32)
        positions = positions / max(1, self.base_seq_length)
        half_dim = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=hidden_states.device, dtype=torch.float32)
            / max(1, half_dim - 1)
        )
        angles = positions[:, None] * frequencies[None, :]
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return hidden_states + embedding[None].to(dtype=hidden_states.dtype)


class ActionConditionEncoder(nn.Module):
    """Encode future actions [B, Ta, action_dim] into condition tokens [B, Ta, inner_dim]."""

    def __init__(self, action_dim: int, inner_dim: int, base_seq_length: int = 64) -> None:
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be positive when use_action_force_adapter=True.")
        self.action_dim = int(action_dim)
        self.inner_dim = int(inner_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.action_dim, self.inner_dim),
            nn.SiLU(),
            nn.Linear(self.inner_dim, self.inner_dim),
        )
        self.position_embed = SinusoidalSequenceEmbedding(self.inner_dim, base_seq_length=base_seq_length)

    def forward(self, future_action_states: torch.Tensor) -> torch.Tensor:
        if future_action_states.ndim != 3 or future_action_states.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected future_action_states [B,Ta,{self.action_dim}], got {tuple(future_action_states.shape)}"
            )
        action_cond_tokens = self.proj(future_action_states)
        return self.position_embed(action_cond_tokens)


class GatedActionForceAdapter(nn.Module):
    """Cross-attend video tokens to action-force context and return a gated residual delta."""

    def __init__(
        self,
        inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        attention_bias: bool,
        attention_out_bias: bool,
        qk_norm: str,
        norm_eps: float,
        norm_elementwise_affine: bool,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(inner_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)
        self.attn = Attention(
            query_dim=inner_dim,
            cross_attention_dim=inner_dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            bias=attention_bias,
            out_bias=attention_out_bias,
            qk_norm=qk_norm,
            processor=LTXVideoAttentionProcessor2_0(),
        )
        self.out = nn.Linear(inner_dim, inner_dim)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        hidden_states: torch.Tensor,
        cond_tokens: torch.Tensor,
        n_view: int,
    ) -> torch.Tensor:
        # hidden_states: [(B*n_view), Lv, C], cond_tokens: [B, K, C]
        adapter_hidden_states = self.attn(
            hidden_states=self.norm(hidden_states),
            encoder_hidden_states=cond_tokens,
            image_rotary_emb=None,
            n_view=n_view,
        )
        gate = self.gate.to(dtype=hidden_states.dtype)
        return torch.tanh(gate) * self.out(adapter_hidden_states)


class ForceStateDynamicEncoder(nn.Module):
    """Encode force history into global contact state and dynamic history tokens."""

    def __init__(
        self,
        force_dim: int,
        inner_dim: int,
        hidden_channels: Tuple[int, ...] = (64, 128, 128),
        kernel_size: int = 3,
        dropout: float = 0.1,
        include_delta: bool = True,
        include_force_norm: bool = True,
        max_dynamic_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.max_dynamic_tokens = max(1, int(max_dynamic_tokens))
        self.dynamic_encoder = FTSequenceTokenEncoder(
            force_dim=force_dim,
            embed_dim=inner_dim,
            hidden_channels=tuple(int(x) for x in hidden_channels),
            kernel_size=int(kernel_size),
            dropout=float(dropout),
            include_delta=bool(include_delta),
            include_force_norm=bool(include_force_norm),
        )
        self.state_proj = nn.Sequential(
            nn.LayerNorm(inner_dim * 2),
            nn.Linear(inner_dim * 2, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.LayerNorm(inner_dim),
        )

    def forward(self, history_force_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dynamic_tokens = self.dynamic_encoder(history_force_states)
        if dynamic_tokens.shape[1] > self.max_dynamic_tokens:
            dynamic_tokens = F.adaptive_avg_pool1d(
                dynamic_tokens.transpose(1, 2),
                self.max_dynamic_tokens,
            ).transpose(1, 2)
        pooled = torch.cat(
            [
                dynamic_tokens.mean(dim=1),
                dynamic_tokens.std(dim=1, unbiased=False),
            ],
            dim=-1,
        )
        force_state_token = self.state_proj(pooled)
        return force_state_token, dynamic_tokens


class ActionForceContextBuilder(nn.Module):
    """Build action-aligned physical context from future actions and force dynamics."""

    def __init__(
        self,
        inner_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        num_heads = max(1, int(num_heads))
        if inner_dim % num_heads != 0:
            num_heads = 1
        self.action_norm = nn.LayerNorm(inner_dim)
        self.force_norm = nn.LayerNorm(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.state_film = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, inner_dim * 2),
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(inner_dim, inner_dim),
        )

    def forward(
        self,
        action_tokens: torch.Tensor,
        force_dynamic_tokens: torch.Tensor,
        force_state_token: torch.Tensor,
    ) -> torch.Tensor:
        query_tokens = self.action_norm(action_tokens)
        force_dynamic_tokens = self.force_norm(force_dynamic_tokens)
        attended_tokens, _ = self.cross_attn(
            query=query_tokens,
            key=force_dynamic_tokens,
            value=force_dynamic_tokens,
            need_weights=False,
        )
        context_tokens = action_tokens + attended_tokens
        gamma, beta = self.state_film(force_state_token).chunk(2, dim=-1)
        context_tokens = context_tokens * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return context_tokens + self.mlp(context_tokens)


class ActionQueryInteractionReadout(nn.Module):
    """Read one interaction token per future action step from visual context."""

    def __init__(
        self,
        inner_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        num_heads = max(1, int(num_heads))
        if inner_dim % num_heads != 0:
            num_heads = 1
        self.query_norm = nn.LayerNorm(inner_dim)
        self.context_norm = nn.LayerNorm(inner_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.mlp = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, inner_dim),
            nn.SiLU(),
            nn.Linear(inner_dim, inner_dim),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_tokens: torch.Tensor,
        n_view: int,
    ) -> torch.Tensor:
        if hidden_states.shape[0] % n_view != 0:
            raise ValueError(f"hidden_states batch {hidden_states.shape[0]} is not divisible by n_view={n_view}.")
        batch_size = hidden_states.shape[0] // n_view
        if query_tokens.shape[0] != batch_size:
            raise ValueError(
                f"query_tokens batch {query_tokens.shape[0]} does not match video batch {batch_size}."
            )

        # [(B*n_view), Lv, C] -> [B, n_view*Lv, C]
        context_tokens = rearrange(hidden_states, "(b v) l c -> b (v l) c", v=n_view)
        residual_query_tokens = query_tokens
        query_tokens = self.query_norm(query_tokens)
        context_tokens = self.context_norm(context_tokens)
        attended_tokens, _ = self.cross_attn(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            need_weights=False,
        )
        interaction_tokens = residual_query_tokens + attended_tokens
        return interaction_tokens + self.mlp(interaction_tokens)


class ForceDecoder(nn.Module):
    """Temporal decoder from interaction tokens [B, T, inner_dim] to force [B, T, force_dim]."""

    def __init__(
        self,
        inner_dim: int,
        force_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else inner_dim
        num_heads = max(1, int(num_heads))
        if inner_dim % num_heads != 0:
            num_heads = 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=inner_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_blocks = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(num_layers)))
        self.force_pred = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, force_dim),
        )
        self.delta_force_pred = nn.Sequential(
            nn.LayerNorm(inner_dim),
            nn.Linear(inner_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, force_dim),
        )

    def forward(self, interaction_tokens: torch.Tensor) -> torch.Tensor:
        force_tokens = self.temporal_blocks(interaction_tokens)
        return self.force_pred(force_tokens)

    def predict_delta(self, interaction_tokens: torch.Tensor) -> torch.Tensor:
        force_tokens = self.temporal_blocks(interaction_tokens)
        return self.delta_force_pred(force_tokens)


@maybe_allow_in_graph
class LTXVideoTransformer3DModel(ModelMixin, ConfigMixin, FromOriginalModelMixin, PeftAdapterMixin):
    """LTX video backbone with an action/force adapter and future-force readout.

    Smoke-test shapes:
        hidden_states: [(B*n_view), T*H*W, in_channels]
        future_action_states: [B, Ta, action_dim]
        history_force_states: [B, Hf, force_dim]
        sample["video"]: unchanged video token output
        sample["interaction_tokens"]: [B, T, inner_dim]
        sample["force"]: [B, T, force_dim]
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        patch_size: int = 1,
        patch_size_t: int = 1,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        cross_attention_dim: int = 2048,
        num_layers: int = 28,
        activation_fn: str = "gelu-approximate",
        qk_norm: str = "rms_norm_across_heads",
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = 4096,
        attention_bias: bool = True,
        attention_out_bias: bool = True,
        use_view_embed: bool = True,
        max_view: int = 3,
        use_action_force_adapter: bool = False,
        action_dim: int = 0,
        force_dim: int = 6,
        adapter_layers: Optional[Tuple[int, ...]] = None,
        num_force_steps: Optional[int] = None,
        force_decoder_hidden_dim: Optional[int] = None,
        force_decoder_num_layers: int = 2,
        force_decoder_num_heads: int = 8,
        force_decoder_dropout: float = 0.0,
        interaction_readout_num_heads: int = 8,
        interaction_readout_dropout: float = 0.0,
        action_position_base_seq_length: int = 64,
        force_history_hidden_channels: Tuple[int, ...] = (64, 128, 128),
        force_history_kernel_size: int = 3,
        force_history_dropout: float = 0.1,
        force_history_include_delta: bool = True,
        force_history_include_norm: bool = True,
        force_dynamic_max_tokens: int = 16,
        action_force_context_num_heads: int = 8,
        action_force_context_dropout: float = 0.0,
        interaction_semantic_dim: int = 256,
        **kwargs,
    ) -> None:
        super().__init__()
        del kwargs
        out_channels = out_channels or in_channels
        inner_dim = num_attention_heads * attention_head_dim

        self.proj_in = nn.Linear(in_channels, inner_dim)

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)
        self.time_embed = AdaLayerNormSingle(inner_dim, use_additional_conditions=False)

        self.use_view_embed = use_view_embed
        if self.use_view_embed:
            self.view_embed = nn.Parameter(torch.randn(max_view, inner_dim))
            self.view_ada = nn.Sequential(nn.SiLU(), nn.Linear(inner_dim, 6 * inner_dim, bias=True))

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.rope = LTXVideoRotaryPosEmbed(
            dim=inner_dim,
            base_num_frames=20,
            base_height=2048,
            base_width=2048,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            theta=10000.0,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                LTXVideoTransformerBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    qk_norm=qk_norm,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    attention_out_bias=attention_out_bias,
                    eps=norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels)
        self.gradient_checkpointing = False

        self.use_action_force_adapter = bool(use_action_force_adapter)
        self.action_dim = int(action_dim)
        self.force_dim = int(force_dim)
        self.action_condition_dim = self.action_dim
        self.force_in_channels = self.force_dim
        self.force_out_channels = self.force_dim
        self.num_force_steps = num_force_steps

        self.adapter_layer_indices = []
        self.af_adapters = nn.ModuleDict()
        if self.use_action_force_adapter:
            if adapter_layers is None:
                second_half_start = max(num_layers // 2, 0)
                adapter_layers = tuple(range(second_half_start, num_layers, 4))
                if not adapter_layers and num_layers > 0:
                    adapter_layers = (num_layers - 1,)
            self.adapter_layer_indices = sorted({int(idx) for idx in adapter_layers if 0 <= int(idx) < num_layers})
            self.adapter_layer_set = set(self.adapter_layer_indices)

            self.action_condition_encoder = ActionConditionEncoder(
                self.action_dim,
                inner_dim,
                base_seq_length=int(action_position_base_seq_length),
            )
            self.force_context_encoder = ForceStateDynamicEncoder(
                force_dim=self.force_dim,
                inner_dim=inner_dim,
                hidden_channels=tuple(int(x) for x in force_history_hidden_channels),
                kernel_size=int(force_history_kernel_size),
                dropout=float(force_history_dropout),
                include_delta=bool(force_history_include_delta),
                include_force_norm=bool(force_history_include_norm),
                max_dynamic_tokens=int(force_dynamic_max_tokens),
            )
            self.action_force_context = ActionForceContextBuilder(
                inner_dim=inner_dim,
                num_heads=int(action_force_context_num_heads),
                dropout=float(action_force_context_dropout),
            )
            self.af_adapters = nn.ModuleDict(
                {
                    str(idx): GatedActionForceAdapter(
                        inner_dim=inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        attention_bias=attention_bias,
                        attention_out_bias=attention_out_bias,
                        qk_norm=qk_norm,
                        norm_eps=norm_eps,
                        norm_elementwise_affine=norm_elementwise_affine,
                    )
                    for idx in self.adapter_layer_indices
                }
            )
            self.interaction_readout = ActionQueryInteractionReadout(
                inner_dim,
                num_heads=int(interaction_readout_num_heads),
                dropout=float(interaction_readout_dropout),
            )
            self.force_decoder = ForceDecoder(
                inner_dim,
                self.force_dim,
                hidden_dim=force_decoder_hidden_dim,
                num_layers=int(force_decoder_num_layers),
                num_heads=int(force_decoder_num_heads),
                dropout=float(force_decoder_dropout),
            )
            semantic_dim = int(interaction_semantic_dim)
            self.interaction_semantic_proj = nn.Sequential(
                nn.LayerNorm(inner_dim),
                nn.Linear(inner_dim, semantic_dim),
            )
            self.language_semantic_proj = nn.Sequential(
                nn.LayerNorm(inner_dim),
                nn.Linear(inner_dim, semantic_dim),
            )
            # Keep the language-side geometry fixed. The auxiliary loss should
            # weakly organize interaction latents around the pretrained text
            # geometry, rather than letting both spaces move toward each other.
            self.language_semantic_proj.requires_grad_(False)
        else:
            self.adapter_layer_set = set()

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _encode_action_force_conditions(
        self,
        future_action_states: Optional[torch.Tensor],
        history_force_states: Optional[torch.Tensor],
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.use_action_force_adapter:
            return None, None, None
        if future_action_states is None or history_force_states is None:
            return None, None, None
        if future_action_states.shape[0] != batch_size or history_force_states.shape[0] != batch_size:
            raise ValueError(
                "Condition batch size must match video batch without views: "
                f"B={batch_size}, action={future_action_states.shape[0]}, force={history_force_states.shape[0]}"
            )

        future_action_states = future_action_states.to(dtype=hidden_dtype)
        history_force_states = history_force_states.to(dtype=hidden_dtype)
        action_cond_tokens = self.action_condition_encoder(future_action_states)
        force_state_token, force_dynamic_tokens = self.force_context_encoder(history_force_states)
        action_force_context_tokens = self.action_force_context(
            action_cond_tokens,
            force_dynamic_tokens,
            force_state_token,
        )
        return action_cond_tokens, force_state_token, action_force_context_tokens

    def forward(
        self,
        hidden_states: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        encoder_attention_mask: torch.Tensor = None,
        n_view: int = 1,
        rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
        return_dict: bool = True,
        return_video: bool = True,
        store_buffer: bool = False,
        video_states_buffer=None,
        video_attention_mask: torch.Tensor = None,
        num_frames: int = None,
        height: int = None,
        width: int = None,
        future_action_states: torch.Tensor = None,
        history_force_states: torch.Tensor = None,
        return_force: bool = False,
        return_interaction: bool = False,
        return_video_features: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        if (return_force or return_interaction) and not self.use_action_force_adapter:
            raise ValueError("return_force/return_interaction requires use_action_force_adapter=True.")
        if self.use_action_force_adapter and (return_force or return_interaction):
            if future_action_states is None:
                raise ValueError("future_action_states is required when return_force=True or return_interaction=True.")
            if history_force_states is None:
                raise ValueError("history_force_states is required when return_force=True or return_interaction=True.")

        run_video_path = return_video or store_buffer or return_force or return_interaction
        action_cond_tokens = None
        force_state_token = None
        action_force_context_tokens = None
        embedded_timestep = None

        if run_video_path:
            if store_buffer:
                video_states_buffer = []

            image_rotary_emb = self.rope(hidden_states, rope_interpolation_scale, num_frames, height, width)

            text_pool_mask = encoder_attention_mask if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2 else None
            if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
                encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
                encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            if video_attention_mask is not None and video_attention_mask.ndim == 2:
                video_attention_mask = (1 - video_attention_mask.to(hidden_states.dtype)) * -10000.0
                video_attention_mask = video_attention_mask.unsqueeze(0)

            batch_size_with_views = hidden_states.size(0)
            batch_size = batch_size_with_views // n_view
            hidden_states = self.proj_in(hidden_states)

            temb, embedded_timestep = self.time_embed(
                timestep.flatten(),
                batch_size=batch_size_with_views,
                hidden_dtype=hidden_states.dtype,
            )
            temb = temb.view(batch_size_with_views, -1, temb.size(-1))
            embedded_timestep = embedded_timestep.view(
                batch_size_with_views, -1, embedded_timestep.size(-1)
            )

            if self.use_view_embed:
                embedded_view = self.view_embed[:n_view].unsqueeze(0).repeat(batch_size, 1, 1)
                embedded_view = rearrange(embedded_view, "b v c -> (b v) c").unsqueeze(1)
                vemb = self.view_ada(embedded_view)
                temb = temb + vemb
                embedded_timestep = embedded_timestep + embedded_view

            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))
            if text_pool_mask is not None:
                mask = text_pool_mask.to(device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
                mask = mask[:, : encoder_hidden_states.shape[1]]
                language_pooled = (encoder_hidden_states * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                language_pooled = encoder_hidden_states.mean(dim=1)

            action_cond_tokens, force_state_token, action_force_context_tokens = self._encode_action_force_conditions(
                future_action_states=future_action_states,
                history_force_states=history_force_states,
                batch_size=batch_size,
                hidden_dtype=hidden_states.dtype,
            )
            if self.use_action_force_adapter and self.adapter_layer_indices and action_force_context_tokens is None:
                logger.debug("Action-force adapter enabled but no conditions were provided for this forward pass.")

        for block_idx, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

                if run_video_path:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        n_view,
                        block_idx % 3 == 0,
                        video_attention_mask,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = video_states_buffer[block_idx]

                if action_force_context_tokens is not None and block_idx in self.adapter_layer_set:
                    adapter_delta = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(self.af_adapters[str(block_idx)]),
                        hidden_states,
                        action_force_context_tokens,
                        n_view,
                        **ckpt_kwargs,
                    )
                    hidden_states = hidden_states + adapter_delta
            else:
                if run_video_path:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                        n_view=n_view,
                        cross_view_attn=block_idx % 3 == 0,
                        self_attention_mask=video_attention_mask,
                    )
                else:
                    hidden_states = video_states_buffer[block_idx]

                if action_force_context_tokens is not None and block_idx in self.adapter_layer_set:
                    hidden_states = hidden_states + self.af_adapters[str(block_idx)](
                        hidden_states,
                        action_force_context_tokens,
                        n_view=n_view,
                    )

            if store_buffer:
                video_states_buffer.append(hidden_states.clone())

        final_output = {}

        if store_buffer:
            final_output["video_states_buffer"] = video_states_buffer

        interaction_tokens = None
        if return_force or return_interaction:
            interaction_tokens = self.interaction_readout(
                hidden_states,
                query_tokens=action_force_context_tokens,
                n_view=n_view,
            )
            if return_interaction:
                final_output["interaction_tokens"] = interaction_tokens
                interaction_pooled = interaction_tokens.mean(dim=1)
                final_output["interaction_semantic"] = self.interaction_semantic_proj(interaction_pooled)
                final_output["language_semantic"] = self.language_semantic_proj(language_pooled)
            if return_force:
                force_pred = self.force_decoder(interaction_tokens)
                final_output["force"] = force_pred

        if return_video_features:
            video_features = hidden_states.mean(dim=1)
            video_features = rearrange(video_features, "(b v) c -> b v c", v=n_view).mean(dim=1)
            final_output["video_features"] = video_features

        if return_video:
            scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift
            final_output["video"] = self.proj_out(hidden_states)

        if not return_dict:
            return (final_output,)

        return Transformer2DModelOutput(sample=final_output)
