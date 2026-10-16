import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentFrameEncoder(nn.Module):
    """Encode predicted VAE latents into one feature token per future step."""

    def __init__(
        self,
        latent_channels: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.latent_channels, int(hidden_dim), kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Dropout2d(float(dropout)),
            nn.Conv2d(int(hidden_dim), int(output_dim), kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, latents: torch.Tensor, target_steps: int) -> torch.Tensor:
        if latents.ndim != 6:
            raise ValueError(f"Expected predicted latents [B,V,C,T,H,W], got {tuple(latents.shape)}")
        batch_size, n_view, channels, n_frames, height, width = latents.shape
        if channels != self.latent_channels:
            raise ValueError(f"Expected latent channels {self.latent_channels}, got {channels}")
        x = latents.to(dtype=self.net[0].weight.dtype)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch_size * n_view * n_frames, channels, height, width)
        x = self.net(x).flatten(1)
        x = x.reshape(batch_size, n_view, n_frames, -1).mean(dim=1)
        if n_frames != target_steps:
            x = F.interpolate(
                x.transpose(1, 2),
                size=target_steps,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return x


class VideoActionAnomalyHead(nn.Module):
    """Temporal MIL anomaly head for action-conditioned video WM outputs.

    This head uses predicted video latents and action-conditioned interaction
    tokens. Raw future actions are already represented in the interaction
    tokens, so this head intentionally has no direct action or force inputs.
    """

    def __init__(
        self,
        interaction_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        dropout: float = 0.1,
        latent_channels: int = 128,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        max_steps: int = 256,
        topk_ratio: float = 0.2,
    ) -> None:
        super().__init__()
        self.interaction_dim = int(interaction_dim)
        self.feature_dim = int(feature_dim)
        self.topk_ratio = float(topk_ratio)

        self.interaction_proj = nn.Sequential(
            nn.LayerNorm(self.interaction_dim),
            nn.Linear(self.interaction_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.feature_dim),
        )
        self.latent_encoder = LatentFrameEncoder(
            latent_channels=int(latent_channels),
            hidden_dim=int(hidden_dim),
            output_dim=self.feature_dim,
            dropout=float(dropout),
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.feature_dim * 2),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, int(max_steps), self.feature_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=max(1, int(temporal_heads)),
            dim_feedforward=int(hidden_dim) * 2,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_model = nn.TransformerEncoder(encoder_layer, num_layers=max(1, int(temporal_layers)))
        self.step_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, 1),
        )

    def _window_from_steps(self, step_logits: torch.Tensor) -> torch.Tensor:
        n_steps = step_logits.shape[1]
        k = max(1, min(n_steps, int(math.ceil(n_steps * self.topk_ratio))))
        return step_logits.topk(k, dim=1).values.mean(dim=1)

    def forward(
        self,
        interaction_tokens: torch.Tensor,
        pred_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if interaction_tokens.ndim != 3 or interaction_tokens.shape[-1] != self.interaction_dim:
            raise ValueError(
                f"Expected interaction_tokens [B,T,{self.interaction_dim}], got {tuple(interaction_tokens.shape)}"
            )
        target_steps = interaction_tokens.shape[1]
        if target_steps > self.pos_embed.shape[1]:
            raise ValueError(f"target_steps={target_steps} exceeds max_steps={self.pos_embed.shape[1]}")

        interaction_tokens = self.interaction_proj(interaction_tokens.float())
        latent_tokens = self.latent_encoder(pred_latents, target_steps=target_steps)
        tokens = self.fuse(torch.cat([interaction_tokens, latent_tokens], dim=-1))
        tokens = tokens + self.pos_embed[:, :target_steps].to(dtype=tokens.dtype)
        tokens = self.temporal_model(tokens)
        step_logits = self.step_head(tokens).squeeze(-1)
        window_logit = self._window_from_steps(step_logits)
        risk_prob = torch.sigmoid(window_logit)
        return {
            "window_logit": window_logit,
            "step_logits": step_logits,
            "risk_prob": risk_prob,
        }


class VideoActionAnomalyHead_v1(VideoActionAnomalyHead):
    """Video-action anomaly head with separate window and risk heads."""

    def __init__(
        self,
        interaction_dim: int,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        dropout: float = 0.1,
        latent_channels: int = 128,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        max_steps: int = 256,
        topk_ratio: float = 0.2,
        window_pool: str = "topk",
        risk_pool: str = "mean",
    ) -> None:
        super().__init__(
            interaction_dim=interaction_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            dropout=dropout,
            latent_channels=latent_channels,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            max_steps=max_steps,
            topk_ratio=topk_ratio,
        )
        self.window_pool = str(window_pool)
        self.risk_pool = str(risk_pool)
        self.window_head = self.step_head
        self.risk_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, 1),
        )

    def _pool_steps(self, step_values: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "mean":
            return step_values.mean(dim=1)
        if mode == "max":
            return step_values.max(dim=1).values
        if mode == "topk":
            return self._window_from_steps(step_values)
        raise ValueError(f"Unsupported pool mode: {mode}")

    def forward(
        self,
        interaction_tokens: torch.Tensor,
        pred_latents: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if interaction_tokens.ndim != 3 or interaction_tokens.shape[-1] != self.interaction_dim:
            raise ValueError(
                f"Expected interaction_tokens [B,T,{self.interaction_dim}], got {tuple(interaction_tokens.shape)}"
            )
        target_steps = interaction_tokens.shape[1]
        if target_steps > self.pos_embed.shape[1]:
            raise ValueError(f"target_steps={target_steps} exceeds max_steps={self.pos_embed.shape[1]}")

        interaction_tokens = self.interaction_proj(interaction_tokens.float())
        latent_tokens = self.latent_encoder(pred_latents, target_steps=target_steps)
        tokens = self.fuse(torch.cat([interaction_tokens, latent_tokens], dim=-1))
        tokens = tokens + self.pos_embed[:, :target_steps].to(dtype=tokens.dtype)
        tokens = self.temporal_model(tokens)
        step_logits = self.window_head(tokens).squeeze(-1)
        risk_logits = self.risk_head(tokens).squeeze(-1)
        window_logit = self._pool_steps(step_logits, self.window_pool)
        risk_prob = torch.sigmoid(self._pool_steps(risk_logits, self.risk_pool))
        return {
            "window_logit": window_logit,
            "step_logits": step_logits,
            "risk_prob": risk_prob,
        }
