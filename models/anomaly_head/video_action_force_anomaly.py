import math
from typing import Dict, Optional

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


class HistoryContextEncoder(nn.Module):
    """Encode history force as a low-authority context used for FiLM conditioning."""

    def __init__(
        self,
        force_dim: int,
        model_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.force_dim = int(force_dim)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(self.force_dim),
            nn.Linear(self.force_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(model_dim)),
            nn.SiLU(),
        )
        self.to_film = nn.Sequential(
            nn.LayerNorm(int(model_dim) * 2),
            nn.Linear(int(model_dim) * 2, int(model_dim) * 2),
        )

    def forward(self, history_force: torch.Tensor) -> torch.Tensor:
        if history_force.ndim != 3 or history_force.shape[-1] != self.force_dim:
            raise ValueError(f"Expected history_force [B,T,{self.force_dim}], got {tuple(history_force.shape)}")
        tokens = self.token_proj(history_force.float())
        context = torch.cat([tokens.mean(dim=1), tokens.std(dim=1, unbiased=False)], dim=-1)
        return self.to_film(context)


class ActionForceAnomalyHead(nn.Module):
    """Temporal MIL anomaly head for frozen action-force WM inference outputs."""

    def __init__(
        self,
        action_dim: int,
        force_dim: int,
        interaction_dim: int,
        video_feature_dim: int = 0,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        dropout: float = 0.1,
        use_video_features: bool = False,
        latent_channels: int = 128,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        max_steps: int = 256,
        topk_ratio: float = 0.2,
        history_context_scale: float = 0.25,
    ) -> None:
        super().__init__()
        del action_dim, video_feature_dim, use_video_features
        self.force_dim = int(force_dim)
        self.interaction_dim = int(interaction_dim)
        self.feature_dim = int(feature_dim)
        self.topk_ratio = float(topk_ratio)
        self.history_context_scale = float(history_context_scale)

        self.force_proj = nn.Sequential(
            nn.LayerNorm(self.force_dim),
            nn.Linear(self.force_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.feature_dim),
        )
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
        fused_dim = self.feature_dim * 3
        self.fuse = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.history_context = HistoryContextEncoder(
            force_dim=self.force_dim,
            model_dim=self.feature_dim,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
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
        self.risk_head = nn.Sequential(
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
        pred_force: torch.Tensor,
        history_force: torch.Tensor,
        interaction_tokens: torch.Tensor,
        pred_latents: torch.Tensor,
        future_actions: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del future_actions, video_features
        if pred_force.ndim != 3 or pred_force.shape[-1] != self.force_dim:
            raise ValueError(f"Expected pred_force [B,T,{self.force_dim}], got {tuple(pred_force.shape)}")
        if interaction_tokens.ndim != 3 or interaction_tokens.shape[-1] != self.interaction_dim:
            raise ValueError(
                f"Expected interaction_tokens [B,T,{self.interaction_dim}], got {tuple(interaction_tokens.shape)}"
            )
        if pred_force.shape[:2] != interaction_tokens.shape[:2]:
            raise ValueError(
                f"pred_force and interaction token shapes must align, got {pred_force.shape[:2]} and {interaction_tokens.shape[:2]}"
            )

        target_steps = pred_force.shape[1]
        force_tokens = self.force_proj(pred_force.float())
        interaction_tokens = self.interaction_proj(interaction_tokens.float())
        latent_tokens = self.latent_encoder(pred_latents, target_steps=target_steps)
        tokens = self.fuse(torch.cat([force_tokens, interaction_tokens, latent_tokens], dim=-1))

        gamma_beta = self.history_context(history_force).to(dtype=tokens.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        tokens = (
            tokens * (1.0 + self.history_context_scale * gamma.unsqueeze(1))
            + self.history_context_scale * beta.unsqueeze(1)
        )

        if target_steps > self.pos_embed.shape[1]:
            raise ValueError(f"target_steps={target_steps} exceeds max_steps={self.pos_embed.shape[1]}")
        tokens = tokens + self.pos_embed[:, :target_steps].to(dtype=tokens.dtype)
        tokens = self.temporal_model(tokens)
        step_logits = self.step_head(tokens).squeeze(-1)
        window_logit = self._window_from_steps(step_logits)
        risk_logit = self.risk_head(tokens.mean(dim=1)).squeeze(-1)
        risk_prob = torch.sigmoid(risk_logit)

        return {
            "window_logit": window_logit,
            "step_logits": step_logits,
            "risk_logit": risk_logit,
            "risk_prob": risk_prob,
        }


class ActionForceAnomalyHead_v1(ActionForceAnomalyHead):
    """Action-force anomaly head with separate step, window, and risk heads.

    The legacy ActionForceAnomalyHead uses the same window logit for binary
    window classification and risk probability. That conflicts when a window has
    sparse anomaly frames: the binary target is 1 while the risk target can be
    small. v1 keeps the feature encoder/temporal model but separates the heads.
    """

    def __init__(
        self,
        action_dim: int,
        force_dim: int,
        interaction_dim: int,
        video_feature_dim: int = 0,
        hidden_dim: int = 256,
        feature_dim: int = 128,
        dropout: float = 0.1,
        use_video_features: bool = False,
        latent_channels: int = 128,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        max_steps: int = 256,
        topk_ratio: float = 0.2,
        history_context_scale: float = 0.25,
        window_pool: str = "topk",
        risk_pool: str = "mean",
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            force_dim=force_dim,
            interaction_dim=interaction_dim,
            video_feature_dim=video_feature_dim,
            hidden_dim=hidden_dim,
            feature_dim=feature_dim,
            dropout=dropout,
            use_video_features=use_video_features,
            latent_channels=latent_channels,
            temporal_layers=temporal_layers,
            temporal_heads=temporal_heads,
            max_steps=max_steps,
            topk_ratio=topk_ratio,
            history_context_scale=history_context_scale,
        )
        self.window_pool = str(window_pool)
        self.risk_pool = str(risk_pool)
        self.window_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, 1),
        )
        self.risk_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.feature_dim, 1),
        )

    def _pool_tokens(self, tokens: torch.Tensor, step_logits: torch.Tensor, mode: str) -> torch.Tensor:
        mode = str(mode)
        if mode == "mean":
            return tokens.mean(dim=1)
        if mode == "max":
            return tokens.max(dim=1).values
        if mode == "topk":
            n_steps = step_logits.shape[1]
            k = max(1, min(n_steps, int(math.ceil(n_steps * self.topk_ratio))))
            indices = step_logits.topk(k, dim=1).indices
            gather_index = indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1])
            return tokens.gather(dim=1, index=gather_index).mean(dim=1)
        raise ValueError(f"Unsupported pooling mode: {mode}")

    def forward(
        self,
        pred_force: torch.Tensor,
        history_force: torch.Tensor,
        interaction_tokens: torch.Tensor,
        pred_latents: torch.Tensor,
        future_actions: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del future_actions, video_features
        if pred_force.ndim != 3 or pred_force.shape[-1] != self.force_dim:
            raise ValueError(f"Expected pred_force [B,T,{self.force_dim}], got {tuple(pred_force.shape)}")
        if interaction_tokens.ndim != 3 or interaction_tokens.shape[-1] != self.interaction_dim:
            raise ValueError(
                f"Expected interaction_tokens [B,T,{self.interaction_dim}], got {tuple(interaction_tokens.shape)}"
            )
        if pred_force.shape[:2] != interaction_tokens.shape[:2]:
            raise ValueError(
                f"pred_force and interaction token shapes must align, got {pred_force.shape[:2]} and {interaction_tokens.shape[:2]}"
            )

        target_steps = pred_force.shape[1]
        force_tokens = self.force_proj(pred_force.float())
        interaction_tokens = self.interaction_proj(interaction_tokens.float())
        latent_tokens = self.latent_encoder(pred_latents, target_steps=target_steps)
        tokens = self.fuse(torch.cat([force_tokens, interaction_tokens, latent_tokens], dim=-1))

        gamma_beta = self.history_context(history_force).to(dtype=tokens.dtype)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        tokens = (
            tokens * (1.0 + self.history_context_scale * gamma.unsqueeze(1))
            + self.history_context_scale * beta.unsqueeze(1)
        )

        if target_steps > self.pos_embed.shape[1]:
            raise ValueError(f"target_steps={target_steps} exceeds max_steps={self.pos_embed.shape[1]}")
        tokens = tokens + self.pos_embed[:, :target_steps].to(dtype=tokens.dtype)
        tokens = self.temporal_model(tokens)

        step_logits = self.step_head(tokens).squeeze(-1)
        window_tokens = self._pool_tokens(tokens, step_logits, self.window_pool)
        risk_tokens = self._pool_tokens(tokens, step_logits, self.risk_pool)
        window_logit = self.window_head(window_tokens).squeeze(-1)
        risk_logit = self.risk_head(risk_tokens).squeeze(-1)
        risk_prob = torch.sigmoid(risk_logit)

        return {
            "window_logit": window_logit,
            "step_logits": step_logits,
            "risk_logit": risk_logit,
            "risk_prob": risk_prob,
        }

