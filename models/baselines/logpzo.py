import math

import torch
from torch import nn
import torch.nn.functional as F

from models.baselines.pca_kmeans import build_pca_features


def _group_count(channels, preferred=8):
    for groups in range(min(int(preferred), int(channels)), 0, -1):
        if int(channels) % groups == 0:
            return groups
    return 1


def extract_video_latent_tokens(pred_latents):
    """Pool WM predicted video latents into logpZO observations.

    Accepted shapes:
        [B, V, C, T, H, W] -> [B, T, C]
        [B, C, T, H, W] -> [B, T, C]
        [B, T, C] -> [B, T, C]
        [B, C] -> [B, 1, C]
    """
    if pred_latents.ndim == 6:
        return pred_latents.float().mean(dim=(1, 4, 5)).transpose(1, 2).contiguous()
    if pred_latents.ndim == 5:
        return pred_latents.float().mean(dim=(3, 4)).transpose(1, 2).contiguous()
    if pred_latents.ndim == 3:
        return pred_latents.float().contiguous()
    if pred_latents.ndim == 2:
        return pred_latents.float().unsqueeze(1).contiguous()
    raise ValueError(f"Unsupported pred_latents shape for logpZO: {tuple(pred_latents.shape)}")


def _fixed_projection_matrix(in_dim, out_dim, device, dtype):
    rows = torch.arange(1, int(in_dim) + 1, device=device, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(1, int(out_dim) + 1, device=device, dtype=torch.float32).unsqueeze(0)
    proj = torch.sin(rows * cols * 12.9898 + rows * 78.233 + cols * 37.719)
    proj = torch.where(proj >= 0, torch.ones_like(proj), -torch.ones_like(proj))
    proj = proj / (float(in_dim) ** 0.5)
    return proj.to(dtype=dtype)


def _fixed_project_sequence(sequence, out_dim=64):
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence [B,T,D], got {tuple(sequence.shape)}")
    sequence = sequence.float()
    proj = _fixed_projection_matrix(
        in_dim=sequence.shape[-1],
        out_dim=int(out_dim),
        device=sequence.device,
        dtype=sequence.dtype,
    )
    return torch.matmul(sequence, proj).contiguous()


def build_observed_logpzo_features(
    video_features,
    state=None,
    history_force_state=None,
    mode="video_state_force",
    force_bins=5,
    mem_size=None,
):
    """Build observed logpZO features as [B,T,D].

    Standard observed modes return one token per window: [B,1,D]. The optional
    video_temporal mode preserves memory-frame tokens as [B,M,D].
    """
    mode = str(mode)
    if mode == "video_temporal":
        if video_features.ndim == 3:
            return video_features.float().contiguous()
        if video_features.ndim != 2:
            raise ValueError(f"Expected video_features [B,D] or [B,T,D], got {tuple(video_features.shape)}")
        if mem_size is None:
            raise ValueError("mode=video_temporal requires mem_size when cached video_features are flattened")
        mem_size = int(mem_size)
        if video_features.shape[-1] % mem_size != 0:
            raise ValueError(
                f"Cannot reshape cached video_features {tuple(video_features.shape)} into "
                f"video_temporal with mem_size={mem_size}"
            )
        return video_features.float().reshape(video_features.shape[0], mem_size, -1).contiguous()

    features = build_pca_features(
        video_features,
        state=state,
        history_force_state=history_force_state,
        mode=mode,
        force_bins=force_bins,
    )
    return features.float().unsqueeze(1).contiguous()


def _valid_token_mask_from_lengths(valid_len, num_tokens, source_steps, device):
    if valid_len is None:
        return torch.ones(1, int(num_tokens), dtype=torch.bool, device=device)
    valid_len = torch.as_tensor(valid_len, device=device).reshape(-1).long()
    starts = torch.arange(int(num_tokens), device=device).long() * int(source_steps) // int(num_tokens)
    return valid_len[:, None] > starts[None, :]


def _masked_bin_stats(sequence, num_bins, valid_len=None, include_delta=True, include_std=True, include_max_abs=True):
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence [B,T,D], got {tuple(sequence.shape)}")
    sequence = sequence.float()
    batch_size, num_steps, dim = sequence.shape
    device = sequence.device
    if valid_len is None:
        valid_len = torch.full((batch_size,), num_steps, device=device, dtype=torch.long)
    else:
        valid_len = torch.as_tensor(valid_len, device=device).reshape(-1).long().clamp(min=0, max=num_steps)
        if valid_len.numel() != batch_size:
            raise ValueError(f"valid_len must have shape [B], got {tuple(valid_len.shape)} for B={batch_size}")

    blocks = []
    for bin_idx in range(int(num_bins)):
        start = int(bin_idx * num_steps // int(num_bins))
        end = int((bin_idx + 1) * num_steps // int(num_bins))
        if end <= start:
            end = min(num_steps, start + 1)
        chunk = sequence[:, start:end]
        local_steps = int(chunk.shape[1])
        local_idx = torch.arange(start, end, device=device).view(1, local_steps)
        mask = local_idx < valid_len.view(batch_size, 1)
        denom = mask.sum(dim=1).clamp_min(1).to(sequence.dtype).view(batch_size, 1)
        mask_f = mask.to(sequence.dtype).unsqueeze(-1)

        mean = (chunk * mask_f).sum(dim=1) / denom
        blocks.append(mean)
        if include_std:
            centered = (chunk - mean[:, None, :]) * mask_f
            blocks.append((centered.pow(2).sum(dim=1) / denom).sqrt())
        if include_max_abs:
            masked_abs = chunk.abs().masked_fill(~mask.unsqueeze(-1), 0.0)
            blocks.append(masked_abs.amax(dim=1))
        if include_delta:
            count = mask.sum(dim=1)
            first_local = torch.zeros(batch_size, device=device, dtype=torch.long)
            last_local = (count - 1).clamp_min(0).long()
            rows = torch.arange(batch_size, device=device)
            first = chunk[rows, first_local]
            last = chunk[rows, last_local]
            delta = torch.where((count > 0).view(batch_size, 1), last - first, torch.zeros(batch_size, dim, device=device))
            blocks.append(delta)
    return torch.cat(blocks, dim=-1).contiguous()


def build_wm_logpzo_features(
    pred_latents,
    pred_force,
    interaction_tokens,
    valid_len=None,
    token_proj_dim=64,
):
    """Build action-force WM logpZO features.

    Returns:
        features: [B,T_lat,D], where T_lat follows pooled pred_latents.
        token_mask: [B,T_lat], derived from valid_len over the 50-step future window.
    """
    if pred_latents is None or pred_force is None or interaction_tokens is None:
        raise ValueError("WM logpZO features require pred_latents, pred_force, and interaction_tokens")
    latent_tokens = extract_video_latent_tokens(pred_latents)
    if pred_force.ndim != 3:
        raise ValueError(f"Expected pred_force [B,T,D], got {tuple(pred_force.shape)}")
    if interaction_tokens.ndim != 3:
        raise ValueError(f"Expected interaction_tokens [B,T,D], got {tuple(interaction_tokens.shape)}")
    if pred_force.shape[0] != latent_tokens.shape[0] or interaction_tokens.shape[0] != latent_tokens.shape[0]:
        raise ValueError("WM logpZO tensors must share batch size")

    num_tokens = int(latent_tokens.shape[1])
    source_steps = int(pred_force.shape[1])
    projected_interaction = _fixed_project_sequence(interaction_tokens, out_dim=int(token_proj_dim))
    force_features = _masked_bin_stats(
        pred_force,
        num_bins=num_tokens,
        valid_len=valid_len,
        include_delta=True,
        include_std=True,
        include_max_abs=True,
    ).reshape(pred_force.shape[0], num_tokens, -1)
    interaction_features = _masked_bin_stats(
        projected_interaction,
        num_bins=num_tokens,
        valid_len=valid_len,
        include_delta=True,
        include_std=False,
        include_max_abs=False,
    ).reshape(projected_interaction.shape[0], num_tokens, -1)
    token_mask = _valid_token_mask_from_lengths(
        valid_len=valid_len,
        num_tokens=num_tokens,
        source_steps=source_steps,
        device=latent_tokens.device,
    )
    if token_mask.shape[0] == 1 and latent_tokens.shape[0] != 1:
        token_mask = token_mask.expand(latent_tokens.shape[0], -1)
    features = torch.cat([latent_tokens.float(), force_features, interaction_features], dim=-1)
    return features.contiguous(), token_mask.contiguous()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalTimeEmbedding dim must be even")
        self.dim = int(dim)

    def forward(self, timesteps):
        timesteps = timesteps.float().reshape(-1)
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(1, half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=timesteps.dtype) * -scale)
        args = timesteps[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class LogpZOBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim, dropout=0.0):
        super().__init__()
        groups = _group_count(hidden_dim)
        self.norm1 = nn.GroupNorm(groups, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.norm2 = nn.GroupNorm(groups, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x, time_emb):
        residual = x
        x = self.conv1(F.silu(self.norm1(x)))
        x = x + self.time_proj(F.silu(time_emb))[:, :, None]
        x = self.conv2(self.dropout(F.silu(self.norm2(x))))
        return x + residual


class LogpZOModel(nn.Module):
    """Flow-matching logpZO baseline over frozen WM latent observations."""

    def __init__(
        self,
        feature_dim=128,
        hidden_dim=256,
        num_layers=4,
        time_embed_dim=128,
        dropout=0.0,
        time_scale=100.0,
        score_reduction="sum",
        normalize_input=True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_scale = float(time_scale)
        self.score_reduction = str(score_reduction)
        self.normalize_input = bool(normalize_input)

        self.input_norm = nn.LayerNorm(self.feature_dim) if self.normalize_input else nn.Identity()
        self.in_proj = nn.Conv1d(self.feature_dim, self.hidden_dim, kernel_size=1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(int(time_embed_dim)),
            nn.Linear(int(time_embed_dim), self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [LogpZOBlock(self.hidden_dim, self.hidden_dim, dropout=dropout) for _ in range(int(num_layers))]
        )
        self.out_norm = nn.GroupNorm(_group_count(self.hidden_dim), self.hidden_dim)
        self.out_proj = nn.Conv1d(self.hidden_dim, self.feature_dim, kernel_size=1)

    def _validate_features(self, features):
        if features.ndim != 3:
            raise ValueError(f"logpZO features must be [B,T,C], got {tuple(features.shape)}")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"logpZO feature_dim={self.feature_dim}, got feature shape {tuple(features.shape)}")
        return features.float()

    def _normalize_features(self, features):
        return self.input_norm(self._validate_features(features))

    def forward(self, features, timesteps, assume_normalized=False):
        features = self._validate_features(features)
        if not assume_normalized:
            features = self.input_norm(features)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=features.device)
        timesteps = timesteps.to(device=features.device, dtype=features.dtype).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(features.shape[0])
        time_emb = self.time_embedding(timesteps * self.time_scale)

        x = features.transpose(1, 2).contiguous()
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, time_emb)
        x = self.out_proj(F.silu(self.out_norm(x)))
        return x.transpose(1, 2).contiguous()

    def flow_matching_loss(self, features, sample_mask=None, token_mask=None):
        x0 = self._normalize_features(features)
        x1 = torch.randn_like(x0)
        cont_t = torch.rand(x0.shape[0], device=x0.device, dtype=x0.dtype)
        vtrue = x1 - x0
        xnow = x0 + cont_t[:, None, None] * vtrue
        vhat = self(xnow, cont_t, assume_normalized=True)
        per_token = (vhat - vtrue).pow(2).mean(dim=-1)
        if token_mask is not None:
            valid = token_mask.to(device=per_token.device, dtype=torch.bool)
            if valid.shape != per_token.shape:
                raise ValueError(f"token_mask shape {tuple(valid.shape)} does not match loss shape {tuple(per_token.shape)}")
            if sample_mask is not None:
                valid = valid & sample_mask.to(device=per_token.device, dtype=torch.bool).reshape(-1, 1)
            if not bool(valid.any().item()):
                return per_token.sum() * 0.0
            return per_token[valid].mean()

        per_sample = per_token.mean(dim=1)
        if sample_mask is None:
            return per_sample.mean()
        sample_mask = sample_mask.to(device=per_sample.device, dtype=torch.bool).reshape(-1)
        if not bool(sample_mask.any().item()):
            return per_sample.sum() * 0.0
        return per_sample[sample_mask].mean()

    @torch.no_grad()
    def score_tokens(self, features):
        x0 = self._normalize_features(features)
        timesteps = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)
        pred_v = self(x0, timesteps, assume_normalized=True)
        z = x0 + pred_v
        return z.pow(2).sum(dim=-1)

    @torch.no_grad()
    def score(self, features, token_mask=None, reduction=None):
        token_scores = self.score_tokens(features)
        reduction = str(reduction or self.score_reduction)
        if token_mask is None:
            if reduction == "max":
                return token_scores.amax(dim=1)
            if reduction == "sum":
                return token_scores.sum(dim=1)
            if reduction == "mean":
                return token_scores.sum(dim=1) / float(features.shape[1] * features.shape[2])
            raise ValueError(f"Unsupported score_reduction={reduction}")

        valid = token_mask.to(device=token_scores.device, dtype=torch.bool)
        if valid.shape != token_scores.shape:
            raise ValueError(f"token_mask shape {tuple(valid.shape)} does not match score shape {tuple(token_scores.shape)}")
        if reduction == "max":
            masked = token_scores.masked_fill(~valid, float("-inf"))
            scores = masked.amax(dim=1)
            return torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        if reduction == "sum":
            return token_scores.masked_fill(~valid, 0.0).sum(dim=1)
        if reduction == "mean":
            denom = valid.sum(dim=1).clamp_min(1).to(token_scores.dtype) * float(features.shape[2])
            return token_scores.masked_fill(~valid, 0.0).sum(dim=1) / denom
        raise ValueError(f"Unsupported score_reduction={reduction}")


class SuccessFailureLogpZOModel(nn.Module):
    """Class-conditional logpZO with separate success and failure density models."""

    def __init__(self, **kwargs):
        super().__init__()
        self.success_model = LogpZOModel(**kwargs)
        self.failure_model = LogpZOModel(**kwargs)
        self.feature_dim = self.success_model.feature_dim
        self.hidden_dim = self.success_model.hidden_dim
        self.score_reduction = self.success_model.score_reduction

    def flow_matching_loss(self, features, success_mask, failure_mask):
        success_loss = self.success_model.flow_matching_loss(features, sample_mask=success_mask)
        failure_loss = self.failure_model.flow_matching_loss(features, sample_mask=failure_mask)
        return success_loss + failure_loss, {
            "success_loss": success_loss.detach(),
            "failure_loss": failure_loss.detach(),
        }

    @torch.no_grad()
    def score_components(self, features):
        success_score = self.success_model.score(features)
        failure_score = self.failure_model.score(features)
        return success_score, failure_score

    @torch.no_grad()
    def score(self, features):
        success_score, failure_score = self.score_components(features)
        return success_score - failure_score
