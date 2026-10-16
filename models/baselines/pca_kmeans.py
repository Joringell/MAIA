import torch
import torch.nn.functional as F


def pooled_memory_video_latents(vae, video, mem_size, dtype=None, generator=None, return_sequence=False):
    """Encode observed memory RGB frames with the frozen VAE and pool to [B, M*C] or [B, M, C]."""
    if video.ndim != 6:
        raise ValueError(f"Expected video [B,C,V,T,H,W], got {tuple(video.shape)}")
    batch_size, _, n_view, num_frames, _, _ = video.shape
    mem_size = min(int(mem_size), int(num_frames))
    if mem_size <= 0:
        raise ValueError("mem_size must be positive")

    mem = video[:, :, :, :mem_size].contiguous()
    mem = mem.permute(0, 2, 3, 1, 4, 5).reshape(batch_size * n_view * mem_size, *video.shape[1:2], *video.shape[-2:])
    mem = mem.unsqueeze(2).to(device=video.device, dtype=vae.dtype)
    latents = vae.encode(mem).latent_dist.sample(generator=generator)
    latents = latents.to(dtype=dtype or video.dtype)

    latents_mean = vae.latents_mean
    latents_std = vae.latents_std
    if latents_mean is not None and latents_std is not None:
        latents_mean = torch.as_tensor(latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents_std = torch.as_tensor(latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
        latents = (latents - latents_mean) / latents_std

    latents = latents.reshape(batch_size, n_view, mem_size, latents.shape[1], *latents.shape[2:])
    # Pool view, latent temporal, and spatial axes; keep the observed memory time axis and channels.
    pooled = latents.float().mean(dim=(1, 4, 5, 6))
    if return_sequence:
        return pooled.contiguous()
    return pooled.flatten(1).contiguous()


def force_summary_features(history_force_state, num_bins=5):
    """Summarize normalized force history [B,H,D] into stable low-dimensional features."""
    if history_force_state.ndim != 3:
        raise ValueError(f"Expected history_force_state [B,H,D], got {tuple(history_force_state.shape)}")
    force = history_force_state.float()
    mean = force.mean(dim=1)
    std = force.std(dim=1, unbiased=False)
    min_value = force.amin(dim=1)
    max_value = force.amax(dim=1)
    last = force[:, -1]
    delta = force[:, -1] - force[:, 0]

    norm = force.norm(dim=-1)
    norm_stats = torch.stack(
        [
            norm.mean(dim=1),
            norm.std(dim=1, unbiased=False),
            norm.amax(dim=1),
        ],
        dim=1,
    )

    num_bins = max(1, int(num_bins))
    pooled = F.adaptive_avg_pool1d(force.transpose(1, 2), num_bins).transpose(1, 2).flatten(1)
    return torch.cat([mean, std, min_value, max_value, last, delta, norm_stats, pooled], dim=1).contiguous()


def build_pca_features(video_features, state=None, history_force_state=None, mode="video_state_force", force_bins=5):
    mode = str(mode)
    blocks = [video_features.float().flatten(1)]
    if mode in {"video_state", "video_state_force"}:
        if state is None:
            raise ValueError(f"mode={mode} requires state")
        blocks.append(state.float().flatten(1))
    if mode in {"video_force", "video_state_force"}:
        if history_force_state is None:
            raise ValueError(f"mode={mode} requires history_force_state")
        blocks.append(force_summary_features(history_force_state, num_bins=force_bins))
    if mode not in {"video", "video_state", "video_force", "video_state_force"}:
        raise ValueError(f"Unsupported PCA feature mode: {mode}")
    return torch.cat(blocks, dim=1).contiguous()


def _temporal_mean_std(sequence, name):
    if sequence.ndim != 3:
        raise ValueError(f"Expected {name} [B,T,D], got {tuple(sequence.shape)}")
    sequence = sequence.float()
    return torch.cat(
        [
            sequence.mean(dim=1),
            sequence.std(dim=1, unbiased=False),
        ],
        dim=1,
    ).contiguous()


def _pred_force_compact_features(pred_force):
    if pred_force.ndim != 3:
        raise ValueError(f"Expected pred_force [B,T,D], got {tuple(pred_force.shape)}")
    pred_force = pred_force.float()
    return torch.cat(
        [
            pred_force.mean(dim=1),
            pred_force.std(dim=1, unbiased=False),
            pred_force[:, -1],
            pred_force[:, -1] - pred_force[:, 0],
        ],
        dim=1,
    ).contiguous()


def _pred_latent_temporal_tokens(pred_latents):
    if pred_latents.ndim != 6:
        raise ValueError(f"Expected pred_latents [B,V,C,T,H,W], got {tuple(pred_latents.shape)}")
    return pred_latents.float().mean(dim=(1, 4, 5)).transpose(1, 2).contiguous()


def _temporal_binned_stats(sequence, num_bins=5, stats=("mean", "std", "delta")):
    if sequence.ndim != 3:
        raise ValueError(f"Expected sequence [B,T,D], got {tuple(sequence.shape)}")
    sequence = sequence.float()
    num_bins = max(1, int(num_bins))
    num_steps = int(sequence.shape[1])
    blocks = []
    for bin_idx in range(num_bins):
        start = int(bin_idx * num_steps // num_bins)
        end = int((bin_idx + 1) * num_steps // num_bins)
        if end <= start:
            end = min(num_steps, start + 1)
        chunk = sequence[:, start:end]
        if "mean" in stats:
            blocks.append(chunk.mean(dim=1))
        if "std" in stats:
            blocks.append(chunk.std(dim=1, unbiased=False))
        if "max_abs" in stats:
            blocks.append(chunk.abs().amax(dim=1))
        if "delta" in stats:
            blocks.append(chunk[:, -1] - chunk[:, 0])
    return torch.cat(blocks, dim=1).contiguous()


def _fixed_projection_matrix(in_dim, out_dim, device, dtype):
    rows = torch.arange(1, int(in_dim) + 1, device=device, dtype=torch.float32).unsqueeze(1)
    cols = torch.arange(1, int(out_dim) + 1, device=device, dtype=torch.float32).unsqueeze(0)
    proj = torch.sin(rows * cols * 12.9898 + rows * 78.233 + cols * 37.719)
    proj = torch.where(proj >= 0, torch.ones_like(proj), -torch.ones_like(proj))
    proj = proj / (float(in_dim) ** 0.5)
    return proj.to(dtype=dtype)


def _fixed_project_sequence(sequence, out_dim=128):
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


def _pred_force_binned_features(pred_force, num_bins=5):
    if pred_force.ndim != 3:
        raise ValueError(f"Expected pred_force [B,T,D], got {tuple(pred_force.shape)}")
    pred_force = pred_force.float()
    abs_force = pred_force.abs()
    delta = pred_force[:, 1:] - pred_force[:, :-1]
    delta_abs = delta.abs()
    norm = pred_force.norm(dim=-1)
    global_stats = [
        pred_force.mean(dim=1),
        pred_force.std(dim=1, unbiased=False),
        pred_force[:, -1],
        pred_force[:, -1] - pred_force[:, 0],
        abs_force.amax(dim=1),
        torch.quantile(abs_force, 0.9, dim=1),
        delta_abs.amax(dim=1) if delta_abs.shape[1] > 0 else torch.zeros_like(pred_force[:, 0]),
        torch.stack(
            [
                norm.mean(dim=1),
                norm.std(dim=1, unbiased=False),
                norm.amax(dim=1),
                torch.quantile(norm, 0.9, dim=1),
            ],
            dim=1,
        ),
    ]
    binned = _temporal_binned_stats(
        pred_force,
        num_bins=num_bins,
        stats=("mean", "std", "max_abs", "delta"),
    )
    return torch.cat([*global_stats, binned], dim=1).contiguous()


def _action_force_cache_binned_features(pred_force, interaction_tokens, pred_latents, num_bins=5, token_proj_dim=64):
    latent_tokens = _pred_latent_temporal_tokens(pred_latents)
    projected_interaction = _fixed_project_sequence(interaction_tokens, out_dim=token_proj_dim)
    return torch.cat(
        [
            _pred_force_binned_features(pred_force, num_bins=num_bins),
            _temporal_binned_stats(projected_interaction, num_bins=num_bins, stats=("mean", "delta")),
            _temporal_binned_stats(latent_tokens, num_bins=num_bins, stats=("mean", "delta")),
        ],
        dim=1,
    ).contiguous()


def build_action_force_cache_features(
    pred_force,
    interaction_tokens,
    pred_latents,
    mode="action_force_cache_mean",
):
    """Fixed unsupervised compression of action-force anomaly-head cache tensors."""
    mode = str(mode)
    if mode == "action_force_cache_bin":
        mode = "action_force_cache_binned"
    if mode not in {"action_force_cache_mean", "action_force_head_v1_cache", "action_force_cache_binned"}:
        raise ValueError(f"Unsupported action-force cache feature mode: {mode}")
    if pred_force is None:
        raise ValueError(f"mode={mode} requires pred_force")
    if interaction_tokens is None:
        raise ValueError(f"mode={mode} requires interaction_tokens")
    if pred_latents is None:
        raise ValueError(f"mode={mode} requires pred_latents")
    if interaction_tokens.ndim != 3:
        raise ValueError(f"Expected interaction_tokens [B,T,D], got {tuple(interaction_tokens.shape)}")

    if mode == "action_force_cache_binned":
        return _action_force_cache_binned_features(
            pred_force=pred_force,
            interaction_tokens=interaction_tokens,
            pred_latents=pred_latents,
            num_bins=5,
            token_proj_dim=64,
        )

    latent_tokens = _pred_latent_temporal_tokens(pred_latents)
    return torch.cat(
        [
            _pred_force_compact_features(pred_force),
            interaction_tokens.float().mean(dim=1),
            _temporal_mean_std(latent_tokens, "pooled_pred_latents"),
        ],
        dim=1,
    ).contiguous()


class PCAKMeansBaseline:
    """Standardized PCA embedding with nearest normal-cluster distance scoring."""

    def __init__(self, pca_dim=64, n_clusters=16, max_iter=100, seed=0, eps=1e-6):
        self.pca_dim = pca_dim
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.seed = int(seed)
        self.eps = float(eps)
        self.mean = None
        self.std = None
        self.components = None
        self.explained_variance = None
        self.centroids = None

    def fit(self, features):
        x = torch.as_tensor(features, dtype=torch.float32).flatten(1)
        if x.shape[0] < 2:
            raise ValueError("PCA-KMeans fit requires at least two normal samples")
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0, unbiased=False).clamp_min(self.eps)
        z = (x - self.mean) / self.std

        _, s, vh = torch.linalg.svd(z, full_matrices=False)
        max_components = min(z.shape[0] - 1, z.shape[1], vh.shape[0])
        if self.pca_dim is None:
            pca_dim = min(64, max_components)
        elif isinstance(self.pca_dim, float) and 0.0 < self.pca_dim < 1.0:
            variance = s.pow(2)
            ratio = variance / variance.sum().clamp_min(self.eps)
            pca_dim = int(torch.searchsorted(torch.cumsum(ratio, dim=0), float(self.pca_dim)).item()) + 1
            pca_dim = min(max(1, pca_dim), max_components)
        else:
            pca_dim = min(max(1, int(self.pca_dim)), max_components)
        self.components = vh[:pca_dim].contiguous()
        variance = s.pow(2) / max(1, z.shape[0] - 1)
        self.explained_variance = variance[:pca_dim].contiguous()

        embeddings = z @ self.components.t()
        self.centroids = self._fit_kmeans(embeddings)
        return self

    def transform(self, features):
        self._check_fitted()
        x = torch.as_tensor(features, dtype=torch.float32).flatten(1)
        z = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return z @ self.components.to(x.device).t()

    def score(self, features):
        self._check_fitted()
        embeddings = self.transform(features)
        centroids = self.centroids.to(embeddings.device)
        distances = torch.cdist(embeddings.float(), centroids.float(), p=2)
        return distances.min(dim=1).values

    def state_dict(self):
        self._check_fitted()
        return {
            "mean": self.mean.cpu(),
            "std": self.std.cpu(),
            "components": self.components.cpu(),
            "explained_variance": self.explained_variance.cpu(),
            "centroids": self.centroids.cpu(),
            "pca_dim": self.pca_dim,
            "n_clusters": self.n_clusters,
            "max_iter": self.max_iter,
            "seed": self.seed,
            "eps": self.eps,
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"].float()
        self.std = state_dict["std"].float()
        self.components = state_dict["components"].float()
        self.explained_variance = state_dict.get("explained_variance")
        if self.explained_variance is not None:
            self.explained_variance = self.explained_variance.float()
        self.centroids = state_dict["centroids"].float()
        self.pca_dim = state_dict.get("pca_dim", self.components.shape[0])
        self.n_clusters = int(state_dict.get("n_clusters", self.centroids.shape[0]))
        self.max_iter = int(state_dict.get("max_iter", self.max_iter))
        self.seed = int(state_dict.get("seed", self.seed))
        self.eps = float(state_dict.get("eps", self.eps))
        return self

    def _fit_kmeans(self, embeddings):
        x = embeddings.float()
        n_samples = int(x.shape[0])
        n_clusters = min(max(1, int(self.n_clusters)), n_samples)
        generator = torch.Generator(device=x.device)
        generator.manual_seed(self.seed)
        initial = torch.randperm(n_samples, generator=generator, device=x.device)[:n_clusters]
        centroids = x[initial].clone()
        labels = None
        for _ in range(max(1, int(self.max_iter))):
            distances = torch.cdist(x, centroids, p=2)
            next_labels = distances.argmin(dim=1)
            if labels is not None and torch.equal(next_labels, labels):
                break
            labels = next_labels
            next_centroids = centroids.clone()
            for cluster_idx in range(n_clusters):
                mask = labels == cluster_idx
                if bool(mask.any().item()):
                    next_centroids[cluster_idx] = x[mask].mean(dim=0)
            centroids = next_centroids
        return centroids.contiguous()

    def _check_fitted(self):
        if self.mean is None or self.std is None or self.components is None or self.centroids is None:
            raise RuntimeError("PCAKMeansBaseline is not fitted")
