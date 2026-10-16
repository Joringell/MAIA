import torch
from torch import nn
import torch.nn.functional as F


def _masked_mean(values, valid_mask):
    valid_mask = valid_mask.to(values).clamp(0, 1)
    denom = valid_mask.sum().clamp_min(1.0)
    return (values * valid_mask).sum() / denom


def _failure_targets(batch, scores, target_mode="episode"):
    target_mode = str(target_mode)
    if target_mode == "frame" and "frame_anomaly_label" in batch:
        frame_labels = batch["frame_anomaly_label"].to(scores).float()
        valid = (frame_labels >= 0) & (batch["valid_mask"].to(scores) > 0)
        if bool(valid.any().item()):
            targets = frame_labels.clamp(0, 1)
            return targets, valid.float()
    if target_mode not in ("episode", "frame"):
        raise ValueError(f"Unsupported SAFE target_mode={target_mode}; use 'episode' or 'frame'")

    labels = batch["episode_label"].float()
    if bool((labels < 0).any().item()):
        raise ValueError("SAFE-style training requires episode_label for every sample")
    targets = 1.0 - labels.to(scores)
    return targets[:, None].expand_as(scores), batch["valid_mask"].to(scores).float()


class SafeMLPFailureModel(nn.Module):
    """Per-timestep MLP failure head over pi05 hidden features."""

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=2,
        dropout=0.0,
        history_steps=1,
        cumsum=False,
        rmean=False,
        target_mode="episode",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_steps = max(1, int(history_steps))
        self.cumsum = bool(cumsum)
        self.rmean = bool(rmean)
        self.target_mode = str(target_mode)

        total_input_dim = self.input_dim * self.history_steps
        layers = []
        if int(num_layers) <= 1:
            layers.append(nn.Linear(total_input_dim, 1))
        else:
            layers.extend([nn.Linear(total_input_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(float(dropout))])
            for _ in range(int(num_layers) - 2):
                layers.extend([nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(float(dropout))])
            layers.append(nn.Linear(self.hidden_dim, 1))
        self.projector = nn.Sequential(*layers)

    def _history_features(self, features):
        if self.history_steps <= 1:
            return features
        batch, time, dim = features.shape
        padded = F.pad(features, (0, 0, self.history_steps - 1, 0), mode="constant", value=0)
        windows = [padded[:, t : t + self.history_steps] for t in range(time)]
        return torch.stack(windows, dim=1).reshape(batch, time, dim * self.history_steps)

    def forward(self, batch):
        features = batch["features"].float()
        scores = torch.sigmoid(self.projector(self._history_features(features))).squeeze(-1)
        if self.cumsum or self.rmean:
            scores = scores.cumsum(dim=1)
            if self.rmean:
                denom = torch.arange(1, scores.shape[1] + 1, device=scores.device, dtype=scores.dtype)
                scores = scores / denom[None]
        return scores

    def loss(self, batch):
        scores = self(batch)
        targets, valid_mask = _failure_targets(batch, scores, target_mode=self.target_mode)
        losses = F.binary_cross_entropy(scores.clamp(1e-6, 1 - 1e-6), targets, reduction="none")
        return _masked_mean(losses, valid_mask)

    @torch.no_grad()
    def episode_scores(self, batch):
        scores = self(batch)
        valid_mask = batch["valid_mask"].bool()
        masked = scores.masked_fill(~valid_mask, float("-inf"))
        return masked.max(dim=1).values


class SafeLSTMFailureModel(nn.Module):
    """Sequence LSTM failure head over pi05 hidden features."""

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=1,
        dropout=0.0,
        history_steps=-1,
        cumsum=False,
        rmean=False,
        target_mode="episode",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.history_steps = int(history_steps)
        self.cumsum = bool(cumsum)
        self.rmean = bool(rmean)
        self.target_mode = str(target_mode)
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            int(num_layers),
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, batch):
        features = batch["features"].float()
        batch_size, time, dim = features.shape
        if dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {dim}")

        if self.history_steps < 0:
            out, _ = self.lstm(features)
        else:
            n = max(1, self.history_steps)
            padded = F.pad(features, (0, 0, n - 1, 0), mode="constant", value=0)
            windows = [padded[:, t : t + n] for t in range(time)]
            seq = torch.stack(windows, dim=1).reshape(batch_size * time, n, dim)
            out, _ = self.lstm(seq)
            out = out[:, -1].reshape(batch_size, time, self.hidden_dim)

        scores = torch.sigmoid(self.fc(self.dropout(out))).squeeze(-1)
        if self.cumsum or self.rmean:
            scores = scores.cumsum(dim=1)
            if self.rmean:
                denom = torch.arange(1, time + 1, device=scores.device, dtype=scores.dtype)
                scores = scores / denom[None]
        return scores

    def loss(self, batch):
        scores = self(batch)
        targets, valid_mask = _failure_targets(batch, scores, target_mode=self.target_mode)
        losses = F.binary_cross_entropy(scores.clamp(1e-6, 1 - 1e-6), targets, reduction="none")
        return _masked_mean(losses, valid_mask)

    @torch.no_grad()
    def episode_scores(self, batch):
        scores = self(batch)
        valid_mask = batch["valid_mask"].bool()
        masked = scores.masked_fill(~valid_mask, float("-inf"))
        return masked.max(dim=1).values
