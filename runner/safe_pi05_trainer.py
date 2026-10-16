import json
import math
import os

import torch
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.pi05_feature_cache_dataset import Pi05FeatureCacheDataset
from models.baselines.safe import (
    SafeLSTMFailureModel,
    SafeMLPFailureModel,
    _failure_targets,
    _masked_mean,
)
from runner.ge_trainer_base import BaseGETrainer, logger
from runner.ge_trainer_video_anomaly import _binary_counts
from runner.observed_baseline_common import _best_f1_metrics, _score_summary, domains_for_split
from utils.memory_utils import free_memory, get_memory_statistics
from utils.model_utils import unwrap_model
from utils.optimizer_utils import get_optimizer


def _dict_cfg(value):
    return dict(value) if isinstance(value, dict) else {}


def _pad_tensor_sequence(values, pad_value=0):
    max_len = max(int(value.shape[0]) for value in values)
    shape = (len(values), max_len) + tuple(values[0].shape[1:])
    output = values[0].new_full(shape, pad_value)
    for row, value in enumerate(values):
        output[row, : value.shape[0]] = value
    return output


def _safe_collate(batch):
    output = {}
    tensor_keys = {
        "features": 0,
        "valid_mask": 0,
        "label": -1,
        "risk_label": -1,
        "frame_label": -1,
        "frame_anomaly_label": -1,
        "interaction_label": -1,
        "frame_index": -1,
        "window_start": -1,
        "window_end": -1,
        "valid_len": 0,
    }
    for key, pad_value in tensor_keys.items():
        if key in batch[0]:
            output[key] = _pad_tensor_sequence([item[key] for item in batch], pad_value=pad_value)

    scalar_keys = (
        "_cache_index",
        "success_labels",
        "episode_label",
        "total_frames",
        "episode_id",
        "source_episode_index",
    )
    for key in scalar_keys:
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch], dim=0)

    string_keys = ("domain_name", "source_h5", "source_h5_basename", "parquet_path", "source", "control_group")
    for key in string_keys:
        if key in batch[0]:
            output[key] = [str(item[key]) for item in batch]
    return output


def _move_batch(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


class Trainer(BaseGETrainer):
    """Train SAFE-style MLP/LSTM failure heads on cached pi05 hidden-state sequences."""

    def _safe_cfg(self):
        return _dict_cfg(getattr(self.args, "safe", {}))

    def _cache_cfg(self):
        return _dict_cfg(getattr(self.args, "pi05_feature", {}))

    def _cache_dir(self):
        cache_dir = getattr(self.args, "pi05_feature_cache_dir", None)
        if cache_dir in (None, ""):
            cache_dir = self._cache_cfg().get("cache_dir")
        if cache_dir in (None, ""):
            raise ValueError("SAFE pi05 training requires pi05_feature.cache_dir or pi05_feature_cache_dir")
        return os.path.expanduser(str(cache_dir))

    def _dataset_for_split(self, split):
        cache_cfg = self._cache_cfg()
        split_cfg = dict(self.args.data.get(split, {}))
        sequence_filter = split_cfg.get(
            "sequence_filter",
            cache_cfg.get(f"{split}_sequence_filter", cache_cfg.get("sequence_filter", "all")),
        )
        logger.info(
            f"{split} SAFE pi05 cache: {self._cache_dir()} "
            f"sequence_filter={sequence_filter}"
        )
        return Pi05FeatureCacheDataset(
            cache_dir=self._cache_dir(),
            split=split,
            max_open_shards=int(cache_cfg.get("max_open_shards", 16)),
            domains=domains_for_split(self.args, split),
            sequence_mode=True,
            sequence_filter=str(sequence_filter),
            min_sequence_length=int(cache_cfg.get("min_sequence_length", 1)),
        )

    def prepare_dataset(self) -> None:
        self.train_dataset = self._dataset_for_split("train")
        self.train_dataloader = self._make_loader(self.train_dataset, shuffle=True)
        logger.info(f">>>>>>>>>>>>>Total SAFE Train Episodes: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")
        if "val" in self.args.data:
            self.prepare_val_dataset()

    def prepare_val_dataset(self):
        self.val_dataset = self._dataset_for_split("val")
        self.val_dataloader = self._make_loader(self.val_dataset, shuffle=False)
        logger.info(f">>>>>>>>>>>>>Total SAFE Validation Episodes: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")

    def _make_loader(self, dataset, shuffle=False):
        loader_kwargs = {}
        if self.args.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = getattr(self.args, "persistent_workers", False)
            loader_kwargs["prefetch_factor"] = getattr(self.args, "prefetch_factor", 2)
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=getattr(self.args, "pin_memory", False),
            multiprocessing_context=None,
            collate_fn=_safe_collate,
            **loader_kwargs,
        )

    @torch.no_grad()
    def _infer_feature_dim(self):
        sample = self.train_dataset[0]
        return int(sample["features"].shape[-1])

    def prepare_models(self):
        logger.info("Initializing SAFE pi05 baseline")
        safe_cfg = self._safe_cfg()
        model_type = str(safe_cfg.get("model_type", "lstm")).lower()
        feature_dim = int(safe_cfg.get("feature_dim", self._infer_feature_dim()))
        kwargs = {
            "input_dim": feature_dim,
            "hidden_dim": int(safe_cfg.get("hidden_dim", 256)),
            "num_layers": int(safe_cfg.get("num_layers", 1 if model_type == "lstm" else 2)),
            "dropout": float(safe_cfg.get("dropout", 0.0)),
            "history_steps": int(safe_cfg.get("history_steps", -1 if model_type == "lstm" else 1)),
            "cumsum": bool(safe_cfg.get("cumsum", False)),
            "rmean": bool(safe_cfg.get("rmean", False)),
            "target_mode": str(safe_cfg.get("target_mode", "episode")),
        }
        if model_type == "mlp":
            self.safe_model = SafeMLPFailureModel(**kwargs)
        elif model_type == "lstm":
            self.safe_model = SafeLSTMFailureModel(**kwargs)
        else:
            raise ValueError(f"Unsupported safe.model_type={model_type}; use 'mlp' or 'lstm'")
        self.safe_model = self.safe_model.to(self.state.accelerator.device)
        self.anomaly_head = self.safe_model
        self.model_type = model_type
        self.safe_feature_dim = feature_dim
        self.normalize_features = bool(safe_cfg.get("normalize_features", True))
        self.feature_mean = None
        self.feature_std = None
        if self.normalize_features:
            self._fit_feature_normalizer()
        self._load_checkpoint()

    @torch.no_grad()
    def _fit_feature_normalizer(self):
        logger.info("Fitting SAFE feature normalizer from train split")
        total = torch.zeros(self.safe_feature_dim, dtype=torch.float64)
        total_sq = torch.zeros(self.safe_feature_dim, dtype=torch.float64)
        count = 0
        for idx in range(len(self.train_dataset)):
            sample = self.train_dataset[idx]
            features = sample["features"].double()
            valid = sample["valid_mask"].bool()
            if not bool(valid.any().item()):
                continue
            valid_features = features[valid]
            total += valid_features.sum(dim=0)
            total_sq += valid_features.square().sum(dim=0)
            count += int(valid_features.shape[0])
        if count <= 0:
            raise ValueError("Cannot fit SAFE feature normalizer from zero valid features")
        mean = total / float(count)
        var = (total_sq / float(count)) - mean.square()
        std = var.clamp_min(1e-12).sqrt()
        self.feature_mean = mean.float()
        self.feature_std = std.float()
        logger.info(
            f"SAFE feature normalizer fitted from {count} frames: "
            f"mean_abs={self.feature_mean.abs().mean().item():.4f}, "
            f"std_mean={self.feature_std.mean().item():.4f}"
        )

    def _load_checkpoint(self):
        checkpoint_path = getattr(self.args, "safe_checkpoint", None)
        if checkpoint_path in (None, ""):
            return
        checkpoint_path = os.path.expanduser(str(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"safe_checkpoint does not exist: {checkpoint_path}")
        logger.info(f"Loading SAFE checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing_keys, unexpected_keys = self.safe_model.load_state_dict(state_dict, strict=False)
        if isinstance(ckpt, dict) and ckpt.get("feature_mean") is not None and ckpt.get("feature_std") is not None:
            self.feature_mean = ckpt["feature_mean"].float()
            self.feature_std = ckpt["feature_std"].float()
            self.normalize_features = bool(ckpt.get("normalize_features", True))
        if missing_keys:
            logger.warning(f"Missing SAFE keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected SAFE keys when loading checkpoint: {unexpected_keys}")

    def prepare_trainable_parameters(self):
        logger.info("Enabling SAFE pi05 baseline")
        self.safe_model.train()
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_optimizer(self):
        logger.info("Initializing SAFE optimizer and lr scheduler")
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
        params = [param for param in self.safe_model.parameters() if param.requires_grad]
        self.state.num_trainable_parameters = sum(param.numel() for param in params)
        logger.info(f"Total SAFE trainable parameters: {self.state.num_trainable_parameters}")
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
        else:
            self.state.train_epochs = max(
                int(self.state.train_epochs or 0),
                math.ceil(int(self.state.train_steps) / max(1, num_update_steps_per_epoch)),
            )
        self.lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.lr_warmup_steps * self.state.accelerator.num_processes,
            num_training_steps=self.state.train_steps * self.state.accelerator.num_processes,
            num_cycles=self.args.lr_num_cycles,
            power=self.args.lr_power,
        )

    def prepare_for_training(self):
        self.safe_model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.safe_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )

    def _normalize_batch(self, batch):
        if not self.normalize_features:
            return batch
        if self.feature_mean is None or self.feature_std is None:
            raise ValueError("SAFE feature normalization is enabled but feature_mean/std are missing")
        mean = self.feature_mean.to(device=batch["features"].device, dtype=batch["features"].dtype)
        std = self.feature_std.to(device=batch["features"].device, dtype=batch["features"].dtype)
        batch = dict(batch)
        batch["features"] = (batch["features"].float() - mean.view(1, 1, -1)) / std.clamp_min(1e-6).view(1, 1, -1)
        return batch

    def _loss_from_batch(self, batch):
        batch = self._normalize_batch(batch)
        scores = self.safe_model(batch)
        target_mode = str(self._safe_cfg().get("target_mode", "episode"))
        targets, valid_mask = _failure_targets(batch, scores, target_mode=target_mode)
        losses = torch.nn.functional.binary_cross_entropy(
            scores.clamp(1e-6, 1 - 1e-6),
            targets,
            reduction="none",
        )
        return _masked_mean(losses, valid_mask), scores.detach(), targets.detach(), valid_mask.detach()

    def _episode_target_from_batch(self, batch):
        labels = batch["episode_label"].float().reshape(-1)
        return 1.0 - labels

    def _normal_mask_for_threshold(self, batch, targets):
        mode = str(self._safe_cfg().get("calibration_normal", "success_normal"))
        if mode == "success_normal":
            return targets < 0.5
        if mode == "all_normal":
            frame_labels = batch.get("frame_anomaly_label")
            valid_mask = batch.get("valid_mask")
            if frame_labels is None or valid_mask is None:
                return targets < 0.5
            frame_labels = frame_labels.float()
            valid_mask = valid_mask.bool()
            anomaly = (frame_labels >= 0.5) & valid_mask
            return ~anomaly.any(dim=1)
        raise ValueError(f"Unsupported safe.calibration_normal={mode}; use 'success_normal' or 'all_normal'")

    @torch.no_grad()
    def validate(self, global_step=0):
        if self.val_dataloader is None:
            return None
        self.safe_model.eval()
        scores = []
        targets = []
        normal_masks = []
        records = []
        model = unwrap_model(self.state.accelerator, self.safe_model)
        for batch in tqdm(
            self.val_dataloader,
            desc="Scoring SAFE validation",
            disable=not self.state.accelerator.is_local_main_process,
        ):
            batch = _move_batch(batch, self.state.accelerator.device)
            batch = self._normalize_batch(batch)
            step_scores = model(batch)
            valid_mask = batch["valid_mask"].bool()
            episode_scores = step_scores.masked_fill(~valid_mask, float("-inf")).max(dim=1).values
            batch_targets = self._episode_target_from_batch(batch).to(episode_scores)
            scores.append(episode_scores.detach().cpu())
            targets.append(batch_targets.detach().cpu())
            normal_masks.append(self._normal_mask_for_threshold(batch, batch_targets).detach().cpu())
            for row in range(episode_scores.shape[0]):
                records.append(
                    {
                        "score": float(episode_scores[row].detach().cpu().item()),
                        "target": float(batch_targets[row].detach().cpu().item()),
                        "domain_name": batch["domain_name"][row],
                        "episode_id": int(batch["episode_id"][row].detach().cpu().item()),
                        "source_h5_basename": batch["source_h5_basename"][row],
                    }
                )
        if not scores:
            self.safe_model.train()
            return None
        scores = torch.cat(scores).float()
        targets = torch.cat(targets).float()
        normal_masks = torch.cat(normal_masks).bool()
        normal_scores = scores[normal_masks]
        threshold = None
        threshold_metrics = None
        if normal_scores.numel() > 0:
            threshold = float(torch.quantile(normal_scores, float(getattr(self.args, "safe_val_quantile", 0.95))).item())
            threshold_metrics = _binary_counts(scores, targets, threshold=threshold)
        payload = {
            "score_direction": "higher_is_more_anomalous",
            "threshold": threshold,
            "threshold_metrics": threshold_metrics,
            "best_f1_metrics": _best_f1_metrics(scores, targets),
            "score_summary": _score_summary(scores, targets),
            "calibration_normal": str(self._safe_cfg().get("calibration_normal", "success_normal")),
        }
        output_dir = os.path.join(self.save_folder, f"step_{global_step}", "validation")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        with open(os.path.join(output_dir, "records.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        torch.save({"scores": scores, "targets": targets, "normal_masks": normal_masks}, os.path.join(output_dir, "scores.pt"))
        self.safe_model.train()
        return payload

    def train(self):
        logger.info("Starting SAFE training on pi05 hidden-state sequences")
        logger.info(f"Memory before training start: {json.dumps(get_memory_statistics(), indent=4)}")
        accelerator = self.state.accelerator
        global_step = int(getattr(self.args, "start_step", 0) or 0)
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=global_step,
            desc="Training SAFE pi05",
            disable=not accelerator.is_local_main_process,
        )
        for epoch in range(self.state.train_epochs):
            if global_step >= self.state.train_steps:
                break
            running_loss = 0.0
            self.safe_model.train()
            for batch in self.train_dataloader:
                batch = _move_batch(batch, accelerator.device)
                with accelerator.accumulate([self.safe_model]):
                    loss, _scores, targets, valid_mask = self._loss_from_batch(batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(self.safe_model.parameters(), self.args.max_grad_norm)
                    else:
                        grad_norm = None
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                loss = accelerator.reduce(loss.detach(), reduction="mean")
                target_ratio = accelerator.reduce(targets.float().mean(), reduction="mean")
                valid_ratio = accelerator.reduce(valid_mask.float().mean(), reduction="mean")
                running_loss += loss.item()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                logs = {
                    "loss": loss.item(),
                    "failure_target_ratio": target_ratio.item(),
                    "valid_ratio": valid_ratio.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)
                if global_step % self.args.steps_to_log == 0 and accelerator.is_main_process and self.writer is not None:
                    self.writer.add_scalar("SAFE/loss", loss.item(), global_step)
                    self.writer.add_scalar("SAFE/failure_target_ratio", target_ratio.item(), global_step)
                if global_step % self.args.steps_to_save == 0:
                    self._save_head(global_step)
                if (
                    getattr(self.args, "steps_to_val", None) is not None
                    and int(self.args.steps_to_val) > 0
                    and global_step % int(self.args.steps_to_val) == 0
                ):
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        self.validate(global_step)
                    accelerator.wait_for_everyone()
                if global_step >= self.state.train_steps:
                    break
            if accelerator.is_main_process and self.writer is not None:
                self.writer.add_scalar("SAFE/epoch_loss", running_loss / max(1, len(self.train_dataloader)), epoch)
        progress_bar.close()
        self._save_head(global_step)
        accelerator.wait_for_everyone()
        free_memory()
        logger.info(f"Memory after training end: {json.dumps(get_memory_statistics(), indent=4)}")
        accelerator.end_training()

    def _save_head(self, global_step):
        self.state.accelerator.wait_for_everyone()
        if not self.state.accelerator.is_main_process:
            return
        model_to_save = unwrap_model(self.state.accelerator, self.safe_model)
        save_dir = os.path.join(self.save_folder, f"step_{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        safe_cfg = self._safe_cfg()
        safe_cfg["feature_dim"] = int(model_to_save.input_dim)
        torch.save(
            {
                "state_dict": model_to_save.state_dict(),
                "safe": safe_cfg,
                "pi05_feature": self._cache_cfg(),
                "normalize_features": bool(self.normalize_features),
                "feature_mean": None if self.feature_mean is None else self.feature_mean.cpu(),
                "feature_std": None if self.feature_std is None else self.feature_std.cpu(),
                "model_type": self.model_type,
                "baseline": "safe_pi05",
                "observation": "pi05 hidden-state sequence cache -> SAFE failure score",
            },
            os.path.join(save_dir, "safe_pi05.pt"),
        )
