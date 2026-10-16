import json
import math
import os

import torch
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from data.anomaly_feature_cache_dataset import AnomalyFeatureCacheDataset
from models.baselines.logpzo import LogpZOModel, build_wm_logpzo_features
from runner.ge_trainer_base import BaseGETrainer, logger
from runner.ge_trainer_video_anomaly import _binary_counts
from runner.observed_baseline_common import domains_for_split
from runner.pca_kmeans_trainer import _best_f1_metrics, _score_summary
from utils.memory_utils import free_memory, get_memory_statistics
from utils.model_utils import unwrap_model
from utils.optimizer_utils import get_optimizer


def _dict_cfg(value):
    return dict(value) if isinstance(value, dict) else {}


class Trainer(BaseGETrainer):
    """Train success-only logpZO on cached action-force WM sequence features."""

    def _wm_feature_config(self):
        return _dict_cfg(getattr(self.args, "wm_feature", {}))

    def _wm_cache_dir(self):
        cfg = self._wm_feature_config()
        cache_dir = cfg.get("cache_dir", getattr(self.args, "anomaly_feature_cache_dir", None))
        if cache_dir in (None, ""):
            raise ValueError("logpZO WM training requires wm_feature.cache_dir")
        return os.path.expanduser(str(cache_dir))

    def _max_open_shards(self):
        cfg = self._wm_feature_config()
        return int(cfg.get("max_open_shards", getattr(self.args, "anomaly_cache_max_open_shards", 4)))

    def _make_dataset(self, split):
        return AnomalyFeatureCacheDataset(
            cache_dir=self._wm_cache_dir(),
            split=split,
            max_open_shards=self._max_open_shards(),
            domains=domains_for_split(self.args, split),
        )

    def prepare_dataset(self) -> None:
        logger.info(f"Training WM logpZO from action-force cache: {self._wm_cache_dir()}")
        self.train_dataset = self._make_dataset("train")
        self.train_dataloader = self._make_loader(self.train_dataset, shuffle=True)
        logger.info(f">>>>>>>>>>>>>Total WM logpZO Train Samples: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")
        if "val" in self.args.data:
            self.prepare_val_dataset()

    def prepare_val_dataset(self):
        val_index_path = os.path.join(self._wm_cache_dir(), "val", "index.json")
        if not os.path.isfile(val_index_path):
            self.val_dataset = None
            self.val_dataloader = None
            logger.warning("No val WM feature cache found; validation will be skipped")
            return
        self.val_dataset = self._make_dataset("val")
        self.val_dataloader = self._make_loader(self.val_dataset, shuffle=False)
        logger.info(f">>>>>>>>>>>>>Total WM logpZO Validation Samples: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")

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
            **loader_kwargs,
        )

    def prepare_models(self):
        logger.info("Initializing WM logpZO baseline")
        self.vae = None
        feature_cfg = self._wm_feature_config()
        self.fit_filter = str(feature_cfg.get("fit_filter", "normal_episode_windows"))
        self.feature_mode = str(feature_cfg.get("feature_mode", "action_force_logpzo_binned"))
        self.token_proj_dim = int(feature_cfg.get("token_proj_dim", 64))

        logpzo_cfg = dict(getattr(self.args, "logpzo", {}))
        logpzo_cfg.setdefault("score_reduction", "max")
        feature_dim = int(logpzo_cfg.get("feature_dim", self._infer_feature_dim()))
        self.logpzo_model = LogpZOModel(
            feature_dim=feature_dim,
            hidden_dim=int(logpzo_cfg.get("hidden_dim", 256)),
            num_layers=int(logpzo_cfg.get("num_layers", 4)),
            time_embed_dim=int(logpzo_cfg.get("time_embed_dim", 128)),
            dropout=float(logpzo_cfg.get("dropout", 0.0)),
            time_scale=float(logpzo_cfg.get("time_scale", 100.0)),
            score_reduction=str(logpzo_cfg.get("score_reduction", "max")),
            normalize_input=bool(logpzo_cfg.get("normalize_input", True)),
        ).to(self.state.accelerator.device)
        self.anomaly_head = self.logpzo_model
        self._load_checkpoint()

    @torch.no_grad()
    def _infer_feature_dim(self):
        if len(self.train_dataset) == 0:
            raise ValueError("Cannot infer WM logpZO feature_dim from an empty train dataset.")
        sample = default_collate([self.train_dataset[0]])
        features, _ = self._features_from_batch(sample)
        return int(features.shape[-1])

    def _load_checkpoint(self):
        checkpoint_path = getattr(self.args, "logpzo_checkpoint", None)
        if checkpoint_path in (None, ""):
            return
        checkpoint_path = os.path.expanduser(str(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"logpzo_checkpoint does not exist: {checkpoint_path}")
        logger.info(f"Loading WM logpZO checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing_keys, unexpected_keys = self.logpzo_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing WM logpZO keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected WM logpZO keys when loading checkpoint: {unexpected_keys}")

    def prepare_trainable_parameters(self):
        logger.info("Training only the WM logpZO density model")
        self.logpzo_model.train()
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_optimizer(self):
        logger.info("Initializing WM logpZO optimizer and lr scheduler")
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
        params = [p for p in self.logpzo_model.parameters() if p.requires_grad]
        self.state.num_trainable_parameters = sum(p.numel() for p in params)
        logger.info(f"Total WM logpZO trainable parameters: {self.state.num_trainable_parameters}")
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
        self.logpzo_model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
            self.logpzo_model,
            self.optimizer,
            self.train_dataloader,
            self.lr_scheduler,
        )

    @torch.no_grad()
    def _features_from_batch(self, batch):
        device = self.state.accelerator.device
        features, token_mask = build_wm_logpzo_features(
            pred_latents=batch["pred_latents"].to(device=device, dtype=torch.float32, non_blocking=True),
            pred_force=batch["pred_force"].to(device=device, dtype=torch.float32, non_blocking=True),
            interaction_tokens=batch["interaction_tokens"].to(device=device, dtype=torch.float32, non_blocking=True),
            valid_len=batch.get("valid_len"),
            token_proj_dim=self.token_proj_dim,
        )
        return features.float().contiguous(), token_mask.to(device=device, dtype=torch.bool, non_blocking=True)

    def _normal_mask_from_batch(self, batch):
        if "label" not in batch:
            raise ValueError("WM logpZO training requires batch['label']")
        device = self.state.accelerator.device
        window_normal = batch["label"].to(device=device, dtype=torch.float32).reshape(-1) >= 0.5
        if self.fit_filter == "normal_windows":
            return window_normal
        if self.fit_filter == "normal_episode_windows":
            if "episode_label" not in batch:
                raise ValueError("wm_feature.fit_filter='normal_episode_windows' requires cached episode labels")
            episode_normal = batch["episode_label"].to(device=device, dtype=torch.float32).reshape(-1) >= 0.5
            return window_normal & episode_normal
        raise ValueError(f"Unsupported wm_feature.fit_filter={self.fit_filter}")

    def _target_from_batch(self, batch):
        labels = batch["label"].float().reshape(-1)
        return 1.0 - labels

    @torch.no_grad()
    def validate(self, global_step=0):
        if self.val_dataloader is None:
            return None
        self.logpzo_model.eval()
        scores = []
        targets = []
        model = unwrap_model(self.state.accelerator, self.logpzo_model)
        for batch in tqdm(
            self.val_dataloader,
            desc="Scoring WM logpZO validation",
            disable=not self.state.accelerator.is_local_main_process,
        ):
            features, token_mask = self._features_from_batch(batch)
            scores.append(model.score(features, token_mask=token_mask).detach().cpu())
            targets.append(self._target_from_batch(batch).cpu())
        if not scores:
            self.logpzo_model.train()
            return None
        scores = torch.cat(scores).float()
        targets = torch.cat(targets).float()
        normal_scores = scores[targets < 0.5]
        threshold = None
        threshold_metrics = None
        if normal_scores.numel() > 0:
            threshold = float(torch.quantile(normal_scores, float(getattr(self.args, "logpzo_val_quantile", 0.95))).item())
            threshold_metrics = _binary_counts(scores, targets, threshold=threshold)
        payload = {
            "score_direction": "higher_is_more_anomalous",
            "threshold": threshold,
            "threshold_metrics": threshold_metrics,
            "best_f1_metrics": _best_f1_metrics(scores, targets),
            "score_summary": _score_summary(scores, targets),
        }
        output_dir = os.path.join(self.save_folder, f"step_{global_step}", "validation")
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        torch.save({"scores": scores, "targets": targets}, os.path.join(output_dir, "scores.pt"))
        self.logpzo_model.train()
        return payload

    def train(self):
        logger.info("Starting WM logpZO training on cached action-force sequence features")
        logger.info(f"Memory before training start: {json.dumps(get_memory_statistics(), indent=4)}")
        accelerator = self.state.accelerator
        global_step = int(getattr(self.args, "start_step", 0) or 0)
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=global_step,
            desc="Training WM logpZO",
            disable=not accelerator.is_local_main_process,
        )
        for epoch in range(self.state.train_epochs):
            if global_step >= self.state.train_steps:
                break
            running_loss = 0.0
            self.logpzo_model.train()
            for batch in self.train_dataloader:
                with accelerator.accumulate([self.logpzo_model]):
                    normal_mask = self._normal_mask_from_batch(batch)
                    features, token_mask = self._features_from_batch(batch)
                    model = unwrap_model(accelerator, self.logpzo_model)
                    loss = model.flow_matching_loss(features, sample_mask=normal_mask, token_mask=token_mask)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(self.logpzo_model.parameters(), self.args.max_grad_norm)
                    else:
                        grad_norm = None
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                loss = accelerator.reduce(loss.detach(), reduction="mean")
                normal_ratio = accelerator.reduce(normal_mask.detach().float().mean(), reduction="mean")
                valid_token_ratio = accelerator.reduce(token_mask.detach().float().mean(), reduction="mean")
                running_loss += loss.item()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                logs = {
                    "loss": loss.item(),
                    "normal_ratio": normal_ratio.item(),
                    "valid_token_ratio": valid_token_ratio.item(),
                    "lr": self.lr_scheduler.get_last_lr()[0],
                }
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm
                progress_bar.set_postfix(logs)
                accelerator.log(logs, step=global_step)
                if global_step % self.args.steps_to_log == 0 and accelerator.is_main_process and self.writer is not None:
                    self.writer.add_scalar("WMLogpZO/loss", loss.item(), global_step)
                    self.writer.add_scalar("WMLogpZO/normal_ratio", normal_ratio.item(), global_step)
                    self.writer.add_scalar("WMLogpZO/valid_token_ratio", valid_token_ratio.item(), global_step)
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
                self.writer.add_scalar("WMLogpZO/epoch_loss", running_loss / max(1, len(self.train_dataloader)), epoch)
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
        model_to_save = unwrap_model(self.state.accelerator, self.logpzo_model)
        save_dir = os.path.join(self.save_folder, f"step_{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        feature_cfg = self._wm_feature_config()
        feature_cfg["token_proj_dim"] = int(self.token_proj_dim)
        logpzo_cfg = dict(getattr(self.args, "logpzo", {}))
        logpzo_cfg["feature_dim"] = int(model_to_save.feature_dim)
        logpzo_cfg.setdefault("score_reduction", str(model_to_save.score_reduction))
        torch.save(
            {
                "state_dict": model_to_save.state_dict(),
                "logpzo": logpzo_cfg,
                "wm_feature": feature_cfg,
                "feature_mode": self.feature_mode,
                "fit_filter": self.fit_filter,
                "baseline": "cached_action_force_wm_logpzo",
                "observation": "pred_latents pooled to T=5 + pred_force binned + projected interaction_tokens binned -> [B,T,D]",
            },
            os.path.join(save_dir, "logpzo_wm.pt"),
        )
