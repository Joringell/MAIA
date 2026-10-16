import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.video_action_anomaly_feature_cache_dataset import VideoActionAnomalyFeatureCacheDataset
from runner.ge_trainer_base import BaseGETrainer, logger
from utils import import_custom_class
from utils.memory_utils import free_memory, get_memory_statistics
from utils.optimizer_utils import get_optimizer


def _binary_counts(scores, targets, threshold=0.5):
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1) >= 0.5
    pred = scores >= float(threshold)
    tp = int((pred & targets).sum().item())
    tn = int((~pred & ~targets).sum().item())
    fp = int((pred & ~targets).sum().item())
    fn = int((~pred & targets).sum().item())
    total = max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_samples": tp + tn + fp + fn,
    }


def _plot_confusion(ax, metrics, title):
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_xticks([0, 1], labels=["normal", "anomaly"])
    ax.set_yticks([0, 1], labels=["normal", "anomaly"])
    if metrics is None or int(metrics.get("num_samples", 0)) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=12)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        return None
    matrix = torch.tensor([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]], dtype=torch.float32)
    image = ax.imshow(matrix.numpy(), cmap="Blues")
    max_value = max(1.0, float(matrix.max().item()))
    row_sums = matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
    row_pct = matrix / row_sums
    for row in range(2):
        for col in range(2):
            value = int(matrix[row, col].item())
            color = "white" if value > 0.55 * max_value else "black"
            ax.text(
                col,
                row,
                f"{value}\n{row_pct[row, col].item() * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=11,
            )
    ax.set_title(f"{title}\nthr={metrics['threshold']:.2f}")
    subtitle = (
        f"acc={metrics['accuracy']:.3f}  P={metrics['precision']:.3f}  "
        f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}"
    )
    ax.text(0.5, -0.25, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=9)
    return image


class Trainer(BaseGETrainer):
    """Train an anomaly head on frozen action-conditioned video WM outputs.

    The frozen WM is the action-only GE path:
      history video + text + future actions -> future video latents + interaction tokens

    Force is intentionally not read or passed anywhere.
    """

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        super().__init__(config_file=config_file, to_log=to_log, output_dir=output_dir)
        if getattr(self.args, "return_force", False):
            raise ValueError("Video-action anomaly trainer must not set return_force=True.")
        self.action_condition_dim = self._resolve_action_dim()
        diffusion_cfg = self.args.diffusion_model["config"]
        diffusion_cfg["use_action_adapter"] = True
        diffusion_cfg["action_dim"] = self.action_condition_dim
        diffusion_cfg["action_condition_dim"] = self.action_condition_dim

    def _resolve_action_dim(self) -> int:
        diffusion_cfg = self.args.diffusion_model["config"]
        action_dim = self.args.data["train"].get("action_dim")
        if action_dim is None:
            action_dim = diffusion_cfg.get("action_condition_dim", diffusion_cfg.get("action_dim"))
        if action_dim is None or int(action_dim) <= 0:
            raise ValueError("action_dim or action_condition_dim must be provided and positive.")
        return int(action_dim)

    def _use_feature_cache(self):
        return bool(getattr(self.args, "anomaly_feature_cache_dir", None))

    def _make_cached_loader(self, split):
        dataset = VideoActionAnomalyFeatureCacheDataset(
            cache_dir=getattr(self.args, "anomaly_feature_cache_dir"),
            split=split,
            max_open_shards=int(getattr(self.args, "anomaly_cache_max_open_shards", 4)),
        )
        loader_kwargs = {}
        pin_memory = getattr(self.args, "pin_memory", False)
        if self.args.dataloader_num_workers > 0:
            loader_kwargs["persistent_workers"] = getattr(self.args, "persistent_workers", False)
            loader_kwargs["prefetch_factor"] = getattr(self.args, "prefetch_factor", 2)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == "train"),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=None,
            **loader_kwargs,
        )
        return dataset, loader

    def prepare_dataset(self) -> None:
        if not self._use_feature_cache():
            return super().prepare_dataset()

        logger.info(f"Training video-action anomaly head from feature cache: {self.args.anomaly_feature_cache_dir}")
        self.train_dataset, self.train_dataloader = self._make_cached_loader("train")
        val_index_path = os.path.join(
            os.path.expanduser(str(self.args.anomaly_feature_cache_dir)),
            "val",
            "index.json",
        )
        if os.path.isfile(val_index_path):
            self.val_dataset, self.val_dataloader = self._make_cached_loader("val")
        else:
            self.val_dataset = None
            self.val_dataloader = None
            logger.warning("No val video-action anomaly feature cache found; validation will be skipped")
        logger.info(f">>>>>>>>>>>>>Total Cached Train Samples: {len(self.train_dataset)}<<<<<<<<<<<<<<<<<<\n")

    def prepare_val_dataset(self) -> None:
        if self._use_feature_cache():
            return
        if not hasattr(self.args, "val_data_class"):
            self.args.val_data_class = self.args.train_data_class
        if not hasattr(self.args, "val_data_class_path"):
            self.args.val_data_class_path = self.args.train_data_class_path
        logger.info(f"Validation Dataset: {self.args.val_data_class}")

        val_dataset_class = import_custom_class(
            self.args.val_data_class,
            self.args.val_data_class_path,
        )
        self.val_dataset = val_dataset_class(**self.args.data["val"])

        val_loader_kwargs = {}
        pin_memory = getattr(self.args, "pin_memory", False)
        if self.args.dataloader_num_workers > 0:
            val_loader_kwargs["persistent_workers"] = getattr(self.args, "persistent_workers", False)
            val_loader_kwargs["prefetch_factor"] = getattr(self.args, "prefetch_factor", 2)
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=getattr(self.args, "val_shuffle", False),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=None,
            **val_loader_kwargs,
        )
        logger.info(f">>>>>>>>>>>>>Total Action Anomaly Validation Samples: {len(self.val_dataset)}<<<<<<<<<<<<<<<<<<\n")

    def prepare_models(self):
        if self._use_feature_cache():
            self.vae = None
            self.text_encoder = None
            self.diffusion_model = None
            self.scheduler = None
        else:
            super().prepare_models()
        self._prepare_anomaly_head()

    def _prepare_anomaly_head(self):
        cfg = self.args.diffusion_model["config"]
        inner_dim = int(cfg["num_attention_heads"]) * int(cfg["attention_head_dim"])
        head_cfg = getattr(self.args, "anomaly_head", {})
        self.anomaly_target = str(getattr(self.args, "anomaly_target", "label"))
        self.anomaly_positive_is = str(getattr(self.args, "anomaly_positive_is", "anomaly"))
        self.anomaly_head_class_path = str(
            getattr(self.args, "anomaly_head_class_path", "models/anomaly_head/video_action_anomaly.py")
        )
        self.anomaly_head_class = str(getattr(self.args, "anomaly_head_class", "VideoActionAnomalyHead_v1"))
        checkpoint_path = getattr(self.args, "anomaly_head_checkpoint", None)
        if checkpoint_path not in (None, "") and os.path.isfile(os.path.expanduser(str(checkpoint_path))):
            ckpt = torch.load(os.path.expanduser(str(checkpoint_path)), map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(ckpt, dict) and ckpt.get("anomaly_head_class"):
                self.anomaly_head_class = str(ckpt["anomaly_head_class"])
                self.anomaly_head_class_path = str(ckpt.get("anomaly_head_class_path", self.anomaly_head_class_path))
            elif not any(str(key).startswith(("window_head.", "risk_head.")) for key in state_dict):
                self.anomaly_head_class = "VideoActionAnomalyHead"
        head_class = import_custom_class(self.anomaly_head_class, self.anomaly_head_class_path)
        head_kwargs = {
            "interaction_dim": int(head_cfg.get("interaction_dim", inner_dim)),
            "hidden_dim": int(head_cfg.get("hidden_dim", 256)),
            "feature_dim": int(head_cfg.get("feature_dim", 128)),
            "dropout": float(head_cfg.get("dropout", 0.1)),
            "latent_channels": int(head_cfg.get("latent_channels", cfg.get("out_channels", 128))),
            "temporal_layers": int(head_cfg.get("temporal_layers", 2)),
            "temporal_heads": int(head_cfg.get("temporal_heads", 4)),
            "max_steps": int(head_cfg.get("max_steps", 256)),
            "topk_ratio": float(head_cfg.get("topk_ratio", 0.2)),
        }
        if self.anomaly_head_class.endswith("_v1"):
            head_kwargs.update(
                {
                    "window_pool": str(head_cfg.get("window_pool", "topk")),
                    "risk_pool": str(head_cfg.get("risk_pool", "mean")),
                }
            )
        self.anomaly_head = head_class(**head_kwargs).to(self.state.accelerator.device)
        self._load_anomaly_head_checkpoint()

    def _load_anomaly_head_checkpoint(self):
        checkpoint_path = getattr(self.args, "anomaly_head_checkpoint", None)
        if checkpoint_path in (None, ""):
            return
        checkpoint_path = os.path.expanduser(str(checkpoint_path))
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"anomaly_head_checkpoint does not exist: {checkpoint_path}")
        logger.info(f"Loading action anomaly head checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        missing_keys, unexpected_keys = self.anomaly_head.load_state_dict(state_dict, strict=False)
        if missing_keys:
            logger.warning(f"Missing anomaly head keys when loading checkpoint: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected anomaly head keys when loading checkpoint: {unexpected_keys}")

    def prepare_trainable_parameters(self):
        logger.info("Freezing action WM and enabling action anomaly head")
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        if self.diffusion_model is not None:
            self.diffusion_model.requires_grad_(False)
            self.diffusion_model.eval()
        self.anomaly_head.train()
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_optimizer(self):
        logger.info("Initializing action anomaly head optimizer and lr scheduler")
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

        params = [p for p in self.anomaly_head.parameters() if p.requires_grad]
        self.state.num_trainable_parameters = sum(p.numel() for p in params)
        logger.info(f"Total action anomaly head trainable parameters: {self.state.num_trainable_parameters}")

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
        self.anomaly_head, self.optimizer, self.train_dataloader, self.lr_scheduler = self.state.accelerator.prepare(
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
            self.diffusion_model,
        )
        pipe.set_progress_bar_config(disable=True)
        return pipe

    def _target_from_batch(self, batch, device):
        if self.anomaly_target == "risk_label":
            if "risk_label" not in batch:
                raise ValueError("anomaly_target='risk_label' requires batch['risk_label']")
            return batch["risk_label"].to(device=device, dtype=torch.float32).reshape(-1)
        if self.anomaly_target != "label":
            raise ValueError(f"Unsupported anomaly_target={self.anomaly_target}")
        if "label" not in batch:
            raise ValueError("anomaly_target='label' requires batch['label']")
        label = batch["label"].to(device=device, dtype=torch.float32).reshape(-1)
        if self.anomaly_positive_is == "anomaly":
            return 1.0 - label
        if self.anomaly_positive_is == "normal":
            return label
        raise ValueError(f"Unsupported anomaly_positive_is={self.anomaly_positive_is}")

    def _prepare_video_action_batch(self, batch, accelerator, weight_dtype):
        video = batch["video"].to(accelerator.device, dtype=weight_dtype, non_blocking=True).contiguous()
        batch_size, _, n_view, _, height, width = video.shape
        video = rearrange(video, "b c v t h w -> (b v) c t h w")
        mem_size = int(self.args.data["train"]["n_previous"])
        action_chunk = int(self.args.data["train"].get("action_chunk", self.args.data["train"].get("future_steps")))
        mem = video[:, :, :mem_size]
        future_actions = batch["actions"][:, -action_chunk:].to(
            accelerator.device,
            dtype=weight_dtype,
            non_blocking=True,
        ).contiguous()
        if future_actions.shape[-1] != self.action_condition_dim:
            raise ValueError(f"Expected action dim {self.action_condition_dim}, got {future_actions.shape[-1]}")
        return mem, future_actions, batch_size, n_view, mem_size, height, width

    def _latent_chunk(self):
        if getattr(self.args, "anomaly_latent_chunk", None) not in (None, ""):
            return int(self.args.anomaly_latent_chunk)
        return int(self.args.data["train"]["chunk"])

    def _run_frozen_inference(
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
        if hasattr(pipe.scheduler, "_step_index"):
            pipe.scheduler._step_index = None
        return pipe.infer(
            image=mem,
            n_prev=mem_size,
            prompt=batch["caption"],
            negative_prompt="",
            height=height,
            width=width,
            chunk=self._latent_chunk(),
            num_inference_steps=int(getattr(self.args, "num_inference_step", 5)),
            guidance_scale=float(getattr(self.args, "anomaly_guidance_scale", 1.0)),
            return_dict=False,
            return_init=False,
            n_view=n_view,
            return_video=False,
            return_interaction=True,
            return_latents=True,
            action_chunk=int(self.args.data["train"].get("action_chunk", self.args.data["train"].get("future_steps"))),
            future_actions=future_actions,
            noise_seed=int(getattr(self.args, "val_noise_seed", getattr(self.args, "seed", 42)) or 42),
            pixel_wise_timestep=bool(getattr(self.args, "pixel_wise_timestep", True)),
            n_chunk=1,
            show_progress=False,
        )[0]

    def _features_from_batch(self, pipe, batch, accelerator, weight_dtype):
        if self._use_feature_cache():
            interaction_tokens = batch["interaction_tokens"].to(
                device=accelerator.device,
                dtype=weight_dtype,
                non_blocking=True,
            ).contiguous()
            pred_latents = batch["pred_latents"].to(
                device=accelerator.device,
                dtype=weight_dtype,
                non_blocking=True,
            ).contiguous()
            return interaction_tokens, pred_latents

        mem, future_actions, batch_size, n_view, mem_size, height, width = self._prepare_video_action_batch(
            batch,
            accelerator,
            weight_dtype,
        )
        with torch.no_grad():
            preds = self._run_frozen_inference(
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
        pred_latents = preds["pred_latents"]
        pred_latents = pred_latents.reshape(
            batch_size,
            n_view,
            pred_latents.shape[1],
            pred_latents.shape[2],
            pred_latents.shape[3],
            pred_latents.shape[4],
        )
        return preds["interaction_tokens"], pred_latents

    def _frame_target_from_batch(self, batch, device):
        frame_target = batch.get("frame_anomaly_label")
        if frame_target is None:
            return None, None
        frame_target = frame_target.to(device=device, dtype=torch.float32)
        frame_mask = frame_target >= 0.0
        if not bool(frame_mask.any().item()):
            return None, None
        return frame_target.clamp(0.0, 1.0), frame_mask

    def _compute_anomaly_loss(self, head_out, target, batch):
        window_logit = head_out["window_logit"]
        step_logits = head_out["step_logits"]
        pos_weight = torch.tensor(
            float(getattr(self.args, "anomaly_pos_weight", 1.0)),
            device=window_logit.device,
            dtype=window_logit.dtype,
        )
        loss_window = F.binary_cross_entropy_with_logits(
            window_logit,
            target.to(window_logit.dtype),
            pos_weight=pos_weight,
        )

        loss_risk = torch.zeros((), device=window_logit.device, dtype=window_logit.dtype)
        if "risk_label" in batch and float(getattr(self.args, "risk_loss_scale", 0.0)) > 0.0:
            risk_target = batch["risk_label"].to(device=window_logit.device, dtype=window_logit.dtype).reshape(-1)
            risk_prob = head_out["risk_prob"].to(window_logit.dtype).clamp(1e-4, 1.0 - 1e-4)
            loss_risk = F.binary_cross_entropy(risk_prob, risk_target)

        loss_step = torch.zeros((), device=window_logit.device, dtype=window_logit.dtype)
        frame_target, frame_mask = self._frame_target_from_batch(batch, window_logit.device)
        if frame_target is not None and float(getattr(self.args, "step_loss_scale", 0.0)) > 0.0:
            if frame_target.shape != step_logits.shape:
                frame_target = F.interpolate(
                    frame_target.unsqueeze(1),
                    size=step_logits.shape[1],
                    mode="nearest",
                ).squeeze(1)
                frame_mask = F.interpolate(
                    frame_mask.float().unsqueeze(1),
                    size=step_logits.shape[1],
                    mode="nearest",
                ).squeeze(1) > 0.5
            per_step = F.binary_cross_entropy_with_logits(
                step_logits,
                frame_target.to(step_logits.dtype),
                reduction="none",
                pos_weight=pos_weight,
            )
            loss_step = per_step[frame_mask].mean()

        loss = (
            float(getattr(self.args, "window_loss_scale", 1.0)) * loss_window
            + float(getattr(self.args, "risk_loss_scale", 0.0)) * loss_risk
            + float(getattr(self.args, "step_loss_scale", 0.0)) * loss_step
        )
        return loss, loss_window.detach(), loss_risk.detach(), loss_step.detach()

    @staticmethod
    def _save_validation_plots(output_dir, scores, targets, interaction_labels, threshold):
        os.makedirs(output_dir, exist_ok=True)
        scores = torch.cat(scores).float().cpu() if scores else torch.empty(0)
        targets = torch.cat(targets).float().cpu() if targets else torch.empty(0)
        interaction_labels = (
            torch.cat(interaction_labels).long().cpu()
            if interaction_labels
            else torch.full((scores.shape[0],), -1, dtype=torch.long)
        )

        overall = _binary_counts(scores, targets, threshold=threshold) if scores.numel() else None
        fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.8), constrained_layout=True)
        image = _plot_confusion(ax, overall, "Validation overall")
        if image is not None:
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        fig.savefig(os.path.join(output_dir, "confusion_overall.png"), dpi=180)
        plt.close(fig)

        by_interaction = {}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
        images = []
        for col, interaction_value in enumerate((0, 1)):
            mask = interaction_labels == interaction_value
            metrics = _binary_counts(scores[mask], targets[mask], threshold=threshold) if bool(mask.any().item()) else None
            by_interaction[str(interaction_value)] = metrics
            image = _plot_confusion(axes[col], metrics, f"interaction={interaction_value}")
            if image is not None:
                images.append(image)
        if images:
            fig.colorbar(images[0], ax=axes, fraction=0.035, pad=0.02)
        fig.suptitle("Validation by interaction label")
        fig.savefig(os.path.join(output_dir, "confusion_by_interaction.png"), dpi=180)
        plt.close(fig)

        payload = {
            "threshold": float(threshold),
            "overall": overall,
            "by_interaction": by_interaction,
        }
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return payload

    def validate(self, pipe, accelerator, global_step):
        if self.val_dataloader is None:
            logger.warning("Skipping video-action anomaly validation because val_dataloader is None")
            return
        self.anomaly_head.eval()
        if self.diffusion_model is not None:
            self.diffusion_model.eval()
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()

        threshold = float(getattr(self.args, "anomaly_val_threshold", 0.5))
        output_dir = os.path.join(self.save_folder, f"step_{global_step}", "validation")

        all_scores = []
        all_targets = []
        all_interactions = []
        weight_dtype = self.state.weight_dtype
        with torch.no_grad():
            for batch in self.val_dataloader:
                interaction_tokens, pred_latents = self._features_from_batch(
                    pipe,
                    batch,
                    accelerator,
                    weight_dtype,
                )
                head_out = self.anomaly_head(
                    interaction_tokens=interaction_tokens,
                    pred_latents=pred_latents,
                )
                target = self._target_from_batch(batch, accelerator.device)
                score = torch.sigmoid(head_out["window_logit"]).detach().float().reshape(-1)
                all_scores.append(score.cpu())
                all_targets.append(target.detach().float().reshape(-1).cpu())
                if "interaction_label" in batch:
                    interaction = batch["interaction_label"].detach().long().reshape(-1).cpu()
                else:
                    interaction = torch.full((score.shape[0],), -1, dtype=torch.long)
                all_interactions.append(interaction)

        payload = self._save_validation_plots(
            output_dir=output_dir,
            scores=all_scores,
            targets=all_targets,
            interaction_labels=all_interactions,
            threshold=threshold,
        )
        overall = payload.get("overall") or {}
        logs = {
            "val_acc": float(overall.get("accuracy", 0.0)),
            "val_precision": float(overall.get("precision", 0.0)),
            "val_recall": float(overall.get("recall", 0.0)),
            "val_f1": float(overall.get("f1", 0.0)),
        }
        accelerator.log(logs, step=global_step)
        if self.writer is not None:
            for key, value in logs.items():
                self.writer.add_scalar(f"VideoActionAnomaly/{key}", value, global_step)
        self.anomaly_head.train()

    def train(self):
        logger.info("Starting frozen action-conditioned GE anomaly-head training")
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
                    "feature cache": getattr(self.args, "anomaly_feature_cache_dir", None),
                    "latent chunk": None if self._use_feature_cache() else self._latent_chunk(),
                },
                indent=4,
            )
        )

        accelerator = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        pipe = None if self._use_feature_cache() else self._make_pipe()
        global_step = int(getattr(self.args, "start_step", 0) or 0)
        progress_bar = tqdm(
            range(0, self.state.train_steps),
            initial=global_step,
            desc="Training video-action anomaly head",
            disable=not accelerator.is_local_main_process,
        )

        for epoch in range(self.state.train_epochs):
            if global_step >= self.state.train_steps:
                break
            self.anomaly_head.train()
            if self.diffusion_model is not None:
                self.diffusion_model.eval()
            if self.vae is not None:
                self.vae.eval()
            if self.text_encoder is not None:
                self.text_encoder.eval()
            running_loss = 0.0

            for batch in self.train_dataloader:
                with accelerator.accumulate([self.anomaly_head]):
                    target = self._target_from_batch(batch, accelerator.device)
                    interaction_tokens, pred_latents = self._features_from_batch(
                        pipe,
                        batch,
                        accelerator,
                        weight_dtype,
                    )
                    head_out = self.anomaly_head(
                        interaction_tokens=interaction_tokens,
                        pred_latents=pred_latents,
                    )
                    loss, loss_window, loss_risk, loss_step = self._compute_anomaly_loss(head_out, target, batch)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        grad_norm = accelerator.clip_grad_norm_(self.anomaly_head.parameters(), self.args.max_grad_norm)
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
                    self.writer.add_scalar("VideoActionAnomaly/loss", loss.item(), global_step)
                    self.writer.add_scalar("VideoActionAnomaly/loss_window", loss_window.item(), global_step)
                    self.writer.add_scalar("VideoActionAnomaly/loss_risk", loss_risk.item(), global_step)
                    self.writer.add_scalar("VideoActionAnomaly/loss_step", loss_step.item(), global_step)
                    self.writer.add_scalar("VideoActionAnomaly/acc", acc.item(), global_step)
                    self.writer.add_scalar("VideoActionAnomaly/target_pos_ratio", target_mean.item(), global_step)

                if global_step % self.args.steps_to_save == 0:
                    self._save_head(global_step)

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
                self.writer.add_scalar(
                    "VideoActionAnomaly/epoch_loss",
                    running_loss / max(1, len(self.train_dataloader)),
                    epoch,
                )

        progress_bar.close()
        self._save_head(global_step)
        accelerator.wait_for_everyone()
        if pipe is not None:
            del pipe
        if self.diffusion_model is not None:
            del self.diffusion_model
        if self.scheduler is not None:
            del self.scheduler
        free_memory()
        logger.info(f"Memory after training end: {json.dumps(get_memory_statistics(), indent=4)}")
        accelerator.end_training()

    def _save_head(self, global_step):
        self.state.accelerator.wait_for_everyone()
        if not self.state.accelerator.is_main_process:
            return
        model_to_save = self.state.accelerator.unwrap_model(self.anomaly_head)
        save_dir = os.path.join(self.save_folder, f"step_{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        torch.save(
            {
                "state_dict": model_to_save.state_dict(),
                "anomaly_head": getattr(self.args, "anomaly_head", {}),
                "anomaly_head_class": getattr(self, "anomaly_head_class", "VideoActionAnomalyHead"),
                "anomaly_head_class_path": getattr(
                    self,
                    "anomaly_head_class_path",
                    "models/anomaly_head/action_anomaly.py",
                ),
                "anomaly_target": self.anomaly_target,
                "anomaly_positive_is": self.anomaly_positive_is,
                "baseline": "video_action",
            },
            os.path.join(save_dir, "anomaly_head.pt"),
        )
