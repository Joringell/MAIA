import json
import os

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

from data.anomaly_feature_cache_dataset import AnomalyFeatureCacheDataset
from data.observed_feature_cache_dataset import ObservedFeatureCacheDataset
from models.baselines.pca_kmeans import build_action_force_cache_features, build_pca_features, pooled_memory_video_latents, PCAKMeansBaseline
from runner.ge_trainer_base import BaseGETrainer, logger
from runner.ge_trainer_video_anomaly import _binary_counts
from runner.observed_baseline_common import data_cfg_for_split, domains_for_split, observed_cache_dir, observed_feature_settings,_best_f1_metrics,_score_summary
from utils import import_custom_class
from utils.memory_utils import free_memory, get_memory_statistics
from utils.model_utils import load_latent_models, load_vae_models


class _IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[int(idx)]
        sample["_source_index"] = torch.tensor(int(idx), dtype=torch.long)
        return sample


def _batch_value(batch, key, row, default=None):
    if key not in batch:
        return default
    value = batch[key]
    if torch.is_tensor(value):
        item = value[row]
        return item.item() if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[row]
    return value


def _cached_metadata_from_batch(batch, row):
    chunk_end = int(_batch_value(batch, "chunk_end", row, -1))
    return {
        "source_index": int(_batch_value(batch, "source_index", row, _batch_value(batch, "_cache_index", row, -1))),
        "episode_dataset_index": int(_batch_value(batch, "episode_dataset_index", row, -1)),
        "episode_id": int(_batch_value(batch, "episode_id", row, -1)),
        "domain": str(_batch_value(batch, "domain_name", row, "")),
        "window_start": int(_batch_value(batch, "window_start", row, -1)),
        "detection_frame": chunk_end,
        "chunk_end": chunk_end,
        "window_end": int(_batch_value(batch, "window_end", row, chunk_end)),
        "valid_len": int(_batch_value(batch, "valid_len", row, -1)),
        "total_frames": int(_batch_value(batch, "total_frames", row, -1)),
    }


def _metadata_for_index(dataset, source_idx):
    source_idx = int(source_idx)
    episode_dataset_idx = source_idx
    chunk_end = None
    if hasattr(dataset, "traverse_index"):
        episode_dataset_idx, chunk_end = dataset.traverse_index[source_idx]
        episode_dataset_idx = int(episode_dataset_idx)
        chunk_end = int(chunk_end)
    item = dataset.dataset[episode_dataset_idx]
    future_steps = int(getattr(dataset, "future_steps", 0) or 0)
    total_frames = int(item[7]) if len(item) > 7 else -1
    if chunk_end is None:
        window_start = -1
        window_end = -1
        valid_len = 0
        detection_frame = -1
    else:
        window_start = int(chunk_end) - future_steps
        if window_start >= 0 and future_steps > 0 and total_frames >= 0:
            window_end = min(window_start + future_steps, total_frames)
            valid_len = max(0, window_end - window_start)
        else:
            window_end = -1
            valid_len = 0
        detection_frame = window_end
    return {
        "source_index": source_idx,
        "episode_dataset_index": episode_dataset_idx,
        "episode_id": int(item[8]) if len(item) > 8 and item[8] is not None else episode_dataset_idx,
        "domain": str(item[3]) if len(item) > 3 else "",
        "window_start": window_start,
        "detection_frame": detection_frame,
        "chunk_end": detection_frame,
        "window_end": window_end,
        "valid_len": valid_len,
        "total_frames": total_frames,
    }


def _dict_cfg(value):
    return dict(value) if isinstance(value, dict) else {}


def _cache_dir_from_cfg(feature_cfg, legacy_dir=None):
    cache_dir = feature_cfg.get("cache_dir", legacy_dir)
    return None if cache_dir in (None, "") else os.path.expanduser(str(cache_dir))


def _max_open_shards_from_cfg(feature_cfg, legacy_value=4):
    return int(feature_cfg.get("max_open_shards", legacy_value))


class Trainer(BaseGETrainer):
    """Fit an observed PCA-KMeans baseline on normal windows only."""

    def __init__(self, config_file, to_log=True, output_dir=None) -> None:
        self.config_file = config_file
        super().__init__(config_file=config_file, to_log=to_log, output_dir=output_dir)

    def _observed_cache_dir(self):
        return observed_cache_dir(self.args)

    def _use_observed_cache(self):
        return self._observed_cache_dir() is not None

    def _data_cfg_for_split(self, split):
        return data_cfg_for_split(self.args, split)

    def _build_feature_specs(self):
        specs = []
        observed_cfg = _dict_cfg(getattr(self.args, "observed_feature", {}))
        observed_dir = _cache_dir_from_cfg(observed_cfg, legacy_dir=self._observed_cache_dir())
        legacy_observed_shards = getattr(self.args, "observed_cache_max_open_shards", 4)
        if observed_dir is not None:
            _, default_mode, default_fit_filter, default_force_bins = observed_feature_settings(self.args)
            spec_cfg = dict(observed_cfg)
            spec_cfg["cache_dir"] = observed_dir
            spec_cfg.setdefault("max_open_shards", _max_open_shards_from_cfg(observed_cfg, legacy_observed_shards))
            specs.append(
                {
                    "name": "observed_feature",
                    "source": "observed_cache",
                    "cfg": spec_cfg,
                    "cache_dir": observed_dir,
                    "max_open_shards": int(spec_cfg["max_open_shards"]),
                    "feature_mode": str(spec_cfg.get("feature_mode", default_mode)),
                    "fit_filter": str(spec_cfg.get("fit_filter", default_fit_filter)),
                    "force_bins": int(spec_cfg.get("force_bins", default_force_bins)),
                    "max_fit_windows": spec_cfg.get("max_fit_windows"),
                }
            )

        wm_cfg = _dict_cfg(getattr(self.args, "wm_feature", {}))
        wm_dir = _cache_dir_from_cfg(wm_cfg)
        if wm_dir is not None:
            spec_cfg = dict(wm_cfg)
            spec_cfg["cache_dir"] = wm_dir
            spec_cfg.setdefault("max_open_shards", _max_open_shards_from_cfg(wm_cfg, 4))
            spec_cfg.setdefault("feature_mode", "action_force_cache_mean")
            spec_cfg.setdefault("fit_filter", "normal_windows")
            specs.append(
                {
                    "name": "wm_feature",
                    "source": "wm_cache",
                    "cfg": spec_cfg,
                    "cache_dir": wm_dir,
                    "max_open_shards": int(spec_cfg["max_open_shards"]),
                    "feature_mode": str(spec_cfg["feature_mode"]),
                    "fit_filter": str(spec_cfg["fit_filter"]),
                    "force_bins": None,
                    "max_fit_windows": spec_cfg.get("max_fit_windows"),
                }
            )

        if not specs:
            feature_cfg, feature_mode, fit_filter, force_bins = observed_feature_settings(self.args)
            specs.append(
                {
                    "name": "observed_feature",
                    "source": "raw_observed",
                    "cfg": dict(feature_cfg),
                    "cache_dir": None,
                    "max_open_shards": 0,
                    "feature_mode": str(feature_mode),
                    "fit_filter": str(fit_filter),
                    "force_bins": int(force_bins),
                    "max_fit_windows": feature_cfg.get("max_fit_windows"),
                }
            )
        return specs

    def prepare_dataset(self) -> None:
        self.feature_specs = self._build_feature_specs()
        self.spec_datasets = {}
        for spec in self.feature_specs:
            dataset = self._make_dataset_for_spec(spec, split="train")
            loader = self._make_loader(dataset, shuffle=True)
            self.spec_datasets[spec["name"]] = {"train_dataset": dataset, "train_dataloader": loader}
            logger.info(
                f">>>>>>>>>>>>>Total PCA-KMeans Train Samples "
                f"({spec['name']}:{spec['source']}): {len(dataset)}<<<<<<<<<<<<<<<<<<\n"
            )
        first = self.feature_specs[0]["name"]
        self.train_dataset = self.spec_datasets[first]["train_dataset"]
        self.train_dataloader = self.spec_datasets[first]["train_dataloader"]
        if "val" in self.args.data:
            self.prepare_val_dataset()

    def prepare_val_dataset(self):
        for spec in self.feature_specs:
            dataset = self._make_dataset_for_spec(spec, split="val")
            loader_dataset = dataset if spec["source"] in {"observed_cache", "wm_cache"} else _IndexedDataset(dataset)
            loader = self._make_loader(loader_dataset, shuffle=False)
            self.spec_datasets[spec["name"]].update({"val_dataset": dataset, "val_dataloader": loader})
            logger.info(
                f">>>>>>>>>>>>>Total PCA-KMeans Validation Samples "
                f"({spec['name']}:{spec['source']}): {len(dataset)}<<<<<<<<<<<<<<<<<<\n"
            )
        first = self.feature_specs[0]["name"]
        self.val_dataset = self.spec_datasets[first].get("val_dataset")
        self.val_dataloader = self.spec_datasets[first].get("val_dataloader")

    def _make_dataset_for_spec(self, spec, split):
        if spec["source"] == "observed_cache":
            logger.info(f"{split} PCA-KMeans observed feature cache: {spec['cache_dir']}")
            return ObservedFeatureCacheDataset(
                cache_dir=spec["cache_dir"],
                split=split,
                max_open_shards=int(spec["max_open_shards"]),
                domains=domains_for_split(self.args, split),
            )
        if spec["source"] == "wm_cache":
            logger.info(f"{split} PCA-KMeans WM feature cache: {spec['cache_dir']}")
            return AnomalyFeatureCacheDataset(
                cache_dir=spec["cache_dir"],
                split=split,
                max_open_shards=int(spec["max_open_shards"]),
                domains=domains_for_split(self.args, split),
            )
        if split == "train":
            logger.info(f"Training Dataset: {self.args.train_data_class}")
            dataset_class = import_custom_class(self.args.train_data_class, self.args.train_data_class_path)
        else:
            if not hasattr(self.args, "val_data_class"):
                self.args.val_data_class = self.args.train_data_class
            if not hasattr(self.args, "val_data_class_path"):
                self.args.val_data_class_path = self.args.train_data_class_path
            logger.info(f"Validation Dataset: {self.args.val_data_class}")
            dataset_class = import_custom_class(self.args.val_data_class, self.args.val_data_class_path)
        return dataset_class(**self._data_cfg_for_split(split))

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
        logger.info("Initializing PCA-KMeans baseline feature branches")
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype
        needs_vae = any(spec["source"] == "raw_observed" for spec in self.feature_specs)
        if not needs_vae:
            logger.info("All PCA-KMeans branches use cached features; VAE inference is skipped")
            self.vae = None
        else:
            logger.info("Initializing frozen VAE for online observed feature extraction")
            vae_class = import_custom_class(self.args.vae_class, getattr(self.args, "vae_class_path", "transformers"))
            if getattr(self.args, "vae_path", False):
                self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
            else:
                self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(
                    device,
                    dtype=dtype,
                ).eval()
            if isinstance(self.vae.latents_mean, list):
                self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
            if isinstance(self.vae.latents_std, list):
                self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
            if getattr(self.args, "enable_slicing", False):
                self.vae.enable_slicing()
            if getattr(self.args, "enable_tiling", False):
                self.vae.enable_tiling()
            self.vae.requires_grad_(False)

        pca_kmeans_cfg = dict(getattr(self.args, "pca_kmeans", {}))
        self.pca_kmeans_cfg = pca_kmeans_cfg
        for spec in self.feature_specs:
            feature_dim = int(pca_kmeans_cfg.get("feature_dim", self._infer_feature_dim(spec)))
            spec["feature_dim"] = feature_dim

    @torch.no_grad()
    def _infer_feature_dim(self, spec):
        dataset = self.spec_datasets[spec["name"]]["train_dataset"]
        if len(dataset) == 0:
            raise ValueError("Cannot infer PCA-KMeans feature_dim from an empty train dataset.")
        sample = default_collate([dataset[0]])
        return int(self._features_from_batch(sample, spec).shape[-1])

    def _new_pca_kmeans_model(self):
        pca_kmeans_cfg = dict(getattr(self, "pca_kmeans_cfg", getattr(self.args, "pca_kmeans", {})))
        return PCAKMeansBaseline(
            pca_dim=pca_kmeans_cfg.get("pca_dim", pca_kmeans_cfg.get("n_components", 64)),
            n_clusters=int(pca_kmeans_cfg.get("n_clusters", 16)),
            max_iter=int(pca_kmeans_cfg.get("max_iter", 100)),
            seed=int(pca_kmeans_cfg.get("seed", getattr(self.args, "seed", 0))),
            eps=float(pca_kmeans_cfg.get("eps", 1e-6)),
        )

    def prepare_trainable_parameters(self):
        logger.info("PCA-KMeans baseline has no trainable parameters")
        self.state.num_trainable_parameters = 0
        if self.vae is not None:
            self.vae.requires_grad_(False)
            self.vae.eval()
        if self.args.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True

    def prepare_optimizer(self):
        self.state.train_epochs = 1
        self.state.train_steps = 1
        self.optimizer = None
        self.lr_scheduler = None

    def prepare_for_training(self):
        return

    def prepare_trackers(self):
        return

    @torch.no_grad()
    def _features_from_batch(self, batch, spec):
        device = self.state.accelerator.device
        dtype = self.state.weight_dtype
        if spec["source"] == "wm_cache":
            return build_action_force_cache_features(
                pred_force=batch["pred_force"].to(device=device, dtype=torch.float32, non_blocking=True),
                interaction_tokens=batch["interaction_tokens"].to(device=device, dtype=torch.float32, non_blocking=True),
                pred_latents=batch["pred_latents"].to(device=device, dtype=torch.float32, non_blocking=True),
                mode=spec["feature_mode"],
            ).float()
        if "video_features" in batch:
            video_features = batch["video_features"].to(device=device, dtype=torch.float32, non_blocking=True)
        else:
            data_cfg = self.args.data["train"]
            mem_size = int(data_cfg.get("n_previous", data_cfg.get("memory_steps", 4)))
            video = batch["video"].to(device=device, dtype=dtype, non_blocking=True).contiguous()
            video_features = pooled_memory_video_latents(self.vae, video, mem_size=mem_size, dtype=dtype)
        state = batch.get("state")
        if state is not None:
            state = state.to(device=device, dtype=torch.float32, non_blocking=True)
        history_force = batch.get("history_force_state")
        if history_force is not None:
            history_force = history_force.to(device=device, dtype=torch.float32, non_blocking=True)
        return build_pca_features(
            video_features,
            state=state,
            history_force_state=history_force,
            mode=spec["feature_mode"],
            force_bins=spec["force_bins"],
        ).float()

    def _normal_mask_from_batch(self, batch, spec):
        if "label" not in batch:
            raise ValueError("PCA-KMeans fitting requires batch['label']")
        device = self.state.accelerator.device
        window_normal = batch["label"].to(device=device, dtype=torch.float32).reshape(-1) >= 0.5
        fit_filter = spec["fit_filter"]
        if fit_filter == "normal_windows":
            return window_normal
        if fit_filter == "normal_episode_windows":
            if "episode_label" not in batch:
                raise ValueError(
                    f"{spec['name']}.fit_filter='normal_episode_windows' requires batch['episode_label']; "
                    "use fit_filter='normal_windows' for WM feature caches unless episode labels are cached."
                )
            episode_normal = batch["episode_label"].to(device=device, dtype=torch.float32).reshape(-1) >= 0.5
            return window_normal & episode_normal
        raise ValueError(f"Unsupported {spec['name']}.fit_filter={fit_filter}")

    def _target_from_batch(self, batch):
        labels = batch["label"].float().reshape(-1)
        return 1.0 - labels

    @torch.no_grad()
    def _collect_fit_features(self, spec):
        features = []
        normal_count = 0
        total_count = 0
        max_fit_value = spec.get("max_fit_windows")
        if max_fit_value in (None, ""):
            max_fit_value = self.pca_kmeans_cfg.get("max_fit_windows")
        max_fit = None if max_fit_value in (None, "") else int(max_fit_value)
        train_dataloader = self.spec_datasets[spec["name"]]["train_dataloader"]
        for batch in tqdm(
            train_dataloader,
            desc=f"Collecting PCA-KMeans fit features ({spec['name']})",
            disable=not self.state.accelerator.is_local_main_process,
        ):
            batch_features = self._features_from_batch(batch, spec).cpu()
            normal_mask = self._normal_mask_from_batch(batch, spec).cpu()
            total_count += int(normal_mask.numel())
            normal_count += int(normal_mask.sum().item())
            if bool(normal_mask.any().item()):
                features.append(batch_features[normal_mask])
            if max_fit is not None and sum(item.shape[0] for item in features) >= max_fit:
                break
        if not features:
            raise ValueError(
                f"No normal windows found for PCA-KMeans fit: "
                f"feature_spec={spec['name']} fit_filter={spec['fit_filter']}"
            )
        features = torch.cat(features, dim=0)
        if max_fit is not None:
            features = features[:max_fit]
        return features, {
            "fit_windows": int(features.shape[0]),
            "feature_dim": int(features.shape[1]),
            "normal_windows_seen": normal_count,
            "total_windows_seen": total_count,
            "feature_spec": spec["name"],
            "feature_source": spec["source"],
            "feature_mode": spec["feature_mode"],
            "fit_filter": spec["fit_filter"],
        }

    @torch.no_grad()
    def validate(self, global_step=0, spec=None, pca_kmeans_model=None):
        spec = spec or self.feature_specs[0]
        spec_state = self.spec_datasets.get(spec["name"], {})
        val_dataloader = spec_state.get("val_dataloader")
        val_dataset = spec_state.get("val_dataset")
        if val_dataloader is None:
            return None
        pca_kmeans_model = pca_kmeans_model or self.pca_kmeans_model
        scores = []
        targets = []
        records = []
        for batch in tqdm(
            val_dataloader,
            desc=f"Scoring PCA-KMeans validation ({spec['name']})",
            disable=not self.state.accelerator.is_local_main_process,
        ):
            features = self._features_from_batch(batch, spec)
            batch_scores = pca_kmeans_model.score(features).detach().cpu().float().reshape(-1)
            batch_targets = self._target_from_batch(batch).cpu().float().reshape(-1)
            scores.append(batch_scores)
            targets.append(batch_targets)

            labels = batch["label"].detach().cpu().float().reshape(-1)
            if "episode_label" in batch:
                episode_labels = batch["episode_label"].detach().cpu().float().reshape(-1)
            else:
                episode_labels = labels
            risk_labels = batch.get("risk_label")
            if risk_labels is not None:
                risk_labels = risk_labels.detach().cpu().float().reshape(-1)
            frame_labels = batch.get("frame_anomaly_label")
            if frame_labels is not None:
                frame_labels = frame_labels.detach().cpu().float()
            interaction_labels = batch.get("interaction_label")
            if interaction_labels is not None:
                interaction_labels = interaction_labels.detach().cpu().long().reshape(-1)
            source_indices = batch.get("_source_index")
            if source_indices is not None:
                source_indices = source_indices.detach().cpu().long().reshape(-1)

            for row in range(int(batch_scores.shape[0])):
                if spec["source"] in {"observed_cache", "wm_cache"}:
                    meta = _cached_metadata_from_batch(batch, row)
                else:
                    if source_indices is None:
                        raise ValueError("PCA-KMeans validation without cache requires _source_index metadata")
                    meta = _metadata_for_index(val_dataset, int(source_indices[row].item()))
                records.append(
                    {
                        **meta,
                        "score": float(batch_scores[row].item()),
                        "label": float(labels[row].item()),
                        "target": float(1.0 - labels[row].item()),
                        "episode_label": float(episode_labels[row].item()),
                        "risk_label": None if risk_labels is None else float(risk_labels[row].item()),
                        "frame_anomaly_label": None if frame_labels is None else frame_labels[row].tolist(),
                        "interaction_label": None if interaction_labels is None else int(interaction_labels[row].item()),
                    }
                )
        if not scores:
            return None
        scores = torch.cat(scores).float()
        targets = torch.cat(targets).float()
        normal_scores = scores[targets < 0.5]
        threshold = None
        threshold_metrics = None
        if normal_scores.numel() > 0:
            threshold = float(torch.quantile(normal_scores, float(getattr(self.args, "pca_kmeans_val_quantile", 0.95))).item())
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
        calibration_dir = os.path.join(self.save_folder, f"step_{global_step}")
        os.makedirs(calibration_dir, exist_ok=True)
        calibration_payload = {
            "records": records,
            "split": "val",
            "score_direction": "higher_is_more_anomalous",
            "num_records": len(records),
            "config_file": self.config_file,
            "feature_spec": spec["name"],
            "feature_source": spec["source"],
            "feature_cache_dir": spec["cache_dir"],
        }
        torch.save(calibration_payload, os.path.join(calibration_dir, "pca_kmeans_calibration_records.pt"))
        with open(os.path.join(calibration_dir, "pca_kmeans_calibration_meta.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": calibration_payload["split"],
                    "score_direction": calibration_payload["score_direction"],
                    "num_records": calibration_payload["num_records"],
                    "config_file": calibration_payload["config_file"],
                    "feature_spec": calibration_payload["feature_spec"],
                    "feature_source": calibration_payload["feature_source"],
                    "feature_cache_dir": calibration_payload["feature_cache_dir"],
                },
                f,
                indent=2,
            )
        return payload

    def train(self):
        logger.info("Fitting PCA-KMeans feature branches on normal windows")
        logger.info(f"Memory before PCA-KMeans fit: {json.dumps(get_memory_statistics(), indent=4)}")
        for step_idx, spec in enumerate(self.feature_specs):
            logger.info(
                f"Fitting PCA-KMeans branch step_{step_idx}: "
                f"{spec['name']} source={spec['source']} mode={spec['feature_mode']}"
            )
            self.pca_kmeans_model = self._new_pca_kmeans_model()
            fit_features, fit_meta = self._collect_fit_features(spec)
            fit_meta["requested_clusters"] = int(self.pca_kmeans_model.n_clusters)
            self.pca_kmeans_model.fit(fit_features)
            fit_meta["actual_clusters"] = int(self.pca_kmeans_model.centroids.shape[0])
            fit_meta["pca_dim"] = int(self.pca_kmeans_model.components.shape[0])
            self._save_pca_kmeans(global_step=step_idx, fit_meta=fit_meta, spec=spec)
            if self.state.accelerator.is_main_process:
                self.validate(global_step=step_idx, spec=spec, pca_kmeans_model=self.pca_kmeans_model)
            free_memory()
        self.state.accelerator.end_training()

    def _save_pca_kmeans(self, global_step, fit_meta, spec):
        if not self.state.accelerator.is_main_process:
            return
        save_dir = os.path.join(self.save_folder, f"step_{global_step}")
        os.makedirs(save_dir, exist_ok=True)
        feature_cfg = dict(spec["cfg"])
        pca_kmeans_cfg = dict(getattr(self.args, "pca_kmeans", {}))
        pca_kmeans_cfg["feature_dim"] = int(spec["feature_dim"])
        payload = {
            "state_dict": self.pca_kmeans_model.state_dict(),
            "pca_kmeans": pca_kmeans_cfg,
            "feature_spec": spec["name"],
            "feature_source": spec["source"],
            "feature_config": feature_cfg,
            "feature_mode": spec["feature_mode"],
            "fit_filter": spec["fit_filter"],
            "fit_meta": fit_meta,
            "baseline": (
                "frozen_wm_feature_pca_kmeans" if spec["source"] == "wm_cache" else "frozen_vae_observed_pca_kmeans"
            ),
            "observation": (
                "frozen world-model prediction features summarized over time"
                if spec["source"] == "wm_cache"
                else "observed memory RGB VAE pooled latent, optional current state, optional history force summary"
            ),
        }
        if spec["force_bins"] is not None:
            payload["force_bins"] = spec["force_bins"]
        if spec["source"] == "wm_cache":
            payload["wm_feature"] = feature_cfg
        else:
            payload["observed_feature"] = feature_cfg
        torch.save(payload, os.path.join(save_dir, "pca_kmeans.pt"))
        with open(os.path.join(save_dir, "fit_meta.json"), "w", encoding="utf-8") as f:
            json.dump(fit_meta, f, indent=2)
