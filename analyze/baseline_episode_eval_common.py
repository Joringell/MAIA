import csv
import json
import os
from collections import defaultdict

_MPL_CACHE_DIR = "/tmp/forcewm_matplotlib_cache"
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

import numpy as np
import torch
from torch.utils.data import Dataset

from analyze.anomaly_eval_common import compute_auprc, compute_auroc, load_frame_labels
from analyze.eval_common import (
    extract_episode_id_from_path,
    load_failure_transition_frame,
    prepare_split_dataset,
)
from models.baselines.pca_kmeans import build_pca_features, pooled_memory_video_latents
from utils import import_custom_class
from utils.model_utils import load_latent_models, load_vae_models
from utils.path_utils import load_resolved_config


STAGES = ("all", "pre", "post")


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(json_safe(row))


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[int(idx)]
        sample["_source_index"] = torch.tensor(int(idx), dtype=torch.long)
        return sample


def load_vae(cfg, device, dtype):
    vae_class = import_custom_class(cfg["vae_class"], cfg.get("vae_class_path", "transformers"))
    if cfg.get("vae_path", False):
        vae = load_vae_models(vae_class, cfg["vae_path"]).to(device, dtype=dtype).eval()
    else:
        vae = load_latent_models(vae_class, cfg["pretrained_model_name_or_path"])["vae"].to(device, dtype=dtype).eval()
    if isinstance(vae.latents_mean, list):
        vae.latents_mean = torch.FloatTensor(vae.latents_mean)
    if isinstance(vae.latents_std, list):
        vae.latents_std = torch.FloatTensor(vae.latents_std)
    if cfg.get("enable_slicing", False):
        vae.enable_slicing()
    if cfg.get("enable_tiling", False):
        vae.enable_tiling()
    vae.requires_grad_(False)
    return vae


def data_cfg_for_split(cfg, split, split_manifest=None, domains=None):
    if split in cfg["data"]:
        data_cfg = dict(cfg["data"][split])
    elif "val" in cfg["data"]:
        data_cfg = dict(cfg["data"]["val"])
        data_cfg["split"] = split
    else:
        data_cfg = dict(cfg["data"]["train"])
        data_cfg["split"] = split
    split_manifest = split_manifest or cfg.get("split_manifest")
    if split_manifest:
        data_cfg["split_manifest"] = split_manifest
        data_cfg["split"] = split
    if domains:
        data_cfg["domains"] = [item.strip() for item in str(domains).split(",") if item.strip()]
    data_cfg["memory_pick_mode"] = "uniform"
    return data_cfg


def build_dataset(cfg, split, split_manifest=None, domains=None):
    class_name = cfg.get("val_data_class", cfg["train_data_class"])
    class_path = cfg.get("val_data_class_path", cfg["train_data_class_path"])
    dataset_class = import_custom_class(class_name, class_path)
    return dataset_class(**data_cfg_for_split(cfg, split, split_manifest=split_manifest, domains=domains))


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


def cached_metadata_from_batch(batch, row):
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


def metadata_for_index(dataset, source_idx):
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


def build_features_from_batch(batch, vae, feature_ckpt, cfg, device, dtype):
    feature_cfg = dict(feature_ckpt.get("observed_feature", cfg.get("observed_feature", {})))
    feature_mode = str(feature_ckpt.get("feature_mode", feature_cfg.get("feature_mode", "video_state_force")))
    force_bins = int(feature_ckpt.get("force_bins", feature_cfg.get("force_bins", 5)))
    if "video_features" in batch:
        video_features = batch["video_features"].to(device=device, dtype=torch.float32, non_blocking=True)
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
            mode=feature_mode,
            force_bins=force_bins,
        )

    data_cfg = cfg["data"].get("train", {})
    mem_size = int(data_cfg.get("n_previous", data_cfg.get("memory_steps", 4)))
    video = batch["video"].to(device=device, dtype=dtype, non_blocking=True).contiguous()
    video_features = pooled_memory_video_latents(vae, video, mem_size=mem_size, dtype=dtype)
    state = batch.get("state")
    if state is not None:
        state = state.to(device=device, dtype=dtype, non_blocking=True)
    history_force = batch.get("history_force_state")
    if history_force is not None:
        history_force = history_force.to(device=device, dtype=dtype, non_blocking=True)
    return build_pca_features(
        video_features,
        state=state,
        history_force_state=history_force,
        mode=feature_mode,
        force_bins=force_bins,
    )


def trajectories_from_records(records):
    episodes = {}
    for row in records:
        key = (str(row["domain"]), int(row["episode_id"]))
        episodes.setdefault(key, []).append(row)
    trajectories = []
    for (domain, episode_id), rows in sorted(episodes.items()):
        rows = sorted(rows, key=lambda item: (int(item["detection_frame"]), int(item["source_index"])))
        trajectories.append(
            {
                "domain": domain,
                "episode_id": int(episode_id),
                "episode_label": float(rows[0]["episode_label"]),
                "scores": torch.tensor([row["score"] for row in rows], dtype=torch.float32),
                "targets": torch.tensor([row["target"] for row in rows], dtype=torch.float32),
                "frame_indices": [int(row["detection_frame"]) for row in rows],
            }
        )
    return trajectories


def records_to_frame_trajectories(records):
    episodes = {}
    for row in records:
        key = (str(row["domain"]), int(row["episode_id"]))
        episodes.setdefault(key, []).append(row)

    trajectories = []
    for (domain, episode_id), rows in sorted(episodes.items()):
        rows = sorted(rows, key=lambda item: (int(item["window_start"]), int(item["source_index"])))
        total_frames = max(int(row.get("total_frames", 0)) for row in rows)
        if total_frames <= 0:
            total_frames = max(int(row.get("window_end", -1)) for row in rows)
        if total_frames <= 0:
            continue

        score_sum = torch.zeros(total_frames, dtype=torch.float32)
        score_count = torch.zeros(total_frames, dtype=torch.float32)
        target_sum = torch.zeros(total_frames, dtype=torch.float32)
        target_count = torch.zeros(total_frames, dtype=torch.float32)
        for row in rows:
            start = max(0, int(row["window_start"]))
            valid_len = max(1, int(row.get("valid_len", 1)))
            end = min(total_frames, start + valid_len)
            if end <= start:
                continue
            score_sum[start:end] += float(row["score"])
            score_count[start:end] += 1.0
            target_sum[start:end] += float(row["target"])
            target_count[start:end] += 1.0

        valid = score_count > 0
        if not bool(valid.any().item()):
            continue
        trajectories.append(
            {
                "domain": domain,
                "episode_id": int(episode_id),
                "episode_label": float(rows[0]["episode_label"]),
                "scores": score_sum[valid] / score_count[valid],
                "targets": target_sum[valid] / target_count[valid].clamp_min(1.0),
                "frame_indices": torch.nonzero(valid, as_tuple=False).reshape(-1).tolist(),
            }
        )
    return trajectories


def attach_trajectory_starts(records):
    starts = {}
    for row in records:
        start = int(row.get("window_start", -1))
        if start < 0:
            continue
        key = (str(row["domain"]), int(row["episode_id"]))
        starts[key] = min(start, starts.get(key, start))
    for row in records:
        key = (str(row["domain"]), int(row["episode_id"]))
        row["trajectory_start"] = int(starts.get(key, max(0, int(row.get("window_start", 0)))))
    return records


def stack_variable_length(sequences, length=None):
    if not sequences:
        return torch.empty(0, 0), torch.empty(0, 0, dtype=torch.bool)
    if length is None:
        length = max(int(seq.numel()) for seq in sequences)
    values = torch.zeros(len(sequences), int(length), dtype=torch.float32)
    mask = torch.zeros(len(sequences), int(length), dtype=torch.bool)
    for idx, seq in enumerate(sequences):
        n = min(int(seq.numel()), int(length))
        if n > 0:
            values[idx, :n] = seq[:n].float()
            mask[idx, :n] = True
    return values, mask


def binary_metrics_from_predictions(predictions, targets):
    pred = torch.as_tensor(predictions, dtype=torch.float32).reshape(-1) >= 0.5
    targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1) >= 0.5
    tp = int((pred & targets).sum().item())
    tn = int((~pred & ~targets).sum().item())
    fp = int((pred & ~targets).sum().item())
    fn = int((~pred & targets).sum().item())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_samples": tp + tn + fp + fn,
    }


def binary_metrics(scores, targets, threshold):
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1) >= 0.5
    pred = scores >= float(threshold)
    return {
        "threshold": float(threshold),
        **binary_metrics_from_predictions(pred.float(), targets.float()),
    }


def best_f1_metrics(scores, targets):
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)
    if scores.numel() == 0:
        return None
    best = None
    for threshold in torch.unique(scores).sort().values:
        metrics = binary_metrics(scores, targets, float(threshold.item()))
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    return best


def grouped_records(records, key, default="unknown"):
    groups = {}
    for record in records:
        value = record.get(key, default)
        if value is None:
            value = default
        groups.setdefault(str(value), []).append(record)
    return groups


def records_scores_targets(records):
    return (
        torch.tensor([record["score"] for record in records], dtype=torch.float32),
        torch.tensor([record["target"] for record in records], dtype=torch.float32),
    )


def cp_window_metrics_for_records(records, cp_payload):
    if not cp_payload or not cp_payload.get("enabled") or not records:
        return None
    upper_band = torch.tensor(cp_payload["upper_band"], dtype=torch.float32)
    alarm_margin = float(cp_payload.get("alarm_margin", 0.0))
    valid_band = cp_payload.get("valid_band_mask")
    if valid_band is None:
        valid_band = torch.ones_like(upper_band, dtype=torch.bool)
    else:
        valid_band = torch.tensor(valid_band, dtype=torch.bool)
    predictions = []
    targets = []
    for trajectory in trajectories_from_records(records):
        length = min(int(trajectory["scores"].numel()), int(upper_band.numel()))
        if length <= 0:
            continue
        row_valid = valid_band[:length]
        if not bool(row_valid.any().item()):
            continue
        scores = trajectory["scores"][:length][row_valid]
        row_targets = trajectory["targets"][:length][row_valid]
        band = upper_band[:length][row_valid]
        predictions.extend((scores > band + alarm_margin).float().tolist())
        targets.extend((row_targets >= 0.5).float().tolist())
    if not predictions:
        return None
    return binary_metrics_from_predictions(torch.tensor(predictions), torch.tensor(targets))


def fill_invalid_band_by_repeat(upper_band, valid):
    upper_band = upper_band.clone().float()
    valid = valid.clone().bool()
    if upper_band.numel() == 0 or bool(valid.all().item()):
        return upper_band, valid, False
    valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
    if valid_indices.numel() == 0:
        return upper_band, valid, False

    first_valid = int(valid_indices[0].item())
    upper_band[:first_valid] = upper_band[first_valid]
    last_valid = first_valid
    for idx in range(first_valid + 1, int(upper_band.numel())):
        if bool(valid[idx].item()):
            last_valid = idx
        else:
            upper_band[idx] = upper_band[last_valid]
    return upper_band, torch.ones_like(valid, dtype=torch.bool), True


def compute_cp_band(calibration_records, eval_records, alpha=0.1, method="pointwise", train_ratio=0.5):
    calibration_trajectories = trajectories_from_records(calibration_records)
    normal = [item["scores"] for item in calibration_trajectories if float(item["episode_label"]) >= 0.5]
    eval_trajectories = trajectories_from_records(eval_records)
    if len(normal) < 1 or not eval_trajectories:
        return {
            "enabled": False,
            "reason": "Need at least one normal calibration episode and one eval episode.",
            "num_normal_calibration_episodes": len(normal),
            "num_eval_episodes": len(eval_trajectories),
        }
    band_len = max([int(seq.numel()) for seq in normal] + [int(item["scores"].numel()) for item in eval_trajectories])
    normal_values, normal_mask = stack_variable_length(normal, length=band_len)
    valid = normal_mask.any(dim=0)

    if method == "pointwise":
        upper_band = torch.zeros(band_len, dtype=torch.float32)
        for time_idx in range(band_len):
            values = normal_values[normal_mask[:, time_idx], time_idx]
            if values.numel() > 0:
                upper_band[time_idx] = torch.quantile(values, 1.0 - float(alpha))
    elif method == "functional":
        if normal_values.shape[0] < 2:
            return {
                "enabled": False,
                "reason": "Functional CP needs at least two normal calibration episodes.",
                "num_normal_calibration_episodes": int(normal_values.shape[0]),
                "num_eval_episodes": len(eval_trajectories),
            }
        n_train = max(1, int(round(normal_values.shape[0] * float(train_ratio))))
        n_train = min(n_train, normal_values.shape[0] - 1)
        train, train_mask = normal_values[:n_train], normal_mask[:n_train]
        cal, cal_mask = normal_values[n_train:], normal_mask[n_train:]
        center = (train * train_mask.float()).sum(dim=0) / train_mask.float().sum(dim=0).clamp_min(1.0)
        center_valid = train_mask.any(dim=0)
        abs_err = (train - center.unsqueeze(0)).abs().masked_fill(~train_mask, float("-inf"))
        modulation = abs_err.max(dim=0).values.clamp_min(1e-6)
        nonconformity = []
        for cal_row, mask_row in zip(cal, cal_mask):
            row_valid = mask_row & center_valid
            if bool(row_valid.any().item()):
                nonconformity.append(((cal_row[row_valid] - center[row_valid]) / modulation[row_valid]).max())
        if not nonconformity:
            return {"enabled": False, "reason": "No calibration overlap for functional CP."}
        q = torch.quantile(torch.stack(nonconformity), 1.0 - float(alpha))
        upper_band = center + q * modulation
        valid = center_valid
    else:
        raise ValueError(f"Unsupported CP method: {method}")

    original_valid = valid.clone()
    original_valid_timesteps = int(original_valid.sum().item())
    upper_band, valid, band_filled = fill_invalid_band_by_repeat(upper_band, valid)

    episode_predictions = []
    episode_targets = []
    detection_indices = []
    alarm_margin = 0.0
    for item in eval_trajectories:
        length = min(int(item["scores"].numel()), int(upper_band.numel()))
        scores = item["scores"][:length]
        targets = item["targets"][:length]
        row_valid = valid[:length]
        if not bool(row_valid.any().item()):
            continue
        exceed = scores[row_valid] > upper_band[:length][row_valid] + alarm_margin
        pred = float(exceed.any().item())
        target = float((targets >= 0.5).any().item())
        episode_predictions.append(pred)
        episode_targets.append(target)
        valid_indices = torch.nonzero(row_valid, as_tuple=False).reshape(-1)
        if pred > 0.5 and target > 0.5:
            first = int(torch.nonzero(exceed, as_tuple=False)[0].item())
            detection_step = int(valid_indices[first].item())
            frame_indices = item.get("frame_indices", [])
            detection_indices.append(int(frame_indices[detection_step]) if detection_step < len(frame_indices) else detection_step)
    window_metrics = cp_window_metrics_for_records(
        eval_records,
        {"enabled": True, "upper_band": upper_band.tolist(), "valid_band_mask": valid.tolist()},
    )

    return {
        "enabled": True,
        "method": method,
        "alpha": float(alpha),
        "train_ratio": float(train_ratio) if method == "functional" else None,
        "num_normal_calibration_episodes": int(normal_values.shape[0]),
        "num_eval_episodes": len(eval_trajectories),
        "trajectory_length": int(upper_band.numel()),
        "valid_band_timesteps": int(valid.sum().item()),
        "original_valid_band_timesteps": original_valid_timesteps,
        "band_fill_method": "repeat_last_valid" if band_filled else "none",
        "band_mean": float(upper_band[valid].mean().item()),
        "band_std": float(upper_band[valid].std(unbiased=False).item()),
        "mean_detection_index": (
            float(torch.tensor(detection_indices, dtype=torch.float32).mean().item()) if detection_indices else None
        ),
        "episode_metrics": binary_metrics_from_predictions(episode_predictions, episode_targets),
        "window_metrics": window_metrics,
        "upper_band": upper_band.tolist(),
        "valid_band_mask": valid.tolist(),
        "original_valid_band_mask": original_valid.tolist(),
    }


def save_episode_plots(output_dir, eval_records, cp_payload, max_plots=None):
    if not cp_payload or not cp_payload.get("enabled"):
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = os.path.join(output_dir, "episode_plots")
    os.makedirs(plot_dir, exist_ok=True)
    upper_band = torch.tensor(cp_payload["upper_band"], dtype=torch.float32)
    alarm_margin = float(cp_payload.get("alarm_margin", 0.0))
    paths = []
    trajectories = trajectories_from_records(eval_records)
    if max_plots is not None:
        trajectories = trajectories[: int(max_plots)]

    for item in trajectories:
        scores = item["scores"].float()
        targets = item["targets"].float()
        length = min(int(scores.numel()), int(upper_band.numel()))
        if length <= 0:
            continue
        frame_indices = item.get("frame_indices", list(range(int(scores.numel()))))
        x = torch.tensor(frame_indices[:length], dtype=torch.long).numpy()
        score_np = scores[:length].numpy()
        band_np = upper_band[:length].numpy()
        target_np = targets[:length].numpy()

        fig, ax = plt.subplots(1, 1, figsize=(10.5, 4.2), dpi=150)
        anomaly = target_np >= 0.5
        label_used = False
        start = None
        for idx, value in enumerate(list(anomaly) + [False]):
            if bool(value) and start is None:
                start = idx
            elif not bool(value) and start is not None:
                span_start = float(x[start]) - 0.5
                span_end = float(x[idx]) - 0.5 if idx < len(x) else float(x[idx - 1]) + 0.5
                ax.axvspan(
                    span_start,
                    span_end,
                    color="tomato",
                    alpha=0.13,
                    label=None if label_used else "gt anomaly",
                )
                label_used = True
                start = None

        ax.step(x, score_np, where="post", color="royalblue", linewidth=1.8, label="baseline score")
        ax.step(x, band_np, where="post", color="black", linestyle="--", linewidth=1.6, label="CP upper band")
        exceed = score_np > band_np + alarm_margin
        if exceed.any():
            label = "score > band" if alarm_margin == 0.0 else f"score > band + {alarm_margin:g}"
            ax.scatter(x[exceed], score_np[exceed], color="crimson", s=18, zorder=3, label=label)
        ax.set_xlabel("detection frame")
        ax.set_ylabel("baseline anomaly score")
        ax.set_title(
            f"{item['domain']} episode={item['episode_id']} "
            f"episode_label={item['episode_label']:.0f}"
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        safe_domain = str(item["domain"]).replace("/", "_")
        path = os.path.join(plot_dir, f"{safe_domain}_episode_{int(item['episode_id']):06d}.png")
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
    return paths


def score_summary(records):
    scores = torch.tensor([row["score"] for row in records], dtype=torch.float32)
    targets = torch.tensor([row["target"] for row in records], dtype=torch.float32) >= 0.5
    output = {}
    for name, mask in (("all", torch.ones_like(targets, dtype=torch.bool)), ("normal", ~targets), ("anomaly", targets)):
        values = scores[mask]
        if values.numel() == 0:
            output[name] = None
        else:
            output[name] = {
                "count": int(values.numel()),
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()),
                "p50": float(torch.quantile(values, 0.5).item()),
                "p90": float(torch.quantile(values, 0.9).item()),
                "p95": float(torch.quantile(values, 0.95).item()),
                "max": float(values.max().item()),
            }
    return output


def _episode_key(row):
    return (str(row.get("domain", "")), int(row.get("episode_id", row.get("episode_dataset_index", -1))))


def attach_cp_band(eval_records, cp_payload):
    records = [dict(row) for row in eval_records]
    if not cp_payload or not cp_payload.get("enabled"):
        for row in records:
            row["cp_band"] = None
            row["cp_valid"] = False
            row["cp_margin"] = None
            row["cp_alarm"] = None
        return records

    upper_band = np.asarray(cp_payload.get("upper_band", []), dtype=np.float64)
    alarm_margin = float(cp_payload.get("alarm_margin", 0.0))
    valid_band = cp_payload.get("valid_band_mask")
    if valid_band is None:
        valid_band = np.ones(upper_band.shape[0], dtype=bool)
    else:
        valid_band = np.asarray(valid_band, dtype=bool)

    if cp_payload.get("time_axis") == "frame":
        for row in records:
            start = int(row.get("window_start", -1))
            valid_len = int(row.get("valid_len", 0))
            if start < 0 or valid_len <= 0 or upper_band.size == 0:
                row["cp_band"] = None
                row["cp_valid"] = False
                row["cp_margin"] = None
                row["cp_alarm"] = None
                continue
            end = min(start + valid_len, upper_band.shape[0], valid_band.shape[0])
            if end <= start:
                idx = min(max(start, 0), upper_band.shape[0] - 1)
                frame_valid = bool(valid_band[idx])
                band = float(upper_band[idx]) if frame_valid else None
            else:
                frame_valid_mask = valid_band[start:end]
                frame_values = upper_band[start:end][frame_valid_mask]
                frame_valid = frame_values.size > 0
                band = float(np.mean(frame_values)) if frame_valid else None
            if band is None:
                row["cp_band"] = None
                row["cp_valid"] = False
                row["cp_margin"] = None
                row["cp_alarm"] = None
                continue
            score = float(row["score"])
            margin = score - band
            row["cp_band"] = band
            row["cp_valid"] = True
            row["cp_margin"] = margin
            row["cp_alarm_margin"] = alarm_margin
            row["cp_alarm"] = bool(margin > alarm_margin)
        return records

    grouped = defaultdict(list)
    for row in records:
        grouped[_episode_key(row)].append(row)

    for rows in grouped.values():
        rows.sort(key=lambda item: (int(item.get("detection_frame", -1)), int(item.get("source_index", -1))))
        for t, row in enumerate(rows):
            if t >= upper_band.shape[0] or not bool(valid_band[t]):
                row["cp_band"] = None
                row["cp_valid"] = False
                row["cp_margin"] = None
                row["cp_alarm"] = None
                continue
            score = float(row["score"])
            band = float(upper_band[t])
            margin = score - band
            row["cp_band"] = band
            row["cp_valid"] = True
            row["cp_margin"] = margin
            row["cp_alarm_margin"] = alarm_margin
            row["cp_alarm"] = bool(margin > alarm_margin)
    return records


def _stage(row):
    interaction = row.get("interaction_label")
    if interaction is None:
        return "pre"
    return "post" if int(interaction) > 0 else "pre"


def _binary_metrics(scores, targets, threshold=0.0):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid]
    if scores.size == 0:
        return None
    pred = scores >= float(threshold)
    pos = targets == 1
    neg = ~pos
    tp = int(np.logical_and(pred, pos).sum())
    fp = int(np.logical_and(pred, neg).sum())
    tn = int(np.logical_and(~pred, neg).sum())
    fn = int(np.logical_and(~pred, pos).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    tpr = recall
    fpr = fp / max(1, fp + tn)
    tnr = tn / max(1, tn + fp)
    return {
        "threshold": float(threshold),
        "num_samples": int(scores.size),
        "num_positive": int(pos.sum()),
        "num_negative": int(neg.sum()),
        "positive_ratio": float(pos.mean()),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": float((tp + tn) / max(1, scores.size)),
        "precision": float(precision),
        "recall": float(recall),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tnr": float(tnr),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "f1": float(f1),
        "auroc": compute_auroc(scores, targets),
        "auprc": compute_auprc(scores, targets),
    }


def _best_f1(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid]
    if scores.size == 0:
        return None
    best = None
    for threshold in np.unique(scores):
        metrics = _binary_metrics(scores, targets, float(threshold))
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    return best


def build_window_rows(records):
    rows = []
    for row in records:
        target = int(float(row.get("target", 0.0)) >= 0.5)
        stage = _stage(row)
        rows.append(
            {
                "domain": str(row.get("domain", "")),
                "episode_id": int(row.get("episode_id", -1)),
                "episode_dataset_index": int(row.get("episode_dataset_index", -1)),
                "source_index": int(row.get("source_index", -1)),
                "window_start": int(row.get("window_start", -1)),
                "detection_frame": int(row.get("detection_frame", -1)),
                "window_end": int(row.get("window_end", -1)),
                "valid_len": int(row.get("valid_len", -1)),
                "total_frames": int(row.get("total_frames", -1)),
                "stage": stage,
                "raw_score": float(row["score"]),
                "cp_band": row.get("cp_band"),
                "cp_valid": bool(row.get("cp_valid", False)),
                "cp_margin": row.get("cp_margin"),
                "cp_alarm": row.get("cp_alarm"),
                "target": target,
                "label": row.get("label"),
                "risk_label": row.get("risk_label"),
                "interaction_label": row.get("interaction_label"),
                "episode_label": row.get("episode_label"),
            }
        )
    return rows


def compute_domain_stage_metrics(records):
    domains = ["all"] + sorted({str(row.get("domain", "")) for row in records})
    rows = []
    nested = {}
    for domain in domains:
        nested[domain] = {}
        domain_records = records if domain == "all" else [row for row in records if str(row.get("domain", "")) == domain]
        for stage in STAGES:
            if stage == "all":
                selected = domain_records
            else:
                selected = [row for row in domain_records if _stage(row) == stage]
            selected = [row for row in selected if row.get("cp_valid") and row.get("cp_margin") is not None]
            scores = [float(row["cp_margin"]) for row in selected]
            targets = [int(float(row.get("target", 0.0)) >= 0.5) for row in selected]
            manual = _binary_metrics(scores, targets, threshold=0.0)
            best = _best_f1(scores, targets)
            out = {
                "domain": domain,
                "stage": stage,
                "score_name": "cp_margin",
                "threshold": 0.0,
                "threshold_meaning": "cp_margin > alarm_margin equals score > CP band + alarm_margin",
                "num_samples": None if manual is None else manual["num_samples"],
                "num_pos": None if manual is None else manual["num_positive"],
                "num_neg": None if manual is None else manual["num_negative"],
                "positive_ratio": None if manual is None else manual["positive_ratio"],
                "tp": None if manual is None else manual["tp"],
                "fp": None if manual is None else manual["fp"],
                "tn": None if manual is None else manual["tn"],
                "fn": None if manual is None else manual["fn"],
                "tpr_recall": None if manual is None else manual["recall"],
                "fpr": None if manual is None else manual["fpr"],
                "precision": None if manual is None else manual["precision"],
                "balanced_accuracy": None if manual is None else manual["balanced_accuracy"],
                "f1": None if manual is None else manual["f1"],
                "auprc": None if manual is None else manual["auprc"],
                "auroc": None if manual is None else manual["auroc"],
                "best_f1": None if best is None else best["f1"],
                "best_threshold": None if best is None else best["threshold"],
                "best_precision": None if best is None else best["precision"],
                "best_recall": None if best is None else best["recall"],
                "best_fpr": None if best is None else best["fpr"],
                "raw_score_mean": float(np.mean([float(row["score"]) for row in selected])) if selected else None,
                "cp_band_mean": float(np.mean([float(row["cp_band"]) for row in selected])) if selected else None,
                "cp_margin_mean": float(np.mean(scores)) if scores else None,
            }
            nested[domain][stage] = out
            rows.append(out)
    return nested, rows


def build_episode_summary_rows(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[_episode_key(row)].append(row)
    output = []
    for (domain, episode_id), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda item: (int(item.get("detection_frame", -1)), int(item.get("source_index", -1))))
        valid_rows = [row for row in rows if row.get("cp_valid") and row.get("cp_margin") is not None]
        alarms = [row for row in valid_rows if bool(row.get("cp_alarm"))]
        positives = [row for row in valid_rows if int(float(row.get("target", 0.0)) >= 0.5) == 1]
        first_alarm = min((int(row.get("window_start", row.get("detection_frame", -1))) for row in alarms), default=None)
        first_positive = min((int(row.get("window_start", row.get("detection_frame", -1))) for row in positives), default=None)
        max_margin = max((float(row["cp_margin"]) for row in valid_rows), default=None)
        max_score = max((float(row["score"]) for row in rows), default=None)
        output.append(
            {
                "domain": domain,
                "episode_id": episode_id,
                "episode_dataset_index": int(rows[0].get("episode_dataset_index", -1)),
                "total_frames": int(max(int(row.get("total_frames", -1)) for row in rows)),
                "num_windows": len(rows),
                "num_cp_valid_windows": len(valid_rows),
                "has_positive_window": bool(positives),
                "episode_label": rows[0].get("episode_label"),
                "first_positive_window_start": first_positive,
                "first_alarm_window_start": first_alarm,
                "detected_positive_episode": bool(alarms) and bool(positives),
                "normal_false_alarm_episode": bool(alarms) and not bool(positives),
                "missed_positive_episode": bool(positives) and not bool(alarms),
                "window_positive_ratio": float(np.mean([int(float(row.get("target", 0.0)) >= 0.5) for row in rows])) if rows else None,
                "cp_alarm_ratio": float(np.mean([bool(row.get("cp_alarm")) for row in valid_rows])) if valid_rows else None,
                "max_raw_score": max_score,
                "max_cp_margin": max_margin,
            }
        )
    return output


def save_confusion_plot(stage_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = [row for row in stage_rows if row["domain"] == "all"]
    fig, axes = plt.subplots(1, len(STAGES), figsize=(13.5, 4.2), constrained_layout=True)
    if len(STAGES) == 1:
        axes = [axes]
    for ax, stage in zip(axes, STAGES):
        row = next((item for item in rows if item["stage"] == stage), None)
        ax.set_title(f"window - {stage}")
        ax.set_xlabel("pred")
        ax.set_ylabel("gt")
        ax.set_xticks([0, 1], labels=["normal", "anom"])
        ax.set_yticks([0, 1], labels=["normal", "anom"])
        if row is None or row["num_samples"] in (None, 0):
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        matrix = np.asarray([[row["tn"], row["fp"]], [row["fn"], row["tp"]]], dtype=np.int64)
        image = ax.imshow(matrix, cmap="Blues")
        max_value = max(1, int(matrix.max()))
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_pct = matrix / np.maximum(row_sums, 1)
        names = np.asarray([["TN", "FP"], ["FN", "TP"]])
        for r in range(2):
            for c in range(2):
                color = "white" if matrix[r, c] > 0.55 * max_value else "black"
                ax.text(
                    c,
                    r,
                    f"{names[r, c]}\n{matrix[r, c]}\n{row_pct[r, c] * 100:.1f}%",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                )
        ax.text(
            0.5,
            -0.22,
            f"P={row['precision']:.3f} R={row['tpr_recall']:.3f} F1={row['f1']:.3f}\n"
            f"AUROC={row['auroc'] if row['auroc'] is not None else np.nan:.3f} "
            f"AUPRC={row['auprc'] if row['auprc'] is not None else np.nan:.3f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle("Baseline CP Confusion Matrices: cp_margin > alarm_margin")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_domain_stage_plot(stage_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    domains = sorted({row["domain"] for row in stage_rows if row["domain"] != "all"})
    if not domains:
        domains = ["all"]
    row_map = {(row["domain"], row["stage"]): row for row in stage_rows}
    metrics = [
        ("tpr_recall", "Recall / TPR"),
        ("fpr", "FPR"),
        ("balanced_accuracy", "bACC"),
        ("auprc", "AUPRC"),
    ]
    x = np.arange(len(domains), dtype=np.float64)
    width = 0.24
    colors = {"all": "0.35", "pre": "cornflowerblue", "post": "tomato"}
    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(domains) * 0.75), 8.5), constrained_layout=True)
    axes = axes.reshape(-1)
    for ax, (key, title) in zip(axes, metrics):
        for offset, stage in zip((-width, 0.0, width), STAGES):
            values = []
            for domain in domains:
                value = row_map.get((domain, stage), {}).get(key)
                values.append(np.nan if value is None else float(value))
            bars = ax.bar(x + offset, values, width=width, label=stage, color=colors[stage], alpha=0.85)
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    continue
                ax.text(bar.get_x() + bar.get_width() / 2.0, value, f"{value:.2f}", ha="center", va="bottom", rotation=90, fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x, domains, rotation=35, ha="right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        ax.legend(loc="upper left", ncol=3)
    fig.suptitle("Baseline Domain x Stage Metrics")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_score_histogram(records, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, len(STAGES), figsize=(15, 4.2), constrained_layout=True)
    if len(STAGES) == 1:
        axes = [axes]
    for ax, stage in zip(axes, STAGES):
        selected = records if stage == "all" else [row for row in records if _stage(row) == stage]
        selected = [row for row in selected if row.get("cp_valid") and row.get("cp_margin") is not None]
        normal = [float(row["cp_margin"]) for row in selected if int(float(row.get("target", 0.0)) >= 0.5) == 0]
        anomaly = [float(row["cp_margin"]) for row in selected if int(float(row.get("target", 0.0)) >= 0.5) == 1]
        values = np.asarray(normal + anomaly, dtype=np.float64)
        if values.size:
            lo, hi = np.nanpercentile(values, [1.0, 99.0])
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = float(np.nanmin(values)), float(np.nanmax(values) + 1e-6)
            bins = np.linspace(lo, hi, 31)
            ax.hist(normal, bins=bins, alpha=0.65, color="steelblue", label="normal")
            ax.hist(anomaly, bins=bins, alpha=0.65, color="crimson", label="anomaly")
        ax.axvline(0.0, color="black", linestyle=":", linewidth=1.5, label="CP threshold")
        ax.set_title(f"cp margin - {stage}")
        ax.set_xlabel("score - CP band")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        ax.legend(loc="upper right")
    fig.suptitle("Baseline CP Margin Distribution")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _select_video_records(records, top_k):
    grouped = defaultdict(list)
    for row in records:
        grouped[_episode_key(row)].append(row)

    ranked = []
    for key, rows in grouped.items():
        rows = [row for row in rows if row.get("cp_valid") and row.get("cp_margin") is not None]
        if not rows:
            continue
        scores = [float(row["cp_margin"]) for row in rows]
        targets = [int(float(row.get("target", 0.0)) >= 0.5) for row in rows]
        metrics = _binary_metrics(scores, targets, threshold=0.0)
        if metrics is None:
            continue
        max_score = max(float(row.get("score", 0.0)) for row in rows)
        ranked.append(
            (
                (
                    float(metrics["f1"]),
                    float(metrics["recall"]),
                    -int(metrics["fn"]),
                    -int(metrics["fp"]),
                    -float(max_score),
                ),
                key,
                rows,
            )
        )
    ranked.sort(key=lambda item: item[0])
    if top_k is not None:
        ranked = ranked[: int(top_k)]
    return [rows for _rank, _key, rows in ranked]


def _aggregate_baseline_episode(rows, total_frames):
    raw_sum = np.zeros(total_frames, dtype=np.float64)
    band_sum = np.zeros(total_frames, dtype=np.float64)
    margin_sum = np.zeros(total_frames, dtype=np.float64)
    target_sum = np.zeros(total_frames, dtype=np.float64)
    target_count = np.zeros(total_frames, dtype=np.int64)
    count = np.zeros(total_frames, dtype=np.int64)

    for row in rows:
        start = int(row.get("window_start", row.get("detection_frame", -1)))
        valid_len = int(row.get("valid_len", 0))
        if start < 0 or valid_len <= 0:
            continue
        end = min(total_frames, start + valid_len)
        if end <= start:
            continue
        length = end - start
        frame_labels = row.get("frame_anomaly_label")
        if frame_labels is not None:
            frame_labels = np.asarray(frame_labels, dtype=np.float64)[:length]
            valid_label = np.isfinite(frame_labels) & (frame_labels >= 0.0)
            frame_target = np.full(length, np.nan, dtype=np.float64)
            frame_target[valid_label] = frame_labels[valid_label]
        else:
            frame_target = np.full(length, float(row.get("target", 0.0)), dtype=np.float64)
        raw_sum[start:end] += float(row.get("score", 0.0))
        band_sum[start:end] += float(row.get("cp_band", 0.0))
        margin_sum[start:end] += float(row.get("cp_margin", 0.0))
        valid_target = np.isfinite(frame_target)
        target_slice = target_sum[start:end]
        count_slice = target_count[start:end]
        target_slice[valid_target] += frame_target[valid_target]
        count_slice[valid_target] += 1
        count[start:end] += 1

    valid = count > 0
    raw_curve = np.full(total_frames, np.nan, dtype=np.float64)
    band_curve = np.full(total_frames, np.nan, dtype=np.float64)
    margin_curve = np.full(total_frames, np.nan, dtype=np.float64)
    target_curve = np.full(total_frames, np.nan, dtype=np.float64)
    raw_curve[valid] = raw_sum[valid] / count[valid]
    band_curve[valid] = band_sum[valid] / count[valid]
    margin_curve[valid] = margin_sum[valid] / count[valid]
    target_valid = target_count > 0
    target_curve[target_valid] = target_sum[target_valid] / target_count[target_valid]
    alarm_curve = np.zeros(total_frames, dtype=np.float64)
    alarm_margin = 0.0
    if rows:
        alarm_margin = float(rows[0].get("cp_alarm_margin", 0.0))
    alarm_curve[valid] = margin_curve[valid] > alarm_margin
    return raw_curve, band_curve, margin_curve, target_curve, alarm_curve


def _frame_stage_metrics(records, domain, stage):
    grouped = defaultdict(list)
    for row in records:
        if domain != "all" and str(row.get("domain", "")) != str(domain):
            continue
        if stage != "all" and _stage(row) != stage:
            continue
        if not row.get("cp_valid") or row.get("cp_margin") is None:
            continue
        grouped[_episode_key(row)].append(row)

    scores = []
    targets = []
    for rows in grouped.values():
        total_frames = int(max(int(row.get("total_frames", -1)) for row in rows))
        if total_frames <= 0:
            continue
        _raw_curve, _band_curve, margin_curve, target_curve, _alarm_curve = _aggregate_baseline_episode(rows, total_frames)
        valid = np.isfinite(margin_curve) & np.isfinite(target_curve)
        if not np.any(valid):
            continue
        scores.append(margin_curve[valid])
        targets.append((target_curve[valid] >= 0.5).astype(np.int64))
    if not scores:
        return None, None
    scores = np.concatenate(scores)
    targets = np.concatenate(targets)
    return _binary_metrics(scores, targets, threshold=0.0), _best_f1(scores, targets)


def build_unified_stage_metrics(records, stage_rows):
    domains = ["all"] + sorted({str(row.get("domain", "")) for row in records})
    window_map = {(row["domain"], row["stage"]): row for row in stage_rows}
    rows = []
    nested = {}
    for domain in domains:
        nested[domain] = {}
        for stage in STAGES:
            window = window_map.get((domain, stage))
            frame_manual, frame_best = _frame_stage_metrics(records, domain, stage)
            out = {
                "domain": domain,
                "stage": stage,
                "window_num": None if window is None else window.get("num_samples"),
                "window_num_pos": None if window is None else window.get("num_pos"),
                "window_num_neg": None if window is None else window.get("num_neg"),
                "window_pos_ratio": None if window is None else window.get("positive_ratio"),
                "window_auroc": None if window is None else window.get("auroc"),
                "window_auprc": None if window is None else window.get("auprc"),
                "window_tp": None if window is None else window.get("tp"),
                "window_fp": None if window is None else window.get("fp"),
                "window_tn": None if window is None else window.get("tn"),
                "window_fn": None if window is None else window.get("fn"),
                "window_precision": None if window is None else window.get("precision"),
                "window_recall": None if window is None else window.get("tpr_recall"),
                "window_tpr": None if window is None else window.get("tpr_recall"),
                "window_fpr": None if window is None else window.get("fpr"),
                "window_tnr": None if window is None else None if window.get("fpr") is None else 1.0 - float(window.get("fpr")),
                "window_balanced_accuracy": None if window is None else window.get("balanced_accuracy"),
                "window_f1": None if window is None else window.get("f1"),
                "window_best_f1": None if window is None else window.get("best_f1"),
                "window_best_threshold": None if window is None else window.get("best_threshold"),
                "risk_num": None,
                "risk_mae": None,
                "risk_mse": None,
                "risk_spearman": None,
                "frame_num": None if frame_manual is None else frame_manual.get("num_samples"),
                "frame_num_pos": None if frame_manual is None else frame_manual.get("num_positive"),
                "frame_num_neg": None if frame_manual is None else frame_manual.get("num_negative"),
                "frame_pos_ratio": None if frame_manual is None else frame_manual.get("positive_ratio"),
                "frame_auroc": None if frame_manual is None else frame_manual.get("auroc"),
                "frame_auprc": None if frame_manual is None else frame_manual.get("auprc"),
                "frame_tp": None if frame_manual is None else frame_manual.get("tp"),
                "frame_fp": None if frame_manual is None else frame_manual.get("fp"),
                "frame_tn": None if frame_manual is None else frame_manual.get("tn"),
                "frame_fn": None if frame_manual is None else frame_manual.get("fn"),
                "frame_precision": None if frame_manual is None else frame_manual.get("precision"),
                "frame_recall": None if frame_manual is None else frame_manual.get("recall"),
                "frame_tpr": None if frame_manual is None else frame_manual.get("tpr"),
                "frame_fpr": None if frame_manual is None else frame_manual.get("fpr"),
                "frame_tnr": None if frame_manual is None else frame_manual.get("tnr"),
                "frame_balanced_accuracy": None if frame_manual is None else frame_manual.get("balanced_accuracy"),
                "frame_f1": None if frame_manual is None else frame_manual.get("f1"),
                "frame_best_f1": None if frame_best is None else frame_best.get("f1"),
                "frame_best_threshold": None if frame_best is None else frame_best.get("threshold"),
            }
            nested[domain][stage] = out
            rows.append(out)
    return nested, rows


def _find_dataset_index(dataset, domain, episode_id):
    for dataset_index, item in enumerate(dataset.dataset):
        item_domain = str(item[3])
        if item_domain != str(domain):
            continue
        try:
            item_episode = extract_episode_id_from_path(item[2])
        except Exception:
            continue
        if int(item_episode) == int(episode_id):
            return int(dataset_index)
    return None

def save_unified_baseline_outputs(
    output_dir,
    eval_records,
    cp_payload,
    metadata=None,
):
    records = attach_cp_band(eval_records, cp_payload)
    window_rows = build_window_rows(records)
    stage_nested, stage_rows = compute_domain_stage_metrics(records)
    unified_nested, unified_rows = build_unified_stage_metrics(records, stage_rows)
    episode_rows = build_episode_summary_rows(records)

    payload = {
        "metadata": metadata or {},
        "score_direction": "higher_is_more_anomalous",
        "threshold_rule": "CP alarm if raw_score > cp_band + alarm_margin; equivalent to cp_margin > alarm_margin",
        "stage_definition": "all=all windows, pre=interaction_label<=0 or missing, post=interaction_label>0",
        "stage_metrics": stage_nested,
        "unified_stage_metrics": unified_nested,
        "frame_metric_definition": (
            "PCA/PCA-KMeans produce one scalar score per observed window. "
            "Frame metrics broadcast each scalar cp_margin over that window's valid future frames "
            "and aggregate overlapping windows; they are not native per-future-step predictions."
        ),
        "num_eval_windows": len(records),
        "num_cp_valid_windows": int(sum(bool(row.get("cp_valid")) for row in records)),
        "num_episodes": len(episode_rows),
    }

    torch.save(
        {
            "metadata": metadata or {},
            "score_direction": "higher_is_more_anomalous",
            "threshold_rule": "CP alarm if raw_score > cp_band + alarm_margin; equivalent to cp_margin > alarm_margin",
            "cp_payload": cp_payload,
            "records": records,
        },
        os.path.join(output_dir, "episode_records.pt"),
    )
    write_json(os.path.join(output_dir, "metrics_summary.json"), payload)
    write_csv(os.path.join(output_dir, "domain_stage_metrics.csv"), stage_rows)
    write_csv(os.path.join(output_dir, "domain_stage_metrics_unified.csv"), unified_rows)
    write_csv(os.path.join(output_dir, "window_records.csv"), window_rows)
    write_csv(os.path.join(output_dir, "episode_summary.csv"), episode_rows)

    plot_dir = os.path.join(output_dir, "plots", "summary")
    save_domain_stage_plot(stage_rows, os.path.join(plot_dir, "domain_stage_metrics.png"))
    save_confusion_plot(stage_rows, os.path.join(plot_dir, "confusion_matrices.png"))
    save_score_histogram(records, os.path.join(plot_dir, "score_histograms.png"))
    return {
        "metrics_summary": os.path.join(output_dir, "metrics_summary.json"),
        "domain_stage_metrics": os.path.join(output_dir, "domain_stage_metrics.csv"),
        "window_records": os.path.join(output_dir, "window_records.csv"),
        "episode_summary": os.path.join(output_dir, "episode_summary.csv"),
        "summary_plots": plot_dir,
    }
