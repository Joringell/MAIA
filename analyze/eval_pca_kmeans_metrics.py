import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.baseline_episode_eval_common import (
    IndexedDataset,
    best_f1_metrics,
    binary_metrics_from_predictions,
    build_dataset,
    build_features_from_batch,
    compute_cp_band,
    cp_window_metrics_for_records,
    grouped_records,
    load_vae,
    metadata_for_index,
    records_scores_targets,
    save_episode_plots,
    save_unified_baseline_outputs,
    score_summary,
)
from analyze.analyze_cp_early_warning import run_early_warning_analysis
from data.anomaly_feature_cache_dataset import AnomalyFeatureCacheDataset
from data.observed_feature_cache_dataset import ObservedFeatureCacheDataset
from models.baselines.pca_kmeans import build_action_force_cache_features, build_pca_features,  PCAKMeansBaseline
from utils.path_utils import load_resolved_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate observed-feature PCA-KMeans with temporal CP band.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--pca_kmeans_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_split", default="val")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--calibration_records", default=None, help="Optional saved PCA-KMeans calibration records.")
    parser.add_argument(
        "--cp_metrics",
        "--cp_band_metrics",
        dest="cp_metrics",
        default=None,
        help="Reuse an existing pca_kmeans_cp_metrics.json cp_band instead of recalibrating.",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Recompute calibration records instead of loading saved records.",
    )
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--domains", default=None, help="Comma-separated domain subset override.")
    parser.add_argument("--observed_feature_cache_dir", default=None, help="Optional cached observed features.")
    parser.add_argument("--observed_cache_max_open_shards", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cp_alpha", type=float, default=0.1)
    parser.add_argument(
        "--cp_method",
        choices=["pointwise", "functional", "global"],
        default="functional",
        help="Conformal calibration: pointwise, functional, or global fixed threshold.",
    )
    parser.add_argument(
        "--cp_normal_filter",
        choices=["auto", "normal_episode_windows", "normal_windows", "success_episodes"],
        default="auto",
        help="Normal-record filter used by --cp_method global.",
    )
    parser.add_argument("--cp_train_ratio", type=float, default=0.5)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--max_episode_plots", type=int, default=None)
    parser.add_argument("--disable_early_warning", action="store_true")
    parser.add_argument("--early_warning_output_dir", default=None)
    parser.add_argument("--early_window", type=int, default=50)
    parser.add_argument("--transition_window", type=int, default=50)
    parser.add_argument("--min_segment_len", type=int, default=2)
    return parser.parse_args()


def load_pca_kmeans(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model = PCAKMeansBaseline()
    model.load_state_dict(state_dict)
    return model, ckpt


def default_calibration_records_path(checkpoint_path):
    return os.path.join(os.path.dirname(os.path.expanduser(str(checkpoint_path))), "pca_kmeans_calibration_records.pt")


def load_calibration_records(path):
    payload = torch.load(os.path.expanduser(str(path)), map_location="cpu")
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"], payload
    if isinstance(payload, list):
        return payload, {"records": payload}
    raise ValueError(f"Invalid PCA-KMeans calibration records file: {path}")


def load_cp_band_metrics(path):
    with open(os.path.expanduser(str(path)), "r", encoding="utf-8") as f:
        payload = json.load(f)
    cp_payload = payload.get("cp_band", payload)
    if not isinstance(cp_payload, dict) or "upper_band" not in cp_payload:
        raise ValueError(f"{path} does not contain cp_band.upper_band")
    cp_payload = dict(cp_payload)
    cp_payload["reused"] = True
    cp_payload["reused_from"] = os.path.expanduser(str(path))
    return cp_payload


def observed_cache_dir(cfg, args):
    cache_dir = args.observed_feature_cache_dir or cfg.get("observed_feature_cache_dir")
    if cache_dir in (None, ""):
        feature_cfg = cfg.get("observed_feature", {})
        cache_dir = feature_cfg.get("cache_dir") if isinstance(feature_cfg, dict) else None
    return None if cache_dir in (None, "") else os.path.expanduser(str(cache_dir))


def wm_cache_dir(cfg, ckpt):
    feature_cfg = ckpt.get("wm_feature")
    if not isinstance(feature_cfg, dict):
        feature_cfg = ckpt.get("feature_config") if ckpt.get("feature_source") == "wm_cache" else None
    if not isinstance(feature_cfg, dict):
        feature_cfg = cfg.get("wm_feature", {})
    cache_dir = feature_cfg.get("cache_dir") if isinstance(feature_cfg, dict) else None
    return None if cache_dir in (None, "") else os.path.expanduser(str(cache_dir))


def feature_source_from_ckpt(ckpt):
    source = ckpt.get("feature_source")
    if source in {"observed_cache", "wm_cache", "raw_observed"}:
        return source
    if "wm_feature" in ckpt:
        return "wm_cache"
    return "observed_cache"


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


def _domain_allowed(domain, domains):
    if not domains:
        return True
    if isinstance(domains, str):
        allowed = {item.strip() for item in domains.split(",") if item.strip()}
    else:
        allowed = {str(item).strip() for item in domains if str(item).strip()}
    return str(domain) in allowed


def domains_for_split(cfg, split, args_domains=None):
    if args_domains not in (None, ""):
        if isinstance(args_domains, str):
            return [item.strip() for item in args_domains.split(",") if item.strip()]
        return [str(item).strip() for item in args_domains if str(item).strip()]
    if split in cfg.get("data", {}):
        data_cfg = cfg["data"][split]
    elif "val" in cfg.get("data", {}):
        data_cfg = cfg["data"]["val"]
    else:
        data_cfg = cfg["data"]["train"]
    domains = data_cfg.get("domains")
    if domains in (None, ""):
        return None
    if isinstance(domains, str):
        if domains.strip().lower() == "all":
            return None
        return [item.strip() for item in domains.split(",") if item.strip()]
    return [str(item).strip() for item in domains if str(item).strip()]


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


def features_from_batch(batch, vae, ckpt, cfg, device, dtype):
    if feature_source_from_ckpt(ckpt) == "wm_cache":
        feature_cfg = dict(ckpt.get("wm_feature", ckpt.get("feature_config", cfg.get("wm_feature", {}))))
        feature_mode = str(ckpt.get("feature_mode", feature_cfg.get("feature_mode", "action_force_cache_mean")))
        return build_action_force_cache_features(
            pred_force=batch["pred_force"].to(device=device, dtype=torch.float32, non_blocking=True),
            interaction_tokens=batch["interaction_tokens"].to(device=device, dtype=torch.float32, non_blocking=True),
            pred_latents=batch["pred_latents"].to(device=device, dtype=torch.float32, non_blocking=True),
            mode=feature_mode,
        ).float()

    feature_cfg = dict(ckpt.get("observed_feature", ckpt.get("feature_config", cfg.get("observed_feature", {}))))
    feature_mode = str(ckpt.get("feature_mode", feature_cfg.get("feature_mode", "video_state_force")))
    force_bins = int(ckpt.get("force_bins", feature_cfg.get("force_bins", 5)))
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

    feature_ckpt = {"observed_feature": feature_cfg, "feature_mode": feature_mode, "force_bins": force_bins}
    return build_features_from_batch(batch, vae, feature_ckpt, cfg, device, dtype)


@torch.no_grad()
def score_split(cfg, split, vae, pca_kmeans_model, ckpt, args, device, dtype):
    feature_source = feature_source_from_ckpt(ckpt)
    cache_dir = wm_cache_dir(cfg, ckpt) if feature_source == "wm_cache" else observed_cache_dir(cfg, args)
    split_domains = domains_for_split(cfg, split, args.domains)
    if feature_source == "wm_cache" and not cache_dir:
        raise ValueError("WM PCA-KMeans checkpoint requires wm_feature.cache_dir in checkpoint or config.")
    if cache_dir and feature_source == "wm_cache":
        feature_cfg = ckpt.get("wm_feature") if isinstance(ckpt.get("wm_feature"), dict) else ckpt.get("feature_config", {})
        dataset = AnomalyFeatureCacheDataset(
            cache_dir=cache_dir,
            split=split,
            max_open_shards=int(feature_cfg.get("max_open_shards", args.observed_cache_max_open_shards)),
            domains=split_domains,
        )
        wrapped_dataset = dataset
    elif cache_dir:
        dataset = ObservedFeatureCacheDataset(
            cache_dir=cache_dir,
            split=split,
            max_open_shards=int(args.observed_cache_max_open_shards),
            domains=split_domains,
        )
        wrapped_dataset = dataset
    else:
        dataset = build_dataset(cfg, split, split_manifest=args.split_manifest, domains=split_domains)
        wrapped_dataset = IndexedDataset(dataset)
    loader = DataLoader(
        wrapped_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )
    records = []
    total = len(dataset) if args.max_windows is None else min(len(dataset), int(args.max_windows))
    seen = 0
    for batch in tqdm(loader, desc=f"Scoring PCA-KMeans {split}"):
        if seen >= total:
            break
        if "label" in batch:
            batch_size = int(batch["label"].shape[0])
        elif "video_features" in batch:
            batch_size = int(batch["video_features"].shape[0])
        else:
            batch_size = int(batch["video"].shape[0])
        keep = min(batch_size, total - seen)
        features = features_from_batch(batch, vae, ckpt, cfg, device, dtype).float()
        scores = pca_kmeans_model.score(features).detach().cpu().float().reshape(-1)
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
        for row in range(keep):
            if cache_dir:
                meta = cached_metadata_from_batch(batch, row)
            else:
                meta = metadata_for_index(dataset, int(source_indices[row].item()))
            if not _domain_allowed(meta["domain"], split_domains):
                continue
            records.append(
                {
                    **meta,
                    "score": float(scores[row].item()),
                    "label": float(labels[row].item()),
                    "target": float(1.0 - labels[row].item()),
                    "episode_label": float(episode_labels[row].item()),
                    "risk_label": None if risk_labels is None else float(risk_labels[row].item()),
                    "frame_anomaly_label": None if frame_labels is None else frame_labels[row].tolist(),
                    "interaction_label": None if interaction_labels is None else int(interaction_labels[row].item()),
                }
            )
        seen += keep
    return records


def domain_allowed(domain, domains):
    if not domains:
        return True
    allowed = {str(item).strip() for item in domains if str(item).strip()}
    return str(domain) in allowed

def filter_records_by_domains(records, domains):
    if not domains:
        return list(records)
    return [row for row in records if domain_allowed(row.get("domain", ""), domains)]


def trajectory_key(row):
    return (str(row.get("domain", "")), int(row.get("episode_id", row.get("episode_dataset_index", -1))))


def window_trajectories_from_records(records):
    grouped = defaultdict(list)
    for row in records:
        grouped[trajectory_key(row)].append(row)
    trajectories = []
    for (domain, episode_id), rows in sorted(grouped.items()):
        rows = sorted(
            rows,
            key=lambda item: (
                int(item.get("detection_frame", item.get("window_end", -1))),
                int(item.get("window_start", -1)),
                int(item.get("source_index", -1)),
            ),
        )
        trajectories.append(
            {
                "domain": domain,
                "episode_id": int(episode_id),
                "episode_label": float(rows[0].get("episode_label", 1.0)),
                "scores": torch.tensor([float(row["score"]) for row in rows], dtype=torch.float32),
                "targets": torch.tensor([float(row.get("target", 0.0)) for row in rows], dtype=torch.float32),
                "valid_mask": torch.ones(len(rows), dtype=torch.bool),
                "target_mask": torch.ones(len(rows), dtype=torch.bool),
                "frame_indices": [int(row.get("detection_frame", row.get("window_end", idx))) for idx, row in enumerate(rows)],
            }
        )
    return trajectories



def conformal_quantile(scores, alpha):
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    scores = scores[torch.isfinite(scores)]
    if scores.numel() == 0:
        raise ValueError("Cannot compute conformal quantile from empty scores.")
    k = int(torch.ceil(torch.tensor((scores.numel() + 1) * (1.0 - float(alpha)))).item())
    k = min(max(1, k), int(scores.numel()))
    return torch.sort(scores).values[k - 1]




def resolve_cp_normal_filter(args, ckpt):
    if args.cp_normal_filter != "auto":
        return str(args.cp_normal_filter)
    fit_filter = str(ckpt.get("fit_filter") or ckpt.get("observed_feature", {}).get("fit_filter", "normal_episode_windows"))
    if fit_filter in {"normal_episode_windows", "normal_windows"}:
        return fit_filter
    return "normal_episode_windows"


def is_normal_calibration_record(row, normal_filter):
    target = float(row.get("target", 1.0))
    episode_label = float(row.get("episode_label", 0.0))
    if normal_filter == "normal_windows":
        return target < 0.5
    if normal_filter == "normal_episode_windows":
        return target < 0.5 and episode_label >= 0.5
    if normal_filter == "success_episodes":
        return episode_label >= 0.5
    raise ValueError(f"Unsupported cp_normal_filter={normal_filter}")


def tensor_summary(values):
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "p50": float(torch.quantile(values, 0.5).item()),
        "p90": float(torch.quantile(values, 0.9).item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
        "max": float(values.max().item()),
    }


def compute_global_cp_threshold(calibration_records, eval_records, alpha=0.1, normal_filter="normal_episode_windows"):
    eval_trajectories = window_trajectories_from_records(eval_records)
    normal_records = [row for row in calibration_records if is_normal_calibration_record(row, normal_filter)]
    normal_scores = torch.tensor([float(row["score"]) for row in normal_records], dtype=torch.float32)
    normal_scores = normal_scores[torch.isfinite(normal_scores)]
    if normal_scores.numel() == 0 or not eval_trajectories:
        return {
            "enabled": False,
            "reason": "Need at least one normal calibration window and one eval episode.",
            "method": "global_threshold",
            "alpha": float(alpha),
            "normal_filter": str(normal_filter),
            "num_calibration_records": len(calibration_records),
            "num_calibration_normal_records": int(normal_scores.numel()),
            "num_eval_episodes": len(eval_trajectories),
            "time_axis": "window",
        }

    threshold = float(conformal_quantile(normal_scores, alpha).item())
    trajectory_length = max(1, max(int(item["scores"].numel()) for item in eval_trajectories))
    upper_band = torch.full((trajectory_length,), threshold, dtype=torch.float32)
    valid_band = torch.ones(trajectory_length, dtype=torch.bool)

    episode_predictions = []
    episode_targets = []
    detection_indices = []
    for item in eval_trajectories:
        scores = item["scores"].float()
        valid = item["valid_mask"].bool()
        if not bool(valid.any().item()):
            continue
        exceed = scores[valid] >= threshold
        pred = float(exceed.any().item())
        target = float((item["targets"][item["target_mask"].bool()] >= 0.5).any().item())
        episode_predictions.append(pred)
        episode_targets.append(target)
        if pred > 0.5 and target > 0.5:
            valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
            detection_step = int(valid_indices[torch.nonzero(exceed, as_tuple=False)[0].item()].item())
            frame_indices = item.get("frame_indices", [])
            detection_indices.append(int(frame_indices[detection_step]) if detection_step < len(frame_indices) else detection_step)

    window_predictions = [float(float(row["score"]) >= threshold) for row in eval_records]
    window_targets = [float(float(row.get("target", 0.0)) >= 0.5) for row in eval_records]
    normal_episode_keys = {
        (str(row.get("domain", "")), int(row.get("episode_id", row.get("episode_dataset_index", -1))))
        for row in normal_records
    }

    cp_payload = {
        "enabled": True,
        "method": "global_threshold",
        "threshold_mode": "fixed",
        "alpha": float(alpha),
        "normal_filter": str(normal_filter),
        "threshold": threshold,
        "time_axis": "window",
        "window_band_reduction": "global scalar threshold repeated for compatibility",
        "num_calibration_records": len(calibration_records),
        "num_calibration_normal_records": int(normal_scores.numel()),
        "num_calibration_normal_episodes": len(normal_episode_keys),
        "num_eval_episodes": len(eval_trajectories),
        "trajectory_length": int(upper_band.numel()),
        "valid_band_timesteps": int(valid_band.sum().item()),
        "band_mean": threshold,
        "band_std": 0.0,
        "mean_detection_index": float(torch.tensor(detection_indices, dtype=torch.float32).mean().item()) if detection_indices else None,
        "episode_metrics": binary_metrics_from_predictions(episode_predictions, episode_targets),
        "window_metrics": binary_metrics_from_predictions(window_predictions, window_targets),
        "upper_band": upper_band.tolist(),
        "valid_band_mask": valid_band.tolist(),
        "calibration_diagnostics": {
            "normal_score_summary": tensor_summary(normal_scores),
            "top_normal_windows_by_score": sorted(
                [
                    {
                        "domain": str(row.get("domain", "")),
                        "episode_id": int(row.get("episode_id", row.get("episode_dataset_index", -1))),
                        "source_index": int(row.get("source_index", -1)),
                        "window_start": int(row.get("window_start", -1)),
                        "detection_frame": int(row.get("detection_frame", -1)),
                        "score": float(row["score"]),
                        "target": float(row.get("target", 0.0)),
                        "episode_label": float(row.get("episode_label", 0.0)),
                    }
                    for row in normal_records
                ],
                key=lambda item: item["score"],
                reverse=True,
            )[:20],
        },
    }
    return cp_payload
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_resolved_config(args.config_file)
    if args.split_manifest in (None, ""):
        args.split_manifest = cfg.get("split_manifest")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if str(cfg.get("mixed_precision", "no")) == "bf16" and device.type == "cuda" else torch.float32
    pca_kmeans_model, ckpt = load_pca_kmeans(args.pca_kmeans_checkpoint)
    feature_source = feature_source_from_ckpt(ckpt)
    cache_dir = wm_cache_dir(cfg, ckpt) if feature_source == "wm_cache" else observed_cache_dir(cfg, args)
    if feature_source == "wm_cache" and not cache_dir:
        raise ValueError("WM PCA-KMeans checkpoint requires wm_feature.cache_dir in checkpoint or config.")
    vae = None if cache_dir else load_vae(cfg, device, dtype)

    calibration_records_path = args.calibration_records or default_calibration_records_path(args.pca_kmeans_checkpoint)
    calibration_payload = None
    reused_cp_payload = load_cp_band_metrics(args.cp_metrics) if args.cp_metrics not in (None, "") else None
    if reused_cp_payload is not None:
        calibration_records = []
        calibration_source = "reused_cp_band"
    elif not bool(args.recalibrate) and os.path.isfile(os.path.expanduser(str(calibration_records_path))):
        calibration_records, calibration_payload = load_calibration_records(calibration_records_path)
        calibration_source = "saved_records"
    else:
        calibration_records = score_split(cfg, args.calibration_split, vae, pca_kmeans_model, ckpt, args, device, dtype)
        calibration_source = "recomputed"
    eval_records = score_split(cfg, args.eval_split, vae, pca_kmeans_model, ckpt, args, device, dtype)
    if reused_cp_payload is not None:
        cp_payload = reused_cp_payload
    elif str(args.cp_method) == "global":
        calibration_domains = domains_for_split(cfg, args.calibration_split, args.domains)
        eval_domains = domains_for_split(cfg, args.eval_split, args.domains)
        calibration_records = filter_records_by_domains(calibration_records, calibration_domains)
        eval_records = filter_records_by_domains(eval_records, eval_domains)
        if not calibration_records:
            raise ValueError("No calibration records remain for global CP calibration.")
        cp_payload = compute_global_cp_threshold(
            calibration_records,
            eval_records,
            alpha=float(args.cp_alpha),
            normal_filter=resolve_cp_normal_filter(args, ckpt),
        )
    else:
        cp_payload = compute_cp_band(
            calibration_records,
            eval_records,
            alpha=float(args.cp_alpha),
            method=str(args.cp_method),
            train_ratio=float(args.cp_train_ratio),
        )

    by_interaction = grouped_records(eval_records, "interaction_label", default="unknown")
    interaction_payloads = {}
    for interaction, interaction_records in sorted(by_interaction.items()):
        interaction_scores, interaction_targets = records_scores_targets(interaction_records)
        interaction_payloads[interaction] = {
            "num_windows": len(interaction_records),
            "score_summary": score_summary(interaction_records),
            "window_metrics_best_f1": best_f1_metrics(interaction_scores, interaction_targets),
            "cp_window_metrics": cp_window_metrics_for_records(interaction_records, cp_payload),
        }
    episode_plot_paths = save_episode_plots(
        args.output_dir,
        eval_records,
        cp_payload,
        max_plots=args.max_episode_plots,
    )
    unified_output_paths = save_unified_baseline_outputs(
        args.output_dir,
        eval_records,
        cp_payload,
        metadata={
            "script": "analyze/eval_pca_kmeans_metrics.py",
            "config_file": args.config_file,
            "pca_kmeans_checkpoint": args.pca_kmeans_checkpoint,
            "calibration_split": args.calibration_split,
            "calibration_records": calibration_records_path,
            "calibration_source": calibration_source,
            "cp_metrics": args.cp_metrics,
            "cp_method": args.cp_method,
            "eval_split": args.eval_split,
            "split_manifest": args.split_manifest,
            "domains": args.domains,
            "feature_source": feature_source,
            "feature_cache_dir": cache_dir,
            "observed_feature_cache_dir": cache_dir if feature_source != "wm_cache" else None,
            "baseline": ckpt.get("baseline", "frozen_vae_observed_pca_kmeans"),
        },
    )

    payload = {
        "config_file": args.config_file,
        "pca_kmeans_checkpoint": args.pca_kmeans_checkpoint,
        "calibration_split": args.calibration_split,
        "calibration_records": calibration_records_path,
        "calibration_source": calibration_source,
        "cp_metrics": args.cp_metrics,
        "cp_method": args.cp_method,
        "eval_split": args.eval_split,
        "split_manifest": args.split_manifest,
        "domains": args.domains,
        "feature_source": feature_source,
        "feature_cache_dir": cache_dir,
        "observed_feature_cache_dir": cache_dir if feature_source != "wm_cache" else None,
        "score_direction": "higher_is_more_anomalous",
        "baseline": ckpt.get("baseline", "frozen_vae_observed_pca_kmeans"),
        "observed_feature": {
            "feature_mode": ckpt.get("feature_mode"),
            "fit_filter": ckpt.get("fit_filter"),
            "force_bins": ckpt.get("force_bins"),
            "fit_meta": ckpt.get("fit_meta"),
        },
        "pca_kmeans": ckpt.get("pca_kmeans", {}),
        "calibration_score_summary": score_summary(calibration_records) if calibration_records else None,
        "eval_score_summary": score_summary(eval_records),
        "cp_band": cp_payload,
        "cp_episode_metrics": cp_payload.get("episode_metrics") if cp_payload else None,
        "cp_window_metrics": cp_payload.get("window_metrics") if cp_payload else None,
        "interactions": interaction_payloads,
        "episode_plots": episode_plot_paths,
        "unified_episode_eval": unified_output_paths,
    }
    cp_records_path = os.path.join(args.output_dir, "pca_kmeans_cp_records.pt")
    early_warning_paths = None
    if not args.disable_early_warning and cp_payload.get("enabled", False):
        early_warning_dir = args.early_warning_output_dir or os.path.join(args.output_dir, "early_warning_analysis")
        early_warning_paths = run_early_warning_analysis(
            eval_records,
            cp_payload,
            output_dir=early_warning_dir,
            early_window=int(args.early_window),
            transition_window=int(args.transition_window),
            min_segment_len=int(args.min_segment_len),
            cp_records=cp_records_path,
        )
        payload["early_warning_analysis"] = early_warning_paths
    with open(os.path.join(args.output_dir, "pca_kmeans_cp_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    torch.save(
        {
            "calibration_records": calibration_records,
            "calibration_payload": calibration_payload,
            "eval_records": eval_records,
            "cp_band": cp_payload,
        },
        cp_records_path,
    )
    print(json.dumps({k: payload[k] for k in ("cp_episode_metrics", "cp_window_metrics")}, indent=2))
    print(args.output_dir)

if __name__ == "__main__":
    main()
