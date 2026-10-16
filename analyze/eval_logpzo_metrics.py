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

from analyze.analyze_cp_early_warning import run_early_warning_analysis
from analyze.baseline_episode_eval_common import (
    best_f1_metrics,
    binary_metrics_from_predictions,
    compute_cp_band,
    cp_window_metrics_for_records,
    grouped_records,
    records_scores_targets,
    save_episode_plots,
    save_unified_baseline_outputs,
    score_summary,
)
from data.anomaly_feature_cache_dataset import AnomalyFeatureCacheDataset
from data.observed_feature_cache_dataset import ObservedFeatureCacheDataset
from models.baselines.logpzo import (
    LogpZOModel,
    build_observed_logpzo_features,
    build_wm_logpzo_features,
)
from utils.path_utils import load_resolved_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate observed/WM logpZO and write pointwise plus fixed-threshold CP metrics.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--logpzo_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--calibration_split", default="val")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--domains", default=None, help="Comma-separated domain subset override.")
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cp_alpha", type=float, default=0.1)
    parser.add_argument(
        "--pointwise_cp_metrics",
        "--cp_metrics",
        dest="pointwise_cp_metrics",
        default=None,
        help="Reuse an existing pointwise logpzo_cp_metrics.json cp_band instead of recalibrating it.",
    )
    parser.add_argument(
        "--fixed_cp_metrics",
        default=None,
        help="Reuse an existing fixed-threshold logpzo_cp_metrics.json cp_band instead of recalibrating it.",
    )
    parser.add_argument(
        "--cp_normal_filter",
        choices=["auto", "normal_episode_windows", "normal_windows", "success_episodes"],
        default="auto",
    )
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--max_episode_plots", type=int, default=None)
    parser.add_argument("--disable_early_warning", action="store_true")
    parser.add_argument("--early_warning_output_dir", default=None)
    parser.add_argument("--early_window", type=int, default=50)
    parser.add_argument("--transition_window", type=int, default=50)
    parser.add_argument("--min_segment_len", type=int, default=2)
    return parser.parse_args()


def load_logpzo(checkpoint_path, device):
    ckpt = torch.load(os.path.expanduser(str(checkpoint_path)), map_location="cpu")
    logpzo_cfg = dict(ckpt.get("logpzo", {}))
    if "feature_dim" not in logpzo_cfg:
        raise ValueError("logpZO checkpoint must include logpzo.feature_dim")
    model = LogpZOModel(
        feature_dim=int(logpzo_cfg["feature_dim"]),
        hidden_dim=int(logpzo_cfg.get("hidden_dim", 256)),
        num_layers=int(logpzo_cfg.get("num_layers", 4)),
        time_embed_dim=int(logpzo_cfg.get("time_embed_dim", 128)),
        dropout=float(logpzo_cfg.get("dropout", 0.0)),
        time_scale=float(logpzo_cfg.get("time_scale", 100.0)),
        score_reduction=str(logpzo_cfg.get("score_reduction", "sum")),
        normalize_input=bool(logpzo_cfg.get("normalize_input", True)),
    )
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt


def load_cp_band_metrics(path):
    with open(os.path.expanduser(str(path)), "r", encoding="utf-8") as f:
        payload = json.load(f)
    cp_payload = payload.get("cp_band", payload)
    if not isinstance(cp_payload, dict) or "upper_band" not in cp_payload:
        raise ValueError(f"CP metrics file does not contain a valid cp_band.upper_band: {path}")
    cp_payload = dict(cp_payload)
    cp_payload["reused"] = True
    cp_payload["reused_from"] = os.path.expanduser(str(path))
    return cp_payload

def feature_source_from_ckpt(ckpt):
    if isinstance(ckpt.get("wm_feature"), dict):
        return "wm"
    if isinstance(ckpt.get("observed_feature"), dict):
        return "observed"
    raise ValueError("logpZO checkpoint must contain observed_feature or wm_feature.")


def feature_config(cfg, ckpt, source):
    key = "wm_feature" if source == "wm" else "observed_feature"
    out = dict(cfg.get(key, {}))
    out.update(dict(ckpt.get(key, {})))
    return out


def feature_cache_dir(cfg, ckpt, args, source):
    if args.feature_cache_dir not in (None, ""):
        return os.path.expanduser(str(args.feature_cache_dir))
    cfg_section = feature_config(cfg, ckpt, source)
    cache_dir = cfg_section.get("cache_dir")
    if cache_dir in (None, ""):
        raise ValueError(f"{source} logpZO eval requires {source}_feature.cache_dir or --feature_cache_dir.")
    return os.path.expanduser(str(cache_dir))


def domains_for_split(cfg, split, args_domains=None):
    if args_domains not in (None, ""):
        return [item.strip() for item in str(args_domains).split(",") if item.strip()]
    data = cfg.get("data", {})
    data_cfg = data.get(split) or data.get("val") or data.get("train") or {}
    domains = data_cfg.get("domains")
    if domains in (None, ""):
        return None
    if isinstance(domains, str):
        return None if domains.strip().lower() == "all" else [item.strip() for item in domains.split(",") if item.strip()]
    return [str(item).strip() for item in domains if str(item).strip()]


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


def make_dataset(cfg, ckpt, args, source, split):
    cfg_section = feature_config(cfg, ckpt, source)
    cache_dir = feature_cache_dir(cfg, ckpt, args, source)
    kwargs = {
        "cache_dir": cache_dir,
        "split": split,
        "max_open_shards": int(cfg_section.get("max_open_shards", 4)),
        "domains": domains_for_split(cfg, split, args.domains),
    }
    if source == "wm":
        return AnomalyFeatureCacheDataset(**kwargs), cache_dir
    return ObservedFeatureCacheDataset(**kwargs), cache_dir


def observed_features_from_batch(batch, cfg, ckpt, device):
    feature_cfg = feature_config(cfg, ckpt, "observed")
    data_cfg = cfg.get("data", {}).get("train", {})
    mem_size = int(data_cfg.get("n_previous", data_cfg.get("memory_steps", 4)))
    state = batch.get("state")
    history_force = batch.get("history_force_state")
    return build_observed_logpzo_features(
        batch["video_features"].to(device=device, dtype=torch.float32, non_blocking=True),
        state=None if state is None else state.to(device=device, dtype=torch.float32, non_blocking=True),
        history_force_state=None
        if history_force is None
        else history_force.to(device=device, dtype=torch.float32, non_blocking=True),
        mode=str(ckpt.get("feature_mode", feature_cfg.get("feature_mode", "video_state_force"))),
        force_bins=int(ckpt.get("force_bins", feature_cfg.get("force_bins", 5))),
        mem_size=mem_size,
    )


def wm_features_from_batch(batch, cfg, ckpt, device):
    feature_cfg = feature_config(cfg, ckpt, "wm")
    return build_wm_logpzo_features(
        pred_latents=batch["pred_latents"].to(device=device, dtype=torch.float32, non_blocking=True),
        pred_force=batch["pred_force"].to(device=device, dtype=torch.float32, non_blocking=True),
        interaction_tokens=batch["interaction_tokens"].to(device=device, dtype=torch.float32, non_blocking=True),
        valid_len=batch.get("valid_len"),
        token_proj_dim=int(feature_cfg.get("token_proj_dim", 64)),
    )


@torch.no_grad()
def score_split(cfg, ckpt, model, args, source, split, device):
    dataset, cache_dir = make_dataset(cfg, ckpt, args, source, split)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
    )
    records = []
    total = len(dataset) if args.max_windows is None else min(len(dataset), int(args.max_windows))
    seen = 0
    for batch in tqdm(loader, desc=f"Scoring logpZO {source} {split}"):
        if seen >= total:
            break
        batch_size = int(batch["label"].shape[0])
        keep = min(batch_size, total - seen)
        if source == "wm":
            features, token_mask = wm_features_from_batch(batch, cfg, ckpt, device)
            scores = model.score(features.float(), token_mask=token_mask.to(device=device)).detach().cpu().float().reshape(-1)
        else:
            features = observed_features_from_batch(batch, cfg, ckpt, device)
            scores = model.score(features.float()).detach().cpu().float().reshape(-1)

        labels = batch["label"].detach().cpu().float().reshape(-1)
        episode_labels = batch["episode_label"].detach().cpu().float().reshape(-1) if "episode_label" in batch else labels
        risk_labels = batch.get("risk_label")
        risk_labels = None if risk_labels is None else risk_labels.detach().cpu().float().reshape(-1)
        frame_labels = batch.get("frame_anomaly_label")
        frame_labels = None if frame_labels is None else frame_labels.detach().cpu().float()
        interaction_labels = batch.get("interaction_label")
        interaction_labels = None if interaction_labels is None else interaction_labels.detach().cpu().long().reshape(-1)

        for row in range(keep):
            records.append(
                {
                    **cached_metadata_from_batch(batch, row),
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
    return records, cache_dir


def resolve_cp_normal_filter(args, ckpt, source):
    if args.cp_normal_filter != "auto":
        return str(args.cp_normal_filter)
    feature_key = "wm_feature" if source == "wm" else "observed_feature"
    feature_cfg = ckpt.get(feature_key, {}) if isinstance(ckpt.get(feature_key), dict) else {}
    fit_filter = str(ckpt.get("fit_filter", feature_cfg.get("fit_filter", "normal_episode_windows")))
    return fit_filter if fit_filter in {"normal_episode_windows", "normal_windows"} else "normal_episode_windows"


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


def conformal_quantile(scores, alpha):
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    scores = scores[torch.isfinite(scores)]
    if scores.numel() == 0:
        raise ValueError("Cannot compute conformal quantile from empty scores.")
    k = int(torch.ceil(torch.tensor((scores.numel() + 1) * (1.0 - float(alpha)))).item())
    k = min(max(1, k), int(scores.numel()))
    return torch.sort(scores).values[k - 1]


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
                "frame_indices": [int(row.get("detection_frame", row.get("window_end", idx))) for idx, row in enumerate(rows)],
            }
        )
    return trajectories


def compute_global_cp_threshold(calibration_records, eval_records, alpha=0.1, normal_filter="normal_episode_windows"):
    normal_records = [row for row in calibration_records if is_normal_calibration_record(row, normal_filter)]
    normal_scores = torch.tensor([float(row["score"]) for row in normal_records], dtype=torch.float32)
    eval_trajectories = window_trajectories_from_records(eval_records)
    if normal_scores.numel() == 0 or not eval_trajectories:
        return {
            "enabled": False,
            "method": "global_threshold",
            "reason": "Need at least one normal calibration window and one eval episode.",
            "alpha": float(alpha),
            "normal_filter": str(normal_filter),
            "num_calibration_records": len(calibration_records),
            "num_calibration_normal_records": int(normal_scores.numel()),
            "num_eval_episodes": len(eval_trajectories),
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
        exceed = scores >= threshold
        pred = float(exceed.any().item())
        target = float((item["targets"] >= 0.5).any().item())
        episode_predictions.append(pred)
        episode_targets.append(target)
        if pred > 0.5 and target > 0.5:
            detection_step = int(torch.nonzero(exceed, as_tuple=False)[0].item())
            frame_indices = item.get("frame_indices", [])
            detection_indices.append(int(frame_indices[detection_step]) if detection_step < len(frame_indices) else detection_step)

    window_predictions = [float(float(row["score"]) >= threshold) for row in eval_records]
    window_targets = [float(float(row.get("target", 0.0)) >= 0.5) for row in eval_records]
    normal_episode_keys = {trajectory_key(row) for row in normal_records}
    return {
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


def interaction_metrics(eval_records, cp_payload):
    payloads = {}
    for interaction, rows in sorted(grouped_records(eval_records, "interaction_label", default="unknown").items()):
        scores, targets = records_scores_targets(rows)
        payloads[interaction] = {
            "num_windows": len(rows),
            "score_summary": score_summary(rows),
            "window_metrics_best_f1": best_f1_metrics(scores, targets),
            "cp_window_metrics": cp_window_metrics_for_records(rows, cp_payload),
        }
    return payloads


def write_method_outputs(
    method_name,
    output_dir,
    cfg,
    ckpt,
    args,
    source,
    cache_dir,
    calibration_records,
    eval_records,
    cp_payload,
):
    method_dir = os.path.join(output_dir, method_name)
    os.makedirs(method_dir, exist_ok=True)
    cp_records_path = os.path.join(method_dir, "logpzo_cp_records.pt")
    episode_plot_paths = save_episode_plots(method_dir, eval_records, cp_payload, max_plots=args.max_episode_plots)
    unified_output_paths = save_unified_baseline_outputs(
        method_dir,
        eval_records,
        cp_payload,
        metadata={
            "script": "analyze/eval_logpzo_metrics.py",
            "config_file": args.config_file,
            "logpzo_checkpoint": args.logpzo_checkpoint,
            "feature_source": source,
            "feature_cache_dir": cache_dir,
            "calibration_split": args.calibration_split,
            "eval_split": args.eval_split,
            "domains": args.domains,
            "cp_variant": method_name,
            "baseline": ckpt.get("baseline", f"{source}_logpzo"),
        },
    )

    payload = {
        "config_file": args.config_file,
        "logpzo_checkpoint": args.logpzo_checkpoint,
        "feature_source": source,
        "feature_cache_dir": cache_dir,
        "calibration_split": args.calibration_split,
        "eval_split": args.eval_split,
        "domains": args.domains,
        "score_direction": "higher_is_more_anomalous",
        "baseline": ckpt.get("baseline", f"{source}_logpzo"),
        "feature": feature_config(cfg, ckpt, source),
        "logpzo": ckpt.get("logpzo", {}),
        "calibration_score_summary": score_summary(calibration_records) if calibration_records else None,
        "eval_score_summary": score_summary(eval_records),
        "cp_band": cp_payload,
        "cp_episode_metrics": cp_payload.get("episode_metrics") if cp_payload else None,
        "cp_window_metrics": cp_payload.get("window_metrics") if cp_payload else None,
        "interactions": interaction_metrics(eval_records, cp_payload),
        "episode_plots": episode_plot_paths,
        "unified_episode_eval": unified_output_paths,
    }
    if not args.disable_early_warning and cp_payload.get("enabled", False):
        early_warning_dir = args.early_warning_output_dir or os.path.join(method_dir, "early_warning_analysis")
        payload["early_warning_analysis"] = run_early_warning_analysis(
            eval_records,
            cp_payload,
            output_dir=early_warning_dir,
            early_window=int(args.early_window),
            transition_window=int(args.transition_window),
            min_segment_len=int(args.min_segment_len),
            cp_records=cp_records_path,
        )

    with open(os.path.join(method_dir, "logpzo_cp_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    torch.save(
        {
            "calibration_records": calibration_records,
            "eval_records": eval_records,
            "cp_band": cp_payload,
        },
        cp_records_path,
    )
    return payload


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_resolved_config(args.config_file)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_logpzo(args.logpzo_checkpoint, device)
    source = feature_source_from_ckpt(ckpt)
    normal_filter = resolve_cp_normal_filter(args, ckpt, source)

    reused_pointwise_cp = load_cp_band_metrics(args.pointwise_cp_metrics) if args.pointwise_cp_metrics else None
    reused_fixed_cp = load_cp_band_metrics(args.fixed_cp_metrics) if args.fixed_cp_metrics else None
    needs_calibration = reused_pointwise_cp is None or reused_fixed_cp is None

    if needs_calibration:
        calibration_records, calibration_cache_dir = score_split(cfg, ckpt, model, args, source, args.calibration_split, device)
        normal_calibration_records = [row for row in calibration_records if is_normal_calibration_record(row, normal_filter)]
        if not normal_calibration_records:
            raise ValueError(f"No normal calibration records for cp_normal_filter={normal_filter}.")
    else:
        calibration_records = []
        normal_calibration_records = []
        calibration_cache_dir = None

    eval_records, eval_cache_dir = score_split(cfg, ckpt, model, args, source, args.eval_split, device)

    if reused_pointwise_cp is not None:
        pointwise_cp = reused_pointwise_cp
    else:
        pointwise_cp = compute_cp_band(
            normal_calibration_records,
            eval_records,
            alpha=float(args.cp_alpha),
            method="pointwise",
        )
        if pointwise_cp:
            pointwise_cp["normal_filter"] = normal_filter
            pointwise_cp["num_calibration_records_before_filter"] = len(calibration_records)
            pointwise_cp["num_calibration_records_after_filter"] = len(normal_calibration_records)

    if reused_fixed_cp is not None:
        fixed_cp = reused_fixed_cp
    else:
        fixed_cp = compute_global_cp_threshold(
            calibration_records,
            eval_records,
            alpha=float(args.cp_alpha),
            normal_filter=normal_filter,
        )

    pointwise_payload = write_method_outputs(
        "pointwise",
        args.output_dir,
        cfg,
        ckpt,
        args,
        source,
        eval_cache_dir,
        normal_calibration_records,
        eval_records,
        pointwise_cp,
    )
    fixed_payload = write_method_outputs(
        "fixed",
        args.output_dir,
        cfg,
        ckpt,
        args,
        source,
        eval_cache_dir,
        calibration_records,
        eval_records,
        fixed_cp,
    )

    root_payload = {
        "config_file": args.config_file,
        "logpzo_checkpoint": args.logpzo_checkpoint,
        "feature_source": source,
        "calibration_cache_dir": calibration_cache_dir,
        "eval_cache_dir": eval_cache_dir,
        "calibration_split": args.calibration_split,
        "eval_split": args.eval_split,
        "cp_alpha": float(args.cp_alpha),
        "cp_normal_filter": normal_filter,
        "pointwise_cp_metrics": args.pointwise_cp_metrics,
        "fixed_cp_metrics": args.fixed_cp_metrics,
        "outputs": {
            "pointwise": os.path.join(args.output_dir, "pointwise", "logpzo_cp_metrics.json"),
            "fixed": os.path.join(args.output_dir, "fixed", "logpzo_cp_metrics.json"),
            "pointwise_episode_plots": os.path.join(args.output_dir, "pointwise", "episode_plots"),
            "pointwise_summary_plots": os.path.join(args.output_dir, "pointwise", "plots", "summary"),
            "fixed_episode_plots": os.path.join(args.output_dir, "fixed", "episode_plots"),
            "fixed_summary_plots": os.path.join(args.output_dir, "fixed", "plots", "summary"),
        },
        "pointwise": {
            "cp_episode_metrics": pointwise_payload.get("cp_episode_metrics"),
            "cp_window_metrics": pointwise_payload.get("cp_window_metrics"),
        },
        "fixed": {
            "cp_episode_metrics": fixed_payload.get("cp_episode_metrics"),
            "cp_window_metrics": fixed_payload.get("cp_window_metrics"),
        },
    }
    with open(os.path.join(args.output_dir, "logpzo_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(root_payload, f, indent=2)
    torch.save(
        {
            "calibration_records": calibration_records,
            "normal_calibration_records": normal_calibration_records,
            "eval_records": eval_records,
            "pointwise_cp_band": pointwise_cp,
            "fixed_cp_band": fixed_cp,
        },
        os.path.join(args.output_dir, "logpzo_records.pt"),
    )
    print(json.dumps(root_payload, indent=2))


if __name__ == "__main__":
    main()
