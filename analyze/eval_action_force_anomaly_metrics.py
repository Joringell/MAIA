import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_MPL_CACHE_DIR = "/tmp/forcewm_matplotlib_cache"
os.makedirs(_MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", _MPL_CACHE_DIR)

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.anomaly_eval_common import (
    compute_auprc,
    compute_auroc,
    build_episode_metrics_video,
    extract_episode_id,
    load_failure_transition_frame,
    load_frame_labels,
    write_json,
)
from analyze.action_force_anomaly_eval_core import (
    build_cache_episode_groups,
    build_cache_reference_dataset,
    build_eval_dataset,
    build_windows,
    infer_anomaly_batch,
    infer_cached_anomaly_batch,
    resolve_device,
    resolve_shard,
)
from analyze.action_force_anomaly_eval_core import load_anomaly_head
from analyze.eval_common import (
    load_multiview_video_segment,
    prepare_models,
    resolve_checkpoint_path,
    resolve_force_layout,
    resolve_video_template_and_cams,
    str_to_dtype,
)
from data.anomaly_feature_cache_dataset import AnomalyFeatureCacheDataset
from utils.path_utils import load_resolved_config


STAGES = ("all", "pre", "post")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Episode-based action-force anomaly evaluation with complete forward records."
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--anomaly_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--checkpoint_path", default=None, help="Optional WM checkpoint override for online inference.")
    parser.add_argument("--anomaly_feature_cache_dir", default=None, help="Optional feature cache; skips WM inference.")
    parser.add_argument("--disable_cache", action="store_true", help="Force online WM inference even if config has a cache dir.")
    parser.add_argument("--eval_split", default="test", help="Dataset split for online inference.")
    parser.add_argument("--cache_split", default=None, help="Cache split. Defaults to --eval_split.")
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--window_stride", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--mem_mode", choices=["uniform_history", "contiguous", "zeros"], default="uniform_history")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--shard_rank", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--merge_wait_seconds", type=int, default=3600)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--plot_mode", choices=["none", "top_errors", "normal_correct", "all"], default="top_errors")
    parser.add_argument("--top_k_plots", type=int, default=20)
    parser.add_argument("--make_episode_videos", action="store_true")
    parser.add_argument("--video_fps", type=float, default=30.0)
    parser.add_argument("--video_output_width", type=int, default=960)
    parser.add_argument("--video_plot_height", type=int, default=420)
    parser.add_argument(
        "--include_deployment_outputs",
        action="store_true",
        help="Also write deployment/early-warning summaries. Disabled by default because window metrics are the primary evaluation.",
    )
    parser.add_argument(
        "--save_score_histograms",
        action="store_true",
        help="Also save score histogram summary plots.",
    )
    return parser.parse_args()


def resolve_run_id(args):
    if args.run_id:
        return str(args.run_id)
    return os.environ.get("TORCHELASTIC_RUN_ID", "single")


def normalize_optional_path(path):
    if path in (None, ""):
        return None
    value = str(path)
    if value.strip().lower() in {"none", "null"}:
        return None
    return value


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def binary_metrics(scores, targets, threshold):
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
    tnr = tn / max(1, tn + fp)
    fpr = fp / max(1, fp + tn)
    balanced_accuracy = 0.5 * (recall + tnr)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "num_samples": int(scores.size),
        "num_positive": int(pos.sum()),
        "num_negative": int(neg.sum()),
        "positive_ratio": float(pos.mean()),
        "pred_positive_ratio": float(pred.mean()),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / max(1, scores.size)),
        "balanced_accuracy": float(balanced_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "tpr": float(recall),
        "tnr": float(tnr),
        "fpr": float(fpr),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "auroc": compute_auroc(scores, targets),
        "auprc": compute_auprc(scores, targets),
    }


def best_f1_metrics(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid]
    if scores.size == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    total_pos = int((targets == 1).sum())
    total_neg = int((targets == 0).sum())

    pos_cumsum = np.cumsum(sorted_targets == 1)
    fp_cumsum = np.cumsum(sorted_targets == 0)
    end_points = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), scores.size - 1]
    thresholds = sorted_scores[end_points]
    tp = pos_cumsum[end_points].astype(np.float64)
    fp = fp_cumsum[end_points].astype(np.float64)
    fn = float(total_pos) - tp
    tn = float(total_neg) - fp
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    f1 = 2.0 * precision * recall / np.maximum(1e-12, precision + recall)
    accuracy = (tp + tn) / max(1, scores.size)

    manual = binary_metrics(scores, targets, 0.5)
    thresholds = np.concatenate([thresholds, np.asarray([0.5], dtype=np.float64)])
    f1 = np.concatenate([f1, np.asarray([manual["f1"]], dtype=np.float64)])
    accuracy = np.concatenate([accuracy, np.asarray([manual["accuracy"]], dtype=np.float64)])
    tie_break = -np.abs(thresholds - 0.5)
    best_index = int(np.lexsort((tie_break, accuracy, f1))[-1])
    best_threshold = float(thresholds[best_index])
    best_f1 = float(f1[best_index])
    best_acc = float(accuracy[best_index])
    best = binary_metrics(scores, targets, best_threshold)
    best["threshold_mode"] = "best_f1"
    best["f1"] = float(best_f1)
    best["accuracy"] = float(best_acc)
    return best


def recall_constrained_metrics(scores, targets, min_recall=0.8):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid]
    if scores.size == 0:
        return None

    total_pos = int((targets == 1).sum())
    total_neg = int((targets == 0).sum())
    if total_pos == 0:
        return None

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    pos_cumsum = np.cumsum(sorted_targets == 1)
    fp_cumsum = np.cumsum(sorted_targets == 0)
    end_points = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]), scores.size - 1]

    thresholds = sorted_scores[end_points]
    tp = pos_cumsum[end_points].astype(np.float64)
    fp = fp_cumsum[end_points].astype(np.float64)
    fn = float(total_pos) - tp
    tn = float(total_neg) - fp
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / np.maximum(1.0, tp + fn)
    fpr = fp / np.maximum(1.0, fp + tn)
    tnr = tn / np.maximum(1.0, tn + fp)
    balanced_accuracy = 0.5 * (recall + tnr)
    f1 = 2.0 * precision * recall / np.maximum(1e-12, precision + recall)

    feasible = recall >= float(min_recall)
    if not np.any(feasible):
        return None

    feasible_indices = np.flatnonzero(feasible)
    # Primary objective: satisfy recall while minimizing false alarms.
    # Ties prefer higher threshold, then better precision/F1.
    best_local = int(
        np.lexsort(
            (
                f1[feasible_indices],
                precision[feasible_indices],
                thresholds[feasible_indices],
                -fpr[feasible_indices],
            )
        )[-1]
    )
    best_index = int(feasible_indices[best_local])
    best = binary_metrics(scores, targets, float(thresholds[best_index]))
    best["threshold_mode"] = f"recall_at_least_{float(min_recall):.2f}"
    best["target_recall"] = float(min_recall)
    best["recall"] = float(recall[best_index])
    best["tpr"] = float(recall[best_index])
    best["fpr"] = float(fpr[best_index])
    best["precision"] = float(precision[best_index])
    best["balanced_accuracy"] = float(balanced_accuracy[best_index])
    best["f1"] = float(f1[best_index])
    return best


def rank_average(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_corr(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    pred = pred[valid]
    target = target[valid]
    if pred.size < 2 or np.std(pred) == 0.0 or np.std(target) == 0.0:
        return None
    pred_rank = rank_average(pred)
    target_rank = rank_average(target)
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def risk_metrics(pred, target):
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(target)
    pred = pred[valid]
    target = target[valid]
    if pred.size == 0:
        return None
    err = pred - target
    return {
        "num_samples": int(pred.size),
        "mae": float(np.mean(np.abs(err))),
        "mse": float(np.mean(err * err)),
        "pred_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
        "spearman": spearman_corr(pred, target),
    }


def safe_first_true(mask):
    indices = np.flatnonzero(np.asarray(mask, dtype=bool))
    return int(indices[0]) if indices.size else None


def load_transition_frame(parquet_path, frame_anomaly):
    try:
        transition = load_failure_transition_frame(parquet_path)
        if transition is not None:
            return int(transition)
    except Exception:
        pass
    first = safe_first_true(np.asarray(frame_anomaly) > 0)
    return first


def stage_masks(starts, interaction_start):
    starts = np.asarray(starts, dtype=np.int64)
    if interaction_start is None:
        pre = np.ones(starts.shape[0], dtype=bool)
        post = np.zeros(starts.shape[0], dtype=bool)
    else:
        pre = starts < int(interaction_start)
        post = starts >= int(interaction_start)
    return {
        "all": np.ones(starts.shape[0], dtype=bool),
        "pre": pre,
        "post": post,
    }


def aggregate_frames(chunks, total_frames, force_dim):
    prob_sum = np.zeros(total_frames, dtype=np.float64)
    prob_count = np.zeros(total_frames, dtype=np.int64)
    pred_force_sum = np.zeros((total_frames, force_dim), dtype=np.float64)
    gt_force_sum = np.zeros((total_frames, force_dim), dtype=np.float64)
    force_count = np.zeros(total_frames, dtype=np.int64)

    for chunk in chunks:
        start = int(chunk["fix_sidx"])
        valid_len = int(chunk["valid_len"])
        end = min(total_frames, start + valid_len)
        if end <= start:
            continue
        length = end - start
        step_prob = np.asarray(chunk["step_prob"], dtype=np.float64)[:length]
        prob_sum[start:end] += step_prob
        prob_count[start:end] += 1

        pred_force = np.asarray(chunk.get("pred_force_denorm", []), dtype=np.float64).reshape(-1, force_dim)[:length]
        gt_force = np.asarray(chunk.get("gt_force_denorm", []), dtype=np.float64).reshape(-1, force_dim)[:length]
        if pred_force.shape[0] == length and gt_force.shape[0] == length:
            pred_force_sum[start:end] += pred_force
            gt_force_sum[start:end] += gt_force
            force_count[start:end] += 1

    frame_prob = np.full(total_frames, np.nan, dtype=np.float64)
    valid_prob = prob_count > 0
    frame_prob[valid_prob] = prob_sum[valid_prob] / prob_count[valid_prob]

    pred_force = np.full((total_frames, force_dim), np.nan, dtype=np.float64)
    gt_force = np.full((total_frames, force_dim), np.nan, dtype=np.float64)
    valid_force = force_count > 0
    pred_force[valid_force] = pred_force_sum[valid_force] / force_count[valid_force, None]
    gt_force[valid_force] = gt_force_sum[valid_force] / force_count[valid_force, None]
    return {
        "aggregated_prob": frame_prob,
        "prediction_count": prob_count,
        "pred_force_denorm": pred_force,
        "gt_force_denorm": gt_force,
        "force_prediction_count": force_count,
    }


def micro_window_size_from_cfg(cfg):
    return int(cfg.get("anomaly_head", {}).get("micro_window_size", 10))


def build_episode_record(cfg, dataset, dataset_index, chunks, force_dim, window_stride, threshold):
    item = dataset.dataset[int(dataset_index)]
    parquet_path = str(item[2])
    domain = str(item[3])
    total_frames = int(item[7])
    episode_id = extract_episode_id(parquet_path)
    chunks = sorted(chunks, key=lambda chunk: int(chunk["fix_sidx"]))
    if not chunks:
        raise RuntimeError(f"No chunks for dataset_index={dataset_index}")

    _, frame_anomaly = load_frame_labels(parquet_path, total_frames)
    frame_anomaly = frame_anomaly.astype(np.int64)
    interaction_labels = None
    try:
        from analyze.action_force_anomaly_eval_core import load_interaction_labels

        interaction_labels = load_interaction_labels(parquet_path, total_frames).astype(np.int64)
    except Exception:
        interaction_labels = np.zeros(total_frames, dtype=np.int64)

    interaction_start = safe_first_true(interaction_labels > 0)
    transition_frame = load_transition_frame(parquet_path, frame_anomaly)
    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    micro_window_size = micro_window_size_from_cfg(cfg)

    num_windows = len(chunks)
    window_start = np.asarray([int(chunk["fix_sidx"]) for chunk in chunks], dtype=np.int64)
    window_valid_len = np.asarray([int(chunk["valid_len"]) for chunk in chunks], dtype=np.int64)
    window_prob = np.asarray([float(chunk["window_prob"]) for chunk in chunks], dtype=np.float64)
    risk_prob = np.asarray([float(chunk["risk_prob"]) for chunk in chunks], dtype=np.float64)
    window_target = np.zeros(num_windows, dtype=np.int64)
    risk_target = np.zeros(num_windows, dtype=np.float64)
    window_interaction = np.zeros(num_windows, dtype=np.int64)
    mem_idx = []

    step_prob = np.full((num_windows, action_chunk), np.nan, dtype=np.float64)
    step_target = np.full((num_windows, action_chunk), np.nan, dtype=np.float64)
    step_interaction = np.full((num_windows, action_chunk), np.nan, dtype=np.float64)
    pred_force = np.full((num_windows, action_chunk, force_dim), np.nan, dtype=np.float64)
    gt_force = np.full((num_windows, action_chunk, force_dim), np.nan, dtype=np.float64)
    micro_start = []
    micro_valid_len = []
    micro_prob = []
    micro_target = []
    micro_interaction = []
    micro_parent_window = []

    for row, chunk in enumerate(chunks):
        start = int(chunk["fix_sidx"])
        valid_len = int(chunk["valid_len"])
        end = min(total_frames, start + valid_len)
        length = max(0, end - start)
        if length <= 0:
            continue
        frame_slice = frame_anomaly[start:end]
        interaction_slice = interaction_labels[start:end]
        chunk_frame_target = chunk.get("frame_target")
        if chunk_frame_target is not None:
            frame_slice = np.asarray(chunk_frame_target, dtype=np.float64)[:length].astype(np.int64)

        step_prob[row, :length] = np.asarray(chunk["step_prob"], dtype=np.float64)[:length]
        step_target[row, :length] = frame_slice
        step_interaction[row, :length] = interaction_slice

        window_target[row] = int(chunk.get("window_target") if chunk.get("window_target") is not None else frame_slice.max())
        risk_target[row] = float(chunk.get("risk_target") if chunk.get("risk_target") is not None else frame_slice.mean())
        window_interaction[row] = int(chunk.get("interaction_label", int((interaction_slice > 0).any())))
        mem_idx.append(chunk.get("mem_idx", []))

        chunk_pred_force = np.asarray(chunk.get("pred_force_denorm", []), dtype=np.float64).reshape(-1, force_dim)[:length]
        chunk_gt_force = np.asarray(chunk.get("gt_force_denorm", []), dtype=np.float64).reshape(-1, force_dim)[:length]
        if chunk_pred_force.shape[0] == length:
            pred_force[row, :length] = chunk_pred_force
        if chunk_gt_force.shape[0] == length:
            gt_force[row, :length] = chunk_gt_force

        chunk_micro_prob = chunk.get("micro_prob")
        if chunk_micro_prob is not None:
            chunk_micro_prob = np.asarray(chunk_micro_prob, dtype=np.float64).reshape(-1)
            for micro_idx, local_start in enumerate(range(0, length, micro_window_size)):
                local_end = min(length, local_start + micro_window_size)
                if local_end <= local_start or micro_idx >= chunk_micro_prob.shape[0]:
                    continue
                block_frame = frame_slice[local_start:local_end]
                block_interaction = interaction_slice[local_start:local_end]
                micro_start.append(int(start + local_start))
                micro_valid_len.append(int(local_end - local_start))
                micro_prob.append(float(chunk_micro_prob[micro_idx]))
                micro_target.append(int(np.max(block_frame)) if block_frame.size else 0)
                micro_interaction.append(int((block_interaction > 0).any()) if block_interaction.size else 0)
                micro_parent_window.append(int(row))

    frame_aggregate = aggregate_frames(chunks, total_frames=total_frames, force_dim=force_dim)
    deployment = compute_deployment_record(
        window_start=window_start,
        window_prob=window_prob,
        threshold=float(threshold),
        horizon=action_chunk,
        transition_frame=transition_frame,
        has_anomaly=bool(frame_anomaly.any()),
        total_frames=total_frames,
    )

    return {
        "dataset_index": int(dataset_index),
        "episode_id": episode_id,
        "domain": domain,
        "parquet_path": parquet_path,
        "prompt": str(chunks[0].get("prompt", "")),
        "total_frames": int(total_frames),
        "action_chunk": int(action_chunk),
        "window_stride": int(window_stride),
        "has_anomaly": bool(frame_anomaly.any()),
        "interaction_start_frame": interaction_start,
        "failure_transition_frame": transition_frame,
        "windows": {
            "start": window_start,
            "valid_len": window_valid_len,
            "prob": window_prob,
            "target": window_target,
            "risk_prob": risk_prob,
            "risk_target": risk_target,
            "interaction_target": window_interaction,
            "mem_idx": mem_idx,
            "step_prob": step_prob,
            "step_target": step_target,
            "step_interaction_target": step_interaction,
            "pred_force_denorm": pred_force,
            "gt_force_denorm": gt_force,
        },
        "micro_windows": {
            "start": np.asarray(micro_start, dtype=np.int64),
            "valid_len": np.asarray(micro_valid_len, dtype=np.int64),
            "prob": np.asarray(micro_prob, dtype=np.float64),
            "target": np.asarray(micro_target, dtype=np.int64),
            "interaction_target": np.asarray(micro_interaction, dtype=np.int64),
            "parent_window_index": np.asarray(micro_parent_window, dtype=np.int64),
            "micro_window_size": int(micro_window_size),
        },
        "frames": {
            "index": np.arange(total_frames, dtype=np.int64),
            "anomaly_label": frame_anomaly,
            "interaction_label": interaction_labels,
            "aggregated_prob": frame_aggregate["aggregated_prob"],
            "prediction_count": frame_aggregate["prediction_count"],
            "pred_force_denorm": frame_aggregate["pred_force_denorm"],
            "gt_force_denorm": frame_aggregate["gt_force_denorm"],
            "force_prediction_count": frame_aggregate["force_prediction_count"],
        },
        "deployment": deployment,
    }


def compute_deployment_record(window_start, window_prob, threshold, horizon, transition_frame, has_anomaly, total_frames):
    alarm_mask = np.asarray(window_prob, dtype=np.float64) >= float(threshold)
    first_alarm_idx = safe_first_true(alarm_mask)
    first_alarm_frame = None if first_alarm_idx is None else int(window_start[first_alarm_idx])
    window_false_alarm_rate = float(alarm_mask.mean()) if alarm_mask.size else 0.0

    record = {
        "threshold": float(threshold),
        "horizon": int(horizon),
        "has_anomaly": bool(has_anomaly),
        "first_alarm_frame": first_alarm_frame,
        "first_alarm_window_index": first_alarm_idx,
        "window_alarm_ratio": window_false_alarm_rate,
        "status": "normal_no_alarm",
        "valid_warning_start": None,
        "early_warning_frames": None,
        "detection_delay_frames": None,
        "too_early_frames": None,
    }
    if not has_anomaly or transition_frame is None:
        if first_alarm_frame is not None:
            record["status"] = "normal_false_alarm"
        return record

    onset = int(transition_frame)
    valid_warning_start = max(0, onset - int(horizon))
    record["valid_warning_start"] = int(valid_warning_start)
    if first_alarm_frame is None:
        record["status"] = "missed"
    elif first_alarm_frame < valid_warning_start:
        record["status"] = "too_early_false_alarm"
        record["too_early_frames"] = int(valid_warning_start - first_alarm_frame)
    elif first_alarm_frame < onset:
        record["status"] = "valid_early_warning"
        record["early_warning_frames"] = int(onset - first_alarm_frame)
    else:
        record["status"] = "delayed_detection"
        record["detection_delay_frames"] = int(first_alarm_frame - onset)
    record["onset"] = onset
    record["total_frames"] = int(total_frames)
    return record


def collect_stage_arrays(records, domain, stage):
    window_scores = []
    window_targets = []
    risk_scores = []
    risk_targets = []
    micro_scores = []
    micro_targets = []
    frame_scores = []
    frame_targets = []

    for record in records:
        if domain != "all" and record["domain"] != domain:
            continue
        interaction_start = record["interaction_start_frame"]
        window_starts = record["windows"]["start"]
        frame_indices = record["frames"]["index"]
        window_mask = stage_masks(window_starts, interaction_start)[stage]
        micro_starts = record.get("micro_windows", {}).get("start", np.asarray([], dtype=np.int64))
        micro_mask = stage_masks(micro_starts, interaction_start)[stage] if np.asarray(micro_starts).size else np.asarray([], dtype=bool)
        frame_mask = stage_masks(frame_indices, interaction_start)[stage]

        window_scores.append(record["windows"]["prob"][window_mask])
        window_targets.append(record["windows"]["target"][window_mask])
        risk_scores.append(record["windows"]["risk_prob"][window_mask])
        risk_targets.append(record["windows"]["risk_target"][window_mask])
        if np.asarray(micro_starts).size:
            micro_scores.append(record["micro_windows"]["prob"][micro_mask])
            micro_targets.append(record["micro_windows"]["target"][micro_mask])

        frame_prob = record["frames"]["aggregated_prob"]
        frame_label = record["frames"]["anomaly_label"]
        valid_frame = frame_mask & np.isfinite(frame_prob)
        frame_scores.append(frame_prob[valid_frame])
        frame_targets.append(frame_label[valid_frame])

    def cat(values, dtype=np.float64):
        values = [np.asarray(v, dtype=dtype) for v in values if np.asarray(v).size]
        if not values:
            return np.asarray([], dtype=dtype)
        return np.concatenate(values, axis=0)

    return {
        "window_scores": cat(window_scores, np.float64),
        "window_targets": cat(window_targets, np.int64),
        "risk_scores": cat(risk_scores, np.float64),
        "risk_targets": cat(risk_targets, np.float64),
        "micro_scores": cat(micro_scores, np.float64),
        "micro_targets": cat(micro_targets, np.int64),
        "frame_scores": cat(frame_scores, np.float64),
        "frame_targets": cat(frame_targets, np.int64),
    }


def compute_stage_metrics(records, threshold):
    domains = ["all"] + sorted({record["domain"] for record in records})
    table = []
    nested = {}
    for domain in domains:
        nested[domain] = {}
        for stage in STAGES:
            arrays = collect_stage_arrays(records, domain, stage)
            row = {
                "domain": domain,
                "stage": stage,
                "window_manual": binary_metrics(arrays["window_scores"], arrays["window_targets"], threshold),
                "window_best_f1": best_f1_metrics(arrays["window_scores"], arrays["window_targets"]),
                "window_recall80": recall_constrained_metrics(
                    arrays["window_scores"], arrays["window_targets"], min_recall=0.8
                ),
                "risk": risk_metrics(arrays["risk_scores"], arrays["risk_targets"]),
                "micro_manual": binary_metrics(arrays["micro_scores"], arrays["micro_targets"], threshold),
                "micro_best_f1": best_f1_metrics(arrays["micro_scores"], arrays["micro_targets"]),
                "micro_recall80": recall_constrained_metrics(
                    arrays["micro_scores"], arrays["micro_targets"], min_recall=0.8
                ),
                "frame_manual": binary_metrics(arrays["frame_scores"], arrays["frame_targets"], threshold),
                "frame_best_f1": best_f1_metrics(arrays["frame_scores"], arrays["frame_targets"]),
                "frame_recall80": recall_constrained_metrics(
                    arrays["frame_scores"], arrays["frame_targets"], min_recall=0.8
                ),
            }
            nested[domain][stage] = row
            table.append(row)
    return nested, table


def compute_deployment_metrics(records, threshold):
    domains = ["all"] + sorted({record["domain"] for record in records})
    out = {}
    for domain in domains:
        selected = [record for record in records if domain == "all" or record["domain"] == domain]
        statuses = {}
        early = []
        delay = []
        normal_false_alarm = []
        for record in selected:
            dep = record["deployment"]
            status = dep["status"]
            statuses[status] = statuses.get(status, 0) + 1
            if dep.get("early_warning_frames") is not None:
                early.append(float(dep["early_warning_frames"]))
            if dep.get("detection_delay_frames") is not None:
                delay.append(float(dep["detection_delay_frames"]))
            if not dep.get("has_anomaly", False):
                normal_false_alarm.append(1.0 if status == "normal_false_alarm" else 0.0)
        fail_eps = [record for record in selected if record["deployment"].get("has_anomaly", False)]
        normal_eps = [record for record in selected if not record["deployment"].get("has_anomaly", False)]
        out[domain] = {
            "threshold": float(threshold),
            "num_episodes": len(selected),
            "num_failure_episodes": len(fail_eps),
            "num_normal_episodes": len(normal_eps),
            "status_counts": statuses,
            "valid_early_warning_rate": statuses.get("valid_early_warning", 0) / max(1, len(fail_eps)),
            "miss_rate": statuses.get("missed", 0) / max(1, len(fail_eps)),
            "too_early_false_alarm_rate_on_failure": statuses.get("too_early_false_alarm", 0) / max(1, len(fail_eps)),
            "normal_false_alarm_rate": float(np.mean(normal_false_alarm)) if normal_false_alarm else None,
            "early_warning_frames_mean": float(np.mean(early)) if early else None,
            "early_warning_frames_median": float(np.median(early)) if early else None,
            "detection_delay_frames_mean": float(np.mean(delay)) if delay else None,
            "detection_delay_frames_median": float(np.median(delay)) if delay else None,
        }
    return out


def metric_value(metrics, key):
    if metrics is None:
        return None
    return metrics.get(key)


def flatten_stage_row(row):
    wm = row["window_manual"]
    wb = row["window_best_f1"]
    wr80 = row["window_recall80"]
    risk = row["risk"]
    mm = row["micro_manual"]
    mb = row["micro_best_f1"]
    mr80 = row["micro_recall80"]
    fm = row["frame_manual"]
    fb = row["frame_best_f1"]
    fr80 = row["frame_recall80"]
    return {
        "domain": row["domain"],
        "stage": row["stage"],
        "window_num": metric_value(wm, "num_samples"),
        "window_num_pos": metric_value(wm, "num_positive"),
        "window_num_neg": metric_value(wm, "num_negative"),
        "window_pos_ratio": metric_value(wm, "positive_ratio"),
        "window_auroc": metric_value(wm, "auroc"),
        "window_auprc": metric_value(wm, "auprc"),
        "window_tp": metric_value(wm, "tp"),
        "window_fp": metric_value(wm, "fp"),
        "window_tn": metric_value(wm, "tn"),
        "window_fn": metric_value(wm, "fn"),
        "window_precision": metric_value(wm, "precision"),
        "window_recall": metric_value(wm, "recall"),
        "window_tpr": metric_value(wm, "tpr"),
        "window_fpr": metric_value(wm, "fpr"),
        "window_tnr": metric_value(wm, "tnr"),
        "window_balanced_accuracy": metric_value(wm, "balanced_accuracy"),
        "window_f1": metric_value(wm, "f1"),
        "window_best_f1": metric_value(wb, "f1"),
        "window_best_threshold": metric_value(wb, "threshold"),
        "window_recall80_threshold": metric_value(wr80, "threshold"),
        "window_recall80_recall": metric_value(wr80, "recall"),
        "window_recall80_fpr": metric_value(wr80, "fpr"),
        "window_recall80_precision": metric_value(wr80, "precision"),
        "window_recall80_bacc": metric_value(wr80, "balanced_accuracy"),
        "window_recall80_f1": metric_value(wr80, "f1"),
        "risk_num": metric_value(risk, "num_samples"),
        "risk_mae": metric_value(risk, "mae"),
        "risk_mse": metric_value(risk, "mse"),
        "risk_spearman": metric_value(risk, "spearman"),
        "micro_num": metric_value(mm, "num_samples"),
        "micro_num_pos": metric_value(mm, "num_positive"),
        "micro_num_neg": metric_value(mm, "num_negative"),
        "micro_pos_ratio": metric_value(mm, "positive_ratio"),
        "micro_auroc": metric_value(mm, "auroc"),
        "micro_auprc": metric_value(mm, "auprc"),
        "micro_tp": metric_value(mm, "tp"),
        "micro_fp": metric_value(mm, "fp"),
        "micro_tn": metric_value(mm, "tn"),
        "micro_fn": metric_value(mm, "fn"),
        "micro_precision": metric_value(mm, "precision"),
        "micro_recall": metric_value(mm, "recall"),
        "micro_tpr": metric_value(mm, "tpr"),
        "micro_fpr": metric_value(mm, "fpr"),
        "micro_tnr": metric_value(mm, "tnr"),
        "micro_balanced_accuracy": metric_value(mm, "balanced_accuracy"),
        "micro_f1": metric_value(mm, "f1"),
        "micro_best_f1": metric_value(mb, "f1"),
        "micro_best_threshold": metric_value(mb, "threshold"),
        "micro_recall80_threshold": metric_value(mr80, "threshold"),
        "micro_recall80_recall": metric_value(mr80, "recall"),
        "micro_recall80_fpr": metric_value(mr80, "fpr"),
        "micro_recall80_precision": metric_value(mr80, "precision"),
        "micro_recall80_bacc": metric_value(mr80, "balanced_accuracy"),
        "micro_recall80_f1": metric_value(mr80, "f1"),
        "frame_num": metric_value(fm, "num_samples"),
        "frame_num_pos": metric_value(fm, "num_positive"),
        "frame_num_neg": metric_value(fm, "num_negative"),
        "frame_pos_ratio": metric_value(fm, "positive_ratio"),
        "frame_auroc": metric_value(fm, "auroc"),
        "frame_auprc": metric_value(fm, "auprc"),
        "frame_tp": metric_value(fm, "tp"),
        "frame_fp": metric_value(fm, "fp"),
        "frame_tn": metric_value(fm, "tn"),
        "frame_fn": metric_value(fm, "fn"),
        "frame_precision": metric_value(fm, "precision"),
        "frame_recall": metric_value(fm, "recall"),
        "frame_tpr": metric_value(fm, "tpr"),
        "frame_fpr": metric_value(fm, "fpr"),
        "frame_tnr": metric_value(fm, "tnr"),
        "frame_balanced_accuracy": metric_value(fm, "balanced_accuracy"),
        "frame_f1": metric_value(fm, "f1"),
        "frame_best_f1": metric_value(fb, "f1"),
        "frame_best_threshold": metric_value(fb, "threshold"),
        "frame_recall80_threshold": metric_value(fr80, "threshold"),
        "frame_recall80_recall": metric_value(fr80, "recall"),
        "frame_recall80_fpr": metric_value(fr80, "fpr"),
        "frame_recall80_precision": metric_value(fr80, "precision"),
        "frame_recall80_bacc": metric_value(fr80, "balanced_accuracy"),
        "frame_recall80_f1": metric_value(fr80, "f1"),
    }


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


def build_episode_summary_rows(records, include_deployment=False):
    rows = []
    for record in records:
        window_prob = record["windows"]["prob"]
        micro_prob = record.get("micro_windows", {}).get("prob", np.asarray([], dtype=np.float64))
        post_mask = stage_masks(record["windows"]["start"], record["interaction_start_frame"])["post"]
        micro_post_mask = (
            stage_masks(record["micro_windows"]["start"], record["interaction_start_frame"])["post"]
            if np.asarray(micro_prob).size
            else np.asarray([], dtype=bool)
        )
        dep = record["deployment"]
        row = {
            "domain": record["domain"],
            "episode_id": record["episode_id"],
            "dataset_index": record["dataset_index"],
            "total_frames": record["total_frames"],
            "num_windows": int(window_prob.size),
            "has_anomaly": record["has_anomaly"],
            "interaction_start_frame": record["interaction_start_frame"],
            "failure_transition_frame": record["failure_transition_frame"],
            "max_window_prob": float(np.nanmax(window_prob)) if window_prob.size else None,
            "mean_window_prob": float(np.nanmean(window_prob)) if window_prob.size else None,
            "post_max_window_prob": float(np.nanmax(window_prob[post_mask])) if post_mask.any() else None,
            "window_positive_ratio": float(np.mean(record["windows"]["target"])) if window_prob.size else None,
            "num_micro_windows": int(micro_prob.size),
            "max_micro_prob": float(np.nanmax(micro_prob)) if micro_prob.size else None,
            "mean_micro_prob": float(np.nanmean(micro_prob)) if micro_prob.size else None,
            "post_max_micro_prob": float(np.nanmax(micro_prob[micro_post_mask])) if micro_post_mask.any() else None,
            "micro_positive_ratio": (
                float(np.mean(record["micro_windows"]["target"])) if micro_prob.size else None
            ),
            "risk_target_mean": float(np.nanmean(record["windows"]["risk_target"])) if window_prob.size else None,
            "risk_prob_mean": float(np.nanmean(record["windows"]["risk_prob"])) if window_prob.size else None,
            "first_alarm_frame": dep["first_alarm_frame"],
            "window_alarm_ratio": dep["window_alarm_ratio"],
        }
        if include_deployment:
            row.update(
                {
                    "deployment_status": dep["status"],
                    "early_warning_frames": dep["early_warning_frames"],
                    "detection_delay_frames": dep["detection_delay_frames"],
                    "too_early_frames": dep["too_early_frames"],
                }
            )
        rows.append(row)
    return rows


def build_window_rows(records):
    rows = []
    for record in records:
        starts = record["windows"]["start"]
        stage_by_start = np.full(starts.shape[0], "pre", dtype=object)
        if record["interaction_start_frame"] is not None:
            stage_by_start[starts >= int(record["interaction_start_frame"])] = "post"
        for i, start in enumerate(starts):
            rows.append(
                {
                    "domain": record["domain"],
                    "episode_id": record["episode_id"],
                    "dataset_index": record["dataset_index"],
                    "window_index": i,
                    "window_start": int(start),
                    "valid_len": int(record["windows"]["valid_len"][i]),
                    "stage": str(stage_by_start[i]),
                    "window_prob": float(record["windows"]["prob"][i]),
                    "window_target": int(record["windows"]["target"][i]),
                    "risk_prob": float(record["windows"]["risk_prob"][i]),
                    "risk_target": float(record["windows"]["risk_target"][i]),
                    "interaction_target": int(record["windows"]["interaction_target"][i]),
                }
            )
    return rows


def build_micro_rows(records):
    rows = []
    for record in records:
        micro = record.get("micro_windows", {})
        starts = np.asarray(micro.get("start", []), dtype=np.int64)
        if starts.size == 0:
            continue
        stage_by_start = np.full(starts.shape[0], "pre", dtype=object)
        if record["interaction_start_frame"] is not None:
            stage_by_start[starts >= int(record["interaction_start_frame"])] = "post"
        for i, start in enumerate(starts):
            parent = int(micro["parent_window_index"][i])
            rows.append(
                {
                    "domain": record["domain"],
                    "episode_id": record["episode_id"],
                    "dataset_index": record["dataset_index"],
                    "micro_index": i,
                    "window_index": parent,
                    "window_start": int(record["windows"]["start"][parent]) if parent < len(record["windows"]["start"]) else None,
                    "micro_start": int(start),
                    "valid_len": int(micro["valid_len"][i]),
                    "stage": str(stage_by_start[i]),
                    "micro_prob": float(micro["prob"][i]),
                    "micro_target": int(micro["target"][i]),
                    "interaction_target": int(micro["interaction_target"][i]),
                }
            )
    return rows


def records_have_micro(records):
    for record in records:
        prob = record.get("micro_windows", {}).get("prob", [])
        if np.asarray(prob).size > 0:
            return True
    return False


def build_early_warning_rows(records):
    rows = []
    for record in records:
        dep = record["deployment"]
        row = {
            "domain": record["domain"],
            "episode_id": record["episode_id"],
            "dataset_index": record["dataset_index"],
            "has_anomaly": dep["has_anomaly"],
            "failure_transition_frame": record["failure_transition_frame"],
            "valid_warning_start": dep["valid_warning_start"],
            "first_alarm_frame": dep["first_alarm_frame"],
            "status": dep["status"],
            "early_warning_frames": dep["early_warning_frames"],
            "detection_delay_frames": dep["detection_delay_frames"],
            "too_early_frames": dep["too_early_frames"],
            "window_alarm_ratio": dep["window_alarm_ratio"],
        }
        rows.append(row)
    return rows


def choose_plot_records(records, mode, top_k, threshold=0.5):
    if mode == "none":
        return []
    if mode == "all":
        return records
    if mode == "normal_correct":
        selected = []
        for record in records:
            window_targets = np.asarray(record["windows"]["target"], dtype=np.int64)
            window_probs = np.asarray(record["windows"]["prob"], dtype=np.float64)
            frame_labels = np.asarray(record["frames"]["anomaly_label"], dtype=np.int64)
            has_gt_anomaly = bool(window_targets.size and np.any(window_targets == 1)) or bool(
                frame_labels.size and np.any(frame_labels == 1)
            )
            has_pred_alarm = bool(window_probs.size and np.any(window_probs >= float(threshold)))
            if not has_gt_anomaly and not has_pred_alarm:
                selected.append(record)

        def normal_rank_key(record):
            probs = np.asarray(record["windows"]["prob"], dtype=np.float64)
            if probs.size == 0:
                return (0.0, 0.0, int(record["dataset_index"]))
            return (
                float(np.nanmax(probs)),
                float(np.nanmean(probs)),
                int(record["dataset_index"]),
            )

        return sorted(selected, key=normal_rank_key)[: max(0, int(top_k))]

    elif mode == "top_errors":
        def rank_key(record):
            metrics = binary_metrics(record["windows"]["prob"], record["windows"]["target"], threshold)
            if metrics is None:
                return (1.0, 1.0, 0, 0, 0.0)
            return (
                float(metrics["f1"]),
                float(metrics["recall"]),
                -int(metrics["fn"]),
                -int(metrics["fp"]),
                -float(np.nanmax(record["windows"]["prob"])) if record["windows"]["prob"].size else 0.0,
            )

        ranked = sorted(
            records,
            key=rank_key,
        )
    return ranked[: max(0, int(top_k))]


def save_episode_plot(record, output_path, threshold, include_deployment=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    xw = record["windows"]["start"]
    xf = record["frames"]["index"]
    dep = record["deployment"]

    axes[0].plot(xw, record["windows"]["prob"], color="crimson", linewidth=2.8, marker=".", markersize=5, label="window prob")
    axes[0].plot(xw, record["windows"]["risk_prob"], color="darkorange", linewidth=2.2, marker=".", markersize=4, label="risk prob")
    axes[0].plot(xw, record["windows"]["target"], color="steelblue", linewidth=2.0, drawstyle="steps-post", label="window target")
    axes[0].axhline(float(threshold), color="black", linestyle=":", linewidth=1.5, label="threshold")
    axes[0].set_ylabel("window")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.4)

    axes[1].plot(xf, record["frames"]["aggregated_prob"], color="crimson", linewidth=2.4, label="aggregated frame prob")
    axes[1].plot(xf, record["frames"]["anomaly_label"], color="steelblue", linewidth=2.0, drawstyle="steps-post", label="frame anomaly")
    axes[1].set_ylabel("frame")
    axes[1].set_xlabel("frame index")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.4)

    transition = record["failure_transition_frame"]
    if transition is not None:
        for ax in axes:
            ax.axvline(int(transition), color="purple", linestyle="--", linewidth=1.8, label="failure onset")
        valid_start = dep.get("valid_warning_start")
        if include_deployment and valid_start is not None:
            for ax in axes:
                ax.axvspan(
                    int(valid_start),
                    int(transition),
                    color="gold",
                    alpha=0.22,
                    label="valid warning band",
                )
                ax.axvline(int(valid_start), color="goldenrod", linestyle=":", linewidth=1.4, label="warning band start")
    interaction_start = record["interaction_start_frame"]
    if interaction_start is not None:
        for ax in axes:
            ax.axvline(int(interaction_start), color="seagreen", linestyle="--", linewidth=1.6, alpha=0.85, label="interaction start")
    if dep["first_alarm_frame"] is not None:
        for ax in axes:
            ax.axvline(int(dep["first_alarm_frame"]), color="black", linestyle="-.", linewidth=1.6, label="first alarm")

    window_metrics = binary_metrics(record["windows"]["prob"], record["windows"]["target"], threshold)
    frame_prob = record["frames"]["aggregated_prob"]
    frame_label = record["frames"]["anomaly_label"]
    frame_valid = np.isfinite(frame_prob)
    frame_metrics = binary_metrics(frame_prob[frame_valid], frame_label[frame_valid], threshold)

    def format_metrics(prefix, metrics):
        if metrics is None:
            return f"{prefix}: no valid samples"
        auroc = metrics.get("auroc")
        auprc = metrics.get("auprc")
        return (
            f"{prefix} metrics\n"
            f"AUROC: {_finite_or_nan(auroc):.3f}\n"
            f"AUPRC: {_finite_or_nan(auprc):.3f}\n"
            f"P/R/F1: {metrics['precision']:.3f}/{metrics['recall']:.3f}/{metrics['f1']:.3f}\n"
            f"pos: {metrics['positive_ratio']:.3f}"
        )

    axes[0].text(
        0.995,
        0.97,
        format_metrics("window", window_metrics),
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "boxstyle": "round,pad=0.35"},
    )
    axes[1].text(
        0.995,
        0.97,
        format_metrics("frame", frame_metrics),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.85, "boxstyle": "round,pad=0.35"},
    )

    title = f"{record['domain']} episode={record['episode_id']} idx={record['dataset_index']}"
    if dep["first_alarm_frame"] is not None:
        title += f" first_alarm={dep['first_alarm_frame']}"
    if include_deployment:
        title += f" status={dep['status']}"
    fig.suptitle(title)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        dedup = {}
        for handle, label in zip(handles, labels):
            dedup.setdefault(label, handle)
        ax.legend(
            dedup.values(),
            dedup.keys(),
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=True,
            fontsize=9,
        )
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _finite_or_nan(value):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def save_domain_stage_summary_plot(stage_rows, output_path):
    if value is None:
        return np.nan
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def save_domain_stage_summary_plot(stage_rows, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    flat_rows = [flatten_stage_row(row) for row in stage_rows]
    domains = [row["domain"] for row in flat_rows if row["domain"] != "all"]
    domains = sorted(set(domains))
    if not domains:
        domains = ["all"]

    metrics = [
        ("window_f1", "Window F1"),
        ("micro_f1", "Micro F1"),
        ("frame_f1", "Frame F1"),
        ("risk_mae", "Risk MAE"),
    ]
    colors = {"all": "0.35", "pre": "cornflowerblue", "post": "tomato"}
    x = np.arange(len(domains), dtype=np.float64)
    width = 0.24

    fig, axes = plt.subplots(2, 2, figsize=(max(12, len(domains) * 0.75), 8.5), constrained_layout=True)
    axes = axes.reshape(-1)
    row_map = {(row["domain"], row["stage"]): row for row in flat_rows}
    for ax, (metric_key, title) in zip(axes, metrics):
        for offset, stage in zip((-width, 0.0, width), STAGES):
            values = [
                _finite_or_nan(row_map.get((domain, stage), {}).get(metric_key))
                for domain in domains
            ]
            bars = ax.bar(x + offset, values, width=width, label=stage, color=colors[stage], alpha=0.85)
            for bar, value in zip(bars, values):
                if not np.isfinite(value):
                    continue
                label = f"{value:.2f}" if metric_key != "risk_mae" else f"{value:.3f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    label,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                )
        ax.set_title(title)
        ax.set_xticks(x, domains, rotation=35, ha="right")
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
        if metric_key != "risk_mae":
            ax.set_ylim(0.0, 1.05)
        ax.legend(loc="upper left", ncol=3)
    fig.suptitle("Domain x Stage Summary")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_deployment_summary_plot(deployment_metrics, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    domains = [domain for domain in sorted(deployment_metrics) if domain != "all"]
    if not domains:
        domains = ["all"]
    statuses = [
        "valid_early_warning",
        "delayed_detection",
        "missed",
        "too_early_false_alarm",
        "normal_false_alarm",
        "normal_no_alarm",
    ]
    colors = {
        "valid_early_warning": "seagreen",
        "delayed_detection": "darkorange",
        "missed": "crimson",
        "too_early_false_alarm": "goldenrod",
        "normal_false_alarm": "purple",
        "normal_no_alarm": "0.65",
    }
    x = np.arange(len(domains), dtype=np.float64)
    bottom = np.zeros(len(domains), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(max(10, len(domains) * 0.75), 5.2), constrained_layout=True)
    for status in statuses:
        counts = np.asarray(
            [deployment_metrics.get(domain, {}).get("status_counts", {}).get(status, 0) for domain in domains],
            dtype=np.float64,
        )
        bars = ax.bar(x, counts, bottom=bottom, label=status, color=colors[status], alpha=0.9)
        for bar, count, base in zip(bars, counts, bottom):
            if count <= 0:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                base + count / 2.0,
                str(int(count)),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if count >= 2 else "black",
            )
        bottom += counts
    ax.set_title("Episode Deployment Status")
    ax.set_ylabel("num episodes")
    ax.set_xticks(x, domains, rotation=35, ha="right")
    ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", ncol=2)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_summary_plot(records, output_path, threshold):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), constrained_layout=True)
    use_micro = records_have_micro(records)
    for col, stage in enumerate(STAGES):
        arrays = collect_stage_arrays(records, "all", stage)
        if use_micro:
            panels = [
                ("micro-window", arrays["micro_scores"], arrays["micro_targets"]),
                ("frame", arrays["frame_scores"], arrays["frame_targets"]),
            ]
        else:
            panels = [
                ("window", arrays["window_scores"], arrays["window_targets"]),
                ("frame", arrays["frame_scores"], arrays["frame_targets"]),
            ]
        for row, (name, scores, targets) in enumerate(panels):
            ax = axes[row, col]
            metrics = binary_metrics(scores, targets, threshold)
            ax.set_title(f"{name} - {stage}")
            ax.set_xlabel("pred")
            ax.set_ylabel("gt")
            ax.set_xticks([0, 1], labels=["normal", "anom"])
            ax.set_yticks([0, 1], labels=["normal", "anom"])
            if metrics is None:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_xlim(-0.5, 1.5)
                ax.set_ylim(1.5, -0.5)
                continue
            matrix = np.asarray(
                [
                    [metrics["tn"], metrics["fp"]],
                    [metrics["fn"], metrics["tp"]],
                ],
                dtype=np.int64,
            )
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
                -0.24,
                (
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}\n"
                    f"AUROC={_finite_or_nan(metrics['auroc']):.3f} AUPRC={_finite_or_nan(metrics['auprc']):.3f}"
                ),
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )
            fig.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    fig.suptitle(f"Overall Confusion Matrices at threshold={float(threshold):.3f}")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_score_histogram_plot(records, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    bins = np.linspace(0.0, 1.0, 31)
    use_micro = records_have_micro(records)
    prefixes = ("micro", "frame") if use_micro else ("window", "frame")
    for col, stage in enumerate(STAGES):
        arrays = collect_stage_arrays(records, "all", stage)
        for row, prefix in enumerate(prefixes):
            ax = axes[row, col]
            scores = arrays[f"{prefix}_scores"]
            targets = arrays[f"{prefix}_targets"]
            valid = np.isfinite(scores) & np.isfinite(targets)
            scores = scores[valid]
            targets = targets[valid]
            normal_scores = scores[targets == 0]
            anomaly_scores = scores[targets == 1]
            ax.hist(normal_scores, bins=bins, alpha=0.65, color="steelblue", label="normal")
            ax.hist(anomaly_scores, bins=bins, alpha=0.65, color="crimson", label="anomaly")
            ax.axvline(0.5, color="black", linestyle=":", linewidth=1.4, label="0.5")
            title = "micro-window" if prefix == "micro" else prefix
            ax.set_title(f"{title} scores - {stage}")
            ax.set_xlim(0.0, 1.0)
            ax.set_ylabel("count")
            ax.grid(True, axis="y", linestyle=":", alpha=0.4)
            ax.legend(loc="upper right")
    fig.suptitle("Score Distribution by Stage")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_cache_eval(cfg, args, device, force_dim, shard_rank, num_shards):
    cache_dir = None if args.disable_cache else (args.anomaly_feature_cache_dir or cfg.get("anomaly_feature_cache_dir"))
    cache_split = args.cache_split or args.eval_split
    cache_dataset = AnomalyFeatureCacheDataset(
        cache_dir=cache_dir,
        split=cache_split,
        max_open_shards=int(cfg.get("anomaly_cache_max_open_shards", 4)),
    )
    reference_dataset = build_cache_reference_dataset(cfg, cache_split, split_manifest=args.split_manifest)
    episode_groups = build_cache_episode_groups(cache_dataset, reference_dataset)
    episode_indices = sorted(episode_groups)
    if args.max_episodes is not None:
        episode_indices = episode_indices[: int(args.max_episodes)]
    shard_indices = [idx for pos, idx in enumerate(episode_indices) if pos % num_shards == shard_rank]

    anomaly_head = load_anomaly_head(cfg, args.anomaly_checkpoint, device)
    batch_items = []
    window_stride = int(args.window_stride)
    for dataset_index in shard_indices:
        batch_items.extend(
            (cache_pos, window)
            for cache_pos, window in episode_groups[int(dataset_index)]
            if int(window["fix_sidx"]) % window_stride == 0
        )
    chunks_by_episode = {int(idx): [] for idx in shard_indices}
    total_batches = (len(batch_items) + int(args.batch_size) - 1) // int(args.batch_size)
    for start in range(0, len(batch_items), int(args.batch_size)):
        batch = batch_items[start : start + int(args.batch_size)]
        if shard_rank == 0:
            print(f"cache batch {start // int(args.batch_size) + 1}/{total_batches}, windows={len(batch)}")
        cache_positions = [item[0] for item in batch]
        windows = [item[1] for item in batch]
        chunks = infer_cached_anomaly_batch(
            cfg=cfg,
            cache_dataset=cache_dataset,
            reference_dataset=reference_dataset,
            anomaly_head=anomaly_head,
            device=device,
            cache_positions=cache_positions,
            windows=windows,
            force_dim=force_dim,
        )
        for window, chunk in zip(windows, chunks):
            chunks_by_episode[int(window["dataset_index"])].append(chunk)

    records = [
        build_episode_record(
            cfg=cfg,
            dataset=reference_dataset,
            dataset_index=idx,
            chunks=chunks_by_episode[int(idx)],
            force_dim=force_dim,
            window_stride=int(args.window_stride),
            threshold=float(args.threshold),
        )
        for idx in shard_indices
        if chunks_by_episode[int(idx)]
    ]
    return records, episode_indices, cache_dataset.metadata


def run_online_eval(cfg, args, device, weight_dtype, action_condition_dim, force_dim, shard_rank, num_shards):
    dataset, selected_indices, _manifest_paths = build_eval_dataset(cfg, args.eval_split, split_manifest=args.split_manifest)
    if args.max_episodes is not None:
        selected_indices = selected_indices[: int(args.max_episodes)]
    shard_indices = [int(idx) for pos, idx in enumerate(selected_indices) if pos % num_shards == shard_rank]
    _, _, vae, _diffusion_model, _, pipe = prepare_models(cfg, device, weight_dtype)
    anomaly_head = load_anomaly_head(cfg, args.anomaly_checkpoint, device)

    windows = build_windows(cfg, dataset, shard_indices, int(args.window_stride))
    chunks_by_episode = {int(idx): [] for idx in shard_indices}
    n_previous = int(cfg["data"]["train"]["n_previous"])
    total_batches = (len(windows) + int(args.batch_size) - 1) // int(args.batch_size)
    for start in range(0, len(windows), int(args.batch_size)):
        batch_windows = windows[start : start + int(args.batch_size)]
        if shard_rank == 0:
            print(f"online batch {start // int(args.batch_size) + 1}/{total_batches}, windows={len(batch_windows)}")
        chunks = infer_anomaly_batch(
            cfg=cfg,
            dataset=dataset,
            pipe=pipe,
            anomaly_head=anomaly_head,
            vae=vae,
            args=args,
            device=device,
            weight_dtype=weight_dtype,
            windows=batch_windows,
            n_previous=n_previous,
            action_condition_dim=action_condition_dim,
            force_dim=force_dim,
        )
        for window, chunk in zip(batch_windows, chunks):
            chunks_by_episode[int(window["dataset_index"])].append(chunk)

    records = [
        build_episode_record(
            cfg=cfg,
            dataset=dataset,
            dataset_index=idx,
            chunks=chunks_by_episode[int(idx)],
            force_dim=force_dim,
            window_stride=int(args.window_stride),
            threshold=float(args.threshold),
        )
        for idx in shard_indices
    ]
    return records, selected_indices, None


def save_outputs(records, metadata, args):
    os.makedirs(args.output_dir, exist_ok=True)
    records = sorted(records, key=lambda r: (str(r["domain"]), int(r["episode_id"] if r["episode_id"] is not None else r["dataset_index"])))
    print("Computing stage metrics...")
    stage_nested, stage_rows = compute_stage_metrics(records, threshold=float(args.threshold))
    include_deployment_outputs = bool(getattr(args, "include_deployment_outputs", False))
    deployment = compute_deployment_metrics(records, threshold=float(args.threshold)) if include_deployment_outputs else None

    payload = {
        "metadata": metadata,
        "records": records,
    }
    print("Saving episode_records.pt...")
    torch.save(payload, os.path.join(args.output_dir, "episode_records.pt"))

    summary = {
        "metadata": metadata,
        "stage_metrics": stage_nested,
    }
    if include_deployment_outputs:
        summary["deployment_metrics"] = deployment
    print("Writing json/csv summaries...")
    write_json(os.path.join(args.output_dir, "metrics_summary.json"), json_safe(summary))
    write_csv(os.path.join(args.output_dir, "domain_stage_metrics.csv"), [flatten_stage_row(row) for row in stage_rows])
    write_csv(
        os.path.join(args.output_dir, "episode_summary.csv"),
        build_episode_summary_rows(records, include_deployment=include_deployment_outputs),
    )
    write_csv(os.path.join(args.output_dir, "window_records.csv"), build_window_rows(records))
    write_csv(os.path.join(args.output_dir, "micro_records.csv"), build_micro_rows(records))
    if include_deployment_outputs:
        write_csv(os.path.join(args.output_dir, "early_warning.csv"), build_early_warning_rows(records))

    print("Saving summary plots...")
    summary_plot_dir = os.path.join(args.output_dir, "plots", "summary")
    save_domain_stage_summary_plot(stage_rows, os.path.join(summary_plot_dir, "domain_stage_metrics.png"))
    save_confusion_summary_plot(records, os.path.join(summary_plot_dir, "confusion_matrices.png"), threshold=float(args.threshold))
    if include_deployment_outputs:
        save_deployment_summary_plot(deployment, os.path.join(summary_plot_dir, "deployment_status.png"))
    if bool(getattr(args, "save_score_histograms", False)):
        save_score_histogram_plot(records, os.path.join(summary_plot_dir, "score_histograms.png"))

    plot_records = choose_plot_records(records, args.plot_mode, args.top_k_plots, threshold=float(args.threshold))
    plot_dir = os.path.join(args.output_dir, "plots", "episode_curves")
    print(f"Saving episode plots: {len(plot_records)}")
    for record in plot_records:
        episode_tag = record["episode_id"] if record["episode_id"] is not None else record["dataset_index"]
        name = f"{record['domain']}_episode_{int(episode_tag):06d}_idx{int(record['dataset_index']):05d}.png"
        save_episode_plot(
            record,
            os.path.join(plot_dir, name),
            threshold=float(args.threshold),
            include_deployment=include_deployment_outputs,
        )


def save_episode_videos(records, cfg, metadata, args):
    if not bool(getattr(args, "make_episode_videos", False)):
        return
    eval_split = metadata.get("cache_split") or metadata.get("eval_split") or "test"
    split_manifest = metadata.get("split_manifest")
    dataset, _selected_indices, _manifest_paths = build_eval_dataset(cfg, eval_split, split_manifest=split_manifest)
    records_by_index = {int(record["dataset_index"]): record for record in records}
    top_k = getattr(args, "top_k_plots", 20)
    video_records = choose_plot_records(records, getattr(args, "plot_mode", "top_errors"), top_k, threshold=float(args.threshold))
    video_dir = os.path.join(args.output_dir, "videos", "episode_curves")
    os.makedirs(video_dir, exist_ok=True)
    for record in video_records:
        dataset_index = int(record["dataset_index"])
        if dataset_index not in records_by_index:
            continue
        video_source, cam_names = resolve_video_template_and_cams(cfg, dataset, dataset_index, split=eval_split)
        total_frames = int(record["total_frames"])
        multiview_frames = load_multiview_video_segment(
            video_source=video_source,
            cam_names=cam_names,
            start_frame=0,
            num_frames=total_frames,
        )
        episode_tag = record["episode_id"] if record["episode_id"] is not None else dataset_index
        output_path = os.path.join(
            video_dir,
            f"{record['domain']}_episode_{int(episode_tag):06d}_idx{dataset_index:05d}.mp4",
        )
        build_episode_metrics_video(
            output_path=output_path,
            multiview_frames=multiview_frames,
            window_x=record["windows"]["start"],
            window_prob=record["windows"]["prob"],
            window_target=record["windows"]["target"],
            frame_x=record["frames"]["index"],
            frame_prob=record["frames"]["aggregated_prob"],
            frame_target=record["frames"]["anomaly_label"],
            threshold=float(args.threshold),
            fps=float(args.video_fps),
            output_width=int(args.video_output_width),
            plot_height=int(args.video_plot_height),
            transition_frame=record["failure_transition_frame"],
            interaction_start=record["interaction_start_frame"],
            pred_force=record.get("frames", {}).get("pred_force_denorm"),
            gt_force=record.get("frames", {}).get("gt_force_denorm"),
        )


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.window_stride <= 0:
        raise ValueError("--window_stride must be positive")
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError("--threshold must be in [0, 1]")
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_resolved_config(args.config_file)
    if args.checkpoint_path is not None:
        cfg["load_weights"] = True
        cfg["load_diffusion_model_weights"] = True
        cfg["diffusion_model"]["model_path"] = resolve_checkpoint_path(args.checkpoint_path)
    if args.num_inference_steps is not None:
        cfg["num_inference_step"] = int(args.num_inference_steps)
    if args.split_manifest in (None, ""):
        args.split_manifest = cfg.get("split_manifest")
    if args.cache_split is None:
        args.cache_split = args.eval_split

    action_condition_dim, force_dim = resolve_force_layout(cfg)
    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    if args.window_stride > action_chunk:
        raise ValueError(f"--window_stride must be <= action_chunk={action_chunk}")

    shard_rank, num_shards = resolve_shard(args)
    run_id = resolve_run_id(args)
    device = resolve_device(args.device)
    weight_dtype = str_to_dtype(args.dtype)
    requested_cache_dir = normalize_optional_path(args.anomaly_feature_cache_dir) or normalize_optional_path(
        cfg.get("anomaly_feature_cache_dir")
    )
    cache_dir = None if args.disable_cache else requested_cache_dir

    if cache_dir:
        if shard_rank == 0:
            print(f"Episode eval from cache: {cache_dir}, split={args.cache_split}")
        records, expected_indices, cache_metadata = run_cache_eval(cfg, args, device, force_dim, shard_rank, num_shards)
        mode = "cache"
    else:
        if shard_rank == 0:
            print(f"Episode eval with online WM inference, split={args.eval_split}")
        records, expected_indices, cache_metadata = run_online_eval(
            cfg, args, device, weight_dtype, action_condition_dim, force_dim, shard_rank, num_shards
        )
        mode = "online"

    partial_dir = os.path.join(args.output_dir, "partials")
    os.makedirs(partial_dir, exist_ok=True)
    partial_path = os.path.join(partial_dir, f"episode_records_{run_id}_rank{shard_rank:03d}.pt")
    partial_payload = {
        "metadata": {
            "run_id": run_id,
            "rank": shard_rank,
            "num_shards": num_shards,
            "expected_indices": [int(x) for x in expected_indices],
            "num_records": len(records),
        },
        "records": records,
    }
    torch.save(partial_payload, partial_path)
    if shard_rank != 0:
        return

    partial_paths = [
        os.path.join(partial_dir, f"episode_records_{run_id}_rank{rank:03d}.pt")
        for rank in range(num_shards)
    ]
    deadline = time.time() + int(args.merge_wait_seconds)
    while any(not os.path.exists(path) for path in partial_paths) and time.time() < deadline:
        time.sleep(2)
    available = [path for path in partial_paths if os.path.exists(path)]
    all_records = []
    for path in available:
        part = torch.load(path, map_location="cpu", weights_only=False)
        if part.get("metadata", {}).get("run_id") == run_id:
            all_records.extend(part.get("records", []))

    if len(all_records) != len(expected_indices):
        print(f"Warning: merged {len(all_records)}/{len(expected_indices)} episodes.")

    metadata = {
        "script": "analyze/eval_action_force_anomaly_metrics.py",
        "mode": mode,
        "config_file": args.config_file,
        "wm_checkpoint": cfg["diffusion_model"]["model_path"],
        "anomaly_checkpoint": args.anomaly_checkpoint,
        "feature_cache_dir": cache_dir,
        "cache_split": args.cache_split if cache_dir else None,
        "eval_split": args.eval_split,
        "split_manifest": args.split_manifest,
        "num_inference_steps": int(cfg["num_inference_step"]),
        "window_stride": int(args.window_stride),
        "horizon": int(action_chunk),
        "threshold": float(args.threshold),
        "batch_size": int(args.batch_size),
        "dtype": args.dtype,
        "num_shards": int(num_shards),
        "merged_shards": int(len(available)),
        "run_id": run_id,
        "cache_metadata": cache_metadata,
    }
    save_outputs(all_records, metadata, args)
    if shard_rank == 0:
        save_episode_videos(all_records, cfg, metadata, args)
    print(f"Saved episode metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
