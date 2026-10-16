import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.anomaly_eval_common import extract_episode_id, load_failure_transition_frame, load_frame_labels
from analyze.eval_action_force_anomaly_metrics import (
    build_episode_record as build_unified_episode_record,
    save_episode_videos as save_unified_episode_videos,
    save_outputs as save_unified_outputs,
)
from analyze.action_force_anomaly_eval_core import (
    _best_f1_metrics,
    _binary_metrics_at_threshold,
    _interaction_group_metrics,
    build_cache_episode_groups,
    build_cache_reference_dataset,
    build_eval_dataset,
    build_windows,
    load_window_sample,
    resolve_device,
    resolve_run_id,
    resolve_shard,
    stack_samples,
)
from analyze.eval_common import prepare_models, resolve_checkpoint_path, str_to_dtype
from data.video_anomaly_feature_cache_dataset import VideoAnomalyFeatureCacheDataset
from utils import import_custom_class
from utils.model_utils import count_model_parameters
from utils.path_utils import load_resolved_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate video-latent anomaly metrics over validation episodes.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint_path", default=None, help="Video WM checkpoint override.")
    parser.add_argument("--anomaly_checkpoint", required=True, help="Path to video anomaly_head.pt.")
    parser.add_argument(
        "--anomaly_feature_cache_dir",
        default=None,
        help="Optional precomputed video-latent anomaly feature cache. If set, WM inference is skipped.",
    )
    parser.add_argument("--disable_cache", action="store_true", help="Force online WM inference even if config has a cache dir.")
    parser.add_argument("--cache_split", default=None, help="Cache split name to evaluate. Defaults to --eval_split.")
    parser.add_argument("--eval_split", default="val", help="Online evaluation split name when no feature cache is used.")
    parser.add_argument("--split_manifest", default=None, help="Optional train/val/test episode split manifest.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--mem_mode", choices=["uniform_history", "contiguous", "zeros"], default="uniform_history")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--shard_rank", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--merge_wait_seconds", type=int, default=3600)
    parser.add_argument("--partial_format", choices=["pt", "json"], default="pt")
    parser.add_argument("--write_episode_details", action="store_true")
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--plot_mode", choices=["none", "top_errors", "all"], default="top_errors")
    parser.add_argument("--top_k_plots", type=int, default=20)
    parser.add_argument("--make_episode_videos", action="store_true")
    parser.add_argument("--video_fps", type=float, default=30.0)
    parser.add_argument("--video_output_width", type=int, default=960)
    parser.add_argument("--video_plot_height", type=int, default=420)
    parser.add_argument(
        "--include_deployment_outputs",
        action="store_true",
        help="Also write deployment/early-warning summaries.",
    )
    parser.add_argument(
        "--save_score_histograms",
        action="store_true",
        help="Also save score histogram summary plots.",
    )
    return parser.parse_args()


def load_video_anomaly_head(config, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    head_cfg = dict(ckpt.get("anomaly_head", config.get("anomaly_head", {})))
    diffusion_cfg = config["diffusion_model"]["config"]
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    head_class_name = str(ckpt.get("anomaly_head_class", config.get("anomaly_head_class", "VideoLatentAnomalyHead_v1")))
    head_class_path = str(
        ckpt.get(
            "anomaly_head_class_path",
            config.get("anomaly_head_class_path", "models/anomaly_head/video_anomaly.py"),
        )
    )
    if "anomaly_head_class" not in ckpt and not any(str(key).startswith(("window_head.", "risk_head.")) for key in state_dict):
        head_class_name = "VideoLatentAnomalyHead"
    head_class = import_custom_class(head_class_name, head_class_path)
    head_kwargs = {
        "hidden_dim": int(head_cfg.get("hidden_dim", 256)),
        "feature_dim": int(head_cfg.get("feature_dim", 128)),
        "dropout": float(head_cfg.get("dropout", 0.1)),
        "latent_channels": int(head_cfg.get("latent_channels", diffusion_cfg.get("out_channels", 128))),
        "temporal_layers": int(head_cfg.get("temporal_layers", 2)),
        "temporal_heads": int(head_cfg.get("temporal_heads", 4)),
        "max_steps": int(head_cfg.get("max_steps", 256)),
        "topk_ratio": float(head_cfg.get("topk_ratio", 0.2)),
    }
    if head_class_name.endswith("_v1"):
        head_kwargs.update(
            {
                "window_pool": str(head_cfg.get("window_pool", "topk")),
                "risk_pool": str(head_cfg.get("risk_pool", "mean")),
            }
        )
    head = head_class(**head_kwargs)
    head.load_state_dict(state_dict, strict=True)
    return head.to(device).eval()


def _latent_chunk(cfg, vae):
    if cfg.get("anomaly_latent_chunk") not in (None, ""):
        return int(cfg["anomaly_latent_chunk"])
    future_steps = int(cfg["data"]["train"].get("future_steps", cfg["data"]["train"].get("action_chunk", 1)))
    return future_steps // int(vae.temporal_compression_ratio) + 1


def infer_video_anomaly_batch(cfg, dataset, pipe, anomaly_head, vae, args, device, weight_dtype, windows, n_previous):
    action_chunk = int(cfg["data"]["train"].get("action_chunk", cfg["data"]["train"]["future_steps"]))
    samples = [load_window_sample(dataset, window, n_previous, args.mem_mode) for window in windows]
    batch = stack_samples(samples)
    domain_names = [dataset.dataset[int(window["dataset_index"])][3] for window in windows]

    image = batch["video"][:, :, :, :n_previous].to(device=device, dtype=weight_dtype)
    batch_size, _, n_view, _, height, width = image.shape
    image = rearrange(image, "b c v t h w -> (b v) c t h w")

    if hasattr(pipe.scheduler, "_step_index"):
        pipe.scheduler._step_index = None
    with torch.no_grad():
        preds = pipe.infer(
            image=image,
            prompt=batch["caption"][:batch_size],
            negative_prompt="",
            num_inference_steps=int(cfg["num_inference_step"]),
            guidance_scale=float(args.guidance_scale),
            height=height,
            width=width,
            n_view=n_view,
            n_prev=n_previous,
            chunk=_latent_chunk(cfg, vae),
            return_video=False,
            return_latents=True,
            return_init=False,
            noise_seed=args.noise_seed,
            pixel_wise_timestep=bool(cfg["pixel_wise_timestep"]),
            n_chunk=1,
            show_progress=False,
        )[0]
        pred_latents = preds["pred_latents"].reshape(
            batch_size,
            n_view,
            preds["pred_latents"].shape[1],
            preds["pred_latents"].shape[2],
            preds["pred_latents"].shape[3],
            preds["pred_latents"].shape[4],
        )
        head_out = anomaly_head(pred_latents=pred_latents)

    step_prob = torch.sigmoid(head_out["step_logits"]).detach().cpu().float().numpy()
    if step_prob.shape[1] != action_chunk:
        x_old = np.linspace(0.0, 1.0, step_prob.shape[1])
        x_new = np.linspace(0.0, 1.0, action_chunk)
        step_prob = np.stack([np.interp(x_new, x_old, row) for row in step_prob], axis=0)
    window_prob = torch.sigmoid(head_out["window_logit"]).detach().cpu().float().reshape(-1).numpy()
    risk_prob = head_out["risk_prob"].detach().cpu().float().reshape(-1).numpy()

    frame_label = batch.get("frame_anomaly_label")
    frame_label = frame_label.detach().cpu().float().numpy() if frame_label is not None else None
    risk_target = batch.get("risk_label")
    risk_target = risk_target.detach().cpu().float().reshape(-1).numpy() if risk_target is not None else None
    label = batch.get("label")
    label = label.detach().cpu().float().reshape(-1).numpy() if label is not None else None
    interaction_label = batch.get("interaction_label")
    interaction_label = interaction_label.detach().cpu().long().reshape(-1).numpy() if interaction_label is not None else None

    chunks = []
    for row, window in enumerate(windows):
        valid_len = int(window["valid_len"])
        frame_target = None
        if frame_label is not None:
            frame_target = frame_label[row, :valid_len]
            if np.any(frame_target < 0):
                frame_target = None
        chunks.append(
            {
                "fix_sidx": int(window["fix_sidx"]),
                "valid_len": valid_len,
                "mem_idx": batch["_mem_idx"][row].detach().cpu().long().tolist(),
                "domain_name": domain_names[row],
                "prompt": batch["caption"][row],
                "step_prob": step_prob[row, :valid_len],
                "window_prob": float(window_prob[row]),
                "risk_prob": float(risk_prob[row]),
                "frame_target": frame_target,
                "risk_target": float(risk_target[row]) if risk_target is not None else None,
                "window_target": float(1.0 - label[row]) if label is not None else None,
                "interaction_label": int(interaction_label[row]) if interaction_label is not None else 0,
            }
        )
    return chunks


def _head_dtype(model):
    return next(model.parameters()).dtype


def infer_cached_video_anomaly_batch(cfg, cache_dataset, anomaly_head, device, cache_positions, windows):
    action_chunk = int(cfg["data"]["train"].get("action_chunk", cfg["data"]["train"]["future_steps"]))
    samples = [cache_dataset[int(pos)] for pos in cache_positions]
    batch = stack_samples(samples)
    feature_dtype = _head_dtype(anomaly_head)
    pred_latents = batch["pred_latents"].to(device=device, dtype=feature_dtype).contiguous()

    with torch.no_grad():
        head_out = anomaly_head(pred_latents=pred_latents)

    step_prob = torch.sigmoid(head_out["step_logits"]).detach().cpu().float().numpy()
    if step_prob.shape[1] != action_chunk:
        x_old = np.linspace(0.0, 1.0, step_prob.shape[1])
        x_new = np.linspace(0.0, 1.0, action_chunk)
        step_prob = np.stack([np.interp(x_new, x_old, row) for row in step_prob], axis=0)
    window_prob = torch.sigmoid(head_out["window_logit"]).detach().cpu().float().reshape(-1).numpy()
    risk_prob = head_out["risk_prob"].detach().cpu().float().reshape(-1).numpy()

    frame_label = batch.get("frame_anomaly_label")
    frame_label = frame_label.detach().cpu().float().numpy() if frame_label is not None else None
    risk_target = batch.get("risk_label")
    risk_target = risk_target.detach().cpu().float().reshape(-1).numpy() if risk_target is not None else None
    label = batch.get("label")
    label = label.detach().cpu().float().reshape(-1).numpy() if label is not None else None
    interaction_label = batch.get("interaction_label")
    interaction_label = interaction_label.detach().cpu().long().reshape(-1).numpy() if interaction_label is not None else None

    chunks = []
    for row, window in enumerate(windows):
        valid_len = int(window["valid_len"])
        frame_target = None
        if frame_label is not None:
            frame_target = frame_label[row, :valid_len]
            if np.any(frame_target < 0):
                frame_target = None
        chunks.append(
            {
                "fix_sidx": int(window["fix_sidx"]),
                "valid_len": valid_len,
                "mem_idx": [],
                "domain_name": "",
                "prompt": "",
                "step_prob": step_prob[row, :valid_len],
                "window_prob": float(window_prob[row]),
                "risk_prob": float(risk_prob[row]),
                "frame_target": frame_target,
                "risk_target": float(risk_target[row]) if risk_target is not None else None,
                "window_target": float(1.0 - label[row]) if label is not None else None,
                "interaction_label": int(interaction_label[row]) if interaction_label is not None else 0,
            }
        )
    return chunks


def evaluate_cached_records(cfg, args, device, shard_rank, num_shards):
    cache_dir = args.anomaly_feature_cache_dir or cfg.get("anomaly_feature_cache_dir")
    cache_dataset = VideoAnomalyFeatureCacheDataset(
        cache_dir=cache_dir,
        split=args.cache_split,
        max_open_shards=int(cfg.get("anomaly_cache_max_open_shards", 4)),
    )
    reference_dataset = build_cache_reference_dataset(cfg, args.cache_split, split_manifest=args.split_manifest)
    episode_groups = build_cache_episode_groups(cache_dataset, reference_dataset)
    episode_indices = sorted(episode_groups)
    if args.max_episodes is not None:
        episode_indices = episode_indices[: int(args.max_episodes)]
    shard_episode_indices = [
        int(dataset_index)
        for position, dataset_index in enumerate(episode_indices)
        if position % num_shards == shard_rank
    ]

    anomaly_head = load_video_anomaly_head(cfg, args.anomaly_checkpoint, device)
    chunks_by_episode = {int(dataset_index): [] for dataset_index in shard_episode_indices}
    batch_items = []
    for dataset_index in shard_episode_indices:
        for cache_pos, window in episode_groups[int(dataset_index)]:
            batch_items.append((cache_pos, window))

    total_batches = (len(batch_items) + int(args.batch_size) - 1) // int(args.batch_size)
    for start in range(0, len(batch_items), int(args.batch_size)):
        batch = batch_items[start : start + int(args.batch_size)]
        if shard_rank == 0:
            print(f"cache rank={shard_rank} batch={start // int(args.batch_size) + 1}/{total_batches} windows={len(batch)}")
        cache_positions = [item[0] for item in batch]
        windows = [item[1] for item in batch]
        chunks = infer_cached_video_anomaly_batch(
            cfg=cfg,
            cache_dataset=cache_dataset,
            anomaly_head=anomaly_head,
            device=device,
            cache_positions=cache_positions,
            windows=windows,
        )
        for window, chunk in zip(windows, chunks):
            chunks_by_episode[int(window["dataset_index"])].append(chunk)

    action_chunk = int(cfg["data"]["train"].get("action_chunk", cfg["data"]["train"]["future_steps"]))
    records = [
        build_unified_episode_record(
            cfg=cfg,
            dataset=reference_dataset,
            dataset_index=dataset_index,
            chunks=chunks_by_episode[dataset_index],
            force_dim=1,
            window_stride=int(getattr(reference_dataset, "window_stride", action_chunk)),
            threshold=float(args.threshold),
        )
        for dataset_index in shard_episode_indices
    ]
    return records, len(batch_items), episode_indices, cache_dataset.metadata


def aggregate_video_anomaly_chunks(chunks, total_frames, global_start=0):
    length = total_frames - int(global_start)
    prob_sum = np.zeros(length, dtype=np.float64)
    prob_count = np.zeros(length, dtype=np.int64)
    target_sum = np.zeros(length, dtype=np.float64)
    target_count = np.zeros(length, dtype=np.int64)
    window_probs = []
    window_targets = []

    for chunk in chunks:
        start = int(chunk["fix_sidx"]) - int(global_start)
        valid_len = min(int(chunk["valid_len"]), length - start)
        if valid_len <= 0:
            continue
        end = start + valid_len
        prob_sum[start:end] += chunk["step_prob"][:valid_len]
        prob_count[start:end] += 1
        if chunk.get("frame_target") is not None:
            target_sum[start:end] += chunk["frame_target"][:valid_len]
            target_count[start:end] += 1
        if chunk.get("window_target") is not None:
            window_probs.append(float(chunk["window_prob"]))
            window_targets.append(float(chunk["window_target"]))

    if np.any(prob_count == 0):
        missing = np.flatnonzero(prob_count == 0)
        raise RuntimeError(f"Uncovered anomaly timesteps: {missing[:20].tolist()}")
    step_prob = prob_sum / prob_count
    frame_target = None
    if np.any(target_count > 0):
        frame_target = np.zeros(length, dtype=np.float64)
        valid = target_count > 0
        frame_target[valid] = target_sum[valid] / target_count[valid]
        frame_target[~valid] = np.nan
    return {
        "step_prob": step_prob,
        "frame_target": frame_target,
        "window_probs": np.asarray(window_probs, dtype=np.float64),
        "window_targets": np.asarray(window_targets, dtype=np.float64),
    }


def build_episode_record(cfg, dataset, dataset_index, chunks, window_stride):
    item = dataset.dataset[int(dataset_index)]
    total_frames = int(item[7])
    chunks = sorted(chunks, key=lambda chunk: int(chunk["fix_sidx"]))
    if not chunks:
        raise RuntimeError(f"No windows generated for dataset_index={dataset_index}")
    aggregate = aggregate_video_anomaly_chunks(chunks, total_frames=total_frames)
    parquet_path = item[2]
    _, frame_anomaly = load_frame_labels(parquet_path, total_frames)
    frame_target = aggregate["frame_target"]
    if frame_target is None:
        frame_target = frame_anomaly
    length = min(aggregate["step_prob"].shape[0], frame_target.shape[0])
    step_scores = aggregate["step_prob"][:length]
    step_targets = (frame_target[:length] >= 0.5).astype(np.int64)
    window_scores = aggregate["window_probs"]
    window_targets = aggregate["window_targets"].astype(np.int64)
    step_interaction_labels = load_interaction_labels(parquet_path, total_frames)[:length].astype(np.int64)
    window_interaction_labels = np.asarray(
        [int(chunk.get("interaction_label", 0)) for chunk in chunks if chunk.get("window_target") is not None],
        dtype=np.int64,
    )
    risk_target = float(np.nanmean(step_targets.astype(np.float64)))
    risk_pred = float(np.nanmean(step_scores.astype(np.float64)))
    try:
        transition = load_failure_transition_frame(parquet_path)
    except Exception:
        transition = None
    return {
        "dataset_index": int(dataset_index),
        "episode_id": extract_episode_id(parquet_path),
        "domain": item[3],
        "parquet_path": parquet_path,
        "num_frames": int(length),
        "failure_transition_frame": transition,
        "has_anomaly": bool(step_targets.any()),
        "window_starts": [int(c["fix_sidx"]) for c in chunks],
        "window_stride": int(window_stride),
        "step_metrics": _binary_metrics_at_threshold(step_scores, step_targets, 0.5),
        "window_metrics": _binary_metrics_at_threshold(window_scores, window_targets, 0.5) if window_scores.size else None,
        "risk_metrics": {
            "mae": float(abs(risk_pred - risk_target)),
            "mse": float((risk_pred - risk_target) ** 2),
            "pred": risk_pred,
            "target": risk_target,
        },
        "_step_scores": step_scores,
        "_step_targets": step_targets,
        "_step_interaction_labels": step_interaction_labels,
        "_window_scores": window_scores,
        "_window_targets": window_targets,
        "_window_interaction_labels": window_interaction_labels,
    }


def aggregate_records(records, threshold=0.5):
    step_scores = np.concatenate([r["_step_scores"] for r in records])
    step_targets = np.concatenate([r["_step_targets"] for r in records])
    step_interaction_labels = np.concatenate([r["_step_interaction_labels"] for r in records])
    window_scores = [r["_window_scores"] for r in records if r["_window_scores"].size]
    window_targets = [r["_window_targets"] for r in records if r["_window_targets"].size]
    window_interaction_labels = [r["_window_interaction_labels"] for r in records if r["_window_scores"].size]
    step_metrics = _best_f1_metrics(step_scores, step_targets)
    step_metrics_manual = _binary_metrics_at_threshold(step_scores, step_targets, threshold)
    aggregate = {
        "step_metrics": step_metrics,
        "step_metrics_manual": step_metrics_manual,
        "risk_metrics": {
            "mae": float(np.mean([r["risk_metrics"]["mae"] for r in records])),
            "mse": float(np.mean([r["risk_metrics"]["mse"] for r in records])),
            "pred_mean": float(np.mean([r["risk_metrics"]["pred"] for r in records])),
            "target_mean": float(np.mean([r["risk_metrics"]["target"] for r in records])),
        },
    }
    if window_scores:
        all_window_scores = np.concatenate(window_scores)
        all_window_targets = np.concatenate(window_targets)
        all_window_interaction_labels = np.concatenate(window_interaction_labels)
        aggregate["window_metrics"] = _best_f1_metrics(all_window_scores, all_window_targets)
        aggregate["window_metrics_manual"] = _binary_metrics_at_threshold(all_window_scores, all_window_targets, threshold)
    else:
        all_window_scores = None
        all_window_targets = None
        all_window_interaction_labels = np.asarray([], dtype=np.int64)
        aggregate["window_metrics"] = None
        aggregate["window_metrics_manual"] = None
    step_best_threshold = step_metrics["threshold"] if step_metrics is not None else threshold
    window_best_metrics = aggregate["window_metrics"]
    window_best_threshold = window_best_metrics["threshold"] if window_best_metrics is not None else threshold
    aggregate["interaction_metrics"] = _interaction_group_metrics(
        step_scores=step_scores,
        step_targets=step_targets,
        step_interaction_labels=step_interaction_labels,
        window_scores=all_window_scores,
        window_targets=all_window_targets,
        window_interaction_labels=all_window_interaction_labels,
        step_best_threshold=step_best_threshold,
        window_best_threshold=window_best_threshold,
        manual_threshold=threshold,
    )
    return aggregate


def records_to_jsonable(records):
    output = []
    for record in records:
        converted = {}
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            elif isinstance(value, np.generic):
                converted[key] = value.item()
            else:
                converted[key] = value
        output.append(converted)
    return output


def records_from_jsonable(payload):
    records = []
    array_keys = {
        "_step_scores",
        "_step_targets",
        "_step_interaction_labels",
        "_window_scores",
        "_window_targets",
        "_window_interaction_labels",
    }
    for record in payload:
        restored = dict(record)
        for key in array_keys:
            if key in restored:
                dtype = np.int64 if key.endswith("targets") or key.endswith("labels") else np.float64
                restored[key] = np.asarray(restored[key], dtype=dtype)
        records.append(restored)
    return records


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")

    shard_rank, num_shards = resolve_shard(args)
    run_id = resolve_run_id(args)
    device = resolve_device(args.device)
    weight_dtype = str_to_dtype(args.dtype)

    cfg = load_resolved_config(args.config_file)
    if args.split_manifest in (None, ""):
        args.split_manifest = cfg.get("split_manifest")
    if args.cache_split is None:
        args.cache_split = args.eval_split
    if args.checkpoint_path is not None:
        cfg["load_weights"] = True
        cfg["load_diffusion_model_weights"] = True
        cfg["diffusion_model"]["model_path"] = resolve_checkpoint_path(args.checkpoint_path)
    if args.num_inference_steps is not None:
        cfg["num_inference_step"] = int(args.num_inference_steps)

    action_chunk = int(cfg["data"]["train"].get("action_chunk", cfg["data"]["train"]["future_steps"]))
    window_stride = int(args.window_stride or action_chunk)

    cache_dir = None if args.disable_cache else (args.anomaly_feature_cache_dir or cfg.get("anomaly_feature_cache_dir"))
    cache_metadata = None
    if cache_dir:
        if shard_rank == 0:
            print(f"Evaluating video anomaly head from cache: {cache_dir}")
        records, num_windows, selected_indices, cache_metadata = evaluate_cached_records(
            cfg=cfg,
            args=args,
            device=device,
            shard_rank=shard_rank,
            num_shards=num_shards,
        )
        manifest_paths = {}
    else:
        dataset, selected_indices, manifest_paths = build_eval_dataset(cfg, args.eval_split, split_manifest=args.split_manifest)
        if args.max_episodes is not None:
            selected_indices = selected_indices[: int(args.max_episodes)]

        shard_indices = [
            int(dataset_index)
            for position, dataset_index in enumerate(selected_indices)
            if position % num_shards == shard_rank
        ]

        _, _, vae, diffusion_model, _, pipe = prepare_models(cfg, device, weight_dtype)
        anomaly_head = load_video_anomaly_head(cfg, args.anomaly_checkpoint, device)
        total_params = count_model_parameters(diffusion_model)
        if shard_rank == 0:
            print(f"rank={shard_rank}/{num_shards} device={device} params total={total_params[0]} trainable={total_params[1]}")

        n_previous = int(cfg["data"]["train"]["n_previous"])
        windows = build_windows(cfg, dataset, shard_indices, window_stride)
        chunks_by_episode = {int(dataset_index): [] for dataset_index in shard_indices}
        total_batches = (len(windows) + int(args.batch_size) - 1) // int(args.batch_size)
        for start in range(0, len(windows), int(args.batch_size)):
            batch_windows = windows[start : start + int(args.batch_size)]
            if shard_rank == 0:
                print(f"rank={shard_rank} batch={start // int(args.batch_size) + 1}/{total_batches} windows={len(batch_windows)}")
            batch_chunks = infer_video_anomaly_batch(
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
            )
            for window, chunk in zip(batch_windows, batch_chunks):
                chunks_by_episode[int(window["dataset_index"])].append(chunk)

        records = [
            build_unified_episode_record(
                cfg=cfg,
                dataset=dataset,
                dataset_index=dataset_index,
                chunks=chunks_by_episode[dataset_index],
                force_dim=1,
                window_stride=window_stride,
                threshold=float(args.threshold),
            )
            for dataset_index in shard_indices
        ]
        num_windows = len(windows)

    partial_payload = {
        "metadata": {
            "run_id": run_id,
            "rank": shard_rank,
            "num_shards": num_shards,
            "num_episodes": len(records),
            "num_windows": num_windows,
            "feature_cache_dir": cache_dir,
            "cache_metadata": cache_metadata,
        },
        "records": records if args.partial_format == "pt" else records_to_jsonable(records),
    }
    partial_ext = "pt" if args.partial_format == "pt" else "json"
    partial_dir = os.path.join(args.output_dir, "partials")
    os.makedirs(partial_dir, exist_ok=True)
    partial_path = os.path.join(partial_dir, f"episode_records_{run_id}_rank{shard_rank:03d}.{partial_ext}")
    if args.partial_format == "pt":
        torch.save(partial_payload, partial_path)
    else:
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(partial_payload, f, indent=2)
    if shard_rank != 0:
        return

    partial_paths = [os.path.join(partial_dir, f"episode_records_{run_id}_rank{rank:03d}.{partial_ext}") for rank in range(num_shards)]
    deadline = time.time() + max(0, int(args.merge_wait_seconds))
    while any(not os.path.exists(path) for path in partial_paths) and time.time() < deadline:
        time.sleep(2)

    available_paths = [path for path in partial_paths if os.path.exists(path)]
    all_records = []
    for path in available_paths:
        if args.partial_format == "pt":
            payload_part = torch.load(path, map_location="cpu", weights_only=False)
        else:
            with open(path, "r", encoding="utf-8") as f:
                payload_part = json.load(f)
        metadata = payload_part.get("metadata", {})
        if metadata.get("run_id") != run_id:
            continue
        all_records.extend(payload_part.get("records", []) if args.partial_format == "pt" else records_from_jsonable(payload_part.get("records", [])))
    all_records.sort(key=lambda record: int(record["dataset_index"]))

    if len(all_records) != len(selected_indices):
        print(f"Warning: merged {len(all_records)}/{len(selected_indices)} episodes. Increase --merge_wait_seconds or check failed ranks.")

    manifest_payload = {name: str(path) for name, path in manifest_paths.items()}
    metadata = {
        "script": "analyze/eval_video_anomaly_metrics.py",
        "mode": "cache" if cache_dir else "online",
        "config_file": args.config_file,
        "wm_checkpoint": cfg["diffusion_model"]["model_path"],
        "anomaly_checkpoint": args.anomaly_checkpoint,
        "feature_cache_dir": cache_dir,
        "cache_split": args.cache_split if cache_dir else None,
        "eval_split": args.eval_split,
        "split_manifest": args.split_manifest,
        "manifest_paths": manifest_payload,
        "num_inference_steps": int(cfg["num_inference_step"]),
        "window_stride": int(window_stride),
        "horizon": int(action_chunk),
        "threshold": float(args.threshold),
        "batch_size": int(args.batch_size),
        "dtype": args.dtype,
        "num_shards": int(num_shards),
        "merged_shards": int(len(available_paths)),
        "run_id": run_id,
        "cache_metadata": cache_metadata,
    }
    save_unified_outputs(all_records, metadata, args)
    save_unified_episode_videos(all_records, cfg, metadata, args)
    print(f"Saved video anomaly episode metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
