import argparse
import json
import os
import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import cv2
from einops import rearrange
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.anomaly_eval_common import (
    basic_binary_metrics,
    extract_episode_id,
    load_failure_transition_frame,
    load_frame_labels,
    write_json,
)
from analyze.eval_common import (
    build_mem_idx,
    denorm_force,
    prepare_models,
    prepare_val_dataset,
    resolve_checkpoint_path,
    resolve_force_layout,
    str_to_dtype,
)
from models.pipeline.custom_pipeline_v import retrieve_latents
from analyze.anomaly_eval_common import build_anomaly_alignment_video
from data.anomaly_feature_cache_dataset import AnomalyFeatureCacheDataset
from utils import import_custom_class
from utils.model_utils import count_model_parameters
from utils.path_utils import load_resolved_config


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate action-force anomaly metrics over a dataset split.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint_path", default=None, help="Action-force WM checkpoint override.")
    parser.add_argument("--anomaly_checkpoint", required=True, help="Path to anomaly_head.pt.")
    parser.add_argument(
        "--anomaly_feature_cache_dir",
        default=None,
        help="Optional precomputed action-force anomaly feature cache. If set, WM inference is skipped.",
    )
    parser.add_argument("--cache_split", default="val", help="Cache split name to evaluate.")
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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help=(
            "Manual anomaly threshold for both step and window metrics. "
            "Best-F1 metrics are always computed separately."
        ),
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--shard_rank", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    parser.add_argument("--merge_wait_seconds", type=int, default=3600)
    parser.add_argument(
        "--partial_format",
        choices=["pt", "json"],
        default="pt",
        help="Format for rank partial files. pt is much faster for large per-episode arrays.",
    )
    parser.add_argument(
        "--write_episode_details",
        action="store_true",
        help="Write per-episode records into metrics.json and per_episode_metrics.jsonl.",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        help=(
            "Identifier used to isolate partial files. Defaults to TORCHELASTIC_RUN_ID "
            "or 'single'. Set this when reusing an output directory."
        ),
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def resolve_shard(args):
    rank = args.shard_rank
    world = args.num_shards
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    if world is None:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    if not 0 <= rank < world:
        raise ValueError(f"Invalid shard rank/world: {rank}/{world}")
    return rank, world


def resolve_run_id(args):
    if args.run_id:
        return str(args.run_id)
    return os.environ.get("TORCHELASTIC_RUN_ID", "single")


def stack_samples(samples):
    batch = {}
    keys = samples[0].keys()
    for key in keys:
        values = [sample[key] for sample in samples]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], str):
            batch[key] = values
        else:
            batch[key] = values
    return batch


def load_interaction_labels(parquet_path, total_frames):
    data = pd.read_parquet(parquet_path, columns=None)
    if "interaction_label" in data.columns:
        labels = data["interaction_label"].to_numpy().astype(np.int64)
    else:
        labels = np.zeros(total_frames, dtype=np.int64)
    if labels.shape[0] < total_frames:
        labels = np.pad(labels, (0, total_frames - labels.shape[0]), constant_values=0)
    return labels[:total_frames]


def build_windows(cfg, dataset, selected_indices, window_stride):
    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    if window_stride <= 0 or window_stride > action_chunk:
        raise ValueError(f"window_stride must be in [1, {action_chunk}], got {window_stride}")

    windows = []
    for dataset_index in selected_indices:
        item = dataset.dataset[int(dataset_index)]
        total_frames = int(item[7])
        for fix_sidx in range(0, total_frames, window_stride):
            valid_len = min(action_chunk, max(0, total_frames - fix_sidx))
            if valid_len > 0:
                windows.append(
                    {
                        "dataset_index": int(dataset_index),
                        "fix_sidx": int(fix_sidx),
                        "valid_len": int(valid_len),
                        "total_frames": int(total_frames),
                    }
                )
    return windows


def load_window_sample(dataset, window, n_previous, mem_mode):
    mem_idx = build_mem_idx(int(window["fix_sidx"]), n_previous, mem_mode)
    dataset.fix_epiidx = int(window["dataset_index"])
    dataset.fix_sidx = int(window["fix_sidx"])
    dataset.fix_mem_idx = mem_idx
    sample = dataset[0]
    sample["_mem_idx"] = torch.tensor(mem_idx, dtype=torch.long)
    return sample


def _split_manifest_data_root(split_manifest):
    if split_manifest in (None, ""):
        return None
    manifest_path = Path(split_manifest).expanduser()
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    data_root = manifest.get("data_root")
    if data_root in (None, ""):
        return None
    data_root = Path(data_root).expanduser()
    if not data_root.is_absolute():
        data_root = (manifest_path.parent / data_root).resolve()
    return str(data_root)


def data_cfg_for_split(cfg, split, split_manifest=None):
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
        split_manifest = str(Path(split_manifest).expanduser())
        data_cfg["split_manifest"] = split_manifest
        data_cfg["split"] = split
        manifest_data_root = _split_manifest_data_root(split_manifest)
        if manifest_data_root is not None:
            data_cfg["data_roots"] = [manifest_data_root]
    data_cfg["memory_pick_mode"] = "uniform"
    return data_cfg


def dataset_class_for_split(cfg, split):
    if split == "train":
        return cfg["train_data_class"], cfg["train_data_class_path"]
    return (
        cfg.get("val_data_class", cfg["train_data_class"]),
        cfg.get("val_data_class_path", cfg["train_data_class_path"]),
    )


def build_cache_reference_dataset(cfg, split, split_manifest=None):
    data_cfg = data_cfg_for_split(cfg, split, split_manifest=split_manifest)
    class_name, class_path = dataset_class_for_split(cfg, split)
    dataset_class = import_custom_class(class_name, class_path)
    return dataset_class(**data_cfg)


def build_eval_dataset(cfg, split, split_manifest=None):
    if split == "val" and not split_manifest and not cfg.get("split_manifest"):
        return prepare_val_dataset(cfg)

    data_cfg = data_cfg_for_split(cfg, split, split_manifest=split_manifest)
    class_name, class_path = dataset_class_for_split(cfg, split)
    dataset_class = import_custom_class(class_name, class_path)
    dataset = dataset_class(**data_cfg)
    selected_indices = list(range(len(dataset.dataset)))
    manifest = data_cfg.get("split_manifest")
    domains = {str(item[3]) for item in dataset.dataset}
    manifest_paths = {domain: Path(manifest) for domain in domains} if manifest else {}
    return dataset, selected_indices, manifest_paths


def _entry_int(entry, key, default=None):
    value = entry.get(key, default)
    if value is None:
        return default
    return int(value)


def resolve_cached_window(dataset, entry, cache_pos):
    cache_index = _entry_int(entry, "index", cache_pos)
    episode_idx = _entry_int(entry, "episode_dataset_index")
    fix_sidx = _entry_int(entry, "window_start")
    total_frames = _entry_int(entry, "total_frames")
    future_steps = int(getattr(dataset, "future_steps", getattr(dataset, "action_chunk", 1)))

    if episode_idx is not None and fix_sidx is not None:
        item = dataset.dataset[int(episode_idx)]
        if total_frames is None or total_frames < 0:
            total_frames = int(item[7])
        valid_len = min(future_steps, max(0, int(total_frames) - int(fix_sidx)))
        return {
            "dataset_index": int(episode_idx),
            "fix_sidx": int(fix_sidx),
            "valid_len": int(valid_len),
            "total_frames": int(total_frames),
            "cache_index": int(cache_index),
            "cache_pos": int(cache_pos),
        }

    cache_index = int(cache_index)
    if hasattr(dataset, "traverse_index"):
        episode_idx, chunk_end = dataset.traverse_index[cache_index]
        future_steps = int(getattr(dataset, "future_steps"))
        fix_sidx = int(chunk_end) - future_steps
    else:
        episode_idx = cache_index
        future_steps = int(getattr(dataset, "future_steps", getattr(dataset, "action_chunk", 1)))
        fix_sidx = 0
    item = dataset.dataset[int(episode_idx)]
    total_frames = int(item[7])
    valid_len = min(future_steps, max(0, total_frames - int(fix_sidx)))
    return {
        "dataset_index": int(episode_idx),
        "fix_sidx": int(fix_sidx),
        "valid_len": int(valid_len),
        "total_frames": int(total_frames),
        "cache_index": cache_index,
        "cache_pos": int(cache_pos),
    }


def build_cache_episode_groups(cache_dataset, reference_dataset):
    groups = {}
    for cache_pos, entry in enumerate(cache_dataset.entries):
        window = resolve_cached_window(reference_dataset, entry, cache_pos)
        if window["valid_len"] <= 0:
            continue
        groups.setdefault(int(window["dataset_index"]), []).append((int(cache_pos), window))
    for values in groups.values():
        values.sort(key=lambda item: int(item[1]["fix_sidx"]))
    return groups


def _head_dtype(model):
    return next(model.parameters()).dtype


def infer_cached_anomaly_batch(cfg, cache_dataset, reference_dataset, anomaly_head, device, cache_positions, windows, force_dim):
    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    samples = [cache_dataset[int(pos)] for pos in cache_positions]
    batch = stack_samples(samples)
    feature_dtype = _head_dtype(anomaly_head)

    pred_force_norm = batch["pred_force"].to(device=device, dtype=feature_dtype).contiguous()
    history_force_state = batch["history_force_state"].to(device=device, dtype=feature_dtype).contiguous()
    interaction_tokens = batch["interaction_tokens"].to(device=device, dtype=feature_dtype).contiguous()
    pred_latents = batch["pred_latents"].to(device=device, dtype=feature_dtype).contiguous()
    head_kwargs = {}
    if bool(getattr(anomaly_head, "requires_observed_inputs", False)):
        gt_force = batch.get("gt_force", batch.get("gt_force_norm"))
        gt_latents = batch.get("gt_latents")
        if gt_force is None or not torch.is_tensor(gt_force) or gt_force.numel() == 0:
            raise ValueError("Cached residual anomaly evaluation requires gt_force/gt_force_norm")
        if gt_latents is None or not torch.is_tensor(gt_latents) or gt_latents.numel() == 0:
            raise ValueError("Cached residual anomaly evaluation requires gt_latents; rebuild the feature cache")
        head_kwargs["gt_force"] = gt_force.to(device=device, dtype=feature_dtype).contiguous()
        head_kwargs["gt_latents"] = gt_latents.to(device=device, dtype=feature_dtype).contiguous()

    with torch.no_grad():
        head_out = anomaly_head(
            pred_force=pred_force_norm,
            history_force=history_force_state,
            interaction_tokens=interaction_tokens,
            pred_latents=pred_latents,
            **head_kwargs,
        )

    pred_force_norm_cpu = pred_force_norm.detach().cpu().float()
    if "gt_force_norm" in batch and batch["gt_force_norm"].numel() > 0:
        gt_force_norm = batch["gt_force_norm"].detach().cpu().float()
    else:
        gt_force_norm = []
        for window in windows:
            sample = reference_dataset[int(window["cache_index"])]
            gt_force_norm.append(sample["force_targets"][-action_chunk:].detach().cpu().float())
        gt_force_norm = torch.stack(gt_force_norm, dim=0)

    domain_names = [reference_dataset.dataset[int(window["dataset_index"])][3] for window in windows]
    pred_force_denorm = []
    gt_force_denorm = []
    for row, domain_name in enumerate(domain_names):
        pred_force_denorm.append(denorm_force(reference_dataset, domain_name, pred_force_norm_cpu[row : row + 1]).cpu())
        gt_force_denorm.append(denorm_force(reference_dataset, domain_name, gt_force_norm[row : row + 1]).cpu())
    pred_force_denorm = torch.cat(pred_force_denorm, dim=0).numpy()
    gt_force_denorm = torch.cat(gt_force_denorm, dim=0).numpy()

    step_prob = torch.sigmoid(head_out["step_logits"]).detach().cpu().float().numpy()
    if step_prob.shape[1] != action_chunk:
        x_old = np.linspace(0.0, 1.0, step_prob.shape[1])
        x_new = np.linspace(0.0, 1.0, action_chunk)
        step_prob = np.stack([np.interp(x_new, x_old, row) for row in step_prob], axis=0)
    window_prob = torch.sigmoid(head_out["window_logit"]).detach().cpu().float().reshape(-1).numpy()
    risk_prob = head_out["risk_prob"].detach().cpu().float().reshape(-1).numpy()
    micro_prob = None
    if "micro_window_logits" in head_out:
        micro_prob = torch.sigmoid(head_out["micro_window_logits"]).detach().cpu().float().numpy()

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
                "domain_name": domain_names[row],
                "prompt": "",
                "pred_force_denorm": pred_force_denorm[row : row + 1, :valid_len],
                "gt_force_denorm": gt_force_denorm[row : row + 1, :valid_len],
                "pred_force_norm": pred_force_norm_cpu[row : row + 1, :valid_len].numpy(),
                "gt_force_norm": gt_force_norm[row : row + 1, :valid_len].numpy(),
                "step_prob": step_prob[row, :valid_len],
                "window_prob": float(window_prob[row]),
                "risk_prob": float(risk_prob[row]),
                "micro_prob": None if micro_prob is None else micro_prob[row],
                "frame_target": frame_target,
                "risk_target": float(risk_target[row]) if risk_target is not None else None,
                "window_target": float(1.0 - label[row]) if label is not None else None,
                "interaction_label": int(interaction_label[row]) if interaction_label is not None else 0,
                "video": None,
            }
        )
    return chunks


def evaluate_cached_records(cfg, args, device, force_dim, shard_rank, num_shards):
    cache_dir = args.anomaly_feature_cache_dir or cfg.get("anomaly_feature_cache_dir")
    cache_dataset = AnomalyFeatureCacheDataset(
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

    anomaly_head = load_anomaly_head(cfg, args.anomaly_checkpoint, device)
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

    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    records = [
        build_episode_record(
            cfg=cfg,
            dataset=reference_dataset,
            dataset_index=dataset_index,
            chunks=chunks_by_episode[dataset_index],
            force_dim=force_dim,
            window_stride=int(getattr(reference_dataset, "window_stride", action_chunk)),
        )
        for dataset_index in shard_episode_indices
    ]
    return records, len(batch_items), episode_indices, cache_dataset.metadata


def infer_anomaly_batch(
    cfg,
    dataset,
    pipe,
    anomaly_head,
    vae,
    args,
    device,
    weight_dtype,
    windows,
    n_previous,
    action_condition_dim,
    force_dim,
):
    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    samples = [
        load_window_sample(dataset, window, n_previous, args.mem_mode)
        for window in windows
    ]
    batch = stack_samples(samples)
    domain_names = [
        dataset.dataset[int(window["dataset_index"])][3]
        for window in windows
    ]

    image = batch["video"][:, :, :, :n_previous].to(device=device, dtype=weight_dtype)
    future_video = batch["video"][:, :, :, n_previous:].to(device=device, dtype=weight_dtype)
    batch_size, _, n_view, _, height, width = image.shape
    image = rearrange(image, "b c v t h w -> (b v) c t h w")
    future_video = rearrange(future_video, "b c v t h w -> (b v) c t h w")

    future_actions = batch["actions"][:, -action_chunk:].to(device=device, dtype=weight_dtype).contiguous()
    if future_actions.shape[-1] != action_condition_dim:
        raise ValueError(f"Expected action dim {action_condition_dim}, got {future_actions.shape[-1]}")

    history_force_state = batch.get("history_force_state")
    if history_force_state is None:
        raise KeyError("Dataset sample must contain history_force_state.")
    history_force_state = history_force_state.to(device=device, dtype=weight_dtype).contiguous()

    latent_chunk = int(cfg["data"]["train"]["chunk"])
    if hasattr(pipe.scheduler, "_step_index"):
        pipe.scheduler._step_index = None

    with torch.no_grad():
        preds = pipe.infer(
            image=image,
            prompt=batch["caption"][:batch_size],
            negative_prompt="",
            num_inference_steps=int(cfg["num_inference_step"]),
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=float(args.guidance_scale),
            height=height,
            width=width,
            n_view=n_view,
            return_force=True,
            return_interaction=True,
            return_latents=True,
            n_prev=n_previous,
            chunk=latent_chunk,
            return_video=False,
            return_init=False,
            noise_seed=args.noise_seed,
            action_chunk=action_chunk,
            history_force_state=history_force_state,
            pixel_wise_timestep=bool(cfg["pixel_wise_timestep"]),
            n_chunk=1,
            force_dim=force_dim,
            future_actions=future_actions,
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
        head_kwargs = {}
        if bool(getattr(anomaly_head, "requires_observed_inputs", False)):
            head_kwargs["gt_force"] = batch["force_targets"][:, -action_chunk:].to(
                device=device,
                dtype=weight_dtype,
            ).contiguous()
            head_kwargs["gt_latents"] = encode_gt_latents(
                vae,
                future_video,
                batch_size,
                n_view,
                weight_dtype,
            )
        head_out = anomaly_head(
            pred_force=preds["force"],
            history_force=history_force_state,
            interaction_tokens=preds["interaction_tokens"],
            pred_latents=pred_latents,
            **head_kwargs,
        )

    pred_force_norm = preds["force"].detach().cpu().float()
    gt_force_norm = batch["force_targets"][:, -action_chunk:].detach().cpu().float()
    pred_force_denorm = []
    gt_force_denorm = []
    for row, domain_name in enumerate(domain_names):
        pred_force_denorm.append(denorm_force(dataset, domain_name, pred_force_norm[row : row + 1]).cpu())
        gt_force_denorm.append(denorm_force(dataset, domain_name, gt_force_norm[row : row + 1]).cpu())
    pred_force_denorm = torch.cat(pred_force_denorm, dim=0).numpy()
    gt_force_denorm = torch.cat(gt_force_denorm, dim=0).numpy()

    step_prob = torch.sigmoid(head_out["step_logits"]).detach().cpu().float().numpy()
    if step_prob.shape[1] != action_chunk:
        x_old = np.linspace(0.0, 1.0, step_prob.shape[1])
        x_new = np.linspace(0.0, 1.0, action_chunk)
        step_prob = np.stack([np.interp(x_new, x_old, row) for row in step_prob], axis=0)
    window_prob = torch.sigmoid(head_out["window_logit"]).detach().cpu().float().reshape(-1).numpy()
    risk_prob = head_out["risk_prob"].detach().cpu().float().reshape(-1).numpy()
    micro_prob = None
    if "micro_window_logits" in head_out:
        micro_prob = torch.sigmoid(head_out["micro_window_logits"]).detach().cpu().float().numpy()

    frame_label = batch.get("frame_anomaly_label")
    if frame_label is not None:
        frame_label = frame_label.detach().cpu().float().numpy()
    risk_target = batch.get("risk_label")
    if risk_target is not None:
        risk_target = risk_target.detach().cpu().float().reshape(-1).numpy()
    label = batch.get("label")
    if label is not None:
        label = label.detach().cpu().float().reshape(-1).numpy()
    interaction_label = batch.get("interaction_label")
    if interaction_label is not None:
        interaction_label = interaction_label.detach().cpu().long().reshape(-1).numpy()

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
                "pred_force_denorm": pred_force_denorm[row : row + 1, :valid_len],
                "gt_force_denorm": gt_force_denorm[row : row + 1, :valid_len],
                "pred_force_norm": pred_force_norm[row : row + 1, :valid_len].numpy(),
                "gt_force_norm": gt_force_norm[row : row + 1, :valid_len].numpy(),
                "step_prob": step_prob[row, :valid_len],
                "window_prob": float(window_prob[row]),
                "risk_prob": float(risk_prob[row]),
                "micro_prob": None if micro_prob is None else micro_prob[row],
                "frame_target": frame_target,
                "risk_target": float(risk_target[row]) if risk_target is not None else None,
                "window_target": float(1.0 - label[row]) if label is not None else None,
                "interaction_label": int(interaction_label[row]) if interaction_label is not None else 0,
                "video": None,
            }
        )
    return chunks


def build_episode_record(cfg, dataset, dataset_index, chunks, force_dim, window_stride):
    item = dataset.dataset[int(dataset_index)]
    total_frames = int(item[7])
    chunks = sorted(chunks, key=lambda chunk: int(chunk["fix_sidx"]))
    if not chunks:
        raise RuntimeError(f"No windows generated for dataset_index={dataset_index}")

    global_start = min(int(chunk["fix_sidx"]) for chunk in chunks)
    aggregate = aggregate_anomaly_chunks(
        chunks,
        total_frames=total_frames,
        force_dim=force_dim,
        global_start=global_start,
    )
    parquet_path = item[2]
    _, frame_anomaly = load_frame_labels(parquet_path, total_frames)
    frame_target = aggregate["frame_target"]
    if frame_target is None:
        frame_target = frame_anomaly[global_start:]
    length = min(aggregate["step_prob"].shape[0], frame_target.shape[0], aggregate["pred_force"].shape[0])
    step_scores = aggregate["step_prob"][:length]
    step_targets = (frame_target[:length] >= 0.5).astype(np.int64)
    window_scores = aggregate["window_probs"]
    window_targets = aggregate["window_targets"].astype(np.int64)
    step_interaction_labels = load_interaction_labels(parquet_path, total_frames)[global_start : global_start + length].astype(np.int64)
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
        "evaluation_start_frame": int(global_start),
        "failure_transition_frame": transition,
        "has_anomaly": bool(step_targets.any()),
        "window_starts": [int(c["fix_sidx"]) for c in chunks],
        "window_stride": int(window_stride),
        "force_metrics": force_metrics(
            aggregate["pred_force"][:length],
            aggregate["gt_force"][:length],
            labels=step_targets,
        ),
        "step_metrics": basic_binary_metrics(step_scores, step_targets),
        "window_metrics": basic_binary_metrics(window_scores, window_targets) if window_scores.size else None,
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
        "_force_pred": aggregate["pred_force"][:length],
        "_force_gt": aggregate["gt_force"][:length],
    }


def _binary_metrics_at_threshold(scores, targets, threshold):
    base = basic_binary_metrics(scores, targets)
    if base is None:
        return None
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid]
    threshold = float(threshold)
    pred = scores >= threshold
    pos = targets == 1
    neg = ~pos
    tp = int(np.logical_and(pred, pos).sum())
    fp = int(np.logical_and(pred, neg).sum())
    tn = int(np.logical_and(~pred, neg).sum())
    fn = int(np.logical_and(~pred, pos).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    base.update(
        {
            "accuracy": float((tp + tn) / max(1, scores.size)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "threshold": threshold,
            "threshold_mode": "manual",
            "pred_positive_ratio": float(pred.mean()) if pred.size else 0.0,
        }
    )
    return base


def _best_f1_metrics(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid]
    if scores.size == 0:
        return None
    candidates = np.unique(np.concatenate([scores, np.asarray([0.0, 0.5, 1.0], dtype=np.float64)]))
    candidates = np.clip(candidates, 0.0, 1.0)
    best_metrics = None
    for threshold in candidates:
        metrics = _binary_metrics_at_threshold(scores, targets, float(threshold))
        if best_metrics is None:
            best_metrics = metrics
            continue
        better = (
            metrics["f1"] > best_metrics["f1"] + 1e-12
            or (
                abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12
                and metrics["accuracy"] > best_metrics["accuracy"] + 1e-12
            )
            or (
                abs(metrics["f1"] - best_metrics["f1"]) <= 1e-12
                and abs(metrics["accuracy"] - best_metrics["accuracy"]) <= 1e-12
                and abs(metrics["threshold"] - 0.5) < abs(best_metrics["threshold"] - 0.5)
            )
        )
        if better:
            best_metrics = metrics
    best_metrics["threshold_mode"] = "best_f1"
    return best_metrics


def _metrics_for_mask(scores, targets, mask, threshold, threshold_mode):
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    mask = np.asarray(mask, dtype=bool)
    if scores.shape[0] != mask.shape[0] or targets.shape[0] != mask.shape[0]:
        raise ValueError(
            f"Mask length {mask.shape[0]} must match scores/targets lengths {scores.shape[0]}/{targets.shape[0]}"
        )
    if not bool(mask.any()):
        return None
    metrics = _binary_metrics_at_threshold(scores[mask], targets[mask], threshold)
    if metrics is not None:
        metrics["threshold_mode"] = threshold_mode
    return metrics


def _interaction_group_metrics(
    step_scores,
    step_targets,
    step_interaction_labels,
    window_scores,
    window_targets,
    window_interaction_labels,
    step_best_threshold,
    window_best_threshold,
    manual_threshold,
):
    groups = {}
    for interaction_value in (0, 1):
        step_mask = np.asarray(step_interaction_labels) == interaction_value
        group = {
            "step_metrics": _metrics_for_mask(
                step_scores,
                step_targets,
                step_mask,
                step_best_threshold,
                "best_f1_global",
            ),
            "step_metrics_manual": _metrics_for_mask(
                step_scores,
                step_targets,
                step_mask,
                manual_threshold,
                "manual",
            ),
        }
        if window_scores is not None and np.asarray(window_scores).size:
            window_mask = np.asarray(window_interaction_labels) == interaction_value
            group["window_metrics"] = _metrics_for_mask(
                window_scores,
                window_targets,
                window_mask,
                window_best_threshold,
                "best_f1_global",
            )
            group["window_metrics_manual"] = _metrics_for_mask(
                window_scores,
                window_targets,
                window_mask,
                manual_threshold,
                "manual",
            )
        else:
            group["window_metrics"] = None
            group["window_metrics_manual"] = None
        group["num_step_samples"] = int(step_mask.sum())
        window_interactions = np.asarray(window_interaction_labels)
        group["num_window_samples"] = int((window_interactions == interaction_value).sum())
        groups[str(interaction_value)] = group
    return groups


def aggregate_records(records, threshold=0.5):
    step_scores = np.concatenate([r["_step_scores"] for r in records])
    step_targets = np.concatenate([r["_step_targets"] for r in records])
    step_interaction_labels = np.concatenate([r["_step_interaction_labels"] for r in records])
    window_scores = [r["_window_scores"] for r in records if r["_window_scores"].size]
    window_targets = [r["_window_targets"] for r in records if r["_window_targets"].size]
    window_interaction_labels = [r["_window_interaction_labels"] for r in records if r["_window_scores"].size]
    force_pred = np.concatenate([r["_force_pred"] for r in records], axis=0)
    force_gt = np.concatenate([r["_force_gt"] for r in records], axis=0)
    step_metrics = _best_f1_metrics(step_scores, step_targets)
    step_metrics_manual = _binary_metrics_at_threshold(step_scores, step_targets, threshold)
    aggregate = {
        "step_metrics": step_metrics,
        "step_metrics_manual": step_metrics_manual,
        "force_metrics": force_metrics(force_pred, force_gt, labels=step_targets),
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


def _confusion_matrix_from_metrics(metrics):
    if metrics is None:
        return None
    return np.asarray(
        [
            [int(metrics.get("tn", 0)), int(metrics.get("fp", 0))],
            [int(metrics.get("fn", 0)), int(metrics.get("tp", 0))],
        ],
        dtype=np.int64,
    )


def _format_metric_value(value):
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _plot_single_confusion_matrix(ax, matrix, metrics, title):
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_xticks([0, 1], labels=["normal", "anomaly"])
    ax.set_yticks([0, 1], labels=["normal", "anomaly"])

    if matrix is None:
        ax.set_title(title)
        ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=14)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(1.5, -0.5)
        return

    mode = str(metrics.get("threshold_mode", "manual"))
    threshold = metrics.get("threshold")
    ax.set_title(f"{title}\n{mode}, threshold={_format_metric_value(threshold)}")
    image = ax.imshow(matrix, cmap="Blues")
    max_value = max(1, int(matrix.max()))
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_pct = np.divide(matrix, np.maximum(row_sums, 1), dtype=np.float64)
    for row in range(2):
        for col in range(2):
            color = "white" if matrix[row, col] > 0.55 * max_value else "black"
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n{row_pct[row, col] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=12,
            )
    subtitle = (
        f"acc={_format_metric_value(metrics.get('accuracy'))}  "
        f"P={_format_metric_value(metrics.get('precision'))}  "
        f"R={_format_metric_value(metrics.get('recall'))}  "
        f"F1={_format_metric_value(metrics.get('f1'))}"
    )
    ax.text(0.5, -0.25, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=10)
    return image


def plot_confusion_summary(
    output_path,
    aggregate,
    title,
    step_metrics_key="step_metrics",
    window_metrics_key="window_metrics",
):
    step_metrics = aggregate.get(step_metrics_key)
    window_metrics = aggregate.get(window_metrics_key)
    matrices = [
        (_confusion_matrix_from_metrics(step_metrics), step_metrics, "Step-level"),
        (_confusion_matrix_from_metrics(window_metrics), window_metrics, "Window-level"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), constrained_layout=True)
    images = []
    for ax, (matrix, metrics, title) in zip(axes, matrices):
        image = _plot_single_confusion_matrix(ax, matrix, metrics, title)
        if image is not None:
            images.append(image)
    if images:
        fig.colorbar(images[0], ax=axes, fraction=0.035, pad=0.02)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_interaction_confusion_summary(
    output_path,
    aggregate,
    title,
    step_metrics_key="step_metrics",
    window_metrics_key="window_metrics",
):
    interaction_metrics = aggregate.get("interaction_metrics", {})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    images = []
    for row, interaction_value in enumerate(("0", "1")):
        group = interaction_metrics.get(interaction_value, {})
        plot_items = [
            (group.get(step_metrics_key), f"interaction={interaction_value} Step-level"),
            (group.get(window_metrics_key), f"interaction={interaction_value} Window-level"),
        ]
        for col, (metrics, panel_title) in enumerate(plot_items):
            matrix = _confusion_matrix_from_metrics(metrics)
            image = _plot_single_confusion_matrix(axes[row, col], matrix, metrics, panel_title)
            if image is not None:
                images.append(image)
    if images:
        fig.colorbar(images[0], ax=axes, fraction=0.035, pad=0.02)
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def public_records(records):
    output = []
    for record in records:
        converted = {}
        for key, value in record.items():
            if key.startswith("_"):
                continue
            if isinstance(value, np.ndarray):
                converted[key] = value.tolist()
            elif isinstance(value, np.generic):
                converted[key] = value.item()
            else:
                converted[key] = value
        output.append(converted)
    return output


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
        "_force_pred",
        "_force_gt",
    }
    for record in payload:
        restored = dict(record)
        for key in array_keys:
            if key in restored:
                dtype = np.int64 if key.endswith("targets") or key.endswith("labels") else np.float64
                restored[key] = np.asarray(restored[key], dtype=dtype)
        records.append(restored)
    return records


FORCE_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def load_anomaly_head(config, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    head_cfg = dict(ckpt.get("anomaly_head", config.get("anomaly_head", {})))
    diffusion_cfg = config["diffusion_model"]["config"]
    inner_dim = int(diffusion_cfg["num_attention_heads"]) * int(diffusion_cfg["attention_head_dim"])
    data_cfg = config["data"]["train"]
    action_dim = int(data_cfg.get("action_dim", diffusion_cfg.get("action_condition_dim", diffusion_cfg.get("action_dim"))))
    force_dim = int(data_cfg.get("force_dim", diffusion_cfg.get("force_dim", diffusion_cfg.get("force_out_channels", 6))))
    state_dict = ckpt["state_dict"]
    if "anomaly_head_class" in ckpt:
        head_class_name = str(ckpt["anomaly_head_class"])
    elif any(str(key).startswith(("window_head.", "risk_head.")) for key in state_dict):
        head_class_name = str(config.get("anomaly_head_class", "ActionForceAnomalyHead_v1"))
    else:
        head_class_name = "ActionForceAnomalyHead"
    head_class_path = str(
        ckpt.get(
            "anomaly_head_class_path",
            config.get("anomaly_head_class_path", "models/anomaly_head/action_force_anomaly.py"),
        )
    )
    head_class = import_custom_class(head_class_name, head_class_path)
    head_kwargs = {
        "action_dim": action_dim,
        "force_dim": force_dim,
        "interaction_dim": int(head_cfg.get("interaction_dim", inner_dim)),
        "video_feature_dim": int(head_cfg.get("video_feature_dim", inner_dim)),
        "hidden_dim": int(head_cfg.get("hidden_dim", 256)),
        "feature_dim": int(head_cfg.get("feature_dim", 128)),
        "dropout": float(head_cfg.get("dropout", 0.1)),
        "use_video_features": bool(head_cfg.get("use_video_features", False)),
        "latent_channels": int(head_cfg.get("latent_channels", diffusion_cfg.get("out_channels", 128))),
        "temporal_layers": int(head_cfg.get("temporal_layers", 2)),
        "temporal_heads": int(head_cfg.get("temporal_heads", 4)),
        "max_steps": int(head_cfg.get("max_steps", 256)),
        "topk_ratio": float(head_cfg.get("topk_ratio", 0.2)),
        "history_context_scale": float(head_cfg.get("history_context_scale", 0.25)),
    }
    if head_class_name.endswith("_v1"):
        head_kwargs.update(
            {
                "window_pool": str(head_cfg.get("window_pool", "topk")),
                "risk_pool": str(head_cfg.get("risk_pool", "mean")),
            }
        )
    if head_class_name == "ActionForceVisionResidualAnomalyHead":
        head_kwargs.update(
            {
                "window_pool": str(head_cfg.get("window_pool", "topk")),
                "risk_pool": str(head_cfg.get("risk_pool", "mean")),
                "micro_window_size": int(head_cfg.get("micro_window_size", 10)),
                "causal_residual": bool(head_cfg.get("causal_residual", True)),
            }
        )
    head = head_class(**head_kwargs)
    head.load_state_dict(state_dict, strict=True)
    return head.to(device).eval()


def encode_gt_latents(vae, future_video, batch_size, n_view, dtype):
    image_for_vae = rearrange(future_video, "bv c t h w -> (bv t) c h w").unsqueeze(2)
    gt_latents = retrieve_latents(vae.encode(image_for_vae), None, sample_mode="argmax")
    if hasattr(vae, "latents_mean") and hasattr(vae, "latents_std"):
        latents_mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(gt_latents.device, gt_latents.dtype)
        latents_std = vae.latents_std.view(1, -1, 1, 1, 1).to(gt_latents.device, gt_latents.dtype)
        gt_latents = (gt_latents - latents_mean) / latents_std
    gt_latents = rearrange(gt_latents, "(b v t) c f h w -> b v c (t f) h w", b=batch_size, v=n_view)
    return gt_latents.to(dtype=dtype).contiguous()


def run_anomaly_window(
    cfg,
    dataset,
    pipe,
    anomaly_head,
    vae,
    args,
    device,
    weight_dtype,
    n_previous,
    action_condition_dim,
    force_dim,
    fix_sidx,
    mem_mode,
    total_frames,
    return_video=False,
):
    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    valid_len = min(action_chunk, max(0, int(total_frames) - int(fix_sidx)))
    if valid_len <= 0:
        return None

    mem_idx = build_mem_idx(int(fix_sidx), n_previous, mem_mode)
    dataset.fix_sidx = int(fix_sidx)
    dataset.fix_mem_idx = mem_idx
    sample = dataset[0]
    batch = sample_to_batch(sample)
    domain_name = get_sample_domain_name(dataset, cfg)

    image = batch["video"][:, :, :, :n_previous].to(device=device, dtype=weight_dtype)
    future_video = batch["video"][:, :, :, n_previous:].to(device=device, dtype=weight_dtype)
    batch_size, _, n_view, _, height, width = image.shape
    image = rearrange(image, "b c v t h w -> (b v) c t h w")
    future_video = rearrange(future_video, "b c v t h w -> (b v) c t h w")
    future_actions = batch["actions"][:, -action_chunk:].to(device=device, dtype=weight_dtype).contiguous()
    if future_actions.shape[-1] != action_condition_dim:
        raise ValueError(f"Expected action dim {action_condition_dim}, got {future_actions.shape[-1]}")
    history_force_state = batch.get("history_force_state")
    if history_force_state is None:
        raise KeyError("Dataset sample must contain history_force_state.")
    history_force_state = history_force_state.to(device=device, dtype=weight_dtype).contiguous()
    prompt = batch["caption"][:batch_size]
    latent_chunk = int(cfg["data"]["train"]["chunk"])


    if hasattr(pipe.scheduler, "_step_index"):
        pipe.scheduler._step_index = None
    with torch.no_grad():
        preds = pipe.infer(
            image=image,
            prompt=prompt,
            negative_prompt="",
            num_inference_steps=int(cfg["num_inference_step"]),
            decode_timestep=0.03,
            decode_noise_scale=0.025,
            guidance_scale=float(getattr(args, "guidance_scale", 1.0)),
            height=height,
            width=width,
            n_view=n_view,
            return_force=True,
            return_interaction=True,
            return_latents=True,
            n_prev=n_previous,
            chunk=latent_chunk,
            return_video=return_video,
            return_init=False,
            noise_seed=getattr(args, "noise_seed", None),
            action_chunk=action_chunk,
            history_force_state=history_force_state,
            pixel_wise_timestep=bool(cfg["pixel_wise_timestep"]),
            n_chunk=1,
            force_dim=force_dim,
            future_actions=future_actions,
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
        head_kwargs = {}
        if bool(getattr(anomaly_head, "requires_observed_inputs", False)):
            head_kwargs["gt_force"] = batch["force_targets"][:, -action_chunk:].to(
                device=device,
                dtype=weight_dtype,
            ).contiguous()
            head_kwargs["gt_latents"] = encode_gt_latents(
                vae,
                future_video,
                batch_size,
                n_view,
                weight_dtype,
            )
        head_out = anomaly_head(
            pred_force=preds["force"],
            history_force=history_force_state,
            interaction_tokens=preds["interaction_tokens"],
            pred_latents=pred_latents,
            **head_kwargs,
        )

    pred_force_norm = preds["force"].detach().cpu().float()[:, :valid_len]
    gt_force_norm = batch["force_targets"][:, -action_chunk:].detach().cpu().float()[:, :valid_len]
    pred_force_denorm = denorm_force(dataset, domain_name, pred_force_norm).cpu().numpy()
    gt_force_denorm = denorm_force(dataset, domain_name, gt_force_norm).cpu().numpy()

    step_prob = torch.sigmoid(head_out["step_logits"]).detach().cpu().float().numpy()[0]
    if step_prob.shape[0] != action_chunk:
        x_old = np.linspace(0.0, 1.0, step_prob.shape[0])
        x_new = np.linspace(0.0, 1.0, action_chunk)
        step_prob = np.interp(x_new, x_old, step_prob)
    step_prob = step_prob[:valid_len]
    window_prob = float(torch.sigmoid(head_out["window_logit"]).detach().cpu().float().reshape(-1)[0])
    risk_prob = float(head_out["risk_prob"].detach().cpu().float().reshape(-1)[0])
    micro_prob = None
    if "micro_window_logits" in head_out:
        micro_prob = torch.sigmoid(head_out["micro_window_logits"]).detach().cpu().float().numpy()[0]

    frame_label = batch.get("frame_anomaly_label")
    frame_target = None
    if frame_label is not None:
        frame_target = frame_label.detach().cpu().float().numpy()[0, :valid_len]
        if np.any(frame_target < 0):
            frame_target = None
    risk_target = batch.get("risk_label")
    risk_target = float(risk_target.detach().cpu().float().reshape(-1)[0]) if risk_target is not None else None
    label = batch.get("label")
    window_target = None
    if label is not None:
        window_target = float(1.0 - label.detach().cpu().float().reshape(-1)[0])
    anomaly_frame_prob = None
    if "anomaly_frame_prob" in batch:
        anomaly_frame_prob = batch["anomaly_frame_prob"].detach().cpu().float().numpy()[0, :valid_len]
    anomaly_window_prob = None
    if "anomaly_window_prob" in batch:
        anomaly_window_prob = batch["anomaly_window_prob"].detach().cpu().float().numpy()[0, :valid_len]

    video = preds.get("video")
    if video is not None:
        video = video.detach().cpu().float()

    return {
        "fix_sidx": int(fix_sidx),
        "valid_len": int(valid_len),
        "mem_idx": mem_idx,
        "domain_name": domain_name,
        "prompt": prompt[0],
        "pred_force_denorm": pred_force_denorm,
        "gt_force_denorm": gt_force_denorm,
        "pred_force_norm": pred_force_norm.numpy(),
        "gt_force_norm": gt_force_norm.numpy(),
        "step_prob": step_prob,
        "window_prob": window_prob,
        "risk_prob": risk_prob,
        "micro_prob": micro_prob,
        "frame_target": frame_target,
        "risk_target": risk_target,
        "window_target": window_target,
        "anomaly_frame_prob": anomaly_frame_prob,
        "anomaly_window_prob": anomaly_window_prob,
        "video": video,
    }


def aggregate_anomaly_chunks(chunks, total_frames, force_dim, global_start=0):
    length = total_frames - int(global_start)
    pred_sum = np.zeros((length, force_dim), dtype=np.float64)
    gt_sum = np.zeros((length, force_dim), dtype=np.float64)
    pred_square_sum = np.zeros((length, force_dim), dtype=np.float64)
    force_count = np.zeros(length, dtype=np.int64)
    prob_sum = np.zeros(length, dtype=np.float64)
    prob_count = np.zeros(length, dtype=np.int64)
    target_sum = np.zeros(length, dtype=np.float64)
    target_count = np.zeros(length, dtype=np.int64)
    anomaly_extra_sum = {
        "anomaly_frame_prob": np.zeros(length, dtype=np.float64),
        "anomaly_window_prob": np.zeros(length, dtype=np.float64),
    }
    anomaly_extra_count = {
        "anomaly_frame_prob": np.zeros(length, dtype=np.int64),
        "anomaly_window_prob": np.zeros(length, dtype=np.int64),
    }
    window_probs = []
    window_targets = []

    for chunk in chunks:
        start = int(chunk["fix_sidx"]) - int(global_start)
        valid_len = min(int(chunk["valid_len"]), length - start)
        if valid_len <= 0:
            continue
        end = start + valid_len
        pred = chunk["pred_force_denorm"][0, :valid_len].astype(np.float64)
        gt = chunk["gt_force_denorm"][0, :valid_len].astype(np.float64)
        pred_sum[start:end] += pred
        pred_square_sum[start:end] += pred**2
        gt_sum[start:end] += gt
        force_count[start:end] += 1
        prob_sum[start:end] += chunk["step_prob"][:valid_len]
        prob_count[start:end] += 1
        if chunk.get("frame_target") is not None:
            target_sum[start:end] += chunk["frame_target"][:valid_len]
            target_count[start:end] += 1
        for key in ("anomaly_frame_prob", "anomaly_window_prob"):
            values = chunk.get(key)
            if values is None:
                continue
            values = np.asarray(values[:valid_len], dtype=np.float64)
            valid = np.isfinite(values)
            if np.any(valid):
                valid_positions = np.arange(start, end, dtype=np.int64)[valid]
                anomaly_extra_sum[key][valid_positions] += values[valid]
                anomaly_extra_count[key][valid_positions] += 1
        if chunk.get("window_target") is not None:
            window_probs.append(float(chunk["window_prob"]))
            window_targets.append(float(chunk["window_target"]))

    if np.any(force_count == 0):
        missing = np.flatnonzero(force_count == 0)
        raise RuntimeError(f"Uncovered force timesteps: {missing[:20].tolist()}")
    if np.any(prob_count == 0):
        missing = np.flatnonzero(prob_count == 0)
        raise RuntimeError(f"Uncovered anomaly timesteps: {missing[:20].tolist()}")

    pred_mean = pred_sum / force_count[:, None]
    gt_mean = gt_sum / force_count[:, None]
    pred_std = np.sqrt(np.maximum(pred_square_sum / force_count[:, None] - pred_mean**2, 0.0))
    step_prob = prob_sum / prob_count
    frame_target = None
    if np.any(target_count > 0):
        frame_target = np.zeros(length, dtype=np.float64)
        valid = target_count > 0
        frame_target[valid] = target_sum[valid] / target_count[valid]
        frame_target[~valid] = np.nan
    anomaly_extra = {}
    for key in ("anomaly_frame_prob", "anomaly_window_prob"):
        counts = anomaly_extra_count[key]
        if np.any(counts > 0):
            values = np.full(length, np.nan, dtype=np.float64)
            valid = counts > 0
            values[valid] = anomaly_extra_sum[key][valid] / counts[valid]
            anomaly_extra[key] = values
        else:
            anomaly_extra[key] = None
    return {
        "pred_force": pred_mean,
        "gt_force": gt_mean,
        "pred_std": pred_std,
        "prediction_count": force_count,
        "step_prob": step_prob,
        "frame_target": frame_target,
        "window_probs": np.asarray(window_probs, dtype=np.float64),
        "window_targets": np.asarray(window_targets, dtype=np.float64),
        **anomaly_extra,
    }


def save_force_label_plot(
    path,
    pred_force,
    gt_force,
    step_prob,
    frame_target=None,
    anomaly_frame_prob=None,
    anomaly_window_prob=None,
    transition_frame=None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = min(pred_force.shape[0], gt_force.shape[0], step_prob.shape[0])
    x = np.arange(length)
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
    axes = axes.reshape(-1)
    for dim in range(6):
        ax = axes[dim]
        ax.plot(x, pred_force[:length, dim], color="tomato", linewidth=1.6, label="pred force")
        ax.plot(x, gt_force[:length, dim], color="cornflowerblue", linewidth=1.4, label="gt force")
        if transition_frame is not None and 0 <= transition_frame < length:
            ax.axvline(transition_frame, color="green", linestyle="--", linewidth=1.2)
        ax.set_title(FORCE_LABELS[dim])
        ax.grid(True, linestyle=":", alpha=0.5)
    ax = axes[6]
    ax.plot(x, step_prob[:length], color="tomato", linewidth=1.8, label="pred anomaly prob")
    if frame_target is not None:
        ax.plot(x, frame_target[:length], color="cornflowerblue", linewidth=1.8, drawstyle="steps-post", label="gt anomaly")
    if transition_frame is not None and 0 <= transition_frame < length:
        ax.axvline(transition_frame, color="green", linestyle="--", linewidth=1.2, label="failure onset")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Anomaly")
    ax.grid(True, linestyle=":", alpha=0.5)
    for extra_ax, values, title in (
        (axes[7], anomaly_frame_prob, "Dataset frame prob"),
        (axes[8], anomaly_window_prob, "Dataset window prob"),
    ):
        values = None if values is None else np.asarray(values[:length], dtype=np.float64)
        if values is None or not np.isfinite(values).any():
            extra_ax.axis("off")
            continue
        extra_ax.plot(x, values, color="darkorange", linewidth=1.7, label=title.lower())
        if frame_target is not None:
            extra_ax.plot(
                x,
                frame_target[:length],
                color="cornflowerblue",
                linewidth=1.4,
                drawstyle="steps-post",
                label="gt anomaly",
            )
        if transition_frame is not None and 0 <= transition_frame < length:
            extra_ax.axvline(transition_frame, color="green", linestyle="--", linewidth=1.2)
        extra_ax.set_ylim(-0.05, 1.05)
        extra_ax.set_title(title)
        extra_ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = [], []
    for legend_ax in axes:
        axis_handles, axis_labels = legend_ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.supxlabel("trajectory timestep", y=0.045)
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def save_force_diff_plot(path, pred_force, gt_force, labels=None, transition_frame=None, window_size=50):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = min(pred_force.shape[0], gt_force.shape[0])
    diff = np.abs(pred_force[:length] - gt_force[:length])
    x = np.arange(length)
    window_mean = np.empty_like(diff)
    for start in range(0, length, int(window_size)):
        end = min(start + int(window_size), length)
        window_mean[start:end] = diff[start:end].mean(axis=0, keepdims=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.reshape(-1)
    for dim, ax in enumerate(axes):
        ax.plot(x, diff[:, dim], color="darkorange", linewidth=1.4, label="abs error")
        ax.plot(x, window_mean[:, dim], color="gray", linestyle="--", linewidth=1.6, label="window mean" if dim == 0 else None)
        if labels is not None:
            anomaly = np.asarray(labels[:length]) > 0.5
            if anomaly.any():
                ax.fill_between(x, 0, max(float(diff[:, dim].max()), 1e-6), where=anomaly, color="red", alpha=0.08, label="gt anomaly" if dim == 0 else None)
        if transition_frame is not None and 0 <= transition_frame < length:
            ax.axvline(transition_frame, color="green", linestyle="--", linewidth=1.2)
        ax.set_title(FORCE_LABELS[dim])
        ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels_ = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels_,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.supxlabel("trajectory timestep", y=0.045)
    fig.supylabel("absolute force error")
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def build_action_force_anomaly_video(
    output_path,
    multiview_frames,
    pred_multiview_frames,
    pred_force,
    gt_force,
    step_prob,
    frame_target=None,
    anomaly_frame_prob=None,
    anomaly_window_prob=None,
    fps=30,
    output_width=1280,
    plot_height=620,
    transition_frame=None,
):
    return build_anomaly_alignment_video(
        output_path=output_path,
        multiview_frames=multiview_frames,
        pred_multiview_frames=pred_multiview_frames,
        score=step_prob,
        score_label="pred anomaly prob",
        frame_target=frame_target,
        anomaly_frame_prob=anomaly_frame_prob,
        anomaly_window_prob=anomaly_window_prob,
        pred_force=pred_force,
        gt_force=gt_force,
        fps=fps,
        output_width=output_width,
        plot_height=plot_height,
        transition_frame=transition_frame,
    )


def build_action_force_video(
    output_path,
    multiview_frames,
    pred_multiview_frames,
    pred_force,
    gt_force,
    fps=30,
    output_width=1280,
    plot_height=480,
    transition_frame=None,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = min(multiview_frames.shape[0], pred_force.shape[0], gt_force.shape[0])
    if pred_multiview_frames is not None:
        length = min(length, pred_multiview_frames.shape[0])
    multiview_frames = multiview_frames[:length]
    pred_multiview_frames = None if pred_multiview_frames is None else pred_multiview_frames[:length]
    pred_force = pred_force[:length]
    gt_force = gt_force[:length]

    h, w = multiview_frames.shape[2], multiview_frames.shape[3]
    source_top_width = int(multiview_frames.shape[1] * w)
    output_width = int(output_width) - int(output_width) % 2
    plot_height = int(plot_height) - int(plot_height) % 2
    top_height = max(2, int(round(h * output_width / source_top_width)))
    top_height -= top_height % 2
    video_rows = 2 if pred_multiview_frames is not None else 1
    output_height = top_height * video_rows + plot_height

    fig = plt.figure(figsize=(output_width / 120.0, plot_height / 120.0), dpi=120)
    axes = fig.subplots(2, 3, sharex=True)
    axes = axes.reshape(-1)
    x = np.arange(length)
    vlines = []
    for dim, ax in enumerate(axes):
        ax.plot(x, pred_force[:, dim], color="tomato", linewidth=1.5, label="pred")
        ax.plot(x, gt_force[:, dim], color="cornflowerblue", linewidth=1.3, label="gt")
        if transition_frame is not None and 0 <= transition_frame < length:
            ax.axvline(transition_frame, color="green", linestyle="--", linewidth=1.2)
        vlines.append(ax.axvline(0, color="#444444", linestyle="--", linewidth=1.0))
        ax.set_title(FORCE_LABELS[dim], fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.45)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{output_width}x{output_height}", "-r", str(float(fps)), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", output_path,
    ]
    writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if writer.stdin is None:
        plt.close(fig)
        raise RuntimeError(f"Failed to open ffmpeg stdin for {output_path}")
    try:
        for i in range(length):
            gt_row = np.concatenate([multiview_frames[i, view_idx] for view_idx in range(multiview_frames.shape[1])], axis=1)
            if gt_row.shape[1] != output_width or gt_row.shape[0] != top_height:
                gt_row = cv2.resize(gt_row, (output_width, top_height), interpolation=cv2.INTER_LINEAR)
            cv2.putText(gt_row, "GT video", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            if pred_multiview_frames is not None:
                pred_row = np.concatenate([pred_multiview_frames[i, view_idx] for view_idx in range(pred_multiview_frames.shape[1])], axis=1)
                if pred_row.shape[1] != output_width or pred_row.shape[0] != top_height:
                    pred_row = cv2.resize(pred_row, (output_width, top_height), interpolation=cv2.INTER_LINEAR)
                cv2.putText(pred_row, "Pred video", (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                top_row = np.concatenate([gt_row, pred_row], axis=0)
            else:
                top_row = gt_row
            for vline in vlines:
                vline.set_xdata([x[i], x[i]])
            fig.canvas.draw()
            plot_rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
            if plot_rgb.shape[1] != output_width or plot_rgb.shape[0] != plot_height:
                plot_rgb = cv2.resize(plot_rgb, (output_width, plot_height), interpolation=cv2.INTER_LINEAR)
            canvas = np.ascontiguousarray(np.concatenate([top_row, plot_rgb], axis=0))
            writer.stdin.write(canvas.tobytes())
    finally:
        writer.stdin.close()
        stderr = writer.stderr.read() if writer.stderr is not None else b""
        return_code = writer.wait()
        plt.close(fig)
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr.decode(errors='replace')}")
    return output_path


def force_metrics(pred_force, gt_force, labels=None):
    length = min(pred_force.shape[0], gt_force.shape[0])
    diff = np.abs(pred_force[:length] - gt_force[:length])
    out = {
        "mae": float(diff.mean()),
        "mae_per_dim": diff.mean(axis=0).tolist(),
        "num_frames": int(length),
    }
    if labels is not None:
        labels = np.asarray(labels[:length])
        normal = labels < 0.5
        anomaly = labels >= 0.5
        if normal.any():
            out["mae_normal"] = float(diff[normal].mean())
        if anomaly.any():
            out["mae_anomaly"] = float(diff[anomaly].mean())
    return out

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {args.batch_size}")
    if args.threshold is not None and not 0.0 <= float(args.threshold) <= 1.0:
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")

    shard_rank, num_shards = resolve_shard(args)
    run_id = resolve_run_id(args)
    device = resolve_device(args.device)
    weight_dtype = str_to_dtype(args.dtype)

    cfg = load_resolved_config(args.config_file)
    if args.checkpoint_path is not None:
        cfg["load_weights"] = True
        cfg["load_diffusion_model_weights"] = True
        cfg["diffusion_model"]["model_path"] = resolve_checkpoint_path(args.checkpoint_path)
    if args.num_inference_steps is not None:
        cfg["num_inference_step"] = int(args.num_inference_steps)
    if args.split_manifest in (None, ""):
        args.split_manifest = cfg.get("split_manifest")

    action_condition_dim, force_dim = resolve_force_layout(cfg)
    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    window_stride = int(args.window_stride or action_chunk)

    cache_dir = args.anomaly_feature_cache_dir or cfg.get("anomaly_feature_cache_dir")
    cache_metadata = None
    if cache_dir:
        if shard_rank == 0:
            print(f"Evaluating action-force anomaly head from cache: {cache_dir}")
        records, num_windows, selected_indices, cache_metadata = evaluate_cached_records(
            cfg=cfg,
            args=args,
            device=device,
            force_dim=force_dim,
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
        anomaly_head = load_anomaly_head(cfg, args.anomaly_checkpoint, device)
        total_params = count_model_parameters(diffusion_model)
        if shard_rank == 0:
            print(
                f"rank={shard_rank}/{num_shards} device={device} "
                f"params total={total_params[0]} trainable={total_params[1]}"
            )

        n_previous = int(cfg["data"]["train"]["n_previous"])
        windows = build_windows(cfg, dataset, shard_indices, window_stride)
        chunks_by_episode = {int(dataset_index): [] for dataset_index in shard_indices}
        total_batches = (len(windows) + int(args.batch_size) - 1) // int(args.batch_size)
        for start in range(0, len(windows), int(args.batch_size)):
            batch_windows = windows[start : start + int(args.batch_size)]
            if shard_rank == 0:
                print(
                    f"rank={shard_rank} batch={start // int(args.batch_size) + 1}/{total_batches} "
                    f"windows={len(batch_windows)}"
                )
            batch_chunks = infer_anomaly_batch(
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
            for window, chunk in zip(batch_windows, batch_chunks):
                chunks_by_episode[int(window["dataset_index"])].append(chunk)

        records = [
            build_episode_record(
                cfg=cfg,
                dataset=dataset,
                dataset_index=dataset_index,
                chunks=chunks_by_episode[dataset_index],
                force_dim=force_dim,
                window_stride=window_stride,
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
    partial_path = os.path.join(args.output_dir, f"partial_{run_id}_rank{shard_rank:03d}.{partial_ext}")
    if args.partial_format == "pt":
        torch.save(partial_payload, partial_path)
    else:
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(partial_payload, f, indent=2)

    if shard_rank != 0:
        return

    partial_paths = [
        os.path.join(args.output_dir, f"partial_{run_id}_rank{rank:03d}.{partial_ext}")
        for rank in range(num_shards)
    ]
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
        if args.partial_format == "pt":
            all_records.extend(payload_part.get("records", []))
        else:
            all_records.extend(records_from_jsonable(payload_part.get("records", [])))
    all_records.sort(key=lambda record: int(record["dataset_index"]))

    expected_episodes = len(selected_indices)
    if len(all_records) != expected_episodes:
        print(
            f"Warning: merged {len(all_records)}/{expected_episodes} episodes. "
            "Increase --merge_wait_seconds or check failed ranks."
        )

    records_by_domain = {}
    for record in all_records:
        records_by_domain.setdefault(record["domain"], []).append(record)
    if args.split_manifest and not manifest_paths:
        manifest_paths = {domain: Path(args.split_manifest).expanduser() for domain in records_by_domain}

    output_groups = list(records_by_domain.items())
    if all_records:
        output_groups.append(("all", all_records))

    for domain, records in output_groups:
        domain_dir = os.path.join(args.output_dir, domain)
        os.makedirs(domain_dir, exist_ok=True)
        aggregate = aggregate_records(records, threshold=args.threshold)
        best_f1_confusion_path = plot_confusion_summary(
            os.path.join(domain_dir, "confusion_matrices_best_f1.png"),
            aggregate,
            title="Anomaly Confusion Matrices (best-F1 thresholds)",
        )
        manual_confusion_path = plot_confusion_summary(
            os.path.join(domain_dir, "confusion_matrices_manual.png"),
            aggregate,
            title=f"Anomaly Confusion Matrices (manual threshold {float(args.threshold):.3f})",
            step_metrics_key="step_metrics_manual",
            window_metrics_key="window_metrics_manual",
        )
        interaction_best_f1_confusion_path = plot_interaction_confusion_summary(
            os.path.join(domain_dir, "interaction_confusion_matrices_best_f1.png"),
            aggregate,
            title="Interaction-split Anomaly Confusion Matrices (global best-F1 thresholds)",
        )
        interaction_manual_confusion_path = plot_interaction_confusion_summary(
            os.path.join(domain_dir, "interaction_confusion_matrices_manual.png"),
            aggregate,
            title=f"Interaction-split Anomaly Confusion Matrices (manual threshold {float(args.threshold):.3f})",
            step_metrics_key="step_metrics_manual",
            window_metrics_key="window_metrics_manual",
        )
        manifest_payload = {name: str(path) for name, path in manifest_paths.items()}
        payload = {
            "domain": domain,
            "config_file": args.config_file,
            "wm_checkpoint": cfg["diffusion_model"]["model_path"],
            "anomaly_checkpoint": args.anomaly_checkpoint,
            "eval_split": args.eval_split if not cache_dir else None,
            "split_manifest_path": str(manifest_paths[domain]) if domain in manifest_paths else None,
            "manifest_paths": manifest_payload if domain == "all" else None,
            "num_episodes": len(records),
            "num_shards": num_shards,
            "merged_shards": len(available_paths),
            "run_id": run_id,
            "batch_size": int(args.batch_size),
            "num_inference_steps": int(cfg["num_inference_step"]),
            "guidance_scale": float(args.guidance_scale),
            "noise_seed": int(args.noise_seed),
            "window_stride": int(window_stride),
            "feature_cache_dir": cache_dir,
            "cache_split": args.cache_split if cache_dir else None,
            "split_manifest": args.split_manifest,
            "cache_metadata": cache_metadata if domain == "all" else None,
            "manual_threshold": float(args.threshold),
            "confusion_matrix_plot_best_f1": best_f1_confusion_path,
            "confusion_matrix_plot_manual": manual_confusion_path,
            "interaction_confusion_matrix_plot_best_f1": interaction_best_f1_confusion_path,
            "interaction_confusion_matrix_plot_manual": interaction_manual_confusion_path,
            "aggregate_metrics": aggregate,
        }
        if args.write_episode_details:
            payload["episodes"] = public_records(records)
        write_json(os.path.join(domain_dir, "metrics.json"), payload)
        if args.write_episode_details:
            with open(os.path.join(domain_dir, "per_episode_metrics.jsonl"), "w", encoding="utf-8") as f:
                for record in public_records(records):
                    f.write(json.dumps(record) + "\n")
        print(f"Saved metrics for {domain}: {domain_dir}")

