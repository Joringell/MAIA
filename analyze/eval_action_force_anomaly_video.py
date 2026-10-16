import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.action_force_anomaly_eval_core import (
    aggregate_anomaly_chunks,
    build_action_force_anomaly_video,
    force_metrics,
    load_anomaly_head,
    run_anomaly_window,
    save_force_diff_plot,
    save_force_label_plot,
)
from analyze.anomaly_eval_common import (
    extract_episode_id,
    load_failure_transition_frame,
    load_frame_labels,
    parse_dataset_indices,
    save_label_curve_data,
    save_label_plot,
    write_json,
)
from analyze.eval_common import (
    aggregate_pred_video_chunks,
    load_multiview_video_segment,
    prepare_dataset,
    prepare_models,
    resolve_checkpoint_path,
    resolve_force_layout,
    resolve_video_template_and_cams,
    save_force_curves,
    str_to_dtype,
)
from utils.model_utils import count_model_parameters
from utils.path_utils import load_resolved_config


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _latency_summary(latency_ms):
    values = np.asarray(latency_ms, dtype=np.float64)
    if values.size == 0:
        return None
    return {
        "num_windows": int(values.size),
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
        "fps_equivalent": float(1000.0 / np.mean(values)) if np.mean(values) > 0 else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize action-force WM + anomaly head inference.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint_path", default=None, help="Action-force WM checkpoint override.")
    parser.add_argument("--anomaly_checkpoint", required=True, help="Path to anomaly_head.pt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--dataset_split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--split_manifest", default=None, help="Optional train/val/test episode split manifest.")
    parser.add_argument("--stats_domain", default=None)
    parser.add_argument("--dataset_index", type=int, default=0)
    parser.add_argument("--dataset_indices", nargs="+", default=None, help="Optional multiple dataset indices; accepts spaces or commas.")
    parser.add_argument("--mem_mode", choices=["uniform_history", "contiguous", "zeros"], default="uniform_history")
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--window_stride", type=int, default=None)
    parser.add_argument("--output_width", type=int, default=1280)
    parser.add_argument("--plot_height", type=int, default=620)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--return_video",
        action="store_true",
        help="Decode and save predicted future video. Leave off to measure risk-only latency.",
    )
    return parser.parse_args()


def run_one(args, cfg, dataset_index_arg):
    action_condition_dim, force_dim = resolve_force_layout(cfg)
    device = torch.device(args.device)
    weight_dtype = str_to_dtype(args.dtype)
    (
        dataset,
        _sample,
        initial_mem_idx,
        selected_dataset_index,
        selected_episode_id,
        manifest_path,
        stats_alias,
    ) = prepare_dataset(
        cfg=cfg,
        split=args.dataset_split,
        domain=args.domain,
        dataset_index=dataset_index_arg,
        fix_sidx=0,
        mem_mode=args.mem_mode,
        stats_domain=args.stats_domain,
        split_manifest=args.split_manifest,
    )
    item = dataset.dataset[selected_dataset_index]
    total_frames = int(item[7])
    parquet_path = item[2]
    episode_id = extract_episode_id(parquet_path) or selected_episode_id
    episode_tag = episode_id if episode_id is not None else selected_dataset_index
    episode_output_dir = os.path.join(args.output_dir, f"{args.domain}_{episode_tag}")
    os.makedirs(episode_output_dir, exist_ok=True)

    _, _, vae, diffusion_model, _, pipe = prepare_models(cfg, device, weight_dtype)
    anomaly_head = load_anomaly_head(cfg, args.anomaly_checkpoint, device)
    total_params = count_model_parameters(diffusion_model)
    print(f"Loaded WM params: total={total_params[0]}, trainable={total_params[1]}")

    n_previous = int(cfg["data"]["train"]["n_previous"])
    action_chunk = int(cfg["data"]["train"]["action_chunk"])
    window_stride = args.window_stride or action_chunk
    if window_stride <= 0 or window_stride > action_chunk:
        raise ValueError(f"window_stride must be in [1, {action_chunk}], got {window_stride}")

    chunks = []
    window_latency_records = []
    window_starts = list(range(0, total_frames, window_stride))
    for idx, fix_sidx in enumerate(window_starts):
        print(f"[{idx + 1}/{len(window_starts)}] infer window fix_sidx={fix_sidx}")
        _sync_device(device)
        window_start_time = time.perf_counter()
        chunk = run_anomaly_window(
            cfg=cfg,
            dataset=dataset,
            pipe=pipe,
            anomaly_head=anomaly_head,
            vae=vae,
            args=args,
            device=device,
            weight_dtype=weight_dtype,
            n_previous=n_previous,
            action_condition_dim=action_condition_dim,
            force_dim=force_dim,
            fix_sidx=fix_sidx,
            mem_mode=args.mem_mode,
            total_frames=total_frames,
            return_video=bool(args.return_video),
        )
        _sync_device(device)
        latency_ms = (time.perf_counter() - window_start_time) * 1000.0
        if chunk is not None:
            chunks.append(chunk)
            max_step_prob = float(np.nanmax(chunk["step_prob"])) if len(chunk["step_prob"]) else float("nan")
            mean_step_prob = float(np.nanmean(chunk["step_prob"])) if len(chunk["step_prob"]) else float("nan")
            record = {
                "fix_sidx": int(chunk["fix_sidx"]),
                "valid_len": int(chunk["valid_len"]),
                "latency_ms": float(latency_ms),
                "window_prob": float(chunk["window_prob"]),
                "risk_prob": float(chunk["risk_prob"]),
                "max_step_prob": max_step_prob,
                "mean_step_prob": mean_step_prob,
            }
            window_latency_records.append(record)
            print(
                f"    latency={latency_ms:.1f} ms | "
                f"risk={record['risk_prob']:.4f} | "
                f"window={record['window_prob']:.4f} | "
                f"max_step={record['max_step_prob']:.4f}"
            )
        else:
            window_latency_records.append(
                {
                    "fix_sidx": int(fix_sidx),
                    "valid_len": 0,
                    "latency_ms": float(latency_ms),
                    "window_prob": None,
                    "risk_prob": None,
                    "max_step_prob": None,
                    "mean_step_prob": None,
                }
            )
            print(f"    latency={latency_ms:.1f} ms | skipped empty window")
    if not chunks:
        raise RuntimeError("No inference windows were generated.")

    aggregate = aggregate_anomaly_chunks(chunks, total_frames=total_frames, force_dim=force_dim)
    frame_labels_raw, frame_anomaly = load_frame_labels(parquet_path, total_frames)
    frame_target = aggregate["frame_target"]
    if frame_target is None:
        frame_target = frame_anomaly
    try:
        transition_frame = load_failure_transition_frame(parquet_path)
    except Exception:
        transition_frame = None

    pred_force = aggregate["pred_force"]
    gt_force = aggregate["gt_force"]
    step_prob = aggregate["step_prob"]
    force_label_plot = save_force_label_plot(
        os.path.join(episode_output_dir, "force_label_comparison.png"),
        pred_force,
        gt_force,
        step_prob,
        frame_target=frame_target,
        anomaly_frame_prob=aggregate.get("anomaly_frame_prob"),
        anomaly_window_prob=aggregate.get("anomaly_window_prob"),
        transition_frame=transition_frame,
    )
    force_plot = save_force_curves(
        episode_output_dir,
        pred_force[None],
        gt_force[None],
        failure_transition_frame=transition_frame,
        global_start=0,
    )
    label_plot = save_label_plot(
        os.path.join(episode_output_dir, "label_comparison.png"),
        step_prob,
        frame_target=frame_target,
        transition_frame=transition_frame,
    )
    label_curve_data = save_label_curve_data(
        os.path.join(episode_output_dir, "label_curve_data"),
        step_prob,
        frame_target=frame_target,
        transition_frame=transition_frame,
        extra_curves={
            "dataset_anomaly_frame_prob": aggregate.get("anomaly_frame_prob"),
            "dataset_anomaly_window_prob": aggregate.get("anomaly_window_prob"),
        },
    )
    diff_plot = save_force_diff_plot(
        os.path.join(episode_output_dir, "force_diff.png"),
        pred_force,
        gt_force,
        labels=frame_target,
        transition_frame=transition_frame,
        window_size=action_chunk,
    )

    video_source, cam_names = resolve_video_template_and_cams(cfg, dataset, selected_dataset_index, split=args.dataset_split)
    multiview_frames = load_multiview_video_segment(
        video_source=video_source,
        cam_names=cam_names,
        start_frame=0,
        num_frames=min(total_frames, pred_force.shape[0]),
    )
    video_stride = int(cfg["data"]["train"].get("video_stride", max(1, action_chunk // max(1, int(cfg["data"]["train"]["chunk"])))))
    pred_multiview_frames = aggregate_pred_video_chunks(
        chunks,
        total_frames=min(total_frames, pred_force.shape[0]),
        n_view=multiview_frames.shape[1],
        video_stride=video_stride,
    )
    video_path = build_action_force_anomaly_video(
        output_path=os.path.join(episode_output_dir, "action_force_anomaly_alignment.mp4"),
        multiview_frames=multiview_frames,
        pred_multiview_frames=pred_multiview_frames,
        pred_force=pred_force,
        gt_force=gt_force,
        step_prob=step_prob,
        frame_target=frame_target,
        anomaly_frame_prob=aggregate.get("anomaly_frame_prob"),
        anomaly_window_prob=aggregate.get("anomaly_window_prob"),
        fps=args.fps,
        output_width=args.output_width,
        plot_height=args.plot_height,
        transition_frame=transition_frame,
    )

    meta = {
        "config_file": args.config_file,
        "wm_checkpoint": cfg["diffusion_model"]["model_path"],
        "anomaly_checkpoint": args.anomaly_checkpoint,
        "domain": args.domain,
        "dataset_split": args.dataset_split,
        "dataset_index": int(selected_dataset_index),
        "dataset_index_arg": int(dataset_index_arg),
        "episode_id": None if episode_id is None else int(episode_id),
        "parquet_path": parquet_path,
        "total_frames": int(total_frames),
        "initial_mem_idx": initial_mem_idx,
        "window_starts": [int(c["fix_sidx"]) for c in chunks],
        "window_stride": int(window_stride),
        "num_inference_steps": int(cfg["num_inference_step"]),
        "guidance_scale": float(args.guidance_scale),
        "noise_seed": int(args.noise_seed),
        "failure_transition_frame": transition_frame,
        "stats_alias": stats_alias,
        "split_manifest_path": None if manifest_path is None else str(manifest_path),
        "plots": {
            "force_label": os.path.basename(force_label_plot),
            "force": None if force_plot is None else os.path.basename(force_plot),
            "label": os.path.basename(label_plot),
            "force_diff": os.path.basename(diff_plot),
            "video": os.path.basename(video_path),
        },
        "data": {
            "label_curve_csv": os.path.basename(label_curve_data["csv"]),
            "label_curve_npz": os.path.basename(label_curve_data["npz"]),
        },
        "force_metrics": force_metrics(pred_force, gt_force, labels=frame_target),
        "anomaly": {
            "step_prob_mean": float(np.nanmean(step_prob)),
            "step_prob_max": float(np.nanmax(step_prob)),
            "gt_anomaly_ratio": float(np.nanmean(frame_target)),
            "window_prob_mean": float(np.mean(aggregate["window_probs"])) if aggregate["window_probs"].size else None,
            "has_dataset_anomaly_frame_prob": aggregate.get("anomaly_frame_prob") is not None,
            "has_dataset_anomaly_window_prob": aggregate.get("anomaly_window_prob") is not None,
        },
        "latency": {
            "summary": _latency_summary([record["latency_ms"] for record in window_latency_records]),
            "windows": window_latency_records,
        },
    }
    write_json(os.path.join(episode_output_dir, "meta.json"), meta)
    print(f"Saved video: {video_path}")
    print(f"Saved plots under: {episode_output_dir}")


def main():
    args = parse_args()
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

    dataset_indices = parse_dataset_indices(args.dataset_index, args.dataset_indices)
    for idx, dataset_index in enumerate(dataset_indices):
        if len(dataset_indices) > 1:
            print(f"=== [{idx + 1}/{len(dataset_indices)}] dataset_index={dataset_index} ===")
        run_one(args, cfg, dataset_index)


if __name__ == "__main__":
    main()
