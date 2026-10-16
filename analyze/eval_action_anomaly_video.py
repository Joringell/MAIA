import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.anomaly_eval_common import (
    extract_episode_id,
    load_failure_transition_frame,
    load_frame_labels,
    parse_dataset_indices,
    save_label_curve_data,
    save_label_plot,
    write_json,
)
from analyze.eval_action_anomaly_metrics import (
    _latent_chunk,
    load_action_anomaly_head,
    resolve_action_dim,
)
from analyze.eval_video_anomaly_metrics import aggregate_video_anomaly_chunks
from analyze.eval_video_anomaly_video import build_video_anomaly_video
from analyze.eval_common import (
    aggregate_pred_video_chunks,
    build_mem_idx,
    get_sample_domain_name,
    load_multiview_video_segment,
    prepare_dataset,
    prepare_models,
    resolve_checkpoint_path,
    resolve_video_template_and_cams,
    sample_to_batch,
    str_to_dtype,
)
from utils.model_utils import count_model_parameters
from utils.path_utils import load_resolved_config


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize action-conditioned video anomaly head inference.")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint_path", default=None, help="Action WM checkpoint override.")
    parser.add_argument("--anomaly_checkpoint", required=True, help="Path to action anomaly_head.pt.")
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
    parser.add_argument("--plot_height", type=int, default=360)
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def _config_split_manifest(cfg, split):
    manifest = cfg.get("split_manifest")
    if manifest not in (None, ""):
        return manifest
    data_cfg = cfg.get("data", {})
    for split_name in (split, "val", "train"):
        split_cfg = data_cfg.get(split_name)
        if isinstance(split_cfg, dict):
            manifest = split_cfg.get("split_manifest")
            if manifest not in (None, ""):
                return manifest
    return None


def run_action_anomaly_window(
    cfg,
    dataset,
    pipe,
    anomaly_head,
    vae,
    args,
    device,
    weight_dtype,
    n_previous,
    fix_sidx,
    mem_mode,
    total_frames,
    action_condition_dim,
):
    action_chunk = int(cfg["data"]["train"].get("action_chunk", cfg["data"]["train"]["future_steps"]))
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
    batch_size, _, n_view, _, height, width = image.shape
    image = rearrange(image, "b c v t h w -> (b v) c t h w")
    prompt = batch["caption"][:batch_size]

    future_actions = batch["actions"][:, -action_chunk:].to(device=device, dtype=weight_dtype).contiguous()
    if future_actions.shape[-1] != action_condition_dim:
        raise ValueError(f"Expected action dim {action_condition_dim}, got {future_actions.shape[-1]}")

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
            guidance_scale=float(args.guidance_scale),
            height=height,
            width=width,
            n_view=n_view,
            n_prev=n_previous,
            chunk=_latent_chunk(cfg, vae),
            return_video=True,
            return_interaction=True,
            return_latents=True,
            return_init=False,
            noise_seed=args.noise_seed,
            action_chunk=action_chunk,
            future_actions=future_actions,
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
        head_out = anomaly_head(
            interaction_tokens=preds["interaction_tokens"],
            pred_latents=pred_latents,
        )

    step_prob = torch.sigmoid(head_out["step_logits"]).detach().cpu().float().numpy()[0]
    if step_prob.shape[0] != action_chunk:
        x_old = np.linspace(0.0, 1.0, step_prob.shape[0])
        x_new = np.linspace(0.0, 1.0, action_chunk)
        step_prob = np.interp(x_new, x_old, step_prob)
    step_prob = step_prob[:valid_len]
    window_prob = float(torch.sigmoid(head_out["window_logit"]).detach().cpu().float().reshape(-1)[0])
    risk_prob = float(head_out["risk_prob"].detach().cpu().float().reshape(-1)[0])

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

    video = preds.get("video")
    if video is not None:
        video = video.detach().cpu().float()

    return {
        "fix_sidx": int(fix_sidx),
        "valid_len": int(valid_len),
        "mem_idx": mem_idx,
        "domain_name": domain_name,
        "prompt": prompt[0],
        "step_prob": step_prob,
        "window_prob": window_prob,
        "risk_prob": risk_prob,
        "frame_target": frame_target,
        "risk_target": risk_target,
        "window_target": window_target,
        "video": video,
    }


def run_one(args, cfg, dataset_index_arg):
    device = torch.device(args.device)
    weight_dtype = str_to_dtype(args.dtype)
    action_condition_dim = resolve_action_dim(cfg)
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
    anomaly_head = load_action_anomaly_head(cfg, args.anomaly_checkpoint, device)
    total_params = count_model_parameters(diffusion_model)
    print(f"Loaded WM params: total={total_params[0]}, trainable={total_params[1]}")

    n_previous = int(cfg["data"]["train"]["n_previous"])
    action_chunk = int(cfg["data"]["train"].get("action_chunk", cfg["data"]["train"]["future_steps"]))
    window_stride = int(args.window_stride or action_chunk)
    if window_stride <= 0 or window_stride > action_chunk:
        raise ValueError(f"window_stride must be in [1, {action_chunk}], got {window_stride}")

    chunks = []
    window_starts = list(range(0, total_frames, window_stride))
    for idx, fix_sidx in enumerate(window_starts):
        print(f"[{idx + 1}/{len(window_starts)}] infer window fix_sidx={fix_sidx}")
        chunk = run_action_anomaly_window(
            cfg=cfg,
            dataset=dataset,
            pipe=pipe,
            anomaly_head=anomaly_head,
            vae=vae,
            args=args,
            device=device,
            weight_dtype=weight_dtype,
            n_previous=n_previous,
            fix_sidx=fix_sidx,
            mem_mode=args.mem_mode,
            total_frames=total_frames,
            action_condition_dim=action_condition_dim,
        )
        if chunk is not None:
            chunks.append(chunk)
    if not chunks:
        raise RuntimeError("No inference windows were generated.")

    aggregate = aggregate_video_anomaly_chunks(chunks, total_frames=total_frames)
    _, frame_anomaly = load_frame_labels(parquet_path, total_frames)
    frame_target = aggregate["frame_target"]
    if frame_target is None:
        frame_target = frame_anomaly
    try:
        transition_frame = load_failure_transition_frame(parquet_path)
    except Exception:
        transition_frame = None

    step_prob = aggregate["step_prob"]
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
    )

    video_source, cam_names = resolve_video_template_and_cams(cfg, dataset, selected_dataset_index, split=args.dataset_split)
    total_video_frames = min(total_frames, step_prob.shape[0])
    multiview_frames = load_multiview_video_segment(
        video_source=video_source,
        cam_names=cam_names,
        start_frame=0,
        num_frames=total_video_frames,
    )
    video_stride = int(cfg["data"]["train"].get("video_stride", max(1, action_chunk // max(1, int(cfg["data"]["train"]["chunk"])))))
    pred_multiview_frames = aggregate_pred_video_chunks(
        chunks,
        total_frames=total_video_frames,
        n_view=multiview_frames.shape[1],
        video_stride=video_stride,
    )
    video_path = build_video_anomaly_video(
        output_path=os.path.join(episode_output_dir, "action_anomaly_alignment.mp4"),
        multiview_frames=multiview_frames,
        pred_multiview_frames=pred_multiview_frames,
        step_prob=step_prob,
        frame_target=frame_target,
        fps=args.fps,
        output_width=args.output_width,
        plot_height=args.plot_height,
        transition_frame=transition_frame,
    )

    window_probs = aggregate["window_probs"]
    window_targets = aggregate["window_targets"]
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
            "label": os.path.basename(label_plot),
            "video": os.path.basename(video_path),
        },
        "data": {
            "label_curve_csv": os.path.basename(label_curve_data["csv"]),
            "label_curve_npz": os.path.basename(label_curve_data["npz"]),
        },
        "anomaly": {
            "step_prob_mean": float(np.nanmean(step_prob)),
            "step_prob_max": float(np.nanmax(step_prob)),
            "gt_anomaly_ratio": float(np.nanmean(frame_target)),
            "window_prob_mean": float(np.mean(window_probs)) if window_probs.size else None,
            "window_target_mean": float(np.mean(window_targets)) if window_targets.size else None,
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
        args.split_manifest = _config_split_manifest(cfg, args.dataset_split)

    dataset_indices = parse_dataset_indices(args.dataset_index, args.dataset_indices)
    for idx, dataset_index in enumerate(dataset_indices):
        if len(dataset_indices) > 1:
            print(f"=== [{idx + 1}/{len(dataset_indices)}] dataset_index={dataset_index} ===")
        run_one(args, cfg, dataset_index)


if __name__ == "__main__":
    main()
