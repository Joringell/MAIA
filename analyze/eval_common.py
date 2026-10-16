import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import import_custom_class
from utils.model_utils import (
    load_condition_models,
    load_diffusion_model,
    load_latent_models,
    load_vae_models,
)

def str_to_dtype(name):
    if name == "fp32":
        return torch.float32
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_checkpoint_path(checkpoint_path):
    path = Path(checkpoint_path).expanduser()
    if path.is_file():
        return str(path)
    if not path.is_dir():
        return str(path)

    single_file = path / "diffusion_pytorch_model.safetensors"
    if single_file.exists():
        return str(single_file)

    index_file = path / "diffusion_pytorch_model.safetensors.index.json"
    if index_file.exists():
        return str(path)

    step_dirs = []
    for child in path.iterdir():
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        step_text = child.name[len("step_") :]
        if not step_text.isdigit():
            continue
        child_single = child / "diffusion_pytorch_model.safetensors"
        child_index = child / "diffusion_pytorch_model.safetensors.index.json"
        if child_single.exists() or child_index.exists():
            step_dirs.append((int(step_text), child))

    if step_dirs:
        _, latest_step_dir = max(step_dirs, key=lambda item: item[0])
        latest_single = latest_step_dir / "diffusion_pytorch_model.safetensors"
        if latest_single.exists():
            return str(latest_single)
        return str(latest_step_dir)

    raise FileNotFoundError(
        f"No diffusion checkpoint found under {path}. Pass a .safetensors file, "
        "a step_* directory, or an experiment directory containing step_* checkpoints."
    )


def resolve_force_layout(cfg):
    diffusion_cfg = cfg["diffusion_model"]["config"]
    split_cfg = cfg["data"]["train"]

    force_dim = int(
        split_cfg.get(
            "force_dim",
            diffusion_cfg.get("force_out_channels", 0) or 0,
        )
    )

    action_dim = split_cfg.get("action_dim")
    if action_dim is None:
        action_dim = diffusion_cfg.get("action_condition_dim")
    if action_dim is None:
        raise ValueError("action_dim or action_condition_dim must be provided.")

    action_dim = int(action_dim)
    if action_dim <= 0 or force_dim <= 0:
        raise ValueError(
            f"action and force dimensions must be positive, got {action_dim} and {force_dim}"
        )
    return action_dim, force_dim


def build_mem_idx(start_idx, n_previous, mode):
    if mode == "zeros":
        return [0 for _ in range(n_previous)]

    if start_idx <= 0:
        return [0 for _ in range(n_previous)]

    if mode == "contiguous":
        mem_start = max(0, start_idx - n_previous)
        mem_idx = list(range(mem_start, start_idx))
        if len(mem_idx) < n_previous:
            mem_idx = [mem_idx[0]] * (n_previous - len(mem_idx)) + mem_idx
        return mem_idx[-n_previous:]

    mem_idx = np.linspace(0, start_idx - 1, n_previous).round().astype(np.int32).tolist()
    return mem_idx


def _expand_roots_for_domains(data_cfg):
    domains = list(data_cfg["domains"])
    roots = list(data_cfg["data_roots"])
    if len(roots) == 1 and len(domains) > 1:
        roots = roots * len(domains)
    if len(roots) != len(domains):
        raise ValueError(
            f"data_roots/domains length mismatch: {len(roots)} vs {len(domains)}"
        )
    return domains, roots


def data_config_for_split(cfg, split):
    if split in cfg["data"]:
        return dict(cfg["data"][split])
    if "val" in cfg["data"]:
        data_cfg = dict(cfg["data"]["val"])
    else:
        data_cfg = dict(cfg["data"]["train"])
    data_cfg["split"] = split
    return data_cfg


def select_domain_data_config(cfg, split, domain):
    data_cfg = data_config_for_split(cfg, split)
    domains, roots = _expand_roots_for_domains(data_cfg)

    if domain not in domains:
        # Held-out test domain: use the same LeRobot root as the config and
        # read the full domain directly from <root>/<domain>/.
        data_cfg["domains"] = [domain]
        data_cfg["data_roots"] = [roots[0]]
        return data_cfg

    domain_index = domains.index(domain)
    data_cfg["domains"] = [domain]
    data_cfg["data_roots"] = [roots[domain_index]]
    return data_cfg


def domain_is_configured(cfg, split, domain):
    domains, _ = _expand_roots_for_domains(data_config_for_split(cfg, split))
    return domain in domains


def alias_dataset_statistics(dataset, domain, stats_domain=None, fallback_cfg=None):
    if stats_domain is None and fallback_cfg is not None:
        train_domains = fallback_cfg.get("data", {}).get("train", {}).get("domains", [])
        if train_domains:
            stats_domain = train_domains[0]
    if not stats_domain or stats_domain == domain:
        return None

    aliases = [
        (f"{stats_domain}_{dataset.action_space}", f"{domain}_{dataset.action_space}"),
        (f"{stats_domain}_state_{dataset.action_space}", f"{domain}_state_{dataset.action_space}"),
        (f"{stats_domain}_delta_{dataset.action_space}", f"{domain}_delta_{dataset.action_space}"),
        (f"{stats_domain}_ee_ft", f"{domain}_ee_ft"),
        (f"{stats_domain}_ee_ft_relative", f"{domain}_ee_ft_relative"),
    ]
    copied = []
    for src_key, dst_key in aliases:
        if dst_key in dataset.StatisticInfo:
            continue
        if src_key in dataset.StatisticInfo:
            dataset.StatisticInfo[dst_key] = dataset.StatisticInfo[src_key]
            copied.append((src_key, dst_key))
    return {"stats_domain": stats_domain, "copied": copied}


def split_manifest_path(data_cfg, split, domain):
    split_ratio = data_cfg.get("split_ratio", data_cfg.get("val_ratio"))
    split_seed = int(data_cfg.get("split_seed", 0))
    if split_ratio is None or not 0 < float(split_ratio) < 1:
        raise ValueError(
            f"eval_ft_pd_video requires split_ratio or val_ratio in data.{split} configuration "
            "to read the persisted episode split manifest"
        )
    ratio_text = (
        f"{float(split_ratio):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    )
    filename = f"episode_split_train{ratio_text}_seed{split_seed}.json"
    return Path(data_cfg["data_roots"][0]) / domain / "meta" / filename


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


def apply_split_manifest_data_root(data_cfg, split_manifest):
    data_root = _split_manifest_data_root(split_manifest)
    if data_root is not None:
        data_cfg["data_roots"] = [data_root]
    return data_cfg


def ensure_split_manifest(dataset_class, data_cfg):
    dataset_class(**dict(data_cfg))


def _episode_indices_from_manifest(manifest, split, domain, manifest_path):
    if split in manifest:
        split_spec = manifest[split]
        if isinstance(split_spec, list):
            return [int(value) for value in split_spec]
        if isinstance(split_spec, dict):
            if domain not in split_spec:
                raise KeyError(f"Domain '{domain}' not found in split '{split}' manifest: {manifest_path}")
            return [int(value) for value in split_spec[domain]]
        raise TypeError(f"Unsupported split manifest format for split '{split}': {type(split_spec)}")

    legacy_key = "val_episode_indices" if split == "val" else "train_episode_indices"
    if split == "test" and "test_episode_indices" in manifest:
        legacy_key = "test_episode_indices"
    if legacy_key not in manifest:
        raise KeyError(f"Split '{split}' not found in split manifest: {manifest_path}")
    return [int(value) for value in manifest[legacy_key]]


def read_split_episode_id(dataset_class, data_cfg, split, domain, dataset_index, split_manifest=None):
    if split_manifest not in (None, ""):
        manifest_path = Path(split_manifest).expanduser()
    else:
        manifest_path = split_manifest_path(data_cfg, split, domain)
        if not manifest_path.exists():
            ensure_split_manifest(dataset_class, data_cfg)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest was not generated by the dataset: {manifest_path}"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    episode_indices = _episode_indices_from_manifest(manifest, split, domain, manifest_path)
    if not 0 <= int(dataset_index) < len(episode_indices):
        raise IndexError(
            f"dataset_index={dataset_index} out of range for split '{split}' with "
            f"{len(episode_indices)} episodes in {manifest_path}"
        )
    return episode_indices[int(dataset_index)], manifest_path


def prepare_dataset(
    cfg,
    split,
    domain,
    dataset_index,
    fix_sidx,
    mem_mode,
    stats_domain=None,
    split_manifest=None,
):
    if split == "train":
        dataset_class_name = cfg["train_data_class"]
        dataset_class_path = cfg["train_data_class_path"]
    else:
        dataset_class_name = cfg.get("val_data_class", cfg["train_data_class"])
        dataset_class_path = cfg.get("val_data_class_path", cfg["train_data_class_path"])

    dataset_class = import_custom_class(dataset_class_name, dataset_class_path)
    data_cfg = select_domain_data_config(cfg, split, domain)
    n_previous = int(cfg["data"]["train"]["n_previous"])
    mem_idx = build_mem_idx(fix_sidx, n_previous, mem_mode)

    split_manifest = split_manifest or cfg.get("split_manifest")
    apply_split_manifest_data_root(data_cfg, split_manifest)
    use_full_domain = not domain_is_configured(cfg, split, domain) and not split_manifest
    selected_episode_id = None
    manifest_path = None
    if not use_full_domain:
        selected_episode_id, manifest_path = read_split_episode_id(
            dataset_class,
            data_cfg,
            split,
            domain,
            dataset_index,
            split_manifest=split_manifest,
        )

    data_cfg["split_ratio"] = None
    data_cfg["val_ratio"] = None
    data_cfg["train_dataset"] = True
    if split_manifest:
        data_cfg["split_manifest"] = str(Path(split_manifest).expanduser())
        data_cfg["split"] = split
    data_cfg["fix_sidx"] = int(fix_sidx)
    data_cfg["fix_mem_idx"] = mem_idx
    dataset = dataset_class(**data_cfg)
    stats_alias = alias_dataset_statistics(
        dataset,
        domain=domain,
        stats_domain=stats_domain,
        fallback_cfg=cfg,
    )

    if use_full_domain:
        selected_dataset_index = int(dataset_index)
        if not 0 <= selected_dataset_index < len(dataset.dataset):
            raise IndexError(
                f"dataset_index={dataset_index} out of range for full domain "
                f"'{domain}' with {len(dataset.dataset)} episodes"
            )
        match = re.search(
            r"episode_(\d+)\.parquet$",
            os.path.basename(dataset.dataset[selected_dataset_index][2]),
        )
        selected_episode_id = int(match.group(1)) if match else selected_dataset_index
        manifest_path = None
    else:
        selected_dataset_index = None
        for full_index, item in enumerate(dataset.dataset):
            match = re.search(r"episode_(\d+)\.parquet$", os.path.basename(item[2]))
            if match and int(match.group(1)) == selected_episode_id:
                selected_dataset_index = full_index
                break
        if selected_dataset_index is None:
            raise RuntimeError(
                f"Episode {selected_episode_id:06d} from {manifest_path} "
                f"was not found in domain '{domain}'"
            )

    dataset.fix_epiidx = selected_dataset_index
    sample = dataset[0]
    return (
        dataset,
        sample,
        mem_idx,
        selected_dataset_index,
        selected_episode_id,
        manifest_path,
        stats_alias,
    )


def sample_to_batch(sample):
    batch = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            batch[key] = value.unsqueeze(0)
        elif isinstance(value, str):
            batch[key] = [value]
        else:
            batch[key] = value
    return batch


def prepare_models(cfg, device, weight_dtype):
    tokenizer_class = import_custom_class(
        cfg["tokenizer_class"], cfg.get("tokenizer_class_path", "transformers")
    )
    textenc_class = import_custom_class(
        cfg["textenc_class"], cfg.get("textenc_class_path", "transformers")
    )
    cond_models = load_condition_models(
        tokenizer_class,
        textenc_class,
        cfg.get("tokenizer_pretrained_model_name_or_path", cfg["pretrained_model_name_or_path"]),
        load_weights=True,
    )
    tokenizer = cond_models["tokenizer"]
    text_encoder = cond_models["text_encoder"].to(device, dtype=weight_dtype).eval()

    vae_class = import_custom_class(cfg["vae_class"], cfg.get("vae_class_path", "transformers"))
    if cfg.get("vae_path", False):
        vae = load_vae_models(vae_class, cfg["vae_path"]).to(device, dtype=weight_dtype).eval()
    else:
        vae = load_latent_models(vae_class, cfg["pretrained_model_name_or_path"])["vae"].to(device, dtype=weight_dtype).eval()
    if isinstance(vae.latents_mean, list):
        vae.latents_mean = torch.FloatTensor(vae.latents_mean)
    if isinstance(vae.latents_std, list):
        vae.latents_std = torch.FloatTensor(vae.latents_std)
    if cfg.get("enable_slicing", False):
        vae.enable_slicing()
    if cfg.get("enable_tiling", False):
        vae.enable_tiling()

    diffusion_model_class = import_custom_class(
        cfg["diffusion_model_class"], cfg.get("diffusion_model_class_path", "transformers")
    )
    diffusion_model_config = dict(cfg["diffusion_model"]["config"])
    # return_risk is the only YAML switch. The model constructor keeps its
    # internal enable_risk_head argument for checkpoint compatibility.
    diffusion_model_config["enable_risk_head"] = bool(
        cfg.get("return_risk", False)
    )
    diffusion_model = load_diffusion_model(
        model_cls=diffusion_model_class,
        model_dir=cfg["diffusion_model"]["model_path"],
        load_weights=cfg.get("load_weights", True) and cfg.get("load_diffusion_model_weights", True),
        **diffusion_model_config,
    ).to(device, dtype=weight_dtype).eval()

    diffusion_scheduler_class = import_custom_class(
        cfg["diffusion_scheduler_class"], cfg.get("diffusion_scheduler_class_path", "diffusers")
    )
    if "diffusion_scheduler_args" in cfg:
        scheduler = diffusion_scheduler_class(**cfg["diffusion_scheduler_args"])
    else:
        scheduler = diffusion_scheduler_class()

    pipeline_class = import_custom_class(
        cfg["pipeline_class"], cfg.get("pipeline_class_path", "diffusers")
    )
    pipe = pipeline_class(scheduler, vae, text_encoder, tokenizer, diffusion_model)

    return tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe


def get_sample_domain_name(dataset, fallback_cfg):
    episode_idx = int(dataset.fix_epiidx) if getattr(dataset, "fix_epiidx", None) is not None else 0
    if (
        hasattr(dataset, "dataset")
        and 0 <= episode_idx < len(dataset.dataset)
        and len(dataset.dataset[episode_idx]) > 3
    ):
        return dataset.dataset[episode_idx][3]
    return fallback_cfg["data"]["train"]["domains"][0]



def denorm_force(dataset, domain_name, pred_force):
    if pred_force is None:
        return None
    if domain_name is None:
        return pred_force
    mean, std = dataset.get_force_bias_std(relative=False, domain_name=domain_name)
    mean = mean.to(device=pred_force.device, dtype=pred_force.dtype).view(1, 1, -1)
    std = std.to(device=pred_force.device, dtype=pred_force.dtype).view(1, 1, -1)
    if mean.shape[-1] != pred_force.shape[-1]:
        raise ValueError(
            f"Force stats dim {mean.shape[-1]} does not match tensor dim "
            f"{pred_force.shape[-1]} for domain '{domain_name}'"
        )
    return pred_force * std + mean


def load_failure_transition_frame(parquet_path):
    data = pd.read_parquet(parquet_path, columns=["frame_index", "label"])
    labels = data["label"].to_numpy()
    transitions = np.flatnonzero((labels[:-1] == 1) & (labels[1:] == 0)) + 1
    if transitions.size == 0:
        return None
    return int(data["frame_index"].iloc[int(transitions[0])])


def save_force_curves(
    output_dir,
    pred_force,
    gt_force=None,
    failure_transition_frame=None,
    global_start=0,
):
    if pred_force is None:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.asarray(pred_force)[0]
    gt = np.asarray(gt_force)[0] if gt_force is not None else None
    n_dims = min(6, pred.shape[-1])
    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    x_pred = np.arange(pred.shape[0])

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.reshape(-1)
    for dim in range(6):
        ax = axes[dim]
        if dim < n_dims:
            ax.plot(x_pred, pred[:, dim], label="pred", color="tomato", linewidth=1.8)
            if gt is not None and dim < gt.shape[-1]:
                x_gt = np.arange(gt.shape[0])
                ax.plot(x_gt, gt[:, dim], label="gt", color="cornflowerblue", linewidth=1.6, alpha=0.9)
            transition_x = (
                failure_transition_frame - int(global_start)
                if failure_transition_frame is not None
                else None
            )
            if transition_x is not None and 0 <= transition_x < pred.shape[0]:
                ax.axvline(
                    transition_x,
                    color="green",
                    linestyle="--",
                    linewidth=1.6,
                    label="failure onset" if dim == 0 else None,
                )
            ax.set_title(labels[dim])
            ax.grid(True, linestyle=":", alpha=0.5)
        else:
            ax.axis("off")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=len(handles))
    fig.supxlabel("trajectory timestep")
    fig.supylabel("denormalized force / torque")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(output_dir, "force_curves.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_force_diff_curves(
    output_dir,
    pred_force,
    gt_force,
    failure_transition_frame=None,
    global_start=0,
    force_ylim=5.0,
    torque_ylim=0.5,
    window_size=30,
):
    if pred_force is None or gt_force is None:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.asarray(pred_force)[0]
    gt = np.asarray(gt_force)[0]
    length = min(pred.shape[0], gt.shape[0])
    diff = np.abs(pred[:length] - gt[:length])
    window_mean = np.empty_like(diff)
    for start in range(0, length, window_size):
        end = min(start + window_size, length)
        window_mean[start:end] = diff[start:end].mean(axis=0, keepdims=True)
    n_dims = min(6, diff.shape[-1])
    labels = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
    x = np.arange(length)
    transition_x = (
        failure_transition_frame - int(global_start)
        if failure_transition_frame is not None
        else None
    )
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
    axes = axes.reshape(-1)
    for dim in range(6):
        ax = axes[dim]
        if dim < n_dims:
            ax.plot(x, diff[:, dim], color="darkorange", linewidth=1.6, label="absolute error")
            ax.plot(
                x,
                window_mean[:, dim],
                color="gray",
                linestyle="--",
                linewidth=1.8,
                label=f"{window_size}-step window mean" if dim == 0 else None,
            )
            ax.set_ylim(0.0, force_ylim if dim < 3 else torque_ylim)
            if transition_x is not None and 0 <= transition_x < length:
                ax.axvline(
                    transition_x,
                    color="green",
                    linestyle="--",
                    linewidth=1.6,
                    label="failure onset" if dim == 0 else None,
                )
            ax.set_title(labels[dim])
            ax.grid(True, linestyle=":", alpha=0.5)
        else:
            ax.axis("off")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, legend_labels, loc="upper center", ncol=len(handles))
    fig.supxlabel("trajectory timestep")
    fig.supylabel("absolute prediction error")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = os.path.join(output_dir, "force_diff.png")
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def save_risk_curve(
    output_dir,
    risk_prob,
    risk_target,
    failure_transition_frame=None,
    global_start=0,
    filename="risk_curve.png",
):
    if risk_prob is None or risk_target is None:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = min(len(risk_prob), len(risk_target))
    x = np.arange(length)
    risk_prob = np.asarray(risk_prob)[:length]
    risk_target = np.asarray(risk_target)[:length]
    transition_x = (
        failure_transition_frame - int(global_start)
        if failure_transition_frame is not None
        else None
    )

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(
        x,
        risk_target,
        color="cornflowerblue",
        linewidth=2.0,
        drawstyle="steps-post",
        label="risk label",
    )
    ax.plot(
        x,
        risk_prob,
        color="tomato",
        linewidth=1.8,
        drawstyle="steps-post",
        label="risk probability",
    )
    if transition_x is not None and 0 <= transition_x < length:
        ax.axvline(
            transition_x,
            color="green",
            linestyle="--",
            linewidth=1.6,
            label="failure onset",
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("trajectory timestep")
    ax.set_ylabel("window failure ratio")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper center", ncol=3)
    ax.set_title(f"risk MAE={np.abs(risk_prob - risk_target).mean():.6f}")
    fig.tight_layout()

    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def resolve_video_template_and_cams(cfg, dataset, dataset_index, split="train"):
    data_cfg = cfg.get("data", {})
    cam_split = split
    if cam_split not in data_cfg:
        for fallback in ("val", "train"):
            if fallback in data_cfg:
                cam_split = fallback
                break
    if cam_split not in data_cfg:
        raise KeyError(f"Could not resolve camera config for split={split}; available data splits={list(data_cfg)}")
    cam_names = list(data_cfg[cam_split]["valid_cam"])
    item = dataset.dataset[int(dataset_index)]
    cam_segments = item[9] if len(item) > 9 else None
    if isinstance(cam_segments, dict):
        missing = [cam for cam in cam_names if cam not in cam_segments]
        if missing:
            raise KeyError(
                f"Missing video metadata for cameras {missing}; available={list(cam_segments)}"
            )
        return cam_segments, cam_names
    return item[0], cam_names


def read_video_segment_rgb(video_path, start_frame, num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = max(0, min(int(start_frame), max(0, total - 1)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame[:, :, ::-1].copy())
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames read from {video_path} at start_frame={start_frame}")
    return np.stack(frames, axis=0)


def load_multiview_video_segment(video_source, cam_names, start_frame, num_frames):
    per_cam = []
    actual_len = None
    for cam_name in cam_names:
        frame_start = start_frame
        if isinstance(video_source, dict):
            segment = video_source[cam_name]
            video_path = segment["video_path"]
            offset = float(segment.get("offset", 0.0))
            if offset:
                cap = cv2.VideoCapture(video_path)
                try:
                    if not cap.isOpened():
                        raise RuntimeError(f"Failed to open video: {video_path}")
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                finally:
                    cap.release()
                frame_start = int(round(float(start_frame) + offset * fps))
        elif "{}" in video_source:
            video_path = video_source.format(cam_name)
        else:
            video_path = video_source
        cam_frames = read_video_segment_rgb(video_path, frame_start, num_frames)
        actual_len = cam_frames.shape[0] if actual_len is None else min(actual_len, cam_frames.shape[0])
        per_cam.append(cam_frames)

    per_cam = [frames[:actual_len] for frames in per_cam]
    return np.stack(per_cam, axis=1)





def _tensor_video_to_uint8(video_tensor, n_view):
    video = video_tensor.detach().cpu().float()
    if video.ndim != 5:
        raise ValueError(f"Expected pred video [(B*V),C,T,H,W], got {tuple(video.shape)}")
    video = video[:n_view]
    video = video.clamp(-1, 1)
    video = ((video + 1.0) * 127.5).to(torch.uint8)
    video = video.permute(2, 0, 3, 4, 1).numpy()
    return video


def aggregate_pred_video_chunks(chunks, total_frames, n_view, video_stride):
    frame_store = {}
    for chunk in chunks:
        video = chunk.get("video")
        if video is None:
            continue
        pred_frames = _tensor_video_to_uint8(video, n_view=n_view)
        start = int(chunk["fix_sidx"])
        valid_len = int(chunk["valid_len"])
        if pred_frames.shape[0] <= 0:
            continue
        for local_t in range(valid_len):
            global_t = start + local_t
            if global_t >= total_frames or global_t in frame_store:
                continue
            pred_index = min(pred_frames.shape[0] - 1, local_t // max(1, int(video_stride)))
            frame_store[global_t] = pred_frames[pred_index]
    if not frame_store:
        return None
    first = next(iter(frame_store.values()))
    out = np.zeros((total_frames, first.shape[0], first.shape[1], first.shape[2], first.shape[3]), dtype=np.uint8)
    known = sorted(frame_store)
    last_value = frame_store[known[0]]
    for t in range(total_frames):
        if t in frame_store:
            last_value = frame_store[t]
        out[t] = last_value
    return out


def aggregate_window_chunks(chunks, total_frames, force_dim, global_start=0):
    length = total_frames - int(global_start)
    pred_sum = np.zeros((length, force_dim), dtype=np.float64)
    pred_square_sum = np.zeros((length, force_dim), dtype=np.float64)
    gt_sum = np.zeros((length, force_dim), dtype=np.float64)
    risk_sum = np.zeros(length, dtype=np.float64)
    risk_target_sum = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.int64)
    risk_count = np.zeros(length, dtype=np.int64)

    for chunk in chunks:
        local_start = int(chunk["fix_sidx"]) - int(global_start)
        valid_len = min(int(chunk["valid_len"]), length - local_start)
        if valid_len <= 0:
            continue
        end = local_start + valid_len
        pred = chunk.get("pred_force_denorm")
        gt = chunk.get("gt_force_denorm")
        if pred is not None and gt is not None:
            pred_values = pred[0, :valid_len].astype(np.float64)
            gt_values = gt[0, :valid_len].astype(np.float64)
            pred_sum[local_start:end] += pred_values
            pred_square_sum[local_start:end] += pred_values**2
            gt_sum[local_start:end] += gt_values
            count[local_start:end] += 1
        if chunk.get("risk_prob") is not None and chunk.get("risk_target") is not None:
            risk_sum[local_start:end] += float(chunk["risk_prob"])
            risk_target_sum[local_start:end] += float(chunk["risk_target"])
            risk_count[local_start:end] += 1

    if np.any(count == 0):
        missing = np.flatnonzero(count == 0)
        raise RuntimeError(f"Uncovered force timesteps: {missing[:20].tolist()}")

    pred_mean = pred_sum / count[:, None]
    gt_mean = gt_sum / count[:, None]
    pred_variance = np.maximum(pred_square_sum / count[:, None] - pred_mean**2, 0.0)
    pred_std = np.sqrt(pred_variance)

    risk_prob = None
    risk_target = None
    if np.any(risk_count > 0):
        if np.any(risk_count == 0):
            missing = np.flatnonzero(risk_count == 0)
            raise RuntimeError(f"Uncovered risk timesteps: {missing[:20].tolist()}")
        risk_prob = risk_sum / risk_count
        risk_target = risk_target_sum / risk_count

    return pred_mean, gt_mean, pred_std, count, risk_prob, risk_target


def extract_episode_id_from_path(parquet_path):
    match = re.search(r"episode_(\d+)\.parquet$", os.path.basename(str(parquet_path)))
    if match is None:
        raise ValueError(f"Cannot parse episode id from {parquet_path}")
    return int(match.group(1))


def _manifest_episode_ids(manifest, split, domain, manifest_path):
    if split in manifest:
        split_spec = manifest[split]
        if isinstance(split_spec, list):
            return {int(value) for value in split_spec}
        if isinstance(split_spec, dict):
            if domain not in split_spec:
                raise KeyError(f"Domain '{domain}' not found in split '{split}' manifest: {manifest_path}")
            return {int(value) for value in split_spec[domain]}
        raise TypeError(f"Unsupported split manifest format for split '{split}': {type(split_spec)}")

    legacy_key = "val_episode_indices" if split == "val" else "train_episode_indices"
    if split == "test" and "test_episode_indices" in manifest:
        legacy_key = "test_episode_indices"
    if legacy_key not in manifest:
        raise KeyError(f"Split '{split}' not found in split manifest: {manifest_path}")
    return {int(value) for value in manifest[legacy_key]}


def prepare_split_dataset(cfg, split="val", split_manifest=None):
    dataset_class_name = cfg.get("val_data_class", cfg["train_data_class"]) if split != "train" else cfg["train_data_class"]
    dataset_class_path = cfg.get("val_data_class_path", cfg["train_data_class_path"]) if split != "train" else cfg["train_data_class_path"]
    dataset_class = import_custom_class(dataset_class_name, dataset_class_path)

    data_cfg = data_config_for_split(cfg, split)
    split_manifest = split_manifest or data_cfg.get("split_manifest") or cfg.get("split_manifest")
    if split_manifest:
        apply_split_manifest_data_root(data_cfg, split_manifest)
        data_cfg["split_manifest"] = str(Path(split_manifest).expanduser())
        data_cfg["split"] = split
        dataset = dataset_class(**data_cfg)
        return dataset, list(range(len(dataset.dataset))), {str(item[3]): Path(split_manifest).expanduser() for item in dataset.dataset}

    domains, roots = _expand_roots_for_domains(data_cfg)
    split_ratio = data_cfg.get("split_ratio", data_cfg.get("val_ratio"))
    split_seed = int(data_cfg.get("split_seed", 0))
    split_episode_ids = {}
    manifest_paths = {}
    for data_root, domain in zip(roots, domains):
        manifest_path = split_manifest_path(data_cfg | {"data_roots": [data_root]}, split, domain)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        split_episode_ids[domain] = _manifest_episode_ids(manifest, split, domain, manifest_path)
        manifest_paths[domain] = manifest_path

    data_cfg["split_ratio"] = None
    data_cfg["val_ratio"] = None
    data_cfg["train_dataset"] = True
    dataset = dataset_class(**data_cfg)
    selected_indices = []
    for dataset_index, item in enumerate(dataset.dataset):
        domain = item[3]
        episode_id = extract_episode_id_from_path(item[2])
        if episode_id in split_episode_ids.get(domain, set()):
            selected_indices.append(dataset_index)

    expected = sum(len(values) for values in split_episode_ids.values())
    if len(selected_indices) != expected:
        found = {
            (dataset.dataset[index][3], extract_episode_id_from_path(dataset.dataset[index][2]))
            for index in selected_indices
        }
        missing = [
            (domain, episode_id)
            for domain, episode_ids in split_episode_ids.items()
            for episode_id in sorted(episode_ids)
            if (domain, episode_id) not in found
        ]
        raise RuntimeError(
            f"Only found {len(selected_indices)}/{expected} episodes for split '{split}'; "
            f"missing examples: {missing[:20]}"
        )
    return dataset, selected_indices, manifest_paths


def prepare_val_dataset(cfg):
    return prepare_split_dataset(cfg, split="val")
