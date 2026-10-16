import argparse
import json
import os
import re
import sys
from datetime import datetime

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from runner.ge_trainer_video_action_force_anomaly import Trainer
from utils import import_custom_class


def _ensure_importable_source(source):
    if not source or (not str(source).endswith(".py") and not os.path.isabs(str(source))):
        return
    path = os.path.abspath(str(source))
    directory = os.path.dirname(path)
    if directory and directory not in sys.path:
        sys.path.insert(0, directory)


def _ensure_process_group_for_base_trainer():
    if not dist.is_available() or dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)


class IndexedSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(idx) for idx in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        source_idx = self.indices[int(idx)]
        sample = self.dataset[source_idx]
        sample["_cache_index"] = torch.tensor(source_idx, dtype=torch.long)
        for key, value in _metadata_for_dataset_index(self.dataset, source_idx).items():
            sample.setdefault(key, value)
        return sample


def _dist_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _move_scalar(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _extract_episode_id(parquet_path):
    match = re.search(r"episode_(\d+)\.parquet$", os.path.basename(str(parquet_path)))
    return int(match.group(1)) if match else -1


def _metadata_for_dataset_index(dataset, source_idx):
    source_idx = int(source_idx)
    episode_dataset_idx = source_idx
    chunk_end = None
    if hasattr(dataset, "traverse_index"):
        episode_dataset_idx, chunk_end = dataset.traverse_index[source_idx]
        episode_dataset_idx = int(episode_dataset_idx)
        chunk_end = int(chunk_end)

    meta = {
        "_cache_source_index": torch.tensor(source_idx, dtype=torch.long),
        "_cache_episode_dataset_index": torch.tensor(episode_dataset_idx, dtype=torch.long),
    }
    base_items = getattr(dataset, "dataset", None)
    if base_items is None or episode_dataset_idx < 0 or episode_dataset_idx >= len(base_items):
        return meta

    item = base_items[episode_dataset_idx]
    parquet_path = str(item[2]) if len(item) > 2 else ""
    domain_name = str(item[3]) if len(item) > 3 else ""
    total_frames = int(item[7]) if len(item) > 7 else -1
    episode_id = int(item[8]) if len(item) > 8 and item[8] is not None else _extract_episode_id(parquet_path)
    future_steps = int(getattr(dataset, "future_steps", getattr(dataset, "action_chunk", 0)) or 0)
    if chunk_end is None:
        chunk_end = -1
        window_start = -1
        window_end = -1
        valid_len = 0
    else:
        window_start = int(chunk_end) - future_steps
        if window_start >= 0 and future_steps > 0:
            window_end = min(window_start + future_steps, total_frames)
            valid_len = max(0, window_end - window_start)
        else:
            window_end = -1
            valid_len = 0
        chunk_end = window_end

    meta.update(
        {
            "_cache_domain_name": domain_name,
            "_cache_episode_id": torch.tensor(episode_id, dtype=torch.long),
            "_cache_parquet_path": parquet_path,
            "_cache_window_start": torch.tensor(window_start, dtype=torch.long),
            "_cache_window_end": torch.tensor(window_end, dtype=torch.long),
            "_cache_chunk_end": torch.tensor(int(chunk_end), dtype=torch.long),
            "_cache_valid_len": torch.tensor(valid_len, dtype=torch.long),
            "_cache_total_frames": torch.tensor(total_frames, dtype=torch.long),
        }
    )
    return meta


def _batch_value(batch, key, row, default=None):
    if key not in batch:
        return default
    value = batch[key]
    if torch.is_tensor(value):
        item = value[row]
        return int(item.item()) if item.numel() == 1 else item.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return value[row]
    return value


def _sample_metadata_from_batch(batch, row):
    keys = {
        "source_index": "_cache_source_index",
        "domain_name": "_cache_domain_name",
        "episode_id": "_cache_episode_id",
        "episode_dataset_index": "_cache_episode_dataset_index",
        "parquet_path": "_cache_parquet_path",
        "window_start": "_cache_window_start",
        "window_end": "_cache_window_end",
        "chunk_end": "_cache_chunk_end",
        "valid_len": "_cache_valid_len",
        "total_frames": "_cache_total_frames",
    }
    meta = {}
    for out_key, batch_key in keys.items():
        value = _batch_value(batch, batch_key, row)
        if value is not None:
            meta[out_key] = value
    return meta


def _sample_from_batch(batch, preds, pred_latents, history_force_states, force_targets, gt_latents, row):
    sample = {
        "pred_force": preds["force"][row].detach().cpu().to(torch.float16),
        "gt_force": force_targets[row].detach().cpu().to(torch.float16),
        "gt_force_norm": force_targets[row].detach().cpu().to(torch.float16),
        "interaction_tokens": preds["interaction_tokens"][row].detach().cpu().to(torch.float16),
        "pred_latents": pred_latents[row].detach().cpu().to(torch.float16),
        "history_force_state": history_force_states[row].detach().cpu().to(torch.float16),
        "label": _move_scalar(batch["label"][row]),
        "episode_label": _move_scalar(batch["episode_label"][row]) if "episode_label" in batch else torch.tensor(-1),
        "risk_label": _move_scalar(batch["risk_label"][row]),
        "frame_anomaly_label": batch["frame_anomaly_label"][row].detach().cpu().to(torch.float32),
        "interaction_label": _move_scalar(batch["interaction_label"][row]),
        "meta": _sample_metadata_from_batch(batch, row),
    }
    if gt_latents is not None:
        sample["gt_latents"] = gt_latents[row].detach().cpu().to(torch.float16)
    return sample


def _flush_shard(split_dir, rank, shard_id, samples, sample_indices, entries):
    shard_name = f"rank_{rank:05d}_shard_{shard_id:05d}.pt"
    torch.save({"samples": samples}, os.path.join(split_dir, shard_name))
    for offset, sample_index in enumerate(sample_indices):
        entry = {
            "shard": shard_name,
            "offset": offset,
            "index": int(sample_index),
        }
        entry.update(samples[offset].get("meta", {}))
        entries.append(entry)


def _split_manifest_has_split(split_manifest, split):
    if not split_manifest:
        return False
    with open(split_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return split in manifest


def _data_cfg_for_split(trainer, split):
    if split in trainer.args.data:
        data_cfg = dict(trainer.args.data[split])
    else:
        base_split = "val" if "val" in trainer.args.data else "train"
        data_cfg = dict(trainer.args.data[base_split])
        data_cfg["split"] = split

    split_manifest = data_cfg.get("split_manifest") or getattr(trainer.args, "split_manifest", None)
    if split == "test":
        if not _split_manifest_has_split(split_manifest, split):
            print(f"Skip precompute split '{split}': split_manifest is missing or has no '{split}' split.")
            return None
        data_cfg["split_manifest"] = split_manifest
        data_cfg["split"] = split
    elif split not in trainer.args.data:
        if not _split_manifest_has_split(split_manifest, split):
            print(f"Skip precompute split '{split}': no data.{split} config and no manifest split.")
            return None
        data_cfg["split_manifest"] = split_manifest
        data_cfg["split"] = split

    data_cfg["memory_pick_mode"] = "uniform"
    return data_cfg


def _build_dataset(trainer, split):
    data_cfg = _data_cfg_for_split(trainer, split)
    if data_cfg is None:
        return None
    class_name = trainer.args.train_data_class if split == "train" else getattr(trainer.args, "val_data_class", trainer.args.train_data_class)
    class_path = trainer.args.train_data_class_path if split == "train" else getattr(
        trainer.args,
        "val_data_class_path",
        trainer.args.train_data_class_path,
    )
    _ensure_importable_source(class_path)
    dataset_class = import_custom_class(class_name, class_path)
    return dataset_class(**data_cfg)


def precompute_split(trainer, split, output_dir, batch_size, num_workers, shard_size, save_gt_latents=False):
    rank, world_size = _dist_info()
    dataset = _build_dataset(trainer, split)
    if dataset is None:
        return
    rank_indices = list(range(rank, len(dataset), world_size))
    rank_dataset = IndexedSubset(dataset, rank_indices)
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": getattr(trainer.args, "pin_memory", False),
    }
    # Models are already initialized on CUDA before precompute_split() is called.
    # Linux DataLoader defaults to fork, which can inherit a live CUDA context and
    # abort workers with `CUDA error: initialization error`. Spawn starts clean
    # worker processes and keeps dataset decoding CPU-only.
    if int(num_workers) > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(rank_dataset, **loader_kwargs)

    accelerator = trainer.state.accelerator
    weight_dtype = trainer.state.weight_dtype
    pipe = trainer._make_pipe()
    pipe = pipe.to(accelerator.device) if hasattr(pipe, "to") else pipe

    entries = []
    shard_samples = []
    shard_indices = []
    shard_id = 0

    progress_desc = f"precompute {split} rank {rank}/{world_size}"
    for batch in tqdm(loader, desc=progress_desc, disable=not accelerator.is_local_main_process):
        cache_indices = batch.pop("_cache_index")
        (
            video,
            mem,
            future_video,
            future_actions,
            force_targets,
            history_force_states,
            _normal_mask,
            batch_size_actual,
            n_view,
            mem_size,
        ) = trainer._prepare_action_force_batch(batch, accelerator, weight_dtype)
        _, _, _, height, width = video.shape
        with torch.no_grad():
            preds = trainer._run_frozen_inference(
                pipe,
                mem,
                batch,
                future_actions,
                history_force_states,
                batch_size_actual,
                n_view,
                mem_size,
                height,
                width,
            )
        pred_latents = preds["pred_latents"].reshape(
            batch_size_actual,
            n_view,
            preds["pred_latents"].shape[1],
            preds["pred_latents"].shape[2],
            preds["pred_latents"].shape[3],
            preds["pred_latents"].shape[4],
        )
        gt_latents = None
        if save_gt_latents:
            with torch.no_grad():
                gt_latents = trainer._encode_future_latents(
                    future_video,
                    batch_size_actual,
                    n_view,
                    weight_dtype,
                )

        for row in range(batch_size_actual):
            shard_samples.append(
                _sample_from_batch(
                    batch,
                    preds,
                    pred_latents,
                    history_force_states,
                    force_targets,
                    gt_latents,
                    row,
                )
            )
            shard_indices.append(int(cache_indices[row].item()))
            if len(shard_samples) >= shard_size:
                _flush_shard(split_dir, rank, shard_id, shard_samples, shard_indices, entries)
                shard_samples = []
                shard_indices = []
                shard_id += 1

    if shard_samples:
        _flush_shard(split_dir, rank, shard_id, shard_samples, shard_indices, entries)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config_file": trainer.config_file if hasattr(trainer, "config_file") else None,
        "split": split,
        "num_samples": len(dataset),
        "rank_samples": len(entries),
        "world_size": world_size,
        "memory_pick_mode": "uniform",
        "num_inference_step": int(getattr(trainer.args, "num_inference_step", 5)),
        "noise_seed": None if getattr(trainer.args, "noise_seed", None) is None else int(trainer.args.noise_seed),
        "anomaly_guidance_scale": float(getattr(trainer.args, "anomaly_guidance_scale", 1.0)),
        "diffusion_model_path": trainer.args.diffusion_model.get("model_path"),
        "save_gt_latents": bool(save_gt_latents),
    }

    rank_index_path = os.path.join(split_dir, f"index_rank_{rank:05d}.json")
    with open(rank_index_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": metadata, "entries": entries}, f, indent=2)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if rank == 0:
        merged_entries = []
        for merge_rank in range(world_size):
            with open(os.path.join(split_dir, f"index_rank_{merge_rank:05d}.json"), "r", encoding="utf-8") as f:
                rank_payload = json.load(f)
            merged_entries.extend(rank_payload.get("entries", []))
        merged_entries = sorted(merged_entries, key=lambda item: int(item["index"]))
        metadata["num_samples"] = len(merged_entries)
        metadata["rank_samples"] = None
        with open(os.path.join(split_dir, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"metadata": metadata, "entries": merged_entries}, f, indent=2)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=128)
    parser.add_argument("--noise_seed", type=int, default=42)
    parser.add_argument("--save_gt_latents", action="store_true")
    args = parser.parse_args()

    _ensure_process_group_for_base_trainer()
    trainer = Trainer(args.config_file, to_log=False)
    trainer.config_file = args.config_file
    trainer.args.anomaly_feature_cache_dir = None
    trainer.args.noise_seed = args.noise_seed
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()

    batch_size = args.batch_size or trainer.args.batch_size
    num_workers = trainer.args.dataloader_num_workers if args.num_workers is None else args.num_workers
    os.makedirs(args.output_dir, exist_ok=True)

    for split in args.splits:
        precompute_split(
            trainer=trainer,
            split=split,
            output_dir=args.output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            shard_size=args.shard_size,
            save_gt_latents=args.save_gt_latents,
        )


if __name__ == "__main__":
    main()
