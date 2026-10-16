import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.baselines.pca_kmeans import pooled_memory_video_latents
from runner.pca_kmeans_trainer import Trainer
from utils import import_custom_class
from utils.model_utils import load_latent_models, load_vae_models


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


def _batch_value(batch, key, row):
    if key not in batch:
        return None
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


def _sample_from_batch(batch, video_features, row):
    sample = {
        "video_features": video_features[row].detach().cpu().to(torch.float16),
        "label": _move_scalar(batch["label"][row]),
        "episode_label": _move_scalar(batch["episode_label"][row]),
        "risk_label": _move_scalar(batch["risk_label"][row]),
        "frame_anomaly_label": batch["frame_anomaly_label"][row].detach().cpu().to(torch.float32),
        "interaction_label": _move_scalar(batch["interaction_label"][row]),
        "meta": _sample_metadata_from_batch(batch, row),
    }
    if "state" in batch:
        sample["state"] = batch["state"][row].detach().cpu().to(torch.float16)
    if "history_force_state" in batch:
        sample["history_force_state"] = batch["history_force_state"][row].detach().cpu().to(torch.float16)
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
    if split not in trainer.args.data:
        if not _split_manifest_has_split(split_manifest, split):
            print(f"Skip observed precompute split '{split}': no data.{split} config and no manifest split.")
            return None
        data_cfg["split_manifest"] = split_manifest
        data_cfg["split"] = split
    elif split_manifest:
        data_cfg["split_manifest"] = split_manifest

    # The configured data_roots is authoritative.  A manifest only selects
    # episode indices; its historical data_root metadata must not silently
    # redirect cache generation to another dataset.
    data_cfg["memory_pick_mode"] = "uniform"
    return data_cfg


def _build_dataset(trainer, split):
    data_cfg = _data_cfg_for_split(trainer, split)
    if data_cfg is None:
        return None, None
    class_name = trainer.args.train_data_class if split == "train" else getattr(trainer.args, "val_data_class", trainer.args.train_data_class)
    class_path = trainer.args.train_data_class_path if split == "train" else getattr(
        trainer.args,
        "val_data_class_path",
        trainer.args.train_data_class_path,
    )
    dataset_class = import_custom_class(class_name, class_path)
    return dataset_class(**data_cfg), data_cfg


def _prepare_vae_for_precompute(trainer):
    device = trainer.state.accelerator.device
    dtype = trainer.state.weight_dtype
    vae_class = import_custom_class(trainer.args.vae_class, getattr(trainer.args, "vae_class_path", "transformers"))
    if getattr(trainer.args, "vae_path", False):
        trainer.vae = load_vae_models(vae_class, trainer.args.vae_path).to(device, dtype=dtype).eval()
    else:
        trainer.vae = load_latent_models(vae_class, trainer.args.pretrained_model_name_or_path)["vae"].to(
            device,
            dtype=dtype,
        ).eval()
    if isinstance(trainer.vae.latents_mean, list):
        trainer.vae.latents_mean = torch.FloatTensor(trainer.vae.latents_mean)
    if isinstance(trainer.vae.latents_std, list):
        trainer.vae.latents_std = torch.FloatTensor(trainer.vae.latents_std)
    if getattr(trainer.args, "enable_slicing", False):
        trainer.vae.enable_slicing()
    if getattr(trainer.args, "enable_tiling", False):
        trainer.vae.enable_tiling()
    trainer.vae.requires_grad_(False)


def precompute_split(trainer, split, output_dir, batch_size, num_workers, shard_size):
    rank, world_size = _dist_info()
    dataset, data_cfg = _build_dataset(trainer, split)
    if dataset is None:
        return
    rank_indices = list(range(rank, len(dataset), world_size))
    rank_dataset = IndexedSubset(dataset, rank_indices)
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    loader = DataLoader(
        rank_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=getattr(trainer.args, "pin_memory", False),
    )

    accelerator = trainer.state.accelerator
    weight_dtype = trainer.state.weight_dtype
    mem_size = int(data_cfg.get("n_previous", data_cfg.get("memory_steps", 4)))

    entries = []
    shard_samples = []
    shard_indices = []
    shard_id = 0

    progress_desc = f"precompute observed {split} rank {rank}/{world_size}"
    for batch in tqdm(loader, desc=progress_desc, disable=not accelerator.is_local_main_process):
        cache_indices = batch.pop("_cache_index")
        video = batch["video"].to(device=accelerator.device, dtype=weight_dtype, non_blocking=True).contiguous()
        with torch.no_grad():
            video_features = pooled_memory_video_latents(
                trainer.vae,
                video,
                mem_size=mem_size,
                dtype=weight_dtype,
            )

        batch_size_actual = int(video_features.shape[0])
        for row in range(batch_size_actual):
            shard_samples.append(_sample_from_batch(batch, video_features, row))
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
        "cache_type": "observed_features",
        "contains": ["video_features", "state", "history_force_state", "labels", "metadata"],
        "mem_size": mem_size,
        "vae_path": getattr(trainer.args, "vae_path", None),
        "pretrained_model_name_or_path": getattr(trainer.args, "pretrained_model_name_or_path", None),
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
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--domains", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--shard_size", type=int, default=1024)
    args = parser.parse_args()

    _ensure_process_group_for_base_trainer()
    trainer = Trainer(args.config_file, to_log=False)
    trainer.config_file = args.config_file
    trainer.args.observed_feature_cache_dir = None
    if isinstance(getattr(trainer.args, "observed_feature", None), dict):
        trainer.args.observed_feature["cache_dir"] = None
    if args.split_manifest not in (None, ""):
        trainer.args.split_manifest = args.split_manifest
        for split in ("train", "val", "test"):
            if split in trainer.args.data:
                trainer.args.data[split]["split_manifest"] = args.split_manifest
    if args.domains:
        for split in ("train", "val", "test"):
            if split in trainer.args.data:
                trainer.args.data[split]["domains"] = list(args.domains)
    _prepare_vae_for_precompute(trainer)
    trainer.vae.eval()

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
        )


if __name__ == "__main__":
    main()
