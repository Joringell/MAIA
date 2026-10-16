import argparse
import glob
import json
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


DEFAULT_DATA_ROOT = "/root/CRIA"
DEFAULT_OPENPI_ROOT = "/root/openpi"

LABEL_COLUMNS = (
    "timestamp",
    "timestamp_us",
    "source_id",
    "task_index",
    "episode_label",
    "frame_label",
    "interaction_label",
)


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def _resolve_domain_root(data_root, domain_name):
    data_root = os.path.expanduser(str(data_root))
    direct = os.path.join(data_root, str(domain_name))
    if os.path.exists(os.path.join(direct, "meta", "info.json")):
        return direct

    matches = sorted(
        path
        for path in glob.glob(os.path.join(data_root, "*", str(domain_name)))
        if os.path.exists(os.path.join(path, "meta", "info.json"))
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find domain '{domain_name}' under {data_root}; expected "
            f"{direct}/meta/info.json or {data_root}/*/{domain_name}/meta/info.json"
        )
    if len(matches) > 1:
        raise ValueError(f"Domain '{domain_name}' is ambiguous under {data_root}: {matches}")
    return matches[0]


def _load_domain_episodes(domain_root):
    rows = _read_jsonl(os.path.join(domain_root, "meta", "episodes.jsonl"))
    by_episode = {}
    for row in rows:
        episode_index = int(row["episode_index"])
        if episode_index in by_episode:
            raise ValueError(f"Duplicate episode_index={episode_index} in {domain_root}")
        by_episode[episode_index] = row
    return by_episode


def _read_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest


def _manifest_episode_specs(manifest, split):
    if split not in manifest:
        raise KeyError(f"Split '{split}' not found in split manifest")
    split_spec = manifest[split]
    if not isinstance(split_spec, dict):
        raise TypeError(f"Expected manifest split '{split}' to be domain -> episode list, got {type(split_spec)}")
    specs = []
    for domain_name, episode_indices in split_spec.items():
        for episode_index in episode_indices:
            specs.append((str(domain_name), int(episode_index)))
    return specs


def _episode_parquet_path(domain_root, episode_index):
    matches = sorted(
        glob.glob(os.path.join(domain_root, "data", "**", f"episode_{int(episode_index):06d}.parquet"), recursive=True)
    )
    if not matches:
        matches = sorted(
            glob.glob(os.path.join(domain_root, "data", "**", f"file-{int(episode_index):03d}.parquet"), recursive=True)
        )
    if not matches:
        raise FileNotFoundError(f"Could not find parquet for episode {episode_index} under {domain_root}/data")
    if len(matches) > 1:
        raise ValueError(f"Episode {episode_index} parquet is ambiguous under {domain_root}: {matches}")
    return matches[0]


def _read_episode_frame_table(parquet_path, episode_row):
    df = pd.read_parquet(parquet_path)
    missing_required = [key for key in ("frame_label", "interaction_label") if key not in df.columns]
    if missing_required:
        raise KeyError(
            f"{parquet_path} is missing required label columns {missing_required}. "
            "Apply annotations to parquet before building the pi05 feature cache."
        )
    if "frame_index" not in df.columns:
        df = df.copy()
        df["frame_index"] = np.arange(len(df), dtype=np.int64)
    keep_columns = ["frame_index"] + [key for key in LABEL_COLUMNS if key in df.columns]
    df = df[keep_columns].copy()
    df["frame_index"] = df["frame_index"].astype(np.int64)
    if "episode_label" not in df.columns and "episode_label" in episode_row:
        df["episode_label"] = int(episode_row["episode_label"])
    return df


def _scalar_from_row(row, key, default):
    if key not in row or pd.isna(row[key]):
        return default
    value = row[key]
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _window_meta(frame_table, frame_index, query_stride, episode_row, domain_name, domain_root, parquet_path):
    total_frames = int(episode_row.get("length", len(frame_table)))
    end = min(int(frame_index) + int(query_stride), total_frames)
    window = frame_table[(frame_table["frame_index"] >= int(frame_index)) & (frame_table["frame_index"] < end)]
    query_rows = frame_table[frame_table["frame_index"] == int(frame_index)]
    query_row = query_rows.iloc[0] if len(query_rows) else (window.iloc[0] if len(window) else pd.Series(dtype=object))

    episode_label = int(_scalar_from_row(query_row, "episode_label", episode_row.get("episode_label", -1)))
    frame_label = int(_scalar_from_row(query_row, "frame_label", -1))
    if "frame_label" in window:
        frame_values = window["frame_label"].to_numpy(dtype=np.int64)
    else:
        frame_values = np.zeros((len(window),), dtype=np.int64)
    if "interaction_label" in window:
        interaction_label = int(window["interaction_label"].to_numpy(dtype=np.int64).max()) if len(window) else 0
    else:
        interaction_label = int(_scalar_from_row(query_row, "interaction_label", 0))

    frame_anomaly = np.full((int(query_stride),), -1.0, dtype=np.float32)
    valid_len = max(0, min(int(query_stride), len(frame_values)))
    if valid_len > 0:
        frame_anomaly[:valid_len] = (frame_values[:valid_len] == 0).astype(np.float32)

    meta = {
        "source_index": -1,
        "domain_name": str(domain_name),
        "episode_id": int(episode_row["episode_index"]),
        "episode_dataset_index": int(episode_row["episode_index"]),
        "source_episode_index": int(episode_row["episode_index"]),
        "parquet_path": str(parquet_path),
        "source_h5": str(episode_row.get("source_h5", "")),
        "source_repo_id": str(episode_row.get("source_repo_id", "")),
        "source": str(episode_row.get("source", "")),
        "control_group": str(episode_row.get("control_group", "")),
        "frame_index": int(frame_index),
        "window_start": int(frame_index),
        "window_end": int(end),
        "chunk_end": int(end),
        "valid_len": int(valid_len),
        "total_frames": int(total_frames),
        "timestamp": float(_scalar_from_row(query_row, "timestamp", -1.0)),
        "timestamp_us": int(_scalar_from_row(query_row, "timestamp_us", -1)),
        "source_id": int(_scalar_from_row(query_row, "source_id", episode_row.get("source_id", -1))),
        "task_index": int(_scalar_from_row(query_row, "task_index", -1)),
        "episode_label": int(episode_label),
        "frame_label": int(frame_label),
        "interaction_label": int(interaction_label),
        "domain_root": str(domain_root),
    }
    return meta, frame_anomaly


def _target_frame_indices(frame_table, query_stride):
    values = sorted(int(value) for value in frame_table["frame_index"].to_numpy())
    return [frame for frame in values if frame % int(query_stride) == 0]


def _compress_pre_velocity(pre_velocity):
    pre_velocity = np.asarray(pre_velocity)
    if pre_velocity.ndim >= 3:
        return pre_velocity.mean(axis=tuple(range(pre_velocity.ndim - 1)))
    return pre_velocity


def _to_numpy(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_numpy(item) for item in value]
    return value


def _as_scalar(value: Any) -> Any:
    value = _to_numpy(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def _prepare_image(value: Any) -> Any:
    value = _to_numpy(value)
    if not isinstance(value, np.ndarray):
        return value
    if value.ndim == 3 and value.shape[0] in (1, 3, 4):
        value = np.moveaxis(value, 0, -1)
    if np.issubdtype(value.dtype, np.floating):
        max_value = float(np.nanmax(value)) if value.size else 0.0
        if max_value <= 1.0:
            value = value * 255.0
        value = np.clip(value, 0, 255).astype(np.uint8)
    return value


def _prepare_sample(sample):
    sample = _to_numpy(sample)
    prepared = {}
    for key, value in sample.items():
        if "image" in key or (isinstance(value, np.ndarray) and value.ndim == 3):
            prepared[key] = _prepare_image(value)
        else:
            prepared[key] = value
    return prepared


def _feature_tensor(feature, dtype_name):
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    return torch.as_tensor(np.asarray(feature), dtype=dtype).detach().cpu()


def _make_sample(feature, meta, frame_anomaly, dtype_name):
    episode_label = int(meta.get("episode_label", -1))
    frame_label = int(meta.get("frame_label", -1))
    interaction_label = int(meta.get("interaction_label", -1))
    frame_anomaly = np.asarray(frame_anomaly, dtype=np.float32)
    valid = frame_anomaly >= 0
    risk_label = float(frame_anomaly[valid].mean()) if bool(valid.any()) else 0.0
    window_label = 0 if risk_label > 0.0 else 1
    sample_meta = dict(meta)
    sample_meta["label"] = int(window_label)
    sample_meta["risk_label"] = float(risk_label)
    return {
        "pi05_hidden": _feature_tensor(feature, dtype_name),
        "label": torch.tensor(window_label, dtype=torch.long),
        "episode_label": torch.tensor(episode_label, dtype=torch.long),
        "frame_label": torch.tensor(frame_label, dtype=torch.long),
        "frame_anomaly_label": torch.as_tensor(frame_anomaly, dtype=torch.float32),
        "risk_label": torch.tensor(risk_label, dtype=torch.float32),
        "interaction_label": torch.tensor(interaction_label, dtype=torch.long),
        "meta": sample_meta,
    }


class _Pi05Extractor:
    def __init__(self, args):
        self.args = args
        self.policy = None
        self.train_config = None
        self.data_config = None
        self.modules = None
        self.domain_dataset_cache = {}

    def _ensure_imports(self):
        if self.modules is not None:
            return
        openpi_src = os.path.join(os.path.expanduser(self.args.openpi_root), "src")
        if openpi_src not in sys.path:
            sys.path.insert(0, openpi_src)
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from openpi import transforms as openpi_transforms
        from openpi.policies import policy_config as openpi_policy_config
        from openpi.training import config as openpi_config

        self.modules = {
            "LeRobotDataset": LeRobotDataset,
            "transforms": openpi_transforms,
            "policy_config": openpi_policy_config,
            "config": openpi_config,
        }

    def _load_metadata(self, repo_id, dataset_root):
        self._ensure_imports()
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
        except ImportError:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        try:
            return LeRobotDatasetMetadata(repo_id=repo_id, root=dataset_root)
        except TypeError:
            return LeRobotDatasetMetadata(repo_id)

    def _build_delta_timestamps(self, metadata):
        fps = float(getattr(metadata, "fps"))
        return {
            key: [step / fps for step in range(self.train_config.model.action_horizon)]
            for key in self.data_config.action_sequence_keys
        }

    def _ensure_policy(self):
        if self.policy is not None:
            return
        if not self.args.checkpoint_dir:
            raise ValueError(
                "Missing pi05 features require extraction, but --checkpoint-dir was not provided."
            )
        self._ensure_imports()
        self.train_config = self.modules["config"].get_config(self.args.config)
        self.data_config = self.train_config.data.create(self.train_config.assets_dirs, self.train_config.model)
        self.policy = self.modules["policy_config"].create_trained_policy(
            self.train_config,
            self.args.checkpoint_dir,
            repack_transforms=self.data_config.repack_transforms,
            sample_kwargs={"return_aux": True, "num_steps": int(self.args.num_steps)},
            default_prompt=self.args.default_prompt,
        )

    def _task_prompt_from_dataset(self, dataset, sample):
        if "prompt" in sample:
            return str(_as_scalar(sample["prompt"]))
        if "task" in sample:
            return str(_as_scalar(sample["task"]))
        if "task_index" not in sample:
            return None
        task_index = int(_as_scalar(sample["task_index"]))
        tasks = getattr(getattr(dataset, "meta", None), "tasks", None)
        if isinstance(tasks, dict) and task_index in tasks:
            return str(tasks[task_index])
        return None

    def _domain_dataset(self, domain_root):
        domain_root = os.path.expanduser(str(domain_root))
        if domain_root in self.domain_dataset_cache:
            return self.domain_dataset_cache[domain_root]

        repo_id = os.path.basename(os.path.normpath(domain_root))
        metadata = self._load_metadata(repo_id, domain_root)
        delta_timestamps = self._build_delta_timestamps(metadata)
        dataset = self.modules["LeRobotDataset"](
            repo_id=repo_id,
            root=domain_root,
            delta_timestamps=delta_timestamps,
        )
        self.domain_dataset_cache[domain_root] = (dataset, metadata)
        return dataset, metadata

    def _sample_for_frame(self, dataset, episode_index, frame_index):
        episode_index = int(episode_index)
        frame_index = int(frame_index)
        starts = dataset.episode_data_index["from"]
        ends = dataset.episode_data_index["to"]
        if episode_index >= len(starts):
            raise IndexError(
                f"episode_index={episode_index} is out of bounds for dataset episode_data_index "
                f"with size {len(starts)}. Load the full domain dataset, not a filtered episode view."
            )

        ep_start = int(starts[episode_index].item())
        ep_end = int(ends[episode_index].item())
        candidates = []
        if ep_start <= ep_start + frame_index < ep_end:
            candidates.append(ep_start + frame_index)
        if 0 <= frame_index < len(dataset):
            candidates.append(frame_index)

        seen = set()
        for idx in candidates:
            if idx in seen:
                continue
            seen.add(idx)
            sample = _to_numpy(dataset[int(idx)])
            sample_ep = int(_as_scalar(sample.get("episode_index", episode_index)))
            sample_frame = int(_as_scalar(sample.get("frame_index", frame_index)))
            if sample_ep == episode_index and sample_frame == frame_index:
                return sample

        for idx in range(ep_start, ep_end):
            sample = _to_numpy(dataset[idx])
            if int(_as_scalar(sample.get("frame_index", -1))) == frame_index:
                return sample
        raise IndexError(f"Could not find episode_index={episode_index}, frame_index={frame_index} in LeRobotDataset")

    def extract_episode(self, domain_root, episode_index, target_frames):
        self._ensure_policy()
        repo_id = os.path.basename(os.path.normpath(domain_root))
        dataset, metadata = self._domain_dataset(domain_root)
        prompt_from_task = None
        if self.data_config.prompt_from_task:
            prompt_from_task = self.modules["transforms"].PromptFromLeRobotTask(getattr(metadata, "tasks", {}))

        features = {}
        for frame_index in tqdm(target_frames, desc=f"extract {repo_id} ep {episode_index}", leave=False):
            sample = self._sample_for_frame(dataset, int(episode_index), int(frame_index))
            if prompt_from_task is not None:
                sample = prompt_from_task(sample)
            sample = _prepare_sample(sample)
            if "prompt" not in sample:
                prompt = self.args.default_prompt or self._task_prompt_from_dataset(dataset, sample)
                if prompt is None:
                    raise KeyError(
                        'Sample does not contain "prompt", and no prompt could be inferred. '
                        "Pass --default-prompt or add a prompt field to the dataset sample."
                    )
                sample["prompt"] = prompt
            output = self.policy.infer(sample)
            if "pre_velocity" not in output:
                raise RuntimeError(
                    "Policy output did not include pre_velocity. Use a pi0/pi05 config and return_aux=True."
                )
            features[int(frame_index)] = _compress_pre_velocity(output["pre_velocity"])
        return features


def _flush_shard(split_dir, rank, shard_id, samples, sample_indices, entries):
    shard_name = f"rank_{int(rank):05d}_shard_{int(shard_id):05d}.pt"
    torch.save({"samples": samples}, os.path.join(split_dir, shard_name))
    for offset, sample_index in enumerate(sample_indices):
        entry = {
            "shard": shard_name,
            "offset": int(offset),
            "index": int(sample_index),
        }
        entry.update(samples[offset].get("meta", {}))
        entries.append(entry)


def _rank_partial_index_path(split_dir, rank):
    return os.path.join(split_dir, f"index_rank_{int(rank):05d}.json")


def _rank_partial_summary_path(output_dir, rank):
    return os.path.join(output_dir, f"summary_rank_{int(rank):05d}.json")


def _rank_stats_path(output_dir, rank):
    return os.path.join(output_dir, f"rank_{int(rank):05d}_done.json")


def _wait_for_paths(paths, timeout_seconds):
    deadline = datetime.now().timestamp() + max(0, int(timeout_seconds))
    missing = [path for path in paths if not os.path.exists(path)]
    while missing and datetime.now().timestamp() < deadline:
        import time

        time.sleep(5)
        missing = [path for path in paths if not os.path.exists(path)]
    if missing:
        raise TimeoutError(f"Timed out waiting for rank outputs: {missing}")


def _sum_metadata(partials, split, args):
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_type": "pi05_hidden_features",
        "split": split,
        "split_manifest": os.path.abspath(args.split_manifest),
        "data_root": os.path.abspath(args.data_root),
        "openpi_root": os.path.abspath(args.openpi_root),
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "query_stride": int(args.query_stride),
        "num_steps": int(args.num_steps),
        "feature_dtype": args.feature_dtype,
        "dry_run": bool(args.dry_run),
        "num_shards": int(args.num_shards),
        "rank_partials": [int(item["metadata"].get("shard_rank", -1)) for item in partials],
    }
    metadata["episodes_requested_total"] = int(
        max(int(item["metadata"].get("episodes_requested_total", 0)) for item in partials)
    )
    sum_keys = (
        "episodes_assigned",
        "episodes_extracted",
        "samples_extracted",
        "num_samples",
    )
    for key in sum_keys:
        metadata[key] = int(sum(int(item["metadata"].get(key, 0)) for item in partials))
    return metadata


def _merge_split_index(args, split):
    split_dir = os.path.join(args.output_dir, split)
    partial_paths = [_rank_partial_index_path(split_dir, rank) for rank in range(int(args.num_shards))]
    _wait_for_paths(partial_paths, args.merge_timeout_seconds)
    partials = []
    for path in partial_paths:
        with open(path, "r", encoding="utf-8") as f:
            partials.append(json.load(f))

    entries = []
    for payload in partials:
        entries.extend(payload.get("entries", []))
    entries.sort(
        key=lambda entry: (
            str(entry.get("domain_name", "")),
            int(entry.get("episode_id", entry.get("source_episode_index", -1))),
            str(entry.get("source_h5_basename", "")),
            int(entry.get("frame_index", -1)),
            int(entry.get("index", -1)),
        )
    )
    for index, entry in enumerate(entries):
        entry["index"] = int(index)

    metadata = _sum_metadata(partials, split, args)
    metadata["num_samples"] = len(entries)
    _write_json(os.path.join(split_dir, "index.json"), {"metadata": metadata, "entries": entries})
    return metadata


def precompute_split(args, manifest, split, extractor):
    split_dir = os.path.join(args.output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    all_specs = _manifest_episode_specs(manifest, split)
    if args.max_episodes_per_split is not None:
        all_specs = all_specs[: int(args.max_episodes_per_split)]
    specs = [
        spec
        for position, spec in enumerate(all_specs)
        if position % int(args.num_shards) == int(args.shard_rank)
    ]

    domain_episode_cache = {}
    frame_table_cache = {}
    entries = []
    shard_samples = []
    shard_indices = []
    shard_id = 0
    sample_index = int(args.shard_rank)
    stats = {
        "episodes_requested_total": len(all_specs),
        "episodes_assigned": len(specs),
        "episodes_extracted": 0,
        "samples_extracted": 0,
    }

    desc = f"precompute pi05 {split} rank {int(args.shard_rank)}/{int(args.num_shards)}"
    for domain_name, source_episode_index in tqdm(specs, desc=desc):
        if domain_name not in domain_episode_cache:
            domain_root = _resolve_domain_root(args.data_root, domain_name)
            domain_episode_cache[domain_name] = (domain_root, _load_domain_episodes(domain_root))
        domain_root, domain_episodes = domain_episode_cache[domain_name]
        if source_episode_index not in domain_episodes:
            raise KeyError(f"{domain_name} episode {source_episode_index} is in manifest but not in {domain_root}")
        episode_row = domain_episodes[source_episode_index]
        parquet_path = _episode_parquet_path(domain_root, source_episode_index)
        table_key = (domain_name, source_episode_index)
        if table_key not in frame_table_cache:
            frame_table_cache[table_key] = _read_episode_frame_table(parquet_path, episode_row)
        frame_table = frame_table_cache[table_key]
        target_frames = _target_frame_indices(frame_table, args.query_stride)

        if args.dry_run:
            stats["episodes_extracted"] += 1
            stats["samples_extracted"] += len(target_frames)
            continue
        extracted_features = extractor.extract_episode(domain_root, source_episode_index, target_frames)
        stats["episodes_extracted"] += 1

        for frame_index in target_frames:
            feature = extracted_features[frame_index]
            meta, frame_anomaly = _window_meta(
                frame_table,
                frame_index,
                args.query_stride,
                episode_row,
                domain_name,
                domain_root,
                parquet_path,
            )
            meta["source_index"] = int(sample_index)
            meta["shard_rank"] = int(args.shard_rank)
            sample = _make_sample(feature, meta, frame_anomaly, args.feature_dtype)
            shard_samples.append(sample)
            shard_indices.append(sample_index)
            sample_index += int(args.num_shards)
            stats["samples_extracted"] += 1
            if len(shard_samples) >= int(args.shard_size):
                _flush_shard(split_dir, args.shard_rank, shard_id, shard_samples, shard_indices, entries)
                shard_samples = []
                shard_indices = []
                shard_id += 1

    if shard_samples:
        _flush_shard(split_dir, args.shard_rank, shard_id, shard_samples, shard_indices, entries)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "cache_type": "pi05_hidden_features",
        "split": split,
        "split_manifest": os.path.abspath(args.split_manifest),
        "data_root": os.path.abspath(args.data_root),
        "openpi_root": os.path.abspath(args.openpi_root),
        "config": args.config,
        "checkpoint_dir": args.checkpoint_dir,
        "query_stride": int(args.query_stride),
        "num_steps": int(args.num_steps),
        "feature_dtype": args.feature_dtype,
        "dry_run": bool(args.dry_run),
        "shard_rank": int(args.shard_rank),
        "num_shards": int(args.num_shards),
    }
    metadata.update(stats)
    metadata["num_samples"] = len(entries)
    payload = {"metadata": metadata, "entries": entries}
    _write_json(_rank_partial_index_path(split_dir, args.shard_rank), payload)
    if int(args.num_shards) == 1:
        _write_json(os.path.join(split_dir, "index.json"), payload)
    return metadata

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ForceWM shard/index caches for pi05 hidden states selected by a split manifest."
    )
    parser.add_argument("--split_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--openpi_root", default=DEFAULT_OPENPI_ROOT)
    parser.add_argument("--config", default="pi05_part_gripper")
    parser.add_argument("--checkpoint_dir", default=None)
    parser.add_argument("--default_prompt", default=None)
    parser.add_argument("--query_stride", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--shard_size", type=int, default=128)
    parser.add_argument("--feature_dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--max_episodes_per_split", type=int, default=None)
    parser.add_argument("--shard_rank", type=int, default=None, help="Rank id for multi-process cache extraction.")
    parser.add_argument("--num_shards", type=int, default=None, help="Number of rank processes for multi-process extraction.")
    parser.add_argument("--merge_timeout_seconds", type=int, default=3600)
    parser.add_argument(
        "--no_pin_cuda_visible_device_per_rank",
        action="store_true",
        help="Do not reduce CUDA_VISIBLE_DEVICES to one local device per rank before importing OpenPI/JAX.",
    )
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def _resolve_rank_args(args):
    if args.shard_rank is None:
        args.shard_rank = int(os.environ.get("RANK", "0"))
    if args.num_shards is None:
        args.num_shards = int(os.environ.get("WORLD_SIZE", "1"))
    if int(args.num_shards) < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= int(args.shard_rank) < int(args.num_shards):
        raise ValueError(f"Invalid shard rank/world size: {args.shard_rank}/{args.num_shards}")
    return args


def _pin_cuda_visible_device_per_rank(args):
    if bool(args.no_pin_cuda_visible_device_per_rank):
        return
    if int(args.num_shards) <= 1:
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible or "," not in visible:
        return
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    if not devices:
        return
    local_rank = int(os.environ.get("LOCAL_RANK", str(args.shard_rank)))
    selected = devices[local_rank % len(devices)]
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    print(
        f"Rank {int(args.shard_rank)}/{int(args.num_shards)} uses CUDA_VISIBLE_DEVICES={selected}",
        flush=True,
    )


def _merge_all_outputs(args):
    split_metadata = {}
    for split in args.splits:
        split_metadata[split] = _merge_split_index(args, split)

    _write_json(
        os.path.join(args.output_dir, "summary.json"),
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "splits": split_metadata,
            "num_shards": int(args.num_shards),
        },
    )


def main():
    args = _resolve_rank_args(parse_args())
    _pin_cuda_visible_device_per_rank(args)
    if int(args.query_stride) < 1:
        raise ValueError("--query_stride must be >= 1")
    os.makedirs(args.output_dir, exist_ok=True)

    manifest = _read_manifest(args.split_manifest)
    extractor = _Pi05Extractor(args)
    all_metadata = {}

    for split in args.splits:
        metadata = precompute_split(args, manifest, split, extractor)
        all_metadata[split] = metadata

    _write_json(
        _rank_partial_summary_path(args.output_dir, args.shard_rank),
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "shard_rank": int(args.shard_rank),
            "num_shards": int(args.num_shards),
            "splits": all_metadata,
        },
    )
    _write_json(
        _rank_stats_path(args.output_dir, args.shard_rank),
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "shard_rank": int(args.shard_rank),
            "num_shards": int(args.num_shards),
            "status": "done",
        },
    )

    if int(args.shard_rank) == 0:
        _merge_all_outputs(args)


if __name__ == "__main__":
    main()
