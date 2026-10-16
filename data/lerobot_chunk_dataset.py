import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import traceback
import json
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import Dataset
import glob
from moviepy.editor import VideoFileClip
import torchvision.transforms as transforms
from tqdm import tqdm

from utils import zero_rank_print
from data.utils.utils import intrinsic_transform, gen_crop_config, intrin_crop_transform


def _normalize_tasks(tasks_value):
    if isinstance(tasks_value, np.ndarray):
        tasks = tasks_value.tolist()
    elif isinstance(tasks_value, (list, tuple)):
        tasks = list(tasks_value)
    else:
        tasks = [str(tasks_value)]
    return [str(t) for t in tasks]


def _normalize_feature_keys(keys):
    if isinstance(keys, str):
        keys = [key.strip() for key in keys.split(",") if key.strip()]
    elif isinstance(keys, (list, tuple)):
        keys = [str(key).strip() for key in keys if str(key).strip()]
    else:
        raise TypeError(f"Unsupported feature key type: {type(keys)}")
    if not keys:
        raise ValueError("Feature keys are empty")
    return keys


def _split_manifest_filename(split_ratio, split_seed):
    ratio_text = f"{float(split_ratio):.6f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"episode_split_train{ratio_text}_seed{int(split_seed)}.json"


def _write_json_atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.tmp.{os.getpid()}"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temp_path, path)


def _create_episode_split(domain_name, episode_indices, split_ratio, split_seed, manifest_path):
    episode_indices = sorted(set(int(index) for index in episode_indices))
    if len(episode_indices) < 2:
        return set(episode_indices), set()

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_episode_indices = sorted(
            set(manifest.get("train_episode_indices", []))
            | set(manifest.get("val_episode_indices", []))
        )
        if (
            manifest.get("domain") == domain_name
            and float(manifest.get("split_ratio")) == float(split_ratio)
            and int(manifest.get("split_seed")) == int(split_seed)
            and manifest_episode_indices == episode_indices
        ):
            zero_rank_print(f"Load episode split manifest from {manifest_path}")
            return set(manifest["train_episode_indices"]), set(manifest["val_episode_indices"])
        raise ValueError(
            f"Existing split manifest {manifest_path} does not match the current "
            "domain/split settings or episode set. Delete it explicitly if you want "
            "to regenerate the split."
        )

    shuffled_indices = list(episode_indices)
    random.Random(split_seed).shuffle(shuffled_indices)
    n_train = int(len(shuffled_indices) * split_ratio)
    n_train = max(1, min(len(shuffled_indices) - 1, n_train))
    train_indices = sorted(shuffled_indices[:n_train])
    val_indices = sorted(shuffled_indices[n_train:])

    manifest = {
        "version": 1,
        "domain": domain_name,
        "split_ratio": float(split_ratio),
        "split_seed": int(split_seed),
        "num_episodes": len(episode_indices),
        "train_episode_indices": train_indices,
        "val_episode_indices": val_indices,
    }
    _write_json_atomic(manifest_path, manifest)
    zero_rank_print(f"Wrote episode split manifest to {manifest_path}")
    return set(train_indices), set(val_indices)


def _split_dataset_items(
    dataset,
    split_ratio,
    split_seed,
    train_dataset,
    split_by_episode,
    split_manifest_paths=None,
):
    if split_ratio is None:
        return dataset
    if split_ratio <= 0 or split_ratio >= 1:
        return dataset
    if len(dataset) < 2:
        return dataset

    rng = random.Random(split_seed)

    if split_by_episode:
        episode_keys = []
        domain_episode_indices = {}
        for item in dataset:
            domain_name = item[3]
            episode_index = item[8] if len(item) > 8 else None
            if episode_index is None:
                raise ValueError(
                    "Cannot persist an episode split because the episode index "
                    f"is missing for dataset item: {item}"
                )
            episode_keys.append((domain_name, int(episode_index)))
            domain_episode_indices.setdefault(domain_name, []).append(int(episode_index))

        unique_keys = sorted(set(episode_keys))
        if len(unique_keys) < 2:
            return dataset

        if split_manifest_paths:
            train_keys = set()
            for domain_name, episode_indices in sorted(domain_episode_indices.items()):
                manifest_path = split_manifest_paths[domain_name]
                domain_train, _ = _create_episode_split(
                    domain_name,
                    episode_indices,
                    split_ratio,
                    split_seed,
                    manifest_path,
                )
                train_keys.update((domain_name, episode_index) for episode_index in domain_train)
        else:
            rng.shuffle(unique_keys)
            n_train = int(len(unique_keys) * split_ratio)
            n_train = max(1, min(len(unique_keys) - 1, n_train))
            train_keys = set(unique_keys[:n_train])

        if train_dataset:
            return [item for item, key in zip(dataset, episode_keys) if key in train_keys]
        return [item for item, key in zip(dataset, episode_keys) if key not in train_keys]

    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    n_train = int(len(indices) * split_ratio)
    n_train = max(1, min(len(indices) - 1, n_train))
    train_indices = set(indices[:n_train])
    if train_dataset:
        return [dataset[i] for i in range(len(dataset)) if i in train_indices]
    return [dataset[i] for i in range(len(dataset)) if i not in train_indices]


def _filter_dataset_by_manifest(dataset, split_manifest, split):
    if split_manifest is None or split in (None, "all"):
        return dataset
    with open(split_manifest, "r") as f:
        manifest = json.load(f)
    if split not in manifest:
        raise KeyError(f"Split '{split}' not found in split manifest: {split_manifest}")

    split_spec = manifest[split]
    if isinstance(split_spec, list):
        allowed = {int(ep) for ep in split_spec}
        return [item for item in dataset if int(item[8]) in allowed]
    if isinstance(split_spec, dict):
        allowed = {str(domain): {int(ep) for ep in episodes} for domain, episodes in split_spec.items()}
        return [
            item for item in dataset
            if str(item[3]) in allowed and int(item[8]) in allowed[str(item[3])]
        ]
    raise TypeError(f"Unsupported split manifest format for split '{split}': {type(split_spec)}")


def _domains_from_split_manifest(split_manifest, split=None):
    with open(split_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if split not in (None, "all") and isinstance(manifest.get(split), dict):
        return list(manifest[split].keys())
    if isinstance(manifest.get("domains"), list):
        return [str(domain) for domain in manifest["domains"]]

    domains = []
    for split_name in ("train", "val", "test"):
        split_spec = manifest.get(split_name)
        if isinstance(split_spec, dict):
            for domain in split_spec:
                if str(domain) not in domains:
                    domains.append(str(domain))
    if domains:
        return domains
    raise KeyError(f"Could not infer domains from split manifest: {split_manifest}")


def _domains_need_manifest_resolution(domains):
    if domains is None:
        return True
    if isinstance(domains, str):
        return domains.strip().lower() in ("", "all")
    if isinstance(domains, (list, tuple)):
        if len(domains) == 0:
            return True
        return len(domains) == 1 and str(domains[0]).strip().lower() == "all"
    return False


def _resolve_domain_root(data_root, domain_name):
    direct = os.path.join(data_root, domain_name)
    if os.path.exists(os.path.join(direct, "meta", "info.json")):
        return direct

    matches = sorted(
        path for path in glob.glob(os.path.join(data_root, "*", domain_name))
        if os.path.exists(os.path.join(path, "meta", "info.json"))
    )
    if not matches:
        raise FileNotFoundError(
            f"Could not find domain '{domain_name}' under {data_root}; "
            f"expected {direct}/meta/info.json or {data_root}/*/{domain_name}/meta/info.json"
        )
    if len(matches) > 1:
        raise ValueError(f"Domain '{domain_name}' is ambiguous under {data_root}: {matches}")
    return matches[0]


def _normalize_stats_payload(stats, domain_name=None):
    if "norm_groups" in stats:
        groups = stats["norm_groups"]
        if domain_name is not None and domain_name in groups:
            return groups[domain_name]
        if len(groups) == 1:
            return next(iter(groups.values()))
    if domain_name is not None and domain_name in stats and isinstance(stats[domain_name], dict):
        return stats[domain_name]
    return stats


def _feature_key_for_stats(keys):
    keys = _normalize_feature_keys(keys)
    return keys[0] if len(keys) == 1 else list(keys)


def _read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _format_lerobot_path(template, episode_index, chunks_size, chunk_index=None, file_index=None, video_key=None):
    episode_chunk = int(episode_index) // max(1, int(chunks_size))
    values = {
        "episode_chunk": episode_chunk,
        "episode_index": int(episode_index),
        "chunk_index": episode_chunk if chunk_index is None else int(chunk_index),
        "file_index": int(episode_index) if file_index is None else int(file_index),
        "video_key": video_key,
    }
    return template.format(**values)


def _merge_episode_stats(episodes_stats_rows):
    merged = {}
    for row in episodes_stats_rows:
        stats_by_key = row.get("stats", {})
        for key, stats in stats_by_key.items():
            if not all(name in stats for name in ("min", "max", "mean", "std", "count")):
                continue
            count = float(np.asarray(stats["count"]).reshape(-1)[0])
            if count <= 0:
                continue
            current = {
                "min": np.asarray(stats["min"], dtype=np.float64),
                "max": np.asarray(stats["max"], dtype=np.float64),
                "mean": np.asarray(stats["mean"], dtype=np.float64),
                "std": np.asarray(stats["std"], dtype=np.float64),
                "count": count,
            }
            if key not in merged:
                merged[key] = current
                continue

            old = merged[key]
            total = old["count"] + current["count"]
            delta = current["mean"] - old["mean"]
            mean = old["mean"] + delta * (current["count"] / total)
            var = (
                old["std"] ** 2 * old["count"]
                + current["std"] ** 2 * current["count"]
                + delta**2 * old["count"] * current["count"] / total
            ) / total
            old["min"] = np.minimum(old["min"], current["min"])
            old["max"] = np.maximum(old["max"], current["max"])
            old["mean"] = mean
            old["std"] = np.sqrt(np.maximum(var, 0.0))
            old["count"] = total

    return {
        key: {
            "min": value["min"].astype(np.float32).tolist(),
            "max": value["max"].astype(np.float32).tolist(),
            "mean": value["mean"].astype(np.float32).tolist(),
            "std": value["std"].astype(np.float32).tolist(),
            "count": [int(value["count"])],
        }
        for key, value in merged.items()
    }


class CustomLeRobotChunkDataset(Dataset):
    def __init__(
        self,
        data_roots,
        domains,
        task_recap_file=None,
        step_recap_file=None,
        sample_size=(192, 256),
        preprocess='resize',
        valid_cam=("wrist", "side_1", "side_2",),
        memory_steps=4,
        memory_span_steps=64,
        memory_pick_mode='uniform',
        future_steps=1,
        video_stride=1,
        chunk=None,
        action_chunk=None,
        n_previous=None,
        previous_pick_mode=None,
        sample_n_frames=None,
        random_crop=True,
        dataset_info_cache_path=None,
        action_type='absolute',
        action_space='joint',
        ignore_seek=False,
        train_dataset=True,
        action_key='action',
        state_key='observation.state.q',
        action_dim=None,
        state_dim=None,
        force_key="observation.state.wrench_filtered_chunk",
        force_type="absolute",
        force_dim=6,
        force_mode="last",
        force_substeps_per_step=4,
        force_history_steps=0,
        force_target_key=None,
        force_target_reduce=None,
        force_target_dim=None,
        force_target_type=None,
        use_unified_prompt=False,
        unified_prompt='best quality, consistent and smooth motion, realistic, clear and distinct.',
        fix_epiidx=None,
        fix_sidx=None,
        fix_mem_idx=None,
        stat_file=None,
        norm_group=None,
        action_norm_key=None,
        action_delta_norm_key=None,
        state_norm_key=None,
        force_norm_key=None,
        force_relative_norm_key=None,
        split_manifest=None,
        split="all",
        val_ratio=None,
        split_ratio=None,
        split_seed=0,
        split_by_episode=True,
    ):
        """
        LeRobot chunked dataset reader.

        Expected layout:
          <data_root>/<domain>/meta/episodes/chunk-*/file-*.parquet
          <data_root>/<domain>/data/chunk-*/file-*.parquet
          <data_root>/<domain>/videos/<cam>/chunk-*/file-*.mp4

        split_ratio: train split ratio (0-1); when set, val uses the complement
        split_seed: random seed for split
        split_by_episode: split by episode_index if True, else by dataset index
        """
        zero_rank_print("loading annotations...")

        # Compatibility layer for FT/PD configs that still use the legacy
        # like-dataset naming while switching to chunk datasets.
        if n_previous is not None:
            memory_steps = n_previous
        if previous_pick_mode is not None:
            memory_pick_mode = previous_pick_mode
        if action_chunk is not None:
            future_steps = action_chunk
        if chunk is not None and action_chunk is not None:
            ratio = int(action_chunk) // max(1, int(chunk))
            video_stride = max(1, ratio)
        if force_target_key is not None:
            force_key = force_target_key
        if force_target_reduce is not None:
            force_mode = force_target_reduce
        if force_target_dim is not None:
            force_dim = force_target_dim
        if force_target_type is not None:
            force_type = force_target_type
        if split_ratio is not None and val_ratio is None:
            val_ratio = split_ratio
        del sample_n_frames

        assert action_type in ["delta", "absolute", "relative"]
        self.action_type = action_type
        assert action_space in ["eef", "joint"]
        self.action_space = action_space

        self.action_key = action_key
        self.action_dim = action_dim
        self.state_keys = _normalize_feature_keys(state_key)
        self.state_key = self.state_keys[0] if len(self.state_keys) == 1 else list(self.state_keys)
        self.state_dim = state_dim
        self.force_keys = _normalize_feature_keys(force_key) if force_key is not None else None
        self.force_key = (
            self.force_keys[0] if self.force_keys is not None and len(self.force_keys) == 1
            else list(self.force_keys) if self.force_keys is not None
            else None
        )
        if force_type not in ("absolute", "relative", "absolute_relative"):
            raise ValueError(
                "force_type must be one of {'absolute', 'relative', 'absolute_relative'}, "
                f"got {force_type}"
            )
        if force_mode not in ("substeps", "last", "flat"):
            raise ValueError("force_mode must be one of {'substeps', 'last', 'flat'}")
        self.force_type = force_type
        self.force_dim = int(force_dim)
        self.force_mode = force_mode
        self.force_substeps_per_step = max(1, int(force_substeps_per_step))
        self.force_history_steps = max(0, int(force_history_steps))

        self.random_crop = random_crop

        if not isinstance(valid_cam, (list, tuple)):
            valid_cam = [valid_cam]
        self.valid_cam = list(valid_cam)

        if _domains_need_manifest_resolution(domains):
            if split_manifest is None:
                raise ValueError("domains must be provided unless split_manifest is set")
            domains = _domains_from_split_manifest(split_manifest, split=split)
            zero_rank_print(f"Load domains from split manifest {split_manifest}: {domains}")
        elif isinstance(domains, str):
            domains = [domains]
        else:
            domains = list(domains)

        if len(data_roots) == 1 and len(domains) > 1:
            data_roots = data_roots * len(domains)
        if len(data_roots) != len(domains):
            raise ValueError(f"data_roots and domains length mismatch: {len(data_roots)} vs {len(domains)}")
        self.data_roots = data_roots
        self.dataset = []

        if dataset_info_cache_path is not None and os.path.exists(dataset_info_cache_path):
            zero_rank_print(f"Load Cache Dataset Information from {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "r") as f:
                self.dataset = json.load(f)
        else:
            for _data_root, _domain_name in zip(self.data_roots, domains):
                print(f"Loading {_domain_name} data from {_data_root}")

                domain_root = _resolve_domain_root(_data_root, _domain_name)
                meta_folder = os.path.join(domain_root, "meta")
                data_folder = os.path.join(domain_root, "data")
                video_folder = os.path.join(domain_root, "videos")
                info_path = os.path.join(meta_folder, "info.json")
                with open(info_path, "r") as f:
                    meta_info = json.load(f)
                chunks_size = int(meta_info.get("chunks_size", meta_info.get("chunk_size", 1000)))
                data_template = meta_info.get("data_path", "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
                video_template = meta_info.get(
                    "video_path",
                    "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
                )

                episodes_root = os.path.join(meta_folder, "episodes")
                episode_files = sorted(glob.glob(os.path.join(episodes_root, "**", "file-*.parquet"), recursive=True))
                episodes_jsonl = os.path.join(meta_folder, "episodes.jsonl")
                if episode_files:
                    episodes_df = pd.concat([pd.read_parquet(p) for p in episode_files], ignore_index=True)
                    episode_iter = list(episodes_df.iterrows())
                    progress_iter = tqdm(episode_iter, total=len(episode_iter))

                    for _, row in progress_iter:
                        episode_index = int(row["episode_index"])
                        length = int(row["length"])

                        tasks = _normalize_tasks(row.get("tasks", "task_0"))
                        task_caption = random.choice(tasks) if len(tasks) > 1 else tasks[0]

                        data_chunk = int(row["data/chunk_index"])
                        data_file = int(row["data/file_index"])
                        data_path = os.path.join(
                            domain_root,
                            _format_lerobot_path(
                                data_template,
                                episode_index,
                                chunks_size,
                                chunk_index=data_chunk,
                                file_index=data_file,
                            ),
                        )

                        cam_segments = {}
                        for cam in self.valid_cam:
                            chunk_key = f"videos/{cam}/chunk_index"
                            file_key = f"videos/{cam}/file_index"
                            from_key = f"videos/{cam}/from_timestamp"
                            if chunk_key not in row or file_key not in row or from_key not in row:
                                raise KeyError(f"Missing video metadata for camera {cam} in episodes parquet")
                            cam_chunk = int(row[chunk_key])
                            cam_file = int(row[file_key])
                            cam_from = float(row[from_key])
                            video_path = os.path.join(
                                domain_root,
                                _format_lerobot_path(
                                    video_template,
                                    episode_index,
                                    chunks_size,
                                    chunk_index=cam_chunk,
                                    file_index=cam_file,
                                    video_key=cam,
                                ),
                            )
                            cam_segments[cam] = {"video_path": video_path, "offset": cam_from}

                        info = [
                            video_folder,
                            None,
                            data_path,
                            _domain_name,
                            "",
                            None,
                            task_caption,
                            length,
                            episode_index,
                            cam_segments,
                        ]
                        self.dataset.append(info)
                elif os.path.exists(episodes_jsonl):
                    episodes = _read_jsonl(episodes_jsonl)
                    for row in tqdm(episodes, total=len(episodes)):
                        episode_index = int(row["episode_index"])
                        length = int(row["length"])
                        tasks = _normalize_tasks(row.get("tasks", "task_0"))
                        task_caption = random.choice(tasks) if len(tasks) > 1 else tasks[0]
                        data_path = os.path.join(
                            domain_root,
                            _format_lerobot_path(data_template, episode_index, chunks_size),
                        )
                        cam_segments = {}
                        for cam in self.valid_cam:
                            video_path = os.path.join(
                                domain_root,
                                _format_lerobot_path(
                                    video_template,
                                    episode_index,
                                    chunks_size,
                                    file_index=episode_index,
                                    video_key=cam,
                                ),
                            )
                            cam_segments[cam] = {"video_path": video_path, "offset": 0.0}

                        info = [
                            video_folder,
                            None,
                            data_path,
                            _domain_name,
                            "",
                            None,
                            task_caption,
                            length,
                            episode_index,
                            cam_segments,
                        ]
                        self.dataset.append(info)
                else:
                    raise FileNotFoundError(
                        f"No episode metadata found. Expected parquet under {episodes_root} "
                        f"or JSONL at {episodes_jsonl}"
                    )

        split_manifest_paths = None
        if (
            split_manifest is None
            and val_ratio is not None
            and 0 < float(val_ratio) < 1
            and split_by_episode
        ):
            manifest_name = _split_manifest_filename(val_ratio, split_seed)
            split_manifest_paths = {
                domain_name: os.path.join(data_root, domain_name, "meta", manifest_name)
                for data_root, domain_name in zip(self.data_roots, domains)
            }

        self.dataset = _filter_dataset_by_manifest(self.dataset, split_manifest, split)
        if split_manifest is None and val_ratio is not None:
            self.dataset = _split_dataset_items(
                self.dataset,
                val_ratio,
                split_seed,
                train_dataset if split in ("all", None) else split == "train",
                split_by_episode,
                split_manifest_paths,
            )

        if dataset_info_cache_path is not None and not os.path.exists(dataset_info_cache_path):
            os.makedirs(os.path.dirname(dataset_info_cache_path), exist_ok=True)
            zero_rank_print(f"Save Cache Dataset Information to {dataset_info_cache_path}")
            with open(dataset_info_cache_path, "w") as f:
                json.dump(self.dataset, f)

        self.length = len(self.dataset)
        self.base_length = self.length
        zero_rank_print(f"data scale: {self.length}")

        self.memory_steps = max(1, int(memory_steps))
        self.memory_span_steps = max(self.memory_steps, int(memory_span_steps))
        self.memory_pick_mode = memory_pick_mode
        self.future_steps = max(1, int(future_steps))
        self.video_stride = max(1, int(video_stride))

        self.sample_size = sample_size

        self.preprocess = preprocess
        if preprocess == 'center_crop_resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(min(sample_size)),
                transforms.CenterCrop(sample_size),
            ])
        elif preprocess == 'resize':
            self.pixel_transforms_resize = transforms.Compose([
                transforms.Resize(sample_size),
            ])
        else:
            raise NotImplementedError(f"unsupported preprocess: {preprocess}")

        self.pixel_transforms_norm = transforms.Normalize([0.5] * 3, [0.5] * 3)

        self.task_recap_map = None
        if task_recap_file is not None:
            with open(task_recap_file, 'r', encoding='UTF-8') as f:
                self.task_recap_map = json.load(f)

        self.step_recap_map = None
        if step_recap_file is not None:
            with open(step_recap_file, 'r', encoding='UTF-8') as f:
                self.step_recap_map = json.load(f)

        self.use_unified_prompt = use_unified_prompt

        self.fix_epiidx = fix_epiidx
        self.fix_sidx = fix_sidx
        self.fix_mem_idx = fix_mem_idx

        self.norm_group = norm_group
        self.StatisticInfo = None
        self.SharedNormStats = None
        self.DomainNormStats = {}
        if stat_file is not None:
            with open(stat_file, "r") as f:
                self.StatisticInfo = json.load(f)
            self.SharedNormStats = _normalize_stats_payload(self.StatisticInfo, norm_group)
        else:
            for _data_root, _domain_name in zip(self.data_roots, domains):
                domain_root = _resolve_domain_root(_data_root, _domain_name)
                stats_path = os.path.join(domain_root, "meta", "stats.json")
                if os.path.exists(stats_path):
                    with open(stats_path, "r") as f:
                        self.DomainNormStats[_domain_name] = _normalize_stats_payload(json.load(f), _domain_name)
                else:
                    episodes_stats_path = os.path.join(domain_root, "meta", "episodes_stats.jsonl")
                    if not os.path.exists(episodes_stats_path):
                        raise FileNotFoundError(
                            f"Missing normalization stats: expected {stats_path} "
                            f"or {episodes_stats_path}"
                        )
                    self.DomainNormStats[_domain_name] = _merge_episode_stats(_read_jsonl(episodes_stats_path))

        self.action_norm_key = action_norm_key or _feature_key_for_stats(action_key)
        self.action_delta_norm_key = action_delta_norm_key
        self.state_norm_key = state_norm_key or _feature_key_for_stats(state_key)
        self.force_norm_key = force_norm_key or (_feature_key_for_stats(force_key) if force_key is not None else None)
        self.force_relative_norm_key = force_relative_norm_key

        self.ignore_seek = ignore_seek
        self._data_cache_path = None
        self._data_cache_df = None

    def get_frame_indexes(self, total_frames):
        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx + self.future_steps))
            frame_indexes = action_indexes[self.video_stride - 1::self.video_stride]
            action_indexes = np.clip(action_indexes, a_min=0, a_max=total_frames - 1)
            frame_indexes = np.clip(frame_indexes, a_min=0, a_max=total_frames - 1)
            return self.fix_mem_idx + frame_indexes, self.fix_mem_idx + action_indexes

        chunk_end = random.randint(self.future_steps, total_frames + self.future_steps)

        indexes_start = max(-self.memory_steps, chunk_end - self.memory_span_steps)
        indexes = np.array(list(range(indexes_start, chunk_end)))
        indexes = np.clip(indexes, a_min=1, a_max=total_frames - 1).tolist()
        video_end = indexes[-self.future_steps:]
        mem_candidates = indexes[:-self.future_steps]
        if len(mem_candidates) < self.memory_steps - 1:
            mem_candidates = [1] * (self.memory_steps - 1) + mem_candidates

        if self.memory_pick_mode == 'uniform':
            mem_indexes = [mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates) - 1, self.memory_steps).tolist()]
        elif self.memory_pick_mode == 'random':
            mem_indexes = [mem_candidates[i] for i in sorted(np.random.choice(list(range(0, len(mem_candidates) - 1)), size=self.memory_steps - 1, replace=False).tolist())] + [mem_candidates[-1]]
        else:
            raise NotImplementedError(f"unsupported memory_pick_mode: {self.memory_pick_mode}")

        if not self.ignore_seek:
            frame_indexes = mem_indexes + video_end[self.video_stride - 1::self.video_stride]
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]

        action_indexes = mem_indexes + video_end
        return frame_indexes, action_indexes

    def _norm_stats_for_domain(self, domain_name=None):
        if self.SharedNormStats is not None:
            return self.SharedNormStats
        if domain_name in self.DomainNormStats:
            return self.DomainNormStats[domain_name]
        if len(self.DomainNormStats) == 1:
            return next(iter(self.DomainNormStats.values()))
        raise KeyError(f"Missing norm stats for domain '{domain_name}'")

    def _get_norm_stats(self, key, space=None, domain_name=None, aliases=()):
        norm_stats = self._norm_stats_for_domain(domain_name)
        candidate_keys = []
        if key is not None:
            candidate_keys.append(key)
        candidate_keys.extend(alias for alias in aliases if alias is not None)

        for candidate in candidate_keys:
            if isinstance(candidate, (list, tuple)):
                if all(k in norm_stats for k in candidate):
                    means = []
                    stds = []
                    for k in candidate:
                        stats = norm_stats[k]
                        if "mean" not in stats or "std" not in stats:
                            break
                        means.extend(np.asarray(stats["mean"]).reshape(-1).tolist())
                        stds.extend(np.asarray(stats["std"]).reshape(-1).tolist())
                    else:
                        return {"mean": means, "std": stds}
                continue

            if candidate not in norm_stats:
                continue
            stats = norm_stats[candidate]
            if "mean" not in stats and space is not None and space in stats:
                stats = stats[space]
            if "mean" in stats and "std" in stats:
                return stats

        raise KeyError(
            f"Missing norm stats for keys {candidate_keys} "
            f"in domain '{domain_name}'"
        )

    def get_action_bias_std(self, delta=False, domain_name=None):
        if delta and self.action_delta_norm_key is None:
            stats = self._get_norm_stats(self.action_norm_key, self.action_space, domain_name, aliases=("action",))
            mean = torch.zeros_like(torch.tensor(stats['mean']).unsqueeze(0), dtype=torch.float32)
            std = torch.ones_like(torch.tensor(stats['std']).unsqueeze(0), dtype=torch.float32)
            return mean, std
        key = self.action_delta_norm_key if delta else self.action_norm_key
        aliases = ("action_delta",) if delta else ("action",)
        stats = self._get_norm_stats(key, self.action_space, domain_name, aliases=aliases)
        mean = torch.tensor(stats['mean']).unsqueeze(0)
        std = torch.tensor(stats['std']).unsqueeze(0) + 1e-6
        return mean, std

    def get_state_bias_std(self, domain_name=None):
        stats = self._get_norm_stats(self.state_norm_key, self.action_space, domain_name, aliases=("state",))
        mean = torch.tensor(stats['mean']).unsqueeze(0)
        std = torch.tensor(stats['std']).unsqueeze(0) + 1e-6
        if self.state_dim is not None:
            mean = mean[:, : self.state_dim]
            std = std[:, : self.state_dim]
        return mean, std

    def get_force_bias_std(self, relative=False, domain_name=None):
        if self.force_norm_key is None:
            raise ValueError("force_key is required for force normalization")
        if relative and self.force_relative_norm_key is None:
            stats = self._get_norm_stats(self.force_norm_key, domain_name=domain_name, aliases=("force",))
            std = torch.tensor(stats['std'], dtype=torch.float32).reshape(-1)
            if std.numel() != self.force_dim:
                raise ValueError(
                    f"Force norm stats for {self.force_norm_key} must be {self.force_dim}D, "
                    f"got {std.numel()} values"
                )
            mean = torch.zeros_like(std, dtype=torch.float32)
            std = std + 1e-6
            return mean, std
        key = self.force_relative_norm_key if relative else self.force_norm_key
        aliases = ("force_relative",) if relative else ("force",)
        stats = self._get_norm_stats(key, domain_name=domain_name, aliases=aliases)
        mean = torch.tensor(stats['mean'], dtype=torch.float32).reshape(-1)
        std = torch.tensor(stats['std'], dtype=torch.float32).reshape(-1)
        if mean.numel() != self.force_dim or std.numel() != self.force_dim:
            raise ValueError(
                f"Force norm stats for {key} must be {self.force_dim}D, "
                f"got mean={mean.numel()} std={std.numel()}"
            )
        std = std + 1e-6
        return mean, std

    @staticmethod
    def _feature_row_to_1d(value):
        if isinstance(value, pd.Series):
            value = value.to_numpy(dtype=object)

        if isinstance(value, (list, tuple)):
            parts = [CustomLeRobotChunkDataset._feature_row_to_1d(item) for item in value]
            return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)

        array = np.asarray(value)
        if array.dtype == object and array.ndim > 0:
            parts = [CustomLeRobotChunkDataset._feature_row_to_1d(item) for item in array.tolist()]
            return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
        return array.reshape(-1)

    def _load_feature_array(self, data, keys):
        keys = _normalize_feature_keys(keys)
        features = []
        for key in keys:
            values = data[key]
            if isinstance(values, pd.DataFrame):
                rows = [self._feature_row_to_1d(row) for _, row in values.iterrows()]
            else:
                rows = [self._feature_row_to_1d(value) for value in values.to_numpy()]

            if not rows:
                raise ValueError(f"Feature key '{key}' has no rows")
            row_dim = rows[0].shape[0]
            if any(row.shape[0] != row_dim for row in rows):
                raise ValueError(f"Feature key '{key}' has inconsistent row dimensions")
            features.append(np.stack(rows, axis=0))

        if len(features) == 1:
            return features[0]
        return np.concatenate(features, axis=1)

    def load_state_array(self, data):
        state = self._load_feature_array(data, self.state_keys)
        if self.state_dim is not None:
            state = state[:, : self.state_dim]
        return state

    def load_force_array(self, data):
        if self.force_keys is None:
            raise ValueError("force_key is required for loading force")
        return self._load_feature_array(data, self.force_keys).astype(np.float32)

    def force_window(self, force_array, row_indices, mode=None):
        mode = mode or self.force_mode
        rows = np.asarray(force_array, dtype=np.float32)[row_indices]
        if mode == "substeps":
            rows = rows.reshape(rows.shape[0], -1, self.force_dim)
            if rows.shape[1] < self.force_substeps_per_step:
                raise ValueError(
                    f"Force rows have {rows.shape[1]} substeps, expected at least {self.force_substeps_per_step}"
                )
            return torch.from_numpy(rows[:, -self.force_substeps_per_step :, :])
        if mode == "last":
            rows = rows.reshape(rows.shape[0], -1, self.force_dim)
            return torch.from_numpy(rows[:, -1, :])
        if mode == "flat":
            return torch.from_numpy(rows.reshape(rows.shape[0], -1))
        raise NotImplementedError(f"unsupported force mode: {mode}")

    def normalize_force(self, force_tensor, relative=False, domain_name=None):
        mean, std = self.get_force_bias_std(relative=relative, domain_name=domain_name)
        view_shape = [1] * force_tensor.ndim
        view_shape[-1] = mean.numel()
        mean = mean.to(dtype=force_tensor.dtype).view(*view_shape)
        std = std.to(dtype=force_tensor.dtype).view(*view_shape)
        return (force_tensor - mean) / std

    def force_targets(self, force_array, future_indices, current_index=None, mode=None, domain_name=None):
        absolute = self.force_window(force_array, future_indices, mode=mode or "last")
        if self.force_type == "absolute":
            return self.normalize_force(absolute, relative=False, domain_name=domain_name)
        if current_index is None:
            raise ValueError("current_index is required for relative force targets")
        current = self.force_window(force_array, [current_index], mode=mode or "last")[0]
        relative = absolute - current
        relative = self.normalize_force(relative, relative=True, domain_name=domain_name)
        if self.force_type == "relative":
            return relative
        absolute = self.normalize_force(absolute, relative=False, domain_name=domain_name)
        return torch.cat([absolute, relative], dim=-1)

    def _load_data(self, parquet_path):
        if parquet_path == self._data_cache_path and self._data_cache_df is not None:
            return self._data_cache_df
        df = pd.read_parquet(parquet_path)
        self._data_cache_path = parquet_path
        self._data_cache_df = df
        return df

    def seek_mp4(self, cam_segments, cam_name_list, slices):
        video_list = []
        for cam_name in cam_name_list:
            segment = cam_segments[cam_name]
            video_reader = VideoFileClip(segment["video_path"])
            fps = video_reader.fps
            offset = segment["offset"]
            video = []
            for idx in slices:
                video.append(video_reader.get_frame(offset + float(idx) / fps))
            video = torch.from_numpy(np.stack(video)).permute(3, 0, 1, 2).contiguous()
            video = video.float() / 255.
            video_reader.close()
            video_list.append(video)
        return video_list

    def get_transform(self):
        return self.sample_size, self.pixel_transforms_resize, self.pixel_transforms_norm

    def transform_video(self, videos, specific_transforms_resize, intrinsics, sample_size):
        """
        Crop (optional) and resize the videos using the dataset preprocessing convention.
        """
        v = len(videos)
        new_videos = []
        new_intrinsics = []
        for iv in range(v):
            video = videos[iv]
            c, t, h, w = video.shape
            if self.random_crop:
                h_start, w_start, h_crop, w_crop = gen_crop_config(video)
                video = video[:, :, h_start:h_start + h_crop, w_start:w_start + w_crop]
                if intrinsics is not None:
                    intrinsic = intrin_crop_transform(intrinsics[iv], h_start, w_start)
                h, w = h_crop, w_crop
            if intrinsics is not None:
                intrinsic = intrinsic_transform(intrinsic, (h, w), sample_size, self.preprocess)
                new_intrinsics.append(intrinsic)
            video = specific_transforms_resize(video)
            new_videos.append(video)
        new_videos = torch.stack(new_videos, dim=1)
        if len(new_intrinsics) > 0:
            new_intrinsics = torch.stack(new_intrinsics, dim=0)
        else:
            new_intrinsics = None
        return new_videos, new_intrinsics

    def normalize_video(self, video, specific_transforms_norm):
        """
        Input video should have shape (c, v, t, h, w).
        """
        c, v, t, h, w = video.shape
        video = specific_transforms_norm(video.permute(1, 2, 0, 3, 4).reshape(-1, c, h, w))
        video = video.reshape(v, t, c, h, w).permute(2, 0, 1, 3, 4)
        return video

    def get_batch(self, idx):
        parquet_path = self.dataset[idx][2]
        domain_name = self.dataset[idx][3]
        caption = self.dataset[idx][6]
        total_frames = self.dataset[idx][7]
        episode_index = self.dataset[idx][8]
        cam_segments = self.dataset[idx][9]

        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self.get_frame_indexes(total_frames)

        data = self._load_data(parquet_path)
        data = data[data["episode_index"] == episode_index]

        action_mean, action_std = self.get_action_bias_std(delta=False, domain_name=domain_name)
        state_mean, state_std = self.get_state_bias_std(domain_name=domain_name)

        try:
            action = self._load_feature_array(data, self.action_key)
            state = self.load_state_array(data)
        except Exception as exc:
            raise ValueError("We currently only support action and state data with the shape of T*C!") from exc

        if self.action_dim is not None:
            action = action[:, : self.action_dim]
        action = action.astype(np.float32)
        state = state.astype(np.float32)
        state = torch.FloatTensor(state)[indexes][self.memory_steps - 1:self.memory_steps]
        state = (state - state_mean) / state_std

        if self.action_type == "absolute":
            action = action[indexes].astype(np.float32)
            action = torch.FloatTensor(action)
            if self.action_dim is not None:
                action_mean = action_mean[:, : self.action_dim]
                action_std = action_std[:, : self.action_dim]
            action = (action - action_mean) / action_std
        elif self.action_type == "delta":
            delta_act_meanv, delta_act_stdv = self.get_action_bias_std(delta=True, domain_name=domain_name)
            if self.action_dim is not None:
                delta_act_meanv = delta_act_meanv[:, : self.action_dim]
                delta_act_stdv = delta_act_stdv[:, : self.action_dim]
            action_curr = torch.FloatTensor(action[indexes].astype(np.float32))
            action_last = torch.FloatTensor(action[[max(0, _ - 1) for _ in indexes]].astype(np.float32))
            delta_action = action_curr - action_last
            for eff_idx in (6, 13, 7, 15):
                if delta_action.shape[1] > eff_idx:
                    delta_action[:, eff_idx] = action_last[:, eff_idx]
            delta_action = (delta_action - delta_act_meanv) / delta_act_stdv
            action = delta_action
        elif self.action_type == "relative":
            action_curr = action[indexes].astype(np.float32)
            action = torch.FloatTensor(action_curr)
            if self.action_dim is not None:
                action_mean = action_mean[:, : self.action_dim]
                action_std = action_std[:, : self.action_dim]
            action = (action - action_mean) / action_std
            state = state[..., : action.shape[-1]]
            action = action - state
        else:
            raise NotImplementedError

        videos = self.seek_mp4(cam_segments, self.valid_cam, vid_indexes)
        videos, _ = self.transform_video(videos, specific_transforms_resize, None, sample_size)
        videos = self.normalize_video(videos, specific_transforms_norm)

        return videos, action, caption, state

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.fix_epiidx is not None:
            video, actions, caption, state = self.get_batch(self.fix_epiidx)
        else:
            while True:
                try:
                    video, actions, caption, state = self.get_batch(idx)
                    break
                except Exception:
                    traceback.print_exc()
                    idx = random.randint(0, self.length - 1)

        sample = dict(
            video=video,
            actions=actions,
            caption=caption,
            state=state,
        )
        return sample
