import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import Dataset


_NUMERIC_META_KEYS = {
    "source_index",
    "episode_id",
    "episode_dataset_index",
    "source_episode_index",
    "frame_index",
    "window_start",
    "window_end",
    "chunk_end",
    "valid_len",
    "total_frames",
    "timestamp_us",
    "source_id",
    "task_index",
    "episode_label",
    "frame_label",
    "interaction_label",
}

_FLOAT_META_KEYS = {
    "timestamp",
}

_STRING_META_KEYS = {
    "domain_name",
    "source_h5",
    "source_h5_basename",
    "parquet_path",
    "source_repo_id",
    "domain_root",
    "source",
    "control_group",
}


def _normalize_domains(domains):
    if domains in (None, "", "all"):
        return None
    if isinstance(domains, str):
        values = [item.strip() for item in domains.split(",") if item.strip()]
    else:
        values = [str(item).strip() for item in domains if str(item).strip()]
    return set(values) if values else None


class Pi05FeatureCacheDataset(Dataset):
    """Read pi05 hidden-state feature caches generated from manifest episodes."""

    def __init__(
        self,
        cache_dir,
        split="train",
        max_open_shards=4,
        domains=None,
        sequence_mode=False,
        sequence_filter="all",
        min_sequence_length=1,
        **kwargs,
    ):
        del kwargs
        self.cache_dir = os.path.expanduser(str(cache_dir))
        self.split = str(split)
        self.split_dir = os.path.join(self.cache_dir, self.split)
        self.index_path = os.path.join(self.split_dir, "index.json")
        self.max_open_shards = max(1, int(max_open_shards))
        self.domains = _normalize_domains(domains)
        self.sequence_mode = bool(sequence_mode)
        self.sequence_filter = str(sequence_filter)
        self.min_sequence_length = max(1, int(min_sequence_length))
        self._shard_cache = OrderedDict()

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"Missing pi05 feature cache index: {self.index_path}")
        with open(self.index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        entries = payload.get("entries", [])
        if self.domains is not None:
            entries = [
                entry
                for entry in entries
                if str(entry.get("domain_name", entry.get("domain", ""))) in self.domains
            ]
        self.entries = entries
        self.metadata = payload.get("metadata", {})
        if self.sequence_mode:
            self.sequences = self._build_sequences(self.entries)
            if not self.sequences:
                suffix = "" if self.domains is None else f" for domains={sorted(self.domains)}"
                raise ValueError(f"Empty pi05 feature cache sequences: {self.index_path}{suffix}")
        elif not self.entries:
            suffix = "" if self.domains is None else f" for domains={sorted(self.domains)}"
            raise ValueError(f"Empty pi05 feature cache index: {self.index_path}{suffix}")

    def __len__(self):
        if self.sequence_mode:
            return len(self.sequences)
        return len(self.entries)

    def _sequence_key(self, entry):
        return (
            str(entry.get("domain_name", "")),
            int(entry.get("episode_id", entry.get("source_episode_index", -1))),
            str(entry.get("source_h5_basename", "")),
        )

    def _sequence_allowed(self, entries):
        if len(entries) < self.min_sequence_length:
            return False
        episode_label = int(entries[0].get("episode_label", -1))
        filter_name = self.sequence_filter
        if filter_name in ("all", "", "none"):
            return True
        if filter_name == "success_normal":
            return episode_label >= 1
        if filter_name == "all_normal":
            for entry in entries:
                if float(entry.get("risk_label", 1.0)) > 0.0:
                    return False
                if int(entry.get("interaction_label", 0)) > 0:
                    return False
            return True
        raise ValueError(
            f"Unsupported pi05 sequence_filter={filter_name}; "
            "use one of {'all', 'success_normal', 'all_normal'}"
        )

    def _build_sequences(self, entries):
        grouped = OrderedDict()
        for entry in sorted(entries, key=lambda item: (self._sequence_key(item), int(item.get("frame_index", -1)))):
            grouped.setdefault(self._sequence_key(entry), []).append(entry)
        return [values for values in grouped.values() if self._sequence_allowed(values)]

    def _load_shard(self, shard_name):
        shard_name = str(shard_name)
        if shard_name in self._shard_cache:
            shard = self._shard_cache.pop(shard_name)
            self._shard_cache[shard_name] = shard
            return shard

        shard_path = os.path.join(self.split_dir, shard_name)
        shard = torch.load(shard_path, map_location="cpu")
        self._shard_cache[shard_name] = shard
        while len(self._shard_cache) > self.max_open_shards:
            self._shard_cache.popitem(last=False)
        return shard

    def __getitem__(self, idx):
        if self.sequence_mode:
            return self._getitem_sequence(idx)

        entry = self.entries[int(idx)]
        shard = self._load_shard(entry["shard"])
        sample = shard["samples"][int(entry["offset"])]
        meta = dict(entry)
        meta.update(sample.get("meta", {}))

        output = {
            "_cache_index": torch.tensor(int(entry.get("index", idx)), dtype=torch.long),
            "_cache_shard": str(entry["shard"]),
            "_cache_offset": torch.tensor(int(entry["offset"]), dtype=torch.long),
            "pi05_hidden": sample["pi05_hidden"],
            "label": sample.get("label", torch.tensor(-1, dtype=torch.long)),
            "episode_label": sample.get("episode_label", torch.tensor(-1, dtype=torch.long)),
            "frame_label": sample.get("frame_label", torch.tensor(-1, dtype=torch.long)),
            "frame_anomaly_label": sample.get("frame_anomaly_label", torch.empty(0, dtype=torch.float32)),
            "risk_label": sample.get("risk_label", torch.tensor(-1, dtype=torch.long)),
            "interaction_label": sample.get("interaction_label", torch.tensor(-1, dtype=torch.long)),
        }

        for key in ("actions", "raw_actions", "state"):
            if key in sample:
                output[key] = sample[key]

        for key, value in meta.items():
            if key in {"shard", "offset", "index"}:
                continue
            if key in _NUMERIC_META_KEYS:
                output[key] = torch.tensor(int(value), dtype=torch.long)
            elif key in _FLOAT_META_KEYS:
                output[key] = torch.tensor(float(value), dtype=torch.float64)
            elif key in _STRING_META_KEYS:
                output[key] = "" if value is None else str(value)
        return output

    def _getitem_sequence(self, idx):
        entries = self.sequences[int(idx)]
        samples = []
        metas = []
        for entry in entries:
            shard = self._load_shard(entry["shard"])
            sample = shard["samples"][int(entry["offset"])]
            meta = dict(entry)
            meta.update(sample.get("meta", {}))
            samples.append(sample)
            metas.append(meta)

        features = torch.stack([sample["pi05_hidden"].float() for sample in samples], dim=0)
        frame_labels = torch.tensor([int(meta.get("frame_label", -1)) for meta in metas], dtype=torch.long)
        frame_anomaly = torch.full_like(frame_labels, -1, dtype=torch.float32)
        valid_frame_labels = frame_labels >= 0
        frame_anomaly[valid_frame_labels] = (frame_labels[valid_frame_labels] == 0).float()

        first = metas[0]
        output = {
            "_cache_index": torch.tensor(int(first.get("source_index", first.get("index", idx))), dtype=torch.long),
            "features": features,
            "valid_mask": torch.ones(features.shape[0], dtype=torch.bool),
            "success_labels": torch.tensor(int(first.get("episode_label", -1)), dtype=torch.long),
            "label": torch.tensor([int(sample.get("label", torch.tensor(-1)).item()) for sample in samples], dtype=torch.long),
            "episode_label": torch.tensor(int(first.get("episode_label", -1)), dtype=torch.long),
            "risk_label": torch.tensor([float(sample.get("risk_label", torch.tensor(-1.0)).item()) for sample in samples], dtype=torch.float32),
            "frame_label": frame_labels,
            "frame_anomaly_label": frame_anomaly,
            "interaction_label": torch.tensor([int(meta.get("interaction_label", -1)) for meta in metas], dtype=torch.long),
            "source_index": torch.tensor([int(meta.get("source_index", meta.get("index", -1))) for meta in metas], dtype=torch.long),
            "frame_index": torch.tensor([int(meta.get("frame_index", -1)) for meta in metas], dtype=torch.long),
            "window_start": torch.tensor([int(meta.get("window_start", -1)) for meta in metas], dtype=torch.long),
            "window_end": torch.tensor([int(meta.get("frame_index", -1)) + 1 for meta in metas], dtype=torch.long),
            "valid_len": torch.ones(features.shape[0], dtype=torch.long),
            "total_frames": torch.tensor(int(first.get("total_frames", -1)), dtype=torch.long),
            "episode_id": torch.tensor(int(first.get("episode_id", -1)), dtype=torch.long),
            "source_episode_index": torch.tensor(int(first.get("source_episode_index", -1)), dtype=torch.long),
            "domain_name": str(first.get("domain_name", "")),
            "source_h5": str(first.get("source_h5", "")),
            "source_h5_basename": str(first.get("source_h5_basename", "")),
            "parquet_path": str(first.get("parquet_path", "")),
        }
        if "source" in first:
            output["source"] = str(first.get("source", ""))
        if "control_group" in first:
            output["control_group"] = str(first.get("control_group", ""))
        return output
