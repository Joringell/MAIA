import json
import os
from collections import OrderedDict

import torch
from torch.utils.data import Dataset


_NUMERIC_META_KEYS = {
    "source_index",
    "episode_id",
    "episode_dataset_index",
    "window_start",
    "window_end",
    "chunk_end",
    "total_frames",
}


class VideoActionAnomalyFeatureCacheDataset(Dataset):
    """Read precomputed frozen action-WM features for anomaly-head training."""

    def __init__(self, cache_dir, split="train", max_open_shards=4, **kwargs):
        del kwargs
        self.cache_dir = os.path.expanduser(str(cache_dir))
        self.split = str(split)
        self.split_dir = os.path.join(self.cache_dir, self.split)
        self.index_path = os.path.join(self.split_dir, "index.json")
        self.max_open_shards = max(1, int(max_open_shards))
        self._shard_cache = OrderedDict()

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"Missing video-action anomaly feature cache index: {self.index_path}")
        with open(self.index_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        self.entries = payload.get("entries", [])
        self.metadata = payload.get("metadata", {})
        if not self.entries:
            raise ValueError(f"Empty video-action anomaly feature cache index: {self.index_path}")

    def __len__(self):
        return len(self.entries)

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
        entry = self.entries[int(idx)]
        shard = self._load_shard(entry["shard"])
        sample = shard["samples"][int(entry["offset"])]
        meta = dict(entry)
        meta.update(sample.get("meta", {}))
        output = {
            "_cache_index": torch.tensor(int(entry.get("index", idx)), dtype=torch.long),
            "_cache_shard": str(entry["shard"]),
            "_cache_offset": torch.tensor(int(entry["offset"]), dtype=torch.long),
            "interaction_tokens": sample["interaction_tokens"],
            "pred_latents": sample["pred_latents"],
            "label": sample["label"],
            "risk_label": sample["risk_label"],
            "frame_anomaly_label": sample["frame_anomaly_label"],
            "interaction_label": sample["interaction_label"],
        }
        for key, value in meta.items():
            if key in {"shard", "offset", "index"}:
                continue
            if key in _NUMERIC_META_KEYS:
                output[key] = torch.tensor(int(value), dtype=torch.long)
            elif key in {"domain_name", "parquet_path"}:
                output[key] = "" if value is None else str(value)
        return output
