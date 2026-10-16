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
    "valid_len",
    "total_frames",
}


def _normalize_domains(domains):
    if domains in (None, "", "all"):
        return None
    if isinstance(domains, str):
        return {item.strip() for item in domains.split(",") if item.strip()}
    return {str(item).strip() for item in domains if str(item).strip()}


def _load_episode_label_sidecar(cache_dir, sidecar_name="episode_labels.json"):
    sidecar_path = os.path.join(cache_dir, sidecar_name)
    if not os.path.isfile(sidecar_path):
        return {}
    with open(sidecar_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    labels = payload.get("labels", payload)
    if not isinstance(labels, dict):
        raise ValueError(f"Invalid episode label sidecar format: {sidecar_path}")
    return labels


class AnomalyFeatureCacheDataset(Dataset):
    """Read precomputed frozen-WM features for anomaly-head training."""

    def __init__(
        self,
        cache_dir,
        split="train",
        max_open_shards=4,
        domains=None,
        episode_label_sidecar="episode_labels.json",
        **kwargs,
    ):
        del kwargs
        self.cache_dir = os.path.expanduser(str(cache_dir))
        self.split = str(split)
        self.split_dir = os.path.join(self.cache_dir, self.split)
        self.index_path = os.path.join(self.split_dir, "index.json")
        self.max_open_shards = max(1, int(max_open_shards))
        self.domains = _normalize_domains(domains)
        self.episode_label_sidecar = _load_episode_label_sidecar(
            self.cache_dir,
            sidecar_name=str(episode_label_sidecar),
        ) if episode_label_sidecar else {}
        self._shard_cache = OrderedDict()

        if not os.path.isfile(self.index_path):
            raise FileNotFoundError(f"Missing anomaly feature cache index: {self.index_path}")
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
        if not self.entries:
            raise ValueError(f"Empty anomaly feature cache index: {self.index_path}")

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

    def _lookup_episode_label(self, entry):
        split_labels = self.episode_label_sidecar.get(self.split, {})
        if not isinstance(split_labels, dict):
            return None
        domain = str(entry.get("domain_name", entry.get("domain", "")))
        episode_id = entry.get("episode_id")
        if episode_id is None:
            return None
        domain_labels = split_labels.get(domain, {})
        if not isinstance(domain_labels, dict):
            return None
        return domain_labels.get(str(int(episode_id)))

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
            "pred_force": sample["pred_force"],
            "interaction_tokens": sample["interaction_tokens"],
            "pred_latents": sample["pred_latents"],
            "history_force_state": sample["history_force_state"],
            "label": sample["label"],
            "risk_label": sample["risk_label"],
            "frame_anomaly_label": sample["frame_anomaly_label"],
            "interaction_label": sample["interaction_label"],
        }
        if "episode_label" in sample:
            output["episode_label"] = sample["episode_label"]
        else:
            episode_label = self._lookup_episode_label(meta)
            if episode_label is not None:
                output["episode_label"] = torch.tensor(float(episode_label), dtype=torch.float32)
        gt_force = sample.get("gt_force", sample.get("gt_force_norm", torch.empty(0)))
        output["gt_force"] = gt_force
        output["gt_force_norm"] = gt_force
        output["gt_latents"] = sample.get("gt_latents", sample.get("future_latents", torch.empty(0)))
        for key, value in meta.items():
            if key in {"shard", "offset", "index"}:
                continue
            if key in _NUMERIC_META_KEYS:
                output[key] = torch.tensor(int(value), dtype=torch.long)
            elif key in {"domain_name", "parquet_path"}:
                output[key] = "" if value is None else str(value)
        return output
