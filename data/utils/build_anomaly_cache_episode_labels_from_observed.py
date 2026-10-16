import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import torch


def _load_index(cache_dir, split):
    index_path = os.path.join(os.path.expanduser(str(cache_dir)), str(split), "index.json")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"Missing cache index: {index_path}")
    with open(index_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    entries = payload.get("entries", [])
    if not entries:
        raise ValueError(f"Empty cache index: {index_path}")
    return index_path, entries


class ShardCache:
    def __init__(self, split_dir, max_open_shards=8):
        self.split_dir = split_dir
        self.max_open_shards = max(1, int(max_open_shards))
        self._cache = OrderedDict()

    def load(self, shard_name):
        shard_name = str(shard_name)
        if shard_name in self._cache:
            shard = self._cache.pop(shard_name)
            self._cache[shard_name] = shard
            return shard

        shard_path = os.path.join(self.split_dir, shard_name)
        shard = torch.load(shard_path, map_location="cpu")
        self._cache[shard_name] = shard
        while len(self._cache) > self.max_open_shards:
            self._cache.popitem(last=False)
        return shard


def _episode_key(entry):
    domain = str(entry.get("domain_name", entry.get("domain", "")))
    episode_id = entry.get("episode_id")
    if not domain or episode_id is None:
        raise KeyError(f"Entry is missing domain_name/domain or episode_id: {entry}")
    return domain, str(int(episode_id))


def _as_float_scalar(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        return float(value[0])
    return float(value)


def _read_observed_episode_labels(observed_cache_dir, split, max_open_shards):
    split_dir = os.path.join(os.path.expanduser(str(observed_cache_dir)), str(split))
    _, entries = _load_index(observed_cache_dir, split)
    shards = ShardCache(split_dir, max_open_shards=max_open_shards)
    labels = {}
    counts = {}

    for entry in entries:
        key = _episode_key(entry)
        shard = shards.load(entry["shard"])
        sample = shard["samples"][int(entry["offset"])]
        if "episode_label" not in sample:
            raise KeyError(
                f"Observed cache sample has no episode_label: "
                f"split={split} shard={entry['shard']} offset={entry['offset']}"
            )
        value = _as_float_scalar(sample["episode_label"])
        old = labels.get(key)
        if old is not None and old != value:
            raise ValueError(
                f"Inconsistent episode_label in observed cache for split={split} "
                f"domain={key[0]} episode_id={key[1]}: {old} vs {value}"
            )
        labels[key] = value
        counts[key] = counts.get(key, 0) + 1

    return labels, counts


def _target_episode_keys(target_cache_dir, split):
    _, entries = _load_index(target_cache_dir, split)
    keys = set()
    for entry in entries:
        keys.add(_episode_key(entry))
    return keys, len(entries)


def _nested_labels(label_map, target_keys):
    nested = {}
    for domain, episode_id in sorted(target_keys):
        if (domain, episode_id) not in label_map:
            continue
        nested.setdefault(domain, {})[episode_id] = float(label_map[(domain, episode_id)])
    return nested


def build_sidecar(observed_cache_dir, target_cache_dir, splits, output_name, max_open_shards, strict):
    output_labels = {}
    split_reports = {}

    for split in splits:
        observed_labels, observed_window_counts = _read_observed_episode_labels(
            observed_cache_dir,
            split,
            max_open_shards=max_open_shards,
        )
        target_keys, target_windows = _target_episode_keys(target_cache_dir, split)
        missing = sorted(target_keys - set(observed_labels))
        extra = sorted(set(observed_labels) - target_keys)
        if missing and strict:
            preview = ", ".join(f"{domain}:{episode_id}" for domain, episode_id in missing[:10])
            raise KeyError(
                f"Observed cache has no labels for {len(missing)} target episodes in split={split}: {preview}"
            )

        output_labels[split] = _nested_labels(observed_labels, target_keys)
        split_reports[split] = {
            "target_windows": int(target_windows),
            "target_episodes": int(len(target_keys)),
            "labeled_target_episodes": int(sum(len(v) for v in output_labels[split].values())),
            "observed_episodes": int(len(observed_labels)),
            "missing_target_episodes": int(len(missing)),
            "extra_observed_episodes": int(len(extra)),
            "observed_windows": int(sum(observed_window_counts.values())),
        }

    return {
        "metadata": {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source": "observed_cache",
            "observed_cache_dir": os.path.abspath(os.path.expanduser(str(observed_cache_dir))),
            "target_cache_dir": os.path.abspath(os.path.expanduser(str(target_cache_dir))),
            "key_fields": ["split", "domain_name", "episode_id"],
            "splits": list(splits),
            "output_name": str(output_name),
            "strict": bool(strict),
            "split_reports": split_reports,
        },
        "labels": output_labels,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Copy episode_label from observed feature cache into an anomaly feature cache sidecar."
    )
    parser.add_argument("--observed_cache_dir", required=True)
    parser.add_argument("--target_cache_dir", "--wm_cache_dir", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--output_name", default="episode_labels.json")
    parser.add_argument("--max_open_shards", type=int, default=8)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--dry_run", action="store_true", help="Validate and print the report without writing JSON.")
    args = parser.parse_args()

    payload = build_sidecar(
        observed_cache_dir=args.observed_cache_dir,
        target_cache_dir=args.target_cache_dir,
        splits=args.splits,
        output_name=args.output_name,
        max_open_shards=args.max_open_shards,
        strict=args.strict,
    )

    output_path = os.path.join(os.path.expanduser(str(args.target_cache_dir)), str(args.output_name))
    if args.dry_run:
        print(f"Dry run: would write {output_path}")
    else:
        tmp_path = f"{output_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, output_path)
        print(f"Wrote {output_path}")

    for split, report in payload["metadata"]["split_reports"].items():
        print(
            f"{split}: target_windows={report['target_windows']} "
            f"target_episodes={report['target_episodes']} "
            f"labeled_target_episodes={report['labeled_target_episodes']} "
            f"missing_target_episodes={report['missing_target_episodes']}"
        )


if __name__ == "__main__":
    main()
