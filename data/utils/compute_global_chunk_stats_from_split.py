#!/usr/bin/env python3
"""Compute chunk-dataset normalization statistics from a split manifest.

Only episodes in the selected split are read; use the default ``train`` split
to avoid leaking validation or test data into normalization statistics. The
result contains shared action, action-delta, state, and optional force
statistics across all selected domains and can be passed directly to
``CustomLeRobotChunkDataset`` as ``stat_file``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm


def normalize_feature_keys(keys: str | Sequence[str]) -> list[str]:
    if isinstance(keys, str):
        out = [key.strip() for key in keys.split(",") if key.strip()]
    else:
        out = [str(key).strip() for key in keys if str(key).strip()]
    if not out:
        raise ValueError("feature keys are empty")
    return out


def subsample_rows(array: np.ndarray, max_rows: int | None, rng: np.random.Generator) -> np.ndarray:
    if max_rows is None or max_rows <= 0 or len(array) <= max_rows:
        return array
    indices = rng.choice(len(array), size=max_rows, replace=False)
    return array[np.sort(indices)]


def feature_row_to_1d(value) -> np.ndarray:
    if isinstance(value, pd.Series):
        value = value.to_numpy(dtype=object)

    if isinstance(value, (list, tuple)):
        parts = [feature_row_to_1d(item) for item in value]
        return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)

    array = np.asarray(value)
    if array.dtype == object and array.ndim > 0:
        parts = [feature_row_to_1d(item) for item in array.tolist()]
        return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)
    return array.reshape(-1).astype(np.float32)


def load_feature_array(df: pd.DataFrame, keys: str | Sequence[str]) -> np.ndarray:
    arrays = []
    for key in normalize_feature_keys(keys):
        if key not in df.columns:
            raise KeyError(f"missing feature key {key!r}; available columns include {list(df.columns)[:20]}")
        values = df[key]
        if isinstance(values, pd.DataFrame):
            rows = [feature_row_to_1d(row) for _, row in values.iterrows()]
        else:
            rows = [feature_row_to_1d(value) for value in values.to_numpy()]
        if not rows:
            raise ValueError(f"feature key {key!r} has no rows")
        row_dim = rows[0].shape[0]
        if any(row.shape[0] != row_dim for row in rows):
            raise ValueError(f"feature key {key!r} has inconsistent row dimensions")
        arrays.append(np.stack(rows, axis=0))
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)


def maybe_filter_rows(df: pd.DataFrame, label_filter: str) -> pd.DataFrame:
    if label_filter == "all":
        return df
    if label_filter == "only_normal":
        if "frame_label" not in df.columns:
            raise KeyError("label_filter='only_normal' requires parquet column 'frame_label'")
        return df[df["frame_label"].astype(int) == 1]
    raise ValueError(f"Unsupported label_filter={label_filter!r}; use all or only_normal")


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _discover_domain_path(data_root: Path, domain: str) -> Path:
    direct = data_root / domain
    if (direct / "meta" / "episodes.jsonl").exists():
        return direct

    matches = sorted(path for path in data_root.glob(f"*/*") if path.name == domain)
    matches = [path for path in matches if (path / "meta" / "episodes.jsonl").exists()]
    if not matches:
        raise FileNotFoundError(f"Could not find domain {domain!r} under {data_root}")
    if len(matches) > 1:
        raise ValueError(f"Domain {domain!r} is ambiguous under {data_root}: {matches}")
    return matches[0]


def _episode_parquet_path(domain_path: Path, episode_index: int) -> Path:
    name = f"episode_{int(episode_index):06d}.parquet"
    matches = sorted((domain_path / "data").glob(f"chunk-*/{name}"))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {domain_path / 'data'}/chunk-*")
    if len(matches) > 1:
        raise ValueError(f"Episode parquet is ambiguous for {domain_path.name} episode {episode_index}: {matches}")
    return matches[0]


def parquets_from_manifest(
    split_manifest: Path,
    *,
    split: str,
    data_root: Path | None,
    domains: Sequence[str] | None,
) -> tuple[list[Path], dict]:
    manifest = _read_json(split_manifest)
    split_map = manifest.get(split)
    if not isinstance(split_map, dict):
        raise KeyError(f"Split manifest does not contain split {split!r}")

    root = data_root or Path(manifest.get("data_root", "")).expanduser()
    if not str(root):
        raise ValueError("Manifest has no data_root; pass --data-root explicitly")
    root = root.resolve()

    selected_domains = list(domains) if domains else list(split_map.keys())
    parquet_files: list[Path] = []
    domain_counts = {}
    for domain in selected_domains:
        if domain not in split_map:
            raise KeyError(f"Domain {domain!r} not found in manifest split {split!r}")
        domain_path = _discover_domain_path(root, domain)
        episode_indices = [int(idx) for idx in split_map[domain]]
        domain_counts[domain] = {
            "domain_path": str(domain_path),
            "num_episodes": len(episode_indices),
        }
        for episode_index in episode_indices:
            parquet_files.append(_episode_parquet_path(domain_path, episode_index))
    return parquet_files, {
        "split_manifest": str(split_manifest),
        "manifest_data_root": manifest.get("data_root"),
        "resolved_data_root": str(root),
        "split": split,
        "domains": selected_domains,
        "domain_counts": domain_counts,
        "num_episodes": len(parquet_files),
    }


def _sanitize_std(std: np.ndarray, eps: float) -> np.ndarray:
    return np.maximum(std.astype(np.float32), float(eps))


def stats_payload(
    array: np.ndarray,
    *,
    robust_mode: str,
    clip_percentiles: tuple[float, float],
    eps: float,
) -> dict:
    if array.ndim != 2:
        raise ValueError(f"expected 2D array, got shape {array.shape}")
    if len(array) == 0:
        raise ValueError("cannot compute stats on empty array")

    q01 = np.percentile(array, 1, axis=0)
    q99 = np.percentile(array, 99, axis=0)
    clip_low = np.percentile(array, clip_percentiles[0], axis=0)
    clip_high = np.percentile(array, clip_percentiles[1], axis=0)

    used = array
    if robust_mode == "clip":
        used = np.clip(array, clip_low, clip_high)
    elif robust_mode == "row_filter":
        mask = ((array >= clip_low) & (array <= clip_high)).all(axis=1)
        if mask.any():
            used = array[mask]
    elif robust_mode != "none":
        raise ValueError(f"unknown robust_mode {robust_mode!r}")

    return {
        "mean": np.mean(used, axis=0).astype(np.float32).tolist(),
        "std": _sanitize_std(np.std(used, axis=0), eps).tolist(),
        "q01": q01.astype(np.float32).tolist(),
        "q99": q99.astype(np.float32).tolist(),
        "clip_low": clip_low.astype(np.float32).tolist(),
        "clip_high": clip_high.astype(np.float32).tolist(),
        "count": int(len(used)),
        "raw_count": int(len(array)),
        "robust_mode": robust_mode,
        "clip_percentiles": [float(clip_percentiles[0]), float(clip_percentiles[1])],
    }


def compute_stats(
    parquet_files: Iterable[Path],
    *,
    action_key: str,
    state_key: str | Sequence[str],
    force_key: str | None,
    force_dim: int,
    label_filter: str,
    max_rows_per_file: int | None,
    robust_mode: str,
    clip_percentiles: tuple[float, float],
    seed: int,
    eps: float,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    action_parts = []
    action_delta_parts = []
    state_parts = []
    force_parts = []
    per_domain = defaultdict(lambda: {"episodes": 0, "rows": 0, "force_rows": 0})

    files = list(parquet_files)
    for parquet_path in tqdm(files, desc="Reading train parquets"):
        df = pd.read_parquet(parquet_path)
        df = maybe_filter_rows(df, label_filter)
        if df.empty:
            continue

        domain = parquet_path.parents[2].name
        per_domain[domain]["episodes"] += 1
        per_domain[domain]["rows"] += int(len(df))

        action = load_feature_array(df, action_key)
        state = load_feature_array(df, state_key)
        if len(action) > 1:
            action_delta_parts.append(subsample_rows(action[1:] - action[:-1], max_rows_per_file, rng))
        action_parts.append(subsample_rows(action, max_rows_per_file, rng))
        state_parts.append(subsample_rows(state, max_rows_per_file, rng))

        if force_key is not None:
            force = load_feature_array(df, force_key)
            force = force.reshape(force.shape[0], -1, force_dim)
            force = force.reshape(-1, force_dim)
            per_domain[domain]["force_rows"] += int(len(force))
            force_parts.append(subsample_rows(force, max_rows_per_file, rng))

    if not action_parts:
        raise ValueError("no rows available after filtering")

    payload_kwargs = {
        "robust_mode": robust_mode,
        "clip_percentiles": clip_percentiles,
        "eps": eps,
    }
    out = {
        "action": stats_payload(np.concatenate(action_parts, axis=0), **payload_kwargs),
        "action_delta": stats_payload(np.concatenate(action_delta_parts, axis=0), **payload_kwargs),
        "state": stats_payload(np.concatenate(state_parts, axis=0), **payload_kwargs),
    }
    if force_parts:
        out["force"] = stats_payload(np.concatenate(force_parts, axis=0), **payload_kwargs)

    return out, {"per_domain": dict(sorted(per_domain.items())), "num_loaded_episodes": len(files)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute global chunk stats from train episodes in a split manifest.")
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--split", default="train", help="Manifest split to use. Defaults to train.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override manifest data_root.")
    parser.add_argument("--domains", nargs="+", default=None, help="Optional subset of domains from the manifest split.")
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--state-key", default="observation.state.ee_pose,observation.state.q")
    parser.add_argument("--force-key", default="observation.state.wrench_filtered_chunk")
    parser.add_argument("--force-dim", type=int, default=6)
    parser.add_argument("--label-filter", choices=["all", "only_normal"], default="all")
    parser.add_argument("--max-rows-per-file", type=int, default=None)
    parser.add_argument("--robust-mode", choices=["none", "clip", "row_filter"], default="none")
    parser.add_argument("--clip-percentiles", type=float, nargs=2, default=(0.5, 99.5))
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    low, high = float(args.clip_percentiles[0]), float(args.clip_percentiles[1])
    if not 0.0 <= low < high <= 100.0:
        raise ValueError(f"--clip-percentiles must satisfy 0 <= low < high <= 100, got {args.clip_percentiles}")

    state_key = normalize_feature_keys(args.state_key)
    parquet_files, manifest_meta = parquets_from_manifest(
        args.split_manifest,
        split=args.split,
        data_root=args.data_root,
        domains=args.domains,
    )
    stats, load_meta = compute_stats(
        parquet_files,
        action_key=args.action_key,
        state_key=state_key,
        force_key=args.force_key,
        force_dim=args.force_dim,
        label_filter=args.label_filter,
        max_rows_per_file=args.max_rows_per_file,
        robust_mode=args.robust_mode,
        clip_percentiles=(low, high),
        seed=args.seed,
        eps=args.eps,
    )

    stats["_metadata"] = {
        "generated_by": "compute_global_chunk_stats_from_split.py",
        "action_key": args.action_key,
        "state_key": state_key,
        "force_key": args.force_key,
        "force_dim": int(args.force_dim),
        "label_filter": args.label_filter,
        "max_rows_per_file": args.max_rows_per_file,
        "seed": int(args.seed),
        **manifest_meta,
        **load_meta,
    }

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    with args.save_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote global chunk stats from {args.split} split: {args.save_path}")
    print(f"episodes: {stats['_metadata']['num_loaded_episodes']} domains: {len(stats['_metadata']['domains'])}")


if __name__ == "__main__":
    main()
