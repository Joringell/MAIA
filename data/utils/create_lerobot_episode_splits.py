#!/usr/bin/env python3
"""Create a fixed train/val/test episode split manifest for LeRobot datasets.

The output format is directly consumable by CustomLeRobotChunkDataset via:

    split_manifest: /path/to/manifest.json
    split: train | val | test

By default, each domain contributes 30 test episodes, 20 validation episodes,
and all remaining episodes to training. When both human and pi05_joint sources
exist in a domain, val/test are split evenly across those sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_ROOT = "/root/CRIA"
DEFAULT_SOURCES = ("human", "pi05_joint")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _stable_seed(seed: int, *parts: object) -> int:
    text = "::".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _discover_domain_path(data_root: Path, domain: str) -> Path:
    matches = sorted(data_root.glob(f"*{''}/{domain}"))
    matches = [path for path in matches if (path / "meta" / "episodes.jsonl").exists()]
    if not matches:
        raise FileNotFoundError(f"Could not find domain '{domain}' under {data_root}")
    if len(matches) > 1:
        raise ValueError(f"Domain '{domain}' is ambiguous under {data_root}: {matches}")
    return matches[0]


def _discover_all_domains(data_root: Path) -> list[str]:
    domains = []
    for path in sorted(data_root.glob("*/*")):
        if 'success' in  path.name or 'real' in path.name:
            continue
        if (path / "meta" / "episodes.jsonl").exists():
            domains.append(path.name)
    if not domains:
        raise FileNotFoundError(f"No domains with meta/episodes.jsonl found under {data_root}")
    return domains


def _allocate_even_with_capacity(
    sources: list[str],
    capacities: dict[str, int],
    count: int,
) -> dict[str, int]:
    if count < 0:
        raise ValueError("count must be non-negative")
    total_capacity = sum(capacities.get(source, 0) for source in sources)
    if count > total_capacity:
        raise ValueError(f"Requested {count} episodes but only {total_capacity} are available")
    if count == 0:
        return {source: 0 for source in sources}

    quotas = {source: count // len(sources) for source in sources}
    for source in sources[: count % len(sources)]:
        quotas[source] += 1

    changed = True
    while changed:
        changed = False
        overflow = 0
        for source in sources:
            cap = capacities.get(source, 0)
            if quotas[source] > cap:
                overflow += quotas[source] - cap
                quotas[source] = cap
                changed = True

        while overflow > 0:
            candidates = [source for source in sources if quotas[source] < capacities.get(source, 0)]
            if not candidates:
                raise ValueError("Internal quota allocation error: no source capacity left")
            for source in candidates:
                if overflow == 0:
                    break
                quotas[source] += 1
                overflow -= 1
            changed = True

    return quotas


def _allocate_proportional_with_capacity(
    labels: list[object],
    capacities: dict[object, int],
    count: int,
) -> dict[object, int]:
    total_capacity = sum(capacities[label] for label in labels)
    if count > total_capacity:
        raise ValueError(f"Requested {count} items but only {total_capacity} are available")
    if count == 0:
        return {label: 0 for label in labels}

    raw = {label: count * capacities[label] / total_capacity for label in labels}
    quotas = {label: min(capacities[label], int(raw[label])) for label in labels}
    remaining = count - sum(quotas.values())

    order = sorted(
        labels,
        key=lambda label: (raw[label] - int(raw[label]), capacities[label], str(label)),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for label in order:
            if remaining == 0:
                break
            if quotas[label] < capacities[label]:
                quotas[label] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise ValueError("Internal proportional allocation error: no label capacity left")
    return quotas


def _sample_label_balanced(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    by_label: dict[object, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row.get("episode_label", "unknown")].append(row)

    labels = sorted(by_label, key=lambda value: str(value))
    label_counts = {label: len(by_label[label]) for label in labels}
    quotas = _allocate_proportional_with_capacity(labels, label_counts, count)

    selected = []
    for label in labels:
        candidates = list(by_label[label])
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[label]])
    rng.shuffle(selected)
    return selected


def _source_order(available_sources: Iterable[str], preferred_sources: tuple[str, ...]) -> list[str]:
    available = set(available_sources)
    preferred = [source for source in preferred_sources if source in available]
    others = sorted(available - set(preferred))
    return preferred + others


def _sample_source_balanced(
    rows: list[dict],
    count: int,
    *,
    seed: int,
    domain: str,
    split_name: str,
    preferred_sources: tuple[str, ...],
) -> list[dict]:
    if count > len(rows):
        raise ValueError(f"{domain}: requested {count} {split_name} episodes from only {len(rows)} rows")

    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source", "unknown"))].append(row)

    sources = _source_order(by_source, preferred_sources)
    quotas = _allocate_even_with_capacity(
        sources,
        {source: len(by_source[source]) for source in sources},
        count,
    )

    selected = []
    for source in sources:
        rng = random.Random(_stable_seed(seed, domain, split_name, source))
        selected.extend(_sample_label_balanced(by_source[source], quotas[source], rng))

    rng = random.Random(_stable_seed(seed, domain, split_name, "final_shuffle"))
    rng.shuffle(selected)
    return selected


def _episode_length(row: dict, ann_ep: dict | None = None) -> int:
    length = row.get("length")
    if length is None and ann_ep is not None:
        length = ann_ep.get("num_frames")
    if length is None:
        return 0
    return max(0, int(length))


def _count_unique_range_frames(ranges: list, length: int | None = None) -> int:
    frames = set()
    for item in ranges or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        start, end = int(item[0]), int(item[1])
        if end < start:
            continue
        if length is not None and length > 0:
            start = max(0, min(start, length - 1))
            end = max(0, min(end, length - 1))
        frames.update(range(start, end + 1))
    return len(frames)


def _annotate_manual_frame_counts(rows: list[dict], domain_path: Path) -> list[dict]:
    annotation = _read_json(domain_path / "meta" / "annotation.json")
    annotation_episodes = annotation.get("episodes", {}) or {}

    annotated_rows = []
    for row in rows:
        episode_index = int(row["episode_index"])
        ann_ep = (
            annotation_episodes.get(f"{episode_index:06d}", {})
            or annotation_episodes.get(str(episode_index), {})
            or {}
        )
        manual_anomaly = ((ann_ep.get("anomaly") or {}).get("manual") or {})
        ranges = manual_anomaly.get("ranges", []) or []
        length = _episode_length(row, ann_ep)
        count_from_ranges = _count_unique_range_frames(ranges, length)
        count_from_field = manual_anomaly.get("failure_frame_count")
        count_from_field = None if count_from_field is None else int(count_from_field)

        # Prefer ranges because they are the frame-level annotation used for anomaly windows.
        anomaly_frames = count_from_ranges if ranges else int(count_from_field or 0)
        anomaly_frames = max(0, min(anomaly_frames, length))

        enriched = dict(row)
        enriched["length"] = length
        enriched["manual_anomaly_ranges"] = ranges
        enriched["manual_anomaly_frames"] = anomaly_frames
        enriched["manual_normal_frames"] = max(0, length - anomaly_frames)
        enriched["manual_anomaly_frames_from_ranges"] = count_from_ranges
        enriched["manual_anomaly_frames_from_field"] = count_from_field
        enriched["manual_anomaly_count_matches_ranges"] = (
            count_from_field is None or int(count_from_field) == int(count_from_ranges)
        )
        annotated_rows.append(enriched)
    return annotated_rows


def _frame_summary(rows: list[dict]) -> dict:
    total_frames = sum(int(row.get("length", 0) or 0) for row in rows)
    anomaly_frames = sum(int(row.get("manual_anomaly_frames", 0) or 0) for row in rows)
    normal_frames = sum(int(row.get("manual_normal_frames", 0) or 0) for row in rows)
    mismatch_count = sum(
        not bool(row.get("manual_anomaly_count_matches_ranges", True))
        for row in rows
    )
    return {
        "total_frames": int(total_frames),
        "manual_normal_frames": int(normal_frames),
        "manual_anomaly_frames": int(anomaly_frames),
        "manual_normal_ratio": float(normal_frames / total_frames) if total_frames else None,
        "manual_anomaly_ratio": float(anomaly_frames / total_frames) if total_frames else None,
        "episodes_with_manual_anomaly": int(sum(int(row.get("manual_anomaly_frames", 0) or 0) > 0 for row in rows)),
        "manual_anomaly_count_range_mismatch_episodes": int(mismatch_count),
    }


def _summarize_rows(rows: list[dict]) -> dict:
    return {
        "num_episodes": len(rows),
        "source": dict(sorted(Counter(str(row.get("source", "unknown")) for row in rows).items())),
        "episode_label": dict(sorted(Counter(str(row.get("episode_label", "unknown")) for row in rows).items())),
        "source_label": {
            f"{source}|{label}": count
            for (source, label), count in sorted(
                Counter(
                    (str(row.get("source", "unknown")), str(row.get("episode_label", "unknown")))
                    for row in rows
                ).items()
            )
        },
        "frames": _frame_summary(rows),
        "source_frames": {
            source: _frame_summary([row for row in rows if str(row.get("source", "unknown")) == source])
            for source in sorted(set(str(row.get("source", "unknown")) for row in rows))
        },
        "episode_label_frames": {
            label: _frame_summary([row for row in rows if str(row.get("episode_label", "unknown")) == label])
            for label in sorted(set(str(row.get("episode_label", "unknown")) for row in rows))
        },
    }


def _episode_indices(rows: list[dict]) -> list[int]:
    return sorted(int(row["episode_index"]) for row in rows)


def create_split_manifest(
    *,
    data_root: Path,
    domains: list[str],
    output: Path,
    seed: int,
    train_count: int,
    val_count: int,
    test_count: int,
    preferred_sources: tuple[str, ...],
    overwrite: bool,
) -> dict:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}. Pass --overwrite to replace it.")

    manifest = {
        "version": 1,
        "generated_by": "create_lerobot_episode_splits.py",
        "data_root": str(data_root),
        "seed": int(seed),
        "split_policy": {
            "unit": "episode",
            "per_domain_test_count": int(test_count),
            "per_domain_val_count": int(val_count),
            "train": "remaining episodes",
            "source_balance": "even across available preferred sources per domain",
            "preferred_sources": list(preferred_sources),
            "label_balance": "proportional within each source using episode_label",
        },
        "domains": list(domains),
        "train": {},
        "val": {},
        "test": {},
        "summary": {},
    }

    for domain in domains:
        domain_path = _discover_domain_path(data_root, domain)
        rows = _read_jsonl(domain_path / "meta" / "episodes.jsonl")
        if not rows:
            raise ValueError(f"{domain}: no episodes found")
        rows = _annotate_manual_frame_counts(rows, domain_path)

        test_rows = _sample_source_balanced(
            rows,
            test_count,
            seed=seed,
            domain=domain,
            split_name="test",
            preferred_sources=preferred_sources,
        )
        test_ids = {int(row["episode_index"]) for row in test_rows}
        remaining = [row for row in rows if int(row["episode_index"]) not in test_ids]

        val_rows = _sample_source_balanced(
            remaining,
            val_count,
            seed=seed,
            domain=domain,
            split_name="val",
            preferred_sources=preferred_sources,
        )
        val_ids = {int(row["episode_index"]) for row in val_rows}

        if train_count is None:
            train_rows = [
                row for row in remaining
                if int(row["episode_index"]) not in val_ids
            ]
        else:
            remaining_ = [row for row in remaining if int(row["episode_index"]) not in val_ids]
            train_rows = _sample_source_balanced(
            remaining_,
            train_count,
            seed=seed,
            domain=domain,
            split_name="train",
            preferred_sources=preferred_sources,
        )
        manifest["train"][domain] = _episode_indices(train_rows)
        manifest["val"][domain] = _episode_indices(val_rows)
        manifest["test"][domain] = _episode_indices(test_rows)
        manifest["summary"][domain] = {
            "all": _summarize_rows(rows),
            "train": _summarize_rows(train_rows),
            "val": _summarize_rows(val_rows),
            "test": _summarize_rows(test_rows),
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return manifest


def _print_summary(manifest: dict) -> None:
    print(f"Wrote split manifest: {manifest.get('output', '')}".rstrip())
    print(f"seed={manifest['seed']} domains={len(manifest['domains'])}")
    print(
        f"{'domain':26s} {'train':>5s} {'val':>5s} {'test':>5s}  "
        f"{'train_anom%':>11s} {'val_anom%':>9s} {'test_anom%':>10s}  "
        "val_source             test_source"
    )
    for domain in manifest["domains"]:
        summary = manifest["summary"][domain]
        train_n = summary["train"]["num_episodes"]
        val_n = summary["val"]["num_episodes"]
        test_n = summary["test"]["num_episodes"]
        train_anom = summary["train"]["frames"]["manual_anomaly_ratio"]
        val_anom = summary["val"]["frames"]["manual_anomaly_ratio"]
        test_anom = summary["test"]["frames"]["manual_anomaly_ratio"]
        print(
            f"{domain:26s} {train_n:5d} {val_n:5d} {test_n:5d}  "
            f"{(100.0 * train_anom if train_anom is not None else 0.0):11.3f} "
            f"{(100.0 * val_anom if val_anom is not None else 0.0):9.3f} "
            f"{(100.0 * test_anom if test_anom is not None else 0.0):10.3f}  "
            f"{summary['val']['source']!s:22s} {summary['test']['source']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed train/val/test episode manifest for LeRobot domains."
    )
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Root containing category/domain datasets.")
    parser.add_argument(
        "--domains",
        nargs="+",
        required=True,
        help="Domain names to include, or 'all' to include every domain under data-root.",
    )
    parser.add_argument("--output", default=None, help="Output manifest JSON path.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=None, help="Train episodes per domain.")
    parser.add_argument("--val-count", type=int, default=20, help="Validation episodes per domain.")
    parser.add_argument("--test-count", type=int, default=30, help="Test episodes per domain.")
    parser.add_argument(
        "--preferred-sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help="Sources to balance evenly when available.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).expanduser()
    domains = _discover_all_domains(data_root) if args.domains == ["all"] else list(args.domains)
    output = (
        Path(args.output).expanduser()
        if args.output is not None
        else data_root
        / "splits"
        / f"lerobot_episode_split_seed{args.seed}_val{args.val_count}_test{args.test_count}.json"
    )

    manifest = create_split_manifest(
        data_root=data_root,
        domains=domains,
        output=output,
        seed=args.seed,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        preferred_sources=tuple(args.preferred_sources),
        overwrite=args.overwrite,
    )
    manifest["output"] = str(output)
    _print_summary(manifest)


if __name__ == "__main__":
    main()
