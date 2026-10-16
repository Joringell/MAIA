import json
import os
import torch
from runner.ge_trainer_video_anomaly import _binary_counts

def _best_f1_metrics(scores, targets):
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1)
    if scores.numel() == 0:
        return None
    best = None
    for threshold in torch.unique(scores).sort().values:
        metrics = _binary_counts(scores, targets, threshold=float(threshold.item()))
        if best is None or metrics["f1"] > best["f1"]:
            best = metrics
    return best


def _score_summary(scores, targets):
    scores = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    targets = torch.as_tensor(targets, dtype=torch.float32).reshape(-1) >= 0.5
    output = {}
    for name, mask in (("all", torch.ones_like(targets, dtype=torch.bool)), ("normal", ~targets), ("anomaly", targets)):
        values = scores[mask]
        if values.numel() == 0:
            output[name] = None
            continue
        output[name] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "p50": float(torch.quantile(values, 0.50).item()),
            "p90": float(torch.quantile(values, 0.90).item()),
            "p95": float(torch.quantile(values, 0.95).item()),
            "max": float(values.max().item()),
            "count": int(values.numel()),
        }
    return output


def _section_dict(args, name):
    value = getattr(args, name, {})
    return dict(value) if isinstance(value, dict) else {}


def observed_feature_config(args):
    return _section_dict(args, "observed_feature")


def observed_cache_dir(args):
    cache_dir = getattr(args, "observed_feature_cache_dir", None)
    if cache_dir in (None, ""):
        cache_dir = observed_feature_config(args).get("cache_dir")
    return None if cache_dir in (None, "") else os.path.expanduser(str(cache_dir))


def use_observed_cache(args):
    return observed_cache_dir(args) is not None


def data_cfg_for_split(args, split):
    data_cfg = dict(args.data[split])
    split_manifest = getattr(args, "split_manifest", None)
    if split_manifest not in (None, "") and not data_cfg.get("split_manifest"):
        data_cfg["split_manifest"] = split_manifest
    return data_cfg


def observed_feature_settings(args):
    cfg = observed_feature_config(args)
    return (
        cfg,
        str(cfg.get("feature_mode", "video_state_force")),
        str(cfg.get("fit_filter", "normal_episode_windows")),
        int(cfg.get("force_bins", 5)),
    )


def domains_for_split(args, split):
    if split not in args.data:
        return None
    data_cfg = dict(args.data[split])
    domains = data_cfg.get("domains")
    if domains in (None, ""):
        return None
    if isinstance(domains, str):
        if domains.strip().lower() == "all":
            split_manifest = data_cfg.get("split_manifest") or getattr(args, "split_manifest", None)
            if split_manifest not in (None, ""):
                with open(os.path.expanduser(str(split_manifest)), "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                manifest_domains = manifest.get("domains")
                if manifest_domains:
                    return [str(item).strip() for item in manifest_domains if str(item).strip()]
            return None
        return [item.strip() for item in domains.split(",") if item.strip()]
    return [str(item).strip() for item in domains if str(item).strip()]
