import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze.baseline_episode_eval_common import (
    best_f1_metrics,
    binary_metrics,
    binary_metrics_from_predictions,
    compute_cp_band,
    records_scores_targets,
    save_episode_plots,
    save_unified_baseline_outputs,
    score_summary,
    write_csv,
)
from data.pi05_feature_cache_dataset import Pi05FeatureCacheDataset
from models.baselines.safe import SafeLSTMFailureModel, SafeMLPFailureModel
from utils.path_utils import load_resolved_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SAFE MLP/LSTM on pi05 feature-cache sequences with single-band and split-binary CP."
    )
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--safe_checkpoint", required=True, help="Path to safe_pi05.pt.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--feature_cache_dir", default=None)
    parser.add_argument(
        "--single_band_cp_metrics",
        default=None,
        help="Reuse an existing single_band/safe_pi05_cp_metrics.json for single-band CP instead of recalibrating.",
    )
    parser.add_argument("--calibration_split", default="val")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--domains", default=None, help="Comma-separated domain subset override.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cp_alpha", type=float, default=0.1)
    parser.add_argument(
        "--cp_alarm_margin",
        type=float,
        default=0.0,
        help="Require score > CP band + margin for single-band alarms and red markers.",
    )
    parser.add_argument("--single_band_method", choices=["pointwise", "functional"], default="pointwise")
    parser.add_argument(
        "--cp_variants",
        choices=["single_band", "split_binary", "both"],
        default="both",
        help="single_band is SAFE/logpZO-style upper band; split_binary uses positive and negative class CP thresholds.",
    )
    parser.add_argument(
        "--target_source",
        choices=["frame", "episode"],
        default="frame",
        help="Evaluation target. frame uses current query-frame anomaly label; episode uses 1 - episode_label.",
    )
    parser.add_argument(
        "--single_band_normal_filter",
        choices=["success_episodes", "normal_frames", "success_normal_frames"],
        default="success_episodes",
        help="Calibration records used for the one-sided upper band.",
    )
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_episode_plots", type=int, default=None)
    return parser.parse_args()


def _dict_cfg(value):
    return dict(value) if isinstance(value, dict) else {}


def _domains_for_split(cfg, split, override=None):
    if override not in (None, ""):
        return [item.strip() for item in str(override).split(",") if item.strip()]
    data_cfg = _dict_cfg(_dict_cfg(cfg.get("data", {})).get(split, {}))
    domains = data_cfg.get("domains")
    if domains in (None, ""):
        return None
    if isinstance(domains, str):
        return None if domains.strip().lower() == "all" else [item.strip() for item in domains.split(",") if item.strip()]
    return [str(item).strip() for item in domains if str(item).strip()]


def _sequence_filter_for_split(cfg, ckpt, split):
    cache_cfg = _dict_cfg(cfg.get("pi05_feature", {}))
    cache_cfg.update(_dict_cfg(ckpt.get("pi05_feature", {})))
    split_cfg = _dict_cfg(_dict_cfg(cfg.get("data", {})).get(split, {}))
    return str(
        split_cfg.get(
            "sequence_filter",
            cache_cfg.get(f"{split}_sequence_filter", cache_cfg.get("sequence_filter", "all")),
        )
    )


def _feature_cache_dir(cfg, ckpt, args):
    if args.feature_cache_dir not in (None, ""):
        return os.path.expanduser(str(args.feature_cache_dir))
    cache_cfg = _dict_cfg(cfg.get("pi05_feature", {}))
    cache_cfg.update(_dict_cfg(ckpt.get("pi05_feature", {})))
    cache_dir = cache_cfg.get("cache_dir")
    if cache_dir in (None, ""):
        raise ValueError("SAFE pi05 eval requires pi05_feature.cache_dir or --feature_cache_dir.")
    return os.path.expanduser(str(cache_dir))


def _max_open_shards(cfg, ckpt):
    cache_cfg = _dict_cfg(cfg.get("pi05_feature", {}))
    cache_cfg.update(_dict_cfg(ckpt.get("pi05_feature", {})))
    return int(cache_cfg.get("max_open_shards", 16))


def _make_dataset(cfg, ckpt, args, split):
    cache_dir = _feature_cache_dir(cfg, ckpt, args)
    dataset = Pi05FeatureCacheDataset(
        cache_dir=cache_dir,
        split=split,
        max_open_shards=_max_open_shards(cfg, ckpt),
        domains=_domains_for_split(cfg, split, args.domains),
        sequence_mode=True,
        sequence_filter=_sequence_filter_for_split(cfg, ckpt, split),
        min_sequence_length=1,
    )
    return dataset, cache_dir


def _pad_tensor_sequence(values, pad_value=0):
    max_len = max(int(value.shape[0]) for value in values)
    shape = (len(values), max_len) + tuple(values[0].shape[1:])
    output = values[0].new_full(shape, pad_value)
    for row, value in enumerate(values):
        output[row, : value.shape[0]] = value
    return output


def _collate_sequences(batch):
    output = {}
    tensor_keys = {
        "features": 0,
        "valid_mask": 0,
        "label": -1,
        "risk_label": -1,
        "frame_label": -1,
        "frame_anomaly_label": -1,
        "interaction_label": -1,
        "source_index": -1,
        "frame_index": -1,
        "window_start": -1,
        "window_end": -1,
        "valid_len": 0,
    }
    for key, pad_value in tensor_keys.items():
        if key in batch[0]:
            output[key] = _pad_tensor_sequence([item[key] for item in batch], pad_value=pad_value)

    scalar_keys = (
        "_cache_index",
        "success_labels",
        "episode_label",
        "total_frames",
        "episode_id",
        "source_episode_index",
    )
    for key in scalar_keys:
        if key in batch[0]:
            output[key] = torch.stack([item[key] for item in batch], dim=0)

    string_keys = ("domain_name", "source_h5", "source_h5_basename", "parquet_path", "source", "control_group")
    for key in string_keys:
        if key in batch[0]:
            output[key] = [str(item[key]) for item in batch]
    return output


def load_safe_model(checkpoint_path, device):
    ckpt = torch.load(os.path.expanduser(str(checkpoint_path)), map_location="cpu")
    safe_cfg = _dict_cfg(ckpt.get("safe", {}))
    if "feature_dim" not in safe_cfg:
        raise ValueError("SAFE checkpoint must include safe.feature_dim")
    model_type = str(ckpt.get("model_type", safe_cfg.get("model_type", "lstm"))).lower()
    kwargs = {
        "input_dim": int(safe_cfg["feature_dim"]),
        "hidden_dim": int(safe_cfg.get("hidden_dim", 256)),
        "num_layers": int(safe_cfg.get("num_layers", 1 if model_type == "lstm" else 2)),
        "dropout": float(safe_cfg.get("dropout", 0.0)),
        "history_steps": int(safe_cfg.get("history_steps", -1 if model_type == "lstm" else 1)),
        "cumsum": bool(safe_cfg.get("cumsum", False)),
        "rmean": bool(safe_cfg.get("rmean", False)),
        "target_mode": str(safe_cfg.get("target_mode", "episode")),
    }
    if model_type == "lstm":
        model = SafeLSTMFailureModel(**kwargs)
    elif model_type == "mlp":
        model = SafeMLPFailureModel(**kwargs)
    else:
        raise ValueError(f"Unsupported SAFE model_type={model_type}")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    return model.to(device).eval(), ckpt


def _normalize_features(features, ckpt, device):
    if not bool(ckpt.get("normalize_features", True)):
        return features.float()
    mean = ckpt.get("feature_mean")
    std = ckpt.get("feature_std")
    if mean is None or std is None:
        raise ValueError("SAFE checkpoint has normalize_features=True but no feature_mean/std.")
    mean = mean.to(device=device, dtype=features.dtype).view(1, 1, -1)
    std = std.to(device=device, dtype=features.dtype).clamp_min(1e-6).view(1, 1, -1)
    return (features.float() - mean) / std


def _target_from_row(batch, row, time_idx, target_source):
    episode_label = float(batch["episode_label"][row].detach().cpu().item())
    if target_source == "episode":
        return float(1.0 - episode_label)
    frame_target = float(batch["frame_anomaly_label"][row, time_idx].detach().cpu().item())
    return frame_target if frame_target >= 0.0 else None


@torch.no_grad()
def score_split(cfg, ckpt, model, args, split, device):
    dataset, cache_dir = _make_dataset(cfg, ckpt, args, split)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=False,
        collate_fn=_collate_sequences,
    )
    records = []
    total = len(dataset) if args.max_episodes is None else min(len(dataset), int(args.max_episodes))
    seen_episodes = 0
    for batch in tqdm(loader, desc=f"Scoring SAFE pi05 {split}"):
        if seen_episodes >= total:
            break
        batch_size = int(batch["features"].shape[0])
        keep = min(batch_size, total - seen_episodes)
        features = batch["features"][:keep].to(device=device, dtype=torch.float32, non_blocking=True)
        valid_mask = batch["valid_mask"][:keep].to(device=device, non_blocking=True).bool()
        model_batch = {"features": _normalize_features(features, ckpt, device), "valid_mask": valid_mask}
        scores = model(model_batch).detach().cpu().float()

        for row in range(keep):
            episode_label = float(batch["episode_label"][row].detach().cpu().item())
            time_count = int(valid_mask[row].sum().detach().cpu().item())
            for time_idx in range(time_count):
                target = _target_from_row(batch, row, time_idx, args.target_source)
                if target is None:
                    continue
                frame_index = int(batch["frame_index"][row, time_idx].detach().cpu().item())
                source_index = int(batch.get("source_index", batch["frame_index"])[row, time_idx].detach().cpu().item())
                records.append(
                    {
                        "source_index": source_index,
                        "episode_dataset_index": int(batch["source_episode_index"][row].detach().cpu().item()),
                        "episode_id": int(batch["episode_id"][row].detach().cpu().item()),
                        "domain": str(batch["domain_name"][row]),
                        "source_h5": str(batch.get("source_h5", [""] * keep)[row]),
                        "source_h5_basename": str(batch.get("source_h5_basename", [""] * keep)[row]),
                        "parquet_path": str(batch.get("parquet_path", [""] * keep)[row]),
                        "window_start": frame_index,
                        "detection_frame": frame_index,
                        "chunk_end": frame_index,
                        "window_end": frame_index + 1,
                        "valid_len": 1,
                        "total_frames": int(batch["total_frames"][row].detach().cpu().item()),
                        "score": float(scores[row, time_idx].item()),
                        "target": float(target),
                        "label": float(1.0 - target),
                        "episode_label": episode_label,
                        "risk_label": float(target),
                        "frame_label": int(batch["frame_label"][row, time_idx].detach().cpu().item()),
                        "frame_anomaly_label": [float(target)],
                        "interaction_label": int(batch["interaction_label"][row, time_idx].detach().cpu().item()),
                    }
                )
        seen_episodes += keep
    return records, cache_dir


def load_single_band_cp_payload(path):
    with open(os.path.expanduser(str(path)), "r", encoding="utf-8") as f:
        payload = json.load(f)
    cp_payload = payload.get("cp_band", payload)
    if not isinstance(cp_payload, dict) or "upper_band" not in cp_payload:
        raise ValueError(f"{path} does not contain a reusable single-band cp_band.upper_band.")
    cp_payload = dict(cp_payload)
    cp_payload["reused"] = True
    cp_payload["reused_from"] = os.path.expanduser(str(path))
    return cp_payload


def is_single_band_normal(row, mode):
    target = float(row.get("target", 1.0))
    episode_label = float(row.get("episode_label", 0.0))
    if mode == "success_episodes":
        return episode_label >= 0.5
    if mode == "normal_frames":
        return target < 0.5
    if mode == "success_normal_frames":
        return episode_label >= 0.5 and target < 0.5
    raise ValueError(f"Unsupported single_band_normal_filter={mode}")


def calibration_records_for_single_band(records, mode):
    selected = [row for row in records if is_single_band_normal(row, mode)]
    if mode in {"normal_frames", "success_normal_frames"}:
        # compute_cp_band treats trajectories with episode_label>=0.5 as the normal set.
        selected = [dict(row, episode_label=1.0) for row in selected]
    return selected


def conformal_quantile(values, alpha):
    scores = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    scores = scores[torch.isfinite(scores)]
    if scores.numel() == 0:
        return None
    k = int(torch.ceil(torch.tensor((scores.numel() + 1) * (1.0 - float(alpha)))).item())
    k = min(max(1, k), int(scores.numel()))
    return float(torch.sort(scores).values[k - 1].item())


def compute_split_binary_cp(calibration_records, eval_records, alpha):
    cal_scores = torch.tensor([float(row["score"]) for row in calibration_records], dtype=torch.float32)
    cal_labels = torch.tensor([int(float(row["target"]) >= 0.5) for row in calibration_records], dtype=torch.long)
    pos_scores = cal_scores[cal_labels == 1]
    neg_scores = cal_scores[cal_labels == 0]
    pos_nconf = 1.0 - pos_scores
    neg_nconf = neg_scores
    pos_q = conformal_quantile(pos_nconf, alpha)
    neg_q = conformal_quantile(neg_nconf, alpha)
    if pos_q is None or neg_q is None:
        return {
            "enabled": False,
            "method": "split_binary",
            "reason": "Need both positive and negative calibration records.",
            "num_calibration_records": len(calibration_records),
            "num_positive_calibration_records": int(pos_scores.numel()),
            "num_negative_calibration_records": int(neg_scores.numel()),
        }

    failure_threshold = float(1.0 - pos_q)
    normal_threshold = float(neg_q)
    eval_scores = torch.tensor([float(row["score"]) for row in eval_records], dtype=torch.float32)
    eval_targets = torch.tensor([int(float(row["target"]) >= 0.5) for row in eval_records], dtype=torch.float32)
    include_failure = eval_scores >= failure_threshold
    include_normal = eval_scores <= normal_threshold
    pred_sets = []
    covered = []
    for score, target, inc_fail, inc_norm in zip(eval_scores, eval_targets, include_failure, include_normal):
        pred_set = []
        if bool(inc_norm.item()):
            pred_set.append(0)
        if bool(inc_fail.item()):
            pred_set.append(1)
        pred_sets.append(pred_set)
        covered.append(int(target.item()) in pred_set)

    return {
        "enabled": True,
        "method": "split_binary",
        "alpha": float(alpha),
        "num_calibration_records": len(calibration_records),
        "num_positive_calibration_records": int(pos_scores.numel()),
        "num_negative_calibration_records": int(neg_scores.numel()),
        "positive_nonconformity_threshold": pos_q,
        "negative_nonconformity_threshold": normal_threshold,
        "failure_score_threshold": failure_threshold,
        "normal_score_threshold": normal_threshold,
        "decision_metrics": {
            "failure_in_prediction_set": binary_metrics_from_predictions(include_failure.float(), eval_targets),
            "normal_excluded": binary_metrics_from_predictions((~include_normal).float(), eval_targets),
            "failure_threshold_manual": binary_metrics(eval_scores, eval_targets, failure_threshold),
            "normal_threshold_manual": binary_metrics(eval_scores, eval_targets, normal_threshold),
            "best_f1": best_f1_metrics(eval_scores, eval_targets),
        },
        "prediction_set_metrics": {
            "coverage": float(torch.tensor(covered, dtype=torch.float32).mean().item()) if covered else None,
            "ambiguous_ratio": float((include_failure & include_normal).float().mean().item()) if eval_scores.numel() else None,
            "empty_ratio": float((~include_failure & ~include_normal).float().mean().item()) if eval_scores.numel() else None,
            "singleton_failure_ratio": float((include_failure & ~include_normal).float().mean().item()) if eval_scores.numel() else None,
            "singleton_normal_ratio": float((~include_failure & include_normal).float().mean().item()) if eval_scores.numel() else None,
        },
    }


def score_saturation_summary(records):
    scores = torch.tensor([float(row["score"]) for row in records], dtype=torch.float32)
    targets = torch.tensor([int(float(row["target"]) >= 0.5) for row in records], dtype=torch.bool)
    output = {}
    for name, mask in (
        ("all", torch.ones_like(targets, dtype=torch.bool)),
        ("normal", ~targets),
        ("anomaly", targets),
    ):
        values = scores[mask]
        if values.numel() == 0:
            output[name] = None
            continue
        output[name] = {
            "count": int(values.numel()),
            "score_le_1e-7_ratio": float((values <= 1e-7).float().mean().item()),
            "score_ge_1_minus_1e-7_ratio": float((values >= 1.0 - 1e-7).float().mean().item()),
            "unique_rounded_6": int(torch.unique(torch.round(values * 1_000_000) / 1_000_000).numel()),
        }
    return output


def write_split_binary_records(method_dir, eval_records, cp_payload):
    if not cp_payload.get("enabled"):
        return None
    failure_threshold = float(cp_payload["failure_score_threshold"])
    normal_threshold = float(cp_payload["normal_score_threshold"])
    rows = []
    for row in eval_records:
        score = float(row["score"])
        target = int(float(row.get("target", 0.0)) >= 0.5)
        include_normal = score <= normal_threshold
        include_failure = score >= failure_threshold
        rows.append(
            {
                "domain": str(row.get("domain", "")),
                "episode_id": int(row.get("episode_id", -1)),
                "source_index": int(row.get("source_index", -1)),
                "frame_index": int(row.get("detection_frame", row.get("window_start", -1))),
                "score": score,
                "target": target,
                "include_normal": bool(include_normal),
                "include_failure": bool(include_failure),
                "normal_excluded_alarm": bool(not include_normal),
                "failure_in_set_alarm": bool(include_failure),
                "ambiguous_set": bool(include_normal and include_failure),
                "empty_set": bool((not include_normal) and (not include_failure)),
                "normal_score_threshold": normal_threshold,
                "failure_score_threshold": failure_threshold,
                "episode_label": row.get("episode_label"),
                "frame_label": row.get("frame_label"),
                "interaction_label": row.get("interaction_label"),
            }
        )
    output_path = os.path.join(method_dir, "split_binary_window_records.csv")
    write_csv(output_path, rows)
    return output_path


def write_single_band_outputs(
    output_dir,
    cfg,
    ckpt,
    args,
    cache_dir,
    calibration_records,
    eval_records,
    reused_cp_payload=None,
):
    method_dir = os.path.join(output_dir, "single_band")
    os.makedirs(method_dir, exist_ok=True)
    if reused_cp_payload is None:
        normal_records = calibration_records_for_single_band(calibration_records, args.single_band_normal_filter)
        cp_payload = compute_cp_band(
            normal_records,
            eval_records,
            alpha=float(args.cp_alpha),
            method=str(args.single_band_method),
        )
        cp_payload["single_band_normal_filter"] = str(args.single_band_normal_filter)
        cp_payload["num_calibration_records_before_filter"] = len(calibration_records)
        cp_payload["num_calibration_records_after_filter"] = len(normal_records)
        cp_payload["reused"] = False
    else:
        cp_payload = dict(reused_cp_payload)
    cp_payload["alarm_margin"] = float(args.cp_alarm_margin)
    plot_paths = save_episode_plots(method_dir, eval_records, cp_payload, max_plots=args.max_episode_plots)
    unified_paths = save_unified_baseline_outputs(
        method_dir,
        eval_records,
        cp_payload,
        metadata={
            "script": "analyze/eval_safe_pi05_metrics.py",
            "config_file": args.config_file,
            "safe_checkpoint": args.safe_checkpoint,
            "feature_cache_dir": cache_dir,
            "calibration_split": args.calibration_split,
            "eval_split": args.eval_split,
            "target_source": args.target_source,
            "cp_variant": "single_band",
            "baseline": ckpt.get("baseline", "safe_pi05"),
        },
    )
    payload = {
        "config_file": args.config_file,
        "safe_checkpoint": args.safe_checkpoint,
        "feature_cache_dir": cache_dir,
        "calibration_split": args.calibration_split,
        "eval_split": args.eval_split,
        "target_source": args.target_source,
        "score_direction": "higher_is_more_anomalous",
        "baseline": ckpt.get("baseline", "safe_pi05"),
        "safe": ckpt.get("safe", {}),
        "single_band_cp_metrics": args.single_band_cp_metrics,
        "calibration_score_summary": score_summary(calibration_records) if calibration_records else None,
        "eval_score_summary": score_summary(eval_records),
        "calibration_score_saturation": score_saturation_summary(calibration_records) if calibration_records else None,
        "eval_score_saturation": score_saturation_summary(eval_records),
        "cp_band": cp_payload,
        "cp_episode_metrics": cp_payload.get("episode_metrics") if cp_payload else None,
        "cp_window_metrics": cp_payload.get("window_metrics") if cp_payload else None,
        "episode_plots": plot_paths,
        "unified_episode_eval": unified_paths,
    }
    with open(os.path.join(method_dir, "safe_pi05_cp_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    torch.save(
        {"calibration_records": calibration_records, "eval_records": eval_records, "cp_band": cp_payload},
        os.path.join(method_dir, "safe_pi05_cp_records.pt"),
    )
    return payload


def write_split_binary_outputs(output_dir, cfg, ckpt, args, cache_dir, calibration_records, eval_records):
    del cfg
    method_dir = os.path.join(output_dir, "split_binary")
    os.makedirs(method_dir, exist_ok=True)
    cp_payload = compute_split_binary_cp(calibration_records, eval_records, alpha=float(args.cp_alpha))
    split_records_path = write_split_binary_records(method_dir, eval_records, cp_payload)

    scores, targets = records_scores_targets(eval_records)
    payload = {
        "config_file": args.config_file,
        "safe_checkpoint": args.safe_checkpoint,
        "feature_cache_dir": cache_dir,
        "calibration_split": args.calibration_split,
        "eval_split": args.eval_split,
        "target_source": args.target_source,
        "score_direction": "higher_is_more_anomalous",
        "baseline": ckpt.get("baseline", "safe_pi05"),
        "safe": ckpt.get("safe", {}),
        "calibration_score_summary": score_summary(calibration_records),
        "eval_score_summary": score_summary(eval_records),
        "calibration_score_saturation": score_saturation_summary(calibration_records),
        "eval_score_saturation": score_saturation_summary(eval_records),
        "eval_metrics_best_f1": best_f1_metrics(scores, targets),
        "split_binary_cp": cp_payload,
        "split_binary_window_records": split_records_path,
        "unified_episode_eval": None,
        "unified_episode_eval_note": "Split-binary CP returns prediction sets, not a single upper band. Use split_binary_window_records and split_binary_cp.decision_metrics.",
    }
    with open(os.path.join(method_dir, "safe_pi05_split_binary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    torch.save(
        {"calibration_records": calibration_records, "eval_records": eval_records, "split_binary_cp": cp_payload},
        os.path.join(method_dir, "safe_pi05_split_binary_records.pt"),
    )
    return payload


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_resolved_config(args.config_file)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load_safe_model(args.safe_checkpoint, device)
    reused_single_band_cp = (
        load_single_band_cp_payload(args.single_band_cp_metrics) if args.single_band_cp_metrics not in (None, "") else None
    )
    needs_calibration = args.cp_variants in {"split_binary", "both"} or reused_single_band_cp is None
    if needs_calibration:
        calibration_records, calibration_cache_dir = score_split(cfg, ckpt, model, args, args.calibration_split, device)
    else:
        calibration_records, calibration_cache_dir = [], None
    eval_records, eval_cache_dir = score_split(cfg, ckpt, model, args, args.eval_split, device)
    if needs_calibration and not calibration_records:
        raise ValueError("No calibration records were scored.")
    if not eval_records:
        raise ValueError("No eval records were scored.")

    outputs = {}
    if args.cp_variants in {"single_band", "both"}:
        outputs["single_band"] = write_single_band_outputs(
            args.output_dir,
            cfg,
            ckpt,
            args,
            eval_cache_dir,
            calibration_records,
            eval_records,
            reused_cp_payload=reused_single_band_cp,
        )
    if args.cp_variants in {"split_binary", "both"}:
        outputs["split_binary"] = write_split_binary_outputs(
            args.output_dir,
            cfg,
            ckpt,
            args,
            eval_cache_dir,
            calibration_records,
            eval_records,
        )

    root_payload = {
        "config_file": args.config_file,
        "safe_checkpoint": args.safe_checkpoint,
        "calibration_cache_dir": calibration_cache_dir,
        "eval_cache_dir": eval_cache_dir,
        "calibration_split": args.calibration_split,
        "eval_split": args.eval_split,
        "target_source": args.target_source,
        "cp_alpha": float(args.cp_alpha),
        "cp_alarm_margin": float(args.cp_alarm_margin),
        "cp_variants": args.cp_variants,
        "single_band_cp_metrics": args.single_band_cp_metrics,
        "num_calibration_records": len(calibration_records),
        "num_eval_records": len(eval_records),
        "outputs": {
            "single_band": os.path.join(args.output_dir, "single_band", "safe_pi05_cp_metrics.json"),
            "split_binary": os.path.join(args.output_dir, "split_binary", "safe_pi05_split_binary_metrics.json"),
        },
    }
    with open(os.path.join(args.output_dir, "safe_pi05_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(root_payload, f, indent=2)
    torch.save(
        {
            "calibration_records": calibration_records,
            "eval_records": eval_records,
            "outputs": outputs,
        },
        os.path.join(args.output_dir, "safe_pi05_records.pt"),
    )
    print(json.dumps(root_payload, indent=2))


if __name__ == "__main__":
    main()
