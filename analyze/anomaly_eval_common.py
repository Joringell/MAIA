import json
import os
import re
import csv
import subprocess

import cv2
import numpy as np
import pandas as pd

from analyze.eval_common import load_failure_transition_frame


def extract_episode_id(parquet_path):
    match = re.search(r"episode_(\d+)\.parquet$", os.path.basename(str(parquet_path)))
    return int(match.group(1)) if match else None


def load_frame_labels(parquet_path, total_frames):
    data = pd.read_parquet(parquet_path)
    if "frame_label" in data.columns:
        labels = data["frame_label"].to_numpy().astype(np.int64)
    elif "label" in data.columns:
        labels = data["label"].to_numpy().astype(np.int64)
    else:
        labels = np.ones(total_frames, dtype=np.int64)
    if labels.shape[0] < total_frames:
        labels = np.pad(labels, (0, total_frames - labels.shape[0]), constant_values=1)
    labels = labels[:total_frames]
    return labels, (labels == 0).astype(np.float32)


def save_label_plot(path, step_prob, frame_target=None, transition_frame=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    length = step_prob.shape[0]
    x = np.arange(length)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, step_prob, color="tomato", linewidth=1.8, label="pred anomaly prob")
    if frame_target is not None:
        ax.plot(x, frame_target[:length], color="cornflowerblue", linewidth=1.8, drawstyle="steps-post", label="gt anomaly")
    if transition_frame is not None and 0 <= transition_frame < length:
        ax.axvline(transition_frame, color="green", linestyle="--", linewidth=1.5, label="failure onset")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("trajectory timestep")
    ax.set_ylabel("anomaly")
    ax.grid(True, linestyle=":", alpha=0.5)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(3, len(handles)),
            frameon=False,
            bbox_to_anchor=(0.5, 0.0),
        )
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _curve_for_export(values, length):
    if values is None:
        return np.full(length, np.nan, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] >= length:
        return values[:length]
    out = np.full(length, np.nan, dtype=np.float64)
    out[: values.shape[0]] = values
    return out


def _csv_float(value):
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.10g}"


def save_label_curve_data(path_prefix, step_prob, frame_target=None, transition_frame=None, extra_curves=None):
    step_prob = np.asarray(step_prob, dtype=np.float64).reshape(-1)
    length = int(step_prob.shape[0])
    frame_index = np.arange(length, dtype=np.int64)
    arrays = {
        "frame_index": frame_index,
        "pred_anomaly_prob": step_prob,
        "gt_anomaly": _curve_for_export(frame_target, length),
        "is_failure_onset": np.zeros(length, dtype=np.int64),
    }
    if transition_frame is not None and 0 <= int(transition_frame) < length:
        arrays["is_failure_onset"][int(transition_frame)] = 1
    for name, values in (extra_curves or {}).items():
        if values is not None:
            arrays[str(name)] = _curve_for_export(values, length)

    csv_path = f"{path_prefix}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(arrays))
        writer.writeheader()
        for row_idx in range(length):
            row = {}
            for key, values in arrays.items():
                value = values[row_idx]
                row[key] = int(value) if np.issubdtype(values.dtype, np.integer) else _csv_float(value)
            writer.writerow(row)

    npz_path = f"{path_prefix}.npz"
    np.savez_compressed(
        npz_path,
        **arrays,
        transition_frame=np.asarray(-1 if transition_frame is None else int(transition_frame), dtype=np.int64),
    )
    return {"csv": csv_path, "npz": npz_path}


def parse_dataset_indices(dataset_index, dataset_indices=None):
    raw_values = dataset_indices if dataset_indices is not None else [dataset_index]
    if isinstance(raw_values, (str, int)):
        raw_values = [raw_values]
    indices = []
    for value in raw_values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                indices.append(int(part))
    if not indices:
        raise ValueError("No dataset indices were provided.")
    return indices


def basic_binary_metrics(scores, targets):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    valid = np.isfinite(scores) & np.isfinite(targets)
    scores = scores[valid]
    targets = targets[valid]
    if scores.size == 0:
        return None
    pred = scores >= 0.5
    pos = targets == 1
    neg = ~pos
    tp = int(np.logical_and(pred, pos).sum())
    fp = int(np.logical_and(pred, neg).sum())
    tn = int(np.logical_and(~pred, neg).sum())
    fn = int(np.logical_and(~pred, pos).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float((tp + tn) / max(1, scores.size)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_ratio": float(pos.mean()),
        "pred_positive_ratio": float(pred.mean()),
        "auroc": compute_auroc(scores, targets),
        "auprc": compute_auprc(scores, targets),
    }


def compute_auroc(scores, targets):
    pos = scores[targets == 1]
    neg = scores[targets == 0]
    if pos.size == 0 or neg.size == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(order.size, dtype=np.float64)
    ranks[order] = np.arange(1, order.size + 1)
    pos_ranks = ranks[: pos.size]
    auc = (pos_ranks.sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(auc)


def compute_auprc(scores, targets):
    if np.sum(targets == 1) == 0:
        return None
    order = np.argsort(-scores)
    sorted_targets = targets[order]
    tp = np.cumsum(sorted_targets == 1)
    fp = np.cumsum(sorted_targets == 0)
    precision = tp / np.maximum(1, tp + fp)
    recall = tp / max(1, int(np.sum(targets == 1)))
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[precision[0] if precision.size else 1.0], precision])
    return float(np.trapz(precision, recall))


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

FORCE_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


def _even(value):
    value = int(value)
    return value - value % 2


def _resize_if_needed(image, width, height):
    if image.shape[1] == width and image.shape[0] == height:
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _write_ffmpeg_video(output_path, frames, fps, width, height):
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(float(fps)), "-i", "-",
        "-an",
        "-c:v",
        "libx264",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-bf",
        "0",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags", "+faststart", output_path,
    ]
    writer = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    if writer.stdin is None:
        raise RuntimeError(f"Failed to open ffmpeg stdin for {output_path}")
    try:
        for frame in frames:
            writer.stdin.write(np.ascontiguousarray(frame).tobytes())
    finally:
        writer.stdin.close()
        stderr = writer.stderr.read() if writer.stderr is not None else b""
        return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {output_path}: {stderr.decode(errors='replace')}")
    return output_path


def _video_row(frames, output_width, top_height, label):
    row = np.concatenate([frames[view_idx] for view_idx in range(frames.shape[0])], axis=1)
    row = _resize_if_needed(row, output_width, top_height)
    cv2.putText(row, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return row


def build_anomaly_alignment_video(
    output_path,
    multiview_frames,
    pred_multiview_frames=None,
    score=None,
    score_label="score",
    frame_target=None,
    anomaly_frame_prob=None,
    anomaly_window_prob=None,
    cp_curve=None,
    pred_force=None,
    gt_force=None,
    fps=30,
    output_width=1280,
    plot_height=360,
    transition_frame=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = [multiview_frames.shape[0]]
    for value in (
        pred_multiview_frames,
        score,
        frame_target,
        anomaly_frame_prob,
        anomaly_window_prob,
        cp_curve,
        pred_force,
        gt_force,
    ):
        if value is not None:
            lengths.append(value.shape[0])
    length = int(min(lengths))
    if length <= 0:
        raise ValueError("Cannot build anomaly video with zero frames.")

    multiview_frames = multiview_frames[:length]
    pred_multiview_frames = None if pred_multiview_frames is None else pred_multiview_frames[:length]
    score = None if score is None else np.asarray(score[:length], dtype=np.float64)
    frame_target = None if frame_target is None else np.asarray(frame_target[:length], dtype=np.float64)
    anomaly_frame_prob = None if anomaly_frame_prob is None else np.asarray(anomaly_frame_prob[:length], dtype=np.float64)
    anomaly_window_prob = None if anomaly_window_prob is None else np.asarray(anomaly_window_prob[:length], dtype=np.float64)
    cp_curve = None if cp_curve is None else np.asarray(cp_curve[:length], dtype=np.float64)
    pred_force = None if pred_force is None else np.asarray(pred_force[:length], dtype=np.float64)
    gt_force = None if gt_force is None else np.asarray(gt_force[:length], dtype=np.float64)

    h, w = multiview_frames.shape[2], multiview_frames.shape[3]
    output_width = _even(output_width)
    plot_height = _even(plot_height)
    source_top_width = int(multiview_frames.shape[1] * w)
    top_height = _even(max(2, int(round(h * output_width / source_top_width))))
    video_rows = 2 if pred_multiview_frames is not None else 1
    output_height = _even(top_height * video_rows + plot_height)

    fig = plt.figure(figsize=(output_width / 120.0, plot_height / 120.0), dpi=120)
    x = np.arange(length)
    current_lines = []

    if pred_force is not None and gt_force is not None:
        axes = fig.subplots(3, 3, sharex=True).reshape(-1)
        for dim in range(6):
            ax = axes[dim]
            ax.plot(x, pred_force[:, dim], color="tomato", linewidth=1.5, label="pred force")
            ax.plot(x, gt_force[:, dim], color="cornflowerblue", linewidth=1.3, label="gt force")
            if transition_frame is not None and 0 <= transition_frame < length:
                ax.axvline(int(transition_frame), color="seagreen", linestyle="--", linewidth=1.2)
            current_lines.append(ax.axvline(0, color="#444444", linestyle="--", linewidth=1.0))
            ax.set_title(FORCE_LABELS[dim], fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.45)
        ax = axes[6]
        extra_axes = (axes[7], axes[8])
    else:
        ax = fig.subplots(1, 1)
        extra_axes = ()

    if score is not None:
        ax.plot(x, score, color="tomato", linewidth=1.9, label=score_label)
    if cp_curve is not None:
        ax.plot(x, cp_curve, color="black", linestyle="--", linewidth=1.6, label="CP band")
    if frame_target is not None:
        ax.plot(x, frame_target, color="cornflowerblue", linewidth=1.5, drawstyle="steps-post", label="gt anomaly")
    if transition_frame is not None and 0 <= transition_frame < length:
        ax.axvline(int(transition_frame), color="seagreen", linestyle="--", linewidth=1.5, label="failure onset")
    current_lines.append(ax.axvline(0, color="#444444", linestyle="--", linewidth=1.0, label="current frame"))
    ax.set_ylim(-0.05, 1.05) if score is not None and cp_curve is None else None
    ax.set_xlabel("frame")
    ax.set_ylabel("anomaly" if cp_curve is None else "score")
    ax.grid(True, linestyle=":", alpha=0.5)
    if len(extra_axes) == 2:
        extra_specs = (
            (extra_axes[0], anomaly_frame_prob, "dataset frame prob"),
            (extra_axes[1], anomaly_window_prob, "dataset window prob"),
        )
        for extra_ax, values, title in extra_specs:
            if values is None or not np.isfinite(values).any():
                extra_ax.axis("off")
                continue
            extra_ax.plot(x, values, color="darkorange", linewidth=1.7, label=title)
            if frame_target is not None:
                extra_ax.plot(
                    x,
                    frame_target,
                    color="cornflowerblue",
                    linewidth=1.3,
                    drawstyle="steps-post",
                    label="gt anomaly",
                )
            if transition_frame is not None and 0 <= transition_frame < length:
                extra_ax.axvline(int(transition_frame), color="seagreen", linestyle="--", linewidth=1.2)
            current_lines.append(extra_ax.axvline(0, color="#444444", linestyle="--", linewidth=1.0))
            extra_ax.set_title(title, fontsize=9)
            extra_ax.set_ylim(-0.05, 1.05)
            extra_ax.grid(True, linestyle=":", alpha=0.45)
            extra_ax.set_xlabel("frame")
    handles, labels = [], []
    for legend_ax in fig.axes:
        axis_handles, axis_labels = legend_ax.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(5, len(handles)),
            frameon=False,
            fontsize=8,
            bbox_to_anchor=(0.5, 0.005),
        )
        fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    else:
        fig.tight_layout()

    rendered_frames = []
    try:
        for i in range(length):
            top_row = _video_row(multiview_frames[i], output_width, top_height, "GT video")
            if pred_multiview_frames is not None:
                pred_row = _video_row(pred_multiview_frames[i], output_width, top_height, "Pred video")
                top_row = np.concatenate([top_row, pred_row], axis=0)
            for line in current_lines:
                line.set_xdata([x[i], x[i]])
            fig.canvas.draw()
            plot_rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
            plot_rgb = _resize_if_needed(plot_rgb, output_width, plot_height)
            rendered_frames.append(np.concatenate([top_row, plot_rgb], axis=0))
    finally:
        plt.close(fig)
    return _write_ffmpeg_video(output_path, rendered_frames, fps, output_width, output_height)


def build_episode_metrics_video(
    output_path,
    multiview_frames,
    window_x,
    window_prob,
    window_target,
    frame_x,
    frame_prob,
    frame_target,
    threshold=0.5,
    fps=30,
    output_width=1280,
    plot_height=520,
    transition_frame=None,
    interaction_start=None,
    pred_force=None,
    gt_force=None,
    force_plot_height=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    length = min(multiview_frames.shape[0], len(frame_x), len(frame_prob), len(frame_target))
    if length <= 0:
        raise ValueError("Cannot build episode metrics video with zero frames.")
    multiview_frames = multiview_frames[:length]
    frame_x = np.asarray(frame_x[:length], dtype=np.int64)
    frame_prob = np.asarray(frame_prob[:length], dtype=np.float64)
    frame_target = np.asarray(frame_target[:length], dtype=np.float64)
    window_x = np.asarray(window_x, dtype=np.int64)
    window_prob = np.asarray(window_prob, dtype=np.float64)
    window_target = np.asarray(window_target, dtype=np.float64)
    pred_force = None if pred_force is None else np.asarray(pred_force[:length], dtype=np.float64)
    gt_force = None if gt_force is None else np.asarray(gt_force[:length], dtype=np.float64)
    if pred_force is not None and pred_force.ndim == 1:
        pred_force = pred_force[:, None]
    if gt_force is not None and gt_force.ndim == 1:
        gt_force = gt_force[:, None]
    has_gt_force = gt_force is not None and gt_force.ndim == 2 and gt_force.shape[1] >= 3
    has_pred_force = pred_force is not None and pred_force.ndim == 2 and pred_force.shape[1] >= 3
    has_pred_force = bool(has_pred_force and np.isfinite(pred_force[:, :3]).any())
    has_force = bool(has_gt_force)

    h, w = multiview_frames.shape[2], multiview_frames.shape[3]
    output_width = _even(output_width)
    plot_height = _even(plot_height)
    force_plot_height = _even(force_plot_height if force_plot_height is not None else max(220, int(round(plot_height * 0.55))))
    total_plot_height = _even(plot_height + force_plot_height) if has_force else plot_height
    source_top_width = int(multiview_frames.shape[1] * w)
    top_height = _even(max(2, int(round(h * output_width / source_top_width))))
    output_height = _even(top_height + total_plot_height)

    if has_force:
        fig = plt.figure(figsize=(output_width / 120.0, total_plot_height / 120.0), dpi=120)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.85])
        axes = [fig.add_subplot(gs[0, :]), fig.add_subplot(gs[1, :])]
        force_axes = [fig.add_subplot(gs[2, dim], sharex=axes[1]) for dim in range(3)]
    else:
        fig, axes = plt.subplots(2, 1, figsize=(output_width / 120.0, plot_height / 120.0), dpi=120, sharex=True)
        force_axes = []
    axes[0].plot(window_x, window_prob, color="tomato", linewidth=2.0, marker=".", markersize=4)
    axes[0].plot(window_x, window_target, color="cornflowerblue", linewidth=1.8, drawstyle="steps-post")
    axes[0].axhline(float(threshold), color="black", linestyle=":", linewidth=1.4)
    axes[0].set_ylabel("window")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].grid(True, linestyle=":", alpha=0.45)

    axes[1].plot(frame_x, frame_prob, color="tomato", linewidth=2.0)
    axes[1].plot(frame_x, frame_target, color="cornflowerblue", linewidth=1.8, drawstyle="steps-post")
    axes[1].set_ylabel("frame")
    axes[1].set_xlabel("frame")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, linestyle=":", alpha=0.45)

    if has_force:
        for dim, ax in enumerate(force_axes):
            ax.plot(frame_x, gt_force[:, dim], color="cornflowerblue", linewidth=1.5)
            if has_pred_force:
                ax.plot(frame_x, pred_force[:, dim], color="tomato", linewidth=1.4)
            ax.set_title(FORCE_LABELS[dim], fontsize=9)
            ax.grid(True, linestyle=":", alpha=0.45)
            if dim == 0:
                ax.set_ylabel("force")
            ax.set_xlabel("frame")

    if transition_frame is not None and 0 <= int(transition_frame) < length:
        for ax in list(axes) + force_axes:
            ax.axvline(int(transition_frame), color="seagreen", linestyle="--", linewidth=1.5)
    if interaction_start is not None and 0 <= int(interaction_start) < length:
        for ax in list(axes) + force_axes:
            ax.axvline(int(interaction_start), color="purple", linestyle="--", linewidth=1.3)
    all_axes = list(axes) + force_axes
    current_lines = [ax.axvline(0, color="#444444", linestyle="-.", linewidth=1.2) for ax in all_axes]
    legend_handles = [
        Line2D([0], [0], color="cornflowerblue", linewidth=2.0, label="blue: target / GT force"),
        Line2D([0], [0], color="tomato", linewidth=2.0, label="orange-red: pred / prob"),
        Line2D([0], [0], color="purple", linestyle="--", linewidth=1.6, label="interaction"),
        Line2D([0], [0], color="seagreen", linestyle="--", linewidth=1.6, label="failure onset"),
        Line2D([0], [0], color="#444444", linestyle="-.", linewidth=1.6, label="current"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))

    frames = []
    try:
        for i in range(length):
            top_row = _video_row(multiview_frames[i], output_width, top_height, "GT video")
            for line in current_lines:
                line.set_xdata([frame_x[i], frame_x[i]])
            fig.canvas.draw()
            plot_rgb = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3]
            plot_rgb = _resize_if_needed(plot_rgb, output_width, total_plot_height)
            frames.append(np.concatenate([top_row, plot_rgb], axis=0))
    finally:
        plt.close(fig)
    return _write_ffmpeg_video(output_path, frames, fps, output_width, output_height)
