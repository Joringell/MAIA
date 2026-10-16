import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.lerobot_chunk_dataset import CustomLeRobotChunkDataset
from utils import zero_rank_print


class CustomLeRobotChunkTraverseDataset(CustomLeRobotChunkDataset):
    """Deterministic window traversal for official LeRobot datasets.

    This keeps the GE/video-pretrain memory semantics:
      memory_steps sparse frames sampled from memory_span_steps,
      future_steps continuous action-clock rows,
      video_stride for future video frames.

    If force_key is provided, each future force target uses the last force
    substep of its action-clock row. ``force_mode`` only controls aggregation
    of ``history_force_state``.
    """

    def __init__(
        self,
        *args,
        window_stride=1,
        samples_per_episode=None,
        include_tail_windows=True,
        start_at_zero=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if samples_per_episode not in (None, 1):
            zero_rank_print(
                f"[CustomLeRobotChunkTraverseDataset] ignore samples_per_episode={samples_per_episode}; "
                "traverse dataset expands all valid windows instead."
            )

        self.window_stride = max(1, int(window_stride))
        self.include_tail_windows = bool(include_tail_windows)
        self.start_at_zero = bool(start_at_zero)
        self.traverse_index = []
        self.short_episode_count = 0
        self._build_traverse_index()

    def _enumerate_chunk_ends(self, total_frames):
        total_frames = int(total_frames)
        if self.include_tail_windows:
            first_start = 0 if self.start_at_zero else 1
            if total_frames <= first_start:
                self.short_episode_count += 1
                return [first_start + self.future_steps]
            starts = list(range(first_start, total_frames, self.window_stride))
            return [start + self.future_steps for start in starts]

        max_start = total_frames - self.future_steps
        if max_start >= 1:
            starts = list(range(1, max_start + 1, self.window_stride))
            if starts[-1] != max_start:
                starts.append(max_start)
            return [start + self.future_steps for start in starts]

        self.short_episode_count += 1
        return [max(self.future_steps + 1, total_frames)]

    def _build_traverse_index(self):
        self.traverse_index = []
        self.short_episode_count = 0
        for episode_idx, item in enumerate(self.dataset):
            total_frames = int(item[7])
            for chunk_end in self._enumerate_chunk_ends(total_frames):
                self.traverse_index.append((episode_idx, int(chunk_end)))

        self.length = len(self.traverse_index)
        zero_rank_print(
            f"official traverse data scale: base_episodes={self.base_length}, "
            f"window_stride={self.window_stride}, effective_windows={self.length}, "
            f"include_tail_windows={self.include_tail_windows}, start_at_zero={self.start_at_zero}, "
            f"short_episodes={self.short_episode_count}"
        )

    def _sample_memory_indexes(self, mem_candidates):
        if len(mem_candidates) < self.memory_steps - 1:
            mem_candidates = [1] * (self.memory_steps - 1) + mem_candidates

        if self.memory_pick_mode == "uniform":
            return [
                mem_candidates[int(i)] for i in np.linspace(0, len(mem_candidates) - 1, self.memory_steps).tolist()
            ]
        if self.memory_pick_mode == "random":
            return [
                mem_candidates[i]
                for i in sorted(
                    np.random.choice(
                        list(range(0, len(mem_candidates) - 1)),
                        size=self.memory_steps - 1,
                        replace=False,
                    ).tolist()
                )
            ] + [mem_candidates[-1]]
        raise NotImplementedError(f"unsupported memory_pick_mode: {self.memory_pick_mode}")

    def _get_frame_indexes_from_window_start(self, total_frames, window_start):
        total_frames = int(total_frames)
        max_frame = max(0, total_frames - 1)
        window_start = int(window_start)
        future_indexes = list(range(window_start, window_start + self.future_steps))
        frame_future = future_indexes[self.video_stride - 1 :: self.video_stride]
        future_indexes = np.clip(future_indexes, a_min=0, a_max=max_frame).tolist()
        frame_future = np.clip(frame_future, a_min=0, a_max=max_frame).tolist()

        memory_span = max(self.memory_steps, self.memory_span_steps - self.future_steps)
        indexes_start = max(-self.memory_steps, window_start - memory_span)
        mem_candidates = np.array(list(range(indexes_start, window_start)))
        mem_candidates = np.clip(mem_candidates, a_min=1, a_max=max(1, max_frame)).tolist()
        mem_indexes = self._sample_memory_indexes(mem_candidates)

        if not self.ignore_seek:
            frame_indexes = mem_indexes + frame_future
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]
        return frame_indexes, mem_indexes + future_indexes

    def _get_frame_indexes_from_chunk_end(self, total_frames, chunk_end):
        if self.fix_sidx is not None and self.fix_mem_idx is not None:
            action_indexes = list(range(self.fix_sidx, self.fix_sidx + self.future_steps))
            frame_indexes = action_indexes[self.video_stride - 1 :: self.video_stride]
            action_indexes = np.clip(action_indexes, a_min=0, a_max=total_frames - 1).tolist()
            frame_indexes = np.clip(frame_indexes, a_min=0, a_max=total_frames - 1).tolist()
            mem_indexes = np.clip(self.fix_mem_idx, a_min=0, a_max=total_frames - 1).tolist()
            return mem_indexes + frame_indexes, mem_indexes + action_indexes

        raw_chunk_end = int(chunk_end)
        raw_start = raw_chunk_end - self.future_steps
        if self.include_tail_windows and raw_start >= 0:
            return self._get_frame_indexes_from_window_start(total_frames, raw_start)

        min_chunk_end = self.future_steps + 1
        max_chunk_end = max(min_chunk_end, int(total_frames))
        chunk_end = max(min_chunk_end, min(raw_chunk_end, max_chunk_end))

        indexes_start = max(-self.memory_steps, chunk_end - self.memory_span_steps)
        indexes = np.array(list(range(indexes_start, chunk_end)))
        indexes = np.clip(indexes, a_min=1, a_max=max(1, total_frames - 1)).tolist()
        video_end = indexes[-self.future_steps :]
        mem_candidates = indexes[: -self.future_steps]
        mem_indexes = self._sample_memory_indexes(mem_candidates)

        if not self.ignore_seek:
            frame_indexes = mem_indexes + video_end[self.video_stride - 1 :: self.video_stride]
        else:
            frame_indexes = mem_indexes + mem_indexes[-1:]

        action_indexes = mem_indexes + video_end
        return frame_indexes, action_indexes

    def _resolve_episode_and_chunk_end(self, idx):
        if self.fix_epiidx is not None:
            episode_idx = int(self.fix_epiidx)
            if episode_idx < 0 or episode_idx >= self.base_length:
                raise IndexError(f"fix_epiidx={episode_idx} out of range for dataset size {self.base_length}")
            total_frames = int(self.dataset[episode_idx][7])
            chunk_end = int(self.fix_sidx) + self.future_steps if self.fix_sidx is not None else self._enumerate_chunk_ends(total_frames)[0]
            return episode_idx, chunk_end

        if idx < 0 or idx >= self.length:
            raise IndexError(f"idx={idx} out of range for traverse dataset size {self.length}")
        return self.traverse_index[idx]

    def get_batch(self, idx):
        episode_idx, chunk_end = self._resolve_episode_and_chunk_end(idx)

        parquet_path = self.dataset[episode_idx][2]
        domain_name = self.dataset[episode_idx][3]
        caption = self.dataset[episode_idx][6]
        total_frames = self.dataset[episode_idx][7]
        cam_segments = self.dataset[episode_idx][9]

        sample_size, specific_transforms_resize, specific_transforms_norm = self.get_transform()
        vid_indexes, indexes = self._get_frame_indexes_from_chunk_end(total_frames, chunk_end)

        data = self._load_data(parquet_path)
        future_indexes = indexes[-self.future_steps :]
        window_start = int(chunk_end) - int(self.future_steps)
        valid_len = min(int(self.future_steps), max(0, int(total_frames) - window_start))
        valid_future_indexes = future_indexes[:valid_len]

        label = torch.tensor(1, dtype=torch.long)
        risk_label = torch.tensor(0.0, dtype=torch.float32)
        interaction_label = torch.tensor(0, dtype=torch.long)
        frame_anomaly_label = torch.full((self.future_steps,), -1.0, dtype=torch.float32)
        if "episode_label" not in data.columns:
            raise KeyError("episode_label is required in parquet for episode-level normal filtering")
        episode_label = torch.tensor(int(np.asarray(data["episode_label"].iloc[0]).reshape(-1)[0]), dtype=torch.long)
        if "frame_label" in data.columns and valid_len > 0:
            future_labels = data["frame_label"].iloc[valid_future_indexes].to_numpy()
            label = torch.tensor(int(future_labels.min()), dtype=torch.long)
            risk_label = torch.tensor(float(np.mean(future_labels == 0)), dtype=torch.float32)
            frame_anomaly_label[:valid_len] = torch.FloatTensor((future_labels == 0).astype(np.float32))
        if "interaction_label" in data.columns and valid_len > 0:
            future_interaction_labels = data["interaction_label"].iloc[valid_future_indexes].to_numpy()
            interaction_label = torch.tensor(int(future_interaction_labels.max()), dtype=torch.long)
        anomaly_extra_curves = {}
        for column in ("anomaly_frame_prob", "anomaly_window_prob"):
            if column in data.columns and valid_len > 0:
                values = data[column].iloc[valid_future_indexes].to_numpy(dtype=np.float32)
                curve = torch.full((self.future_steps,), float("nan"), dtype=torch.float32)
                curve[:valid_len] = torch.from_numpy(values)
                anomaly_extra_curves[column] = curve

        action_mean, action_std = self.get_action_bias_std(delta=False, domain_name=domain_name)
        state_mean, state_std = self.get_state_bias_std(domain_name=domain_name)

        try:
            action = self._load_feature_array(data, self.action_key).astype(np.float32)
            state = self.load_state_array(data).astype(np.float32)
        except Exception as exc:
            raise ValueError("We currently only support action/state data with shape T*C.") from exc

        if self.action_dim is not None:
            action = action[:, : self.action_dim]

        state = torch.FloatTensor(state)[indexes][self.memory_steps - 1 : self.memory_steps]
        state = (state - state_mean) / state_std

        if self.action_type == "absolute":
            action = torch.FloatTensor(action[indexes].astype(np.float32))
            if self.action_dim is not None:
                action_mean = action_mean[:, : self.action_dim]
                action_std = action_std[:, : self.action_dim]
            action = (action - action_mean) / action_std
        elif self.action_type == "delta":
            delta_mean, delta_std = self.get_action_bias_std(delta=True, domain_name=domain_name)
            if self.action_dim is not None:
                delta_mean = delta_mean[:, : self.action_dim]
                delta_std = delta_std[:, : self.action_dim]
            action_curr = torch.FloatTensor(action[indexes].astype(np.float32))
            action_last = torch.FloatTensor(action[[max(0, i - 1) for i in indexes]].astype(np.float32))
            action = action_curr - action_last
            for eff_idx in (6, 13, 7, 15):
                if action.shape[1] > eff_idx:
                    action[:, eff_idx] = action_last[:, eff_idx]
            action = (action - delta_mean) / delta_std
        elif self.action_type == "relative":
            action = torch.FloatTensor(action[indexes].astype(np.float32))
            if self.action_dim is not None:
                action_mean = action_mean[:, : self.action_dim]
                action_std = action_std[:, : self.action_dim]
            action = (action - action_mean) / action_std
            action = action - state[..., : action.shape[-1]]
        else:
            raise NotImplementedError(f"unsupported action_type: {self.action_type}")

        force_targets = None
        history_force_state = None
        if self.force_keys is not None:
            force_source = self.load_force_array(data)
            current_index = int(indexes[self.memory_steps - 1])
            force_targets = self.force_targets(
                force_source,
                future_indexes,
                current_index=current_index,
                mode="last",
                domain_name=domain_name,
            )
            if self.force_history_steps > 0:
                if self.force_type != "absolute":
                    raise ValueError(
                        "force_history_steps currently supports force_type='absolute' only"
                    )
                future_start = int(future_indexes[0])
                history_indexes = np.arange(
                    future_start - self.force_history_steps,
                    future_start,
                )
                history_indexes = np.clip(
                    history_indexes,
                    a_min=0,
                    a_max=force_source.shape[0] - 1,
                ).tolist()
                history_force_state = self.force_window(
                    force_source,
                    history_indexes,
                    mode=self.force_mode,
                )
                history_force_state = self.normalize_force(
                    history_force_state,
                    relative=False,
                    domain_name=domain_name,
                )

        videos = self.seek_mp4(cam_segments, self.valid_cam, vid_indexes)
        videos, _ = self.transform_video(videos, specific_transforms_resize, None, sample_size)
        videos = self.normalize_video(videos, specific_transforms_norm)

        sample = {
            "video": videos,
            "actions": action,
            "caption": caption,
            "state": state,
            "label": label,
            "episode_label": episode_label,
            "risk_label": risk_label,
            "frame_anomaly_label": frame_anomaly_label,
            "interaction_label": interaction_label,
        }
        sample.update(anomaly_extra_curves)
        if force_targets is not None:
            sample["force_targets"] = force_targets
        if history_force_state is not None:
            sample["history_force_state"] = history_force_state
        return sample

    def __getitem__(self, idx):
        if self.length == 0:
            raise IndexError("Empty traverse dataset")
        return self.get_batch(idx)
