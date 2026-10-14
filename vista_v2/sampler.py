from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image


class VideoSampler(ABC):
    """仅负责选择已有帧，返回 (PIL 图片列表, 秒数时间戳列表)。"""

    def __init__(self, num_frames=32):
        if isinstance(num_frames, bool) or not isinstance(num_frames, (int, np.integer)) or num_frames <= 0:
            raise ValueError("num_frames 必须是正整数")
        self.num_frames = int(num_frames)

    @abstractmethod
    def sample(self, frames, timestamps):
        """输入按时间排列的 RGB 数组或 PIL 图片及其秒数时间戳。"""
        raise NotImplementedError


class UniformSampler(VideoSampler):
    def sample(self, frames, timestamps):
        timestamps = _validate_candidates(frames, timestamps)
        indices = np.linspace(
            0, len(frames) - 1, min(len(frames), self.num_frames), dtype=int
        )
        return _select(frames, timestamps, indices)


class RandomSampler(VideoSampler):
    def __init__(self, num_frames=32, random_seed=None):
        super().__init__(num_frames)
        self.random_seed = random_seed

    def sample(self, frames, timestamps):
        timestamps = _validate_candidates(frames, timestamps)
        rng = np.random.default_rng(self.random_seed)
        indices = np.sort(rng.choice(
            len(frames), size=min(len(frames), self.num_frames), replace=False
        ))
        return _select(frames, timestamps, indices)


class SceneSampler(VideoSampler):
    def __init__(self, num_frames=32, threshold=0.08,
                 min_scene_seconds=3.0):
        super().__init__(num_frames)
        if not np.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold 必须为非负有限数")
        if not np.isfinite(min_scene_seconds) or min_scene_seconds < 0:
            raise ValueError("min_scene_seconds 必须为非负有限数")
        self.threshold = threshold
        self.min_scene_seconds = min_scene_seconds

    def sample(self, frames, timestamps):
        timestamps = _validate_candidates(frames, timestamps)
        rgb_frames = [
            np.asarray(frame.convert("RGB")) if isinstance(frame, Image.Image)
            else frame for frame in frames
        ]
        changes = self._calculate_changes(rgb_frames)
        boundaries = self._detect_boundaries(
            timestamps, changes
        )
        boundaries, counts = self._allocate_frames(
            timestamps, changes, boundaries
        )
        indices = self._select_from_segments(boundaries, counts)
        return _select(frames, timestamps, indices)

    def _prepare_frame(self, frame):
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 转成浮点数，避免 uint8 相减溢出
        return blurred.astype(np.float32) / 255.0

    def _calculate_changes(self, frames):
        features = [self._prepare_frame(frame) for frame in frames]
        changes = np.zeros(len(features))

        for i in range(1, len(features)):
            changes[i] = np.abs(features[i] - features[i - 1]).mean()

        return changes

    def _detect_boundaries(self, timestamps, changes):
        boundaries = [0]
        for i in range(1, len(timestamps)):
            elapsed = timestamps[i] - timestamps[boundaries[-1]]
            if changes[i] > self.threshold and elapsed >= self.min_scene_seconds:
                boundaries.append(i)
        boundaries.append(len(timestamps))
        return boundaries

    def _allocate_frames(self, timestamps, changes, boundaries):
        """返回调整后的边界和各片段的帧数配额。"""
        n = len(timestamps)
        # 总候选不足时，返回全部，不重复帧
        target = min(self.num_frames, n)
        boundaries = list(boundaries)

        # 情况一：片段过多，删除较弱的内部边界
        if len(boundaries) - 1 > target:
            internal = boundaries[1:-1]

            # 差异降序；差异相同时，较早的边界优先
            strongest = sorted(
                internal,
                key=lambda i: (-float(changes[i]), i),
            )[:target - 1]

            boundaries = [0, *sorted(strongest), n]

        starts = np.array(boundaries[:-1], dtype=int)
        ends = np.array(boundaries[1:], dtype=int)

        # 每段拥有多少张候选帧
        capacities = ends - starts

        # 末段没有下一个边界时间，用典型候选间隔估计末端
        interval = float(np.median(np.diff(timestamps))) if n > 1 else 1.0
        end_times = np.array([
            timestamps[end] if end < n else timestamps[-1] + interval
            for end in ends
        ])
        durations = end_times - timestamps[starts]

        # 每段至少一帧
        counts = np.ones(len(starts), dtype=int)
        remaining = target - int(counts.sum())

        # 情况二：还有额度，补给当前时间覆盖最稀疏的片段
        for _ in range(remaining):
            scores = durations / counts

            # 候选帧已经用完的片段不能继续分配
            scores[counts >= capacities] = -np.inf

            chosen = int(np.argmax(scores))
            counts[chosen] += 1

        return boundaries, counts

    def _select_from_segments(self, boundaries, counts):
        """按各段配额取候选帧索引，结果有序且不重复。"""
        selected = []

        for start, end, count in zip(
            boundaries[:-1], boundaries[1:], counts
        ):
            size = end - start

            # 把本段分为 count 个小区间，取各区间中心
            offsets = (
                (np.arange(count) + 0.5) * size / count
            ).astype(int)

            selected.extend((start + offsets).tolist())

        return np.array(selected, dtype=int)


def read_candidates(video_path, interval=1.0):
    """每隔 interval 秒读取一帧，返回时间戳和 RGB 图像。"""
    if not np.isfinite(interval) or interval <= 0:
        raise ValueError("interval 必须大于 0")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise IOError(f"无法打开视频：{video_path}")

    timestamps = []
    frames = []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if not np.isfinite(fps) or not np.isfinite(total_frames) or fps <= 0 or total_frames <= 0:
            raise ValueError("无法获取有效的视频时长")

        duration = total_frames / fps

        for t in np.arange(0, duration, interval):
            cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000))
            success, frame = cap.read()

            if not success:
                continue

            timestamps.append(float(t))
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    finally:
        cap.release()

    if not frames:
        raise ValueError("未读取到有效帧")


    return np.array(timestamps), frames


def _validate_candidates(frames, timestamps):
    timestamps = np.asarray(timestamps, dtype=float)
    if len(frames) == 0:
        raise ValueError("候选帧不能为空")
    if timestamps.ndim != 1 or len(frames) != len(timestamps):
        raise ValueError("frames 与 timestamps 必须一一对应")
    if not np.all(np.isfinite(timestamps)) or np.any(timestamps < 0):
        raise ValueError("时间戳必须是非负有限数")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("候选帧须按时间排列，时间戳必须严格递增")
    return timestamps


def _select(frames, timestamps, indices):
    selected = [
        frames[i].convert("RGB") if isinstance(frames[i], Image.Image)
        else Image.fromarray(frames[i])
        for i in indices
    ]
    return selected, timestamps[indices].tolist()
