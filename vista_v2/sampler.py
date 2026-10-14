import numpy as np
import cv2

from PIL import Image


class VideoSampler:
    """统一的视频采样入口，返回 (PIL 图像列表, 秒数时间戳列表)。

    示例：
        sampler = VideoSampler("scene", threshold=0.08)
        frames, timestamps = sampler(path, num_frames=16)

    自定义方法须接受 (path, num_frames, **kwargs)，并返回相同格式。
    图片和时间戳应一一对应、按时间递增，候选不足时可少于 num_frames。
    """

    def __init__(self, method="uniform", **kwargs):
        self._methods = {
            "uniform": self.uniform_sample,
            "random": self.random_sample,
            "scene": self.scene_sample,
        }
        self._validate_method(method)
        self.method = method
        self.options = dict(kwargs)

    def _validate_method(self, method):
        if method not in self._methods:
            raise ValueError(
                f"未知采样方法：{method}，可用方法：{', '.join(self._methods)}"
            )

    def register(self, name, function):
        """为当前实例注册新方法；不覆盖已有方法。"""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("方法名称必须是非空字符串")
        if name in self._methods:
            raise ValueError(f"采样方法已存在：{name}")
        if not callable(function):
            raise TypeError("采样方法必须可调用")
        self._methods[name] = function

    def sample(self, path, num_frames=32, *, method=None, **kwargs):
        """调用时可覆盖默认参数；切换方法时不继承原方法的专用参数。"""
        selected = self.method if method is None else method
        self._validate_method(selected)
        options = self.options.copy() if selected == self.method else {}
        options.update(kwargs)
        return self._methods[selected](path, num_frames=num_frames, **options)

    def __call__(self, path, num_frames=32, **kwargs):
        return self.sample(path, num_frames=num_frames, **kwargs)

    @staticmethod
    def uniform_sample(
            path,
            num_frames = 32,
    ):
        """均匀采样，返回 (PIL 图像列表, 秒数时间戳列表)，一一对应。"""
        if not isinstance(num_frames, (int, np.integer)) or num_frames <= 0:
            raise ValueError("num_frames 必须是正整数")

        cap = cv2.VideoCapture(str(path))

        if not cap.isOpened():
            raise FileNotFoundError(path)


        frames = []
        timestamps = []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("无法获取有效帧率，不能计算时间戳")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("视频没有有效帧")

            frames_indices = np.linspace(
                0, total_frames - 1, min(total_frames,num_frames), dtype=int
            )

            for index in frames_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

                success, frame = cap.read()
                if not success:
                    raise IOError(f'Cannot read frame {index} in  {path}')

                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps.append(float(index) / fps)


        finally:
            cap.release()

        return [Image.fromarray(frame) for frame in frames], timestamps

    @staticmethod
    def random_sample(
        path,
        num_frames = 32,
        random_seed = None
    ):
        """随机采样，返回 (PIL 图像列表, 秒数时间戳列表)，按时间排序且不重复；候选不足时返回全部帧。"""
        if not isinstance(num_frames, (int, np.integer)) or num_frames <= 0:
            raise ValueError("num_frames 必须是正整数")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(path)

        frames = []
        timestamps = []
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError("无法获取有效帧率，不能计算时间戳")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                raise ValueError("视频没有有效帧")

            rng = np.random.default_rng(random_seed)
            frames_indices = np.sort(rng.choice(
                total_frames,
                size=min(num_frames, total_frames),
                replace=False,
            ))

            for index in frames_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

                success, frame = cap.read()
                if not success:
                    raise IOError(f'Cannot read frame {index} in  {path}')
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                timestamps.append(float(index) / fps)

        finally:
            cap.release()

        return [Image.fromarray(frame) for frame in frames], timestamps

    @classmethod
    def read_candidates(cls, video_path, interval=1.0):
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


        # visualize_frames(frames) # debug
        return np.array(timestamps), frames

    @classmethod
    def prepare_frame(cls, frame):
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 转成浮点数，避免 uint8 相减溢出
        return blurred.astype(np.float32) / 255.0

    @classmethod
    def calculate_changes(cls, frames):
        features = [cls.prepare_frame(frame) for frame in frames]
        changes = np.zeros(len(features))

        for i in range(1, len(features)):
            changes[i] = np.abs(features[i] - features[i - 1]).mean()

        return changes

    @classmethod
    def detect_boundaries(cls, timestamps, changes,
                          threshold=0.08, min_scene_seconds=3.0):
        """返回片段边界，包含起点 0 和终点 len(timestamps)。"""
        boundaries = [0]

        for i in range(1, len(timestamps)):
            elapsed = timestamps[i] - timestamps[boundaries[-1]]

            if changes[i] > threshold and elapsed >= min_scene_seconds:
                boundaries.append(i)

        boundaries.append(len(timestamps))
        return boundaries

    @classmethod
    def allocate_frames(cls, timestamps, changes, boundaries, k):
        """返回调整后的边界和各片段的帧数配额。"""
        if k <= 0:
            raise ValueError("k 必须大于 0")

        n = len(timestamps)
        if n == 0:
            raise ValueError("候选帧不能为空")

        # 总候选不足时，返回全部，不重复帧
        target = min(k, n)
        boundaries = list(boundaries)


        if len(boundaries) - 1 > target:
            internal = boundaries[1:-1]
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

    @classmethod
    def select_from_segments(cls, boundaries, counts):
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

    @classmethod
    def scene_sample(
        cls,
        path,
        num_frames,
        threshold=0.08,
        interval=1.0,
        min_scene_seconds=3.0,
    ):
        """
        返回: (PIL 图像列表, 秒数时间戳列表)，时间戳为候选采样位置。
        path: 视频路径
        num_frames: 目标帧数；候选不足时返回全部候选
        threshold: 相邻候选帧的画面变化阈值
        interval: 候选帧的采样时间间隔，单位秒
        min_scene_seconds: 检测分段时，相邻分段起点的最小间隔
        """
        if not isinstance(num_frames, (int, np.integer)) or num_frames <= 0:
            raise ValueError("num_frames 必须是正整数")

        # 按前面 read_candidates 的返回顺序
        timestamps, frames = cls.read_candidates(path, interval)

        # 内部已经逐张进行缩放、灰度化等处理；
        # 原始 RGB frames 不会被替换
        changes = cls.calculate_changes(frames)

        timestamps = np.asarray(timestamps, dtype=float)
        changes = np.asarray(changes, dtype=float)

        original_boundaries = cls.detect_boundaries(
            timestamps,
            changes,
            threshold=threshold,
            min_scene_seconds=min_scene_seconds,
        )

        final_boundaries, counts = cls.allocate_frames(
            timestamps,
            changes,
            original_boundaries,
            num_frames,
        )

        indices = cls.select_from_segments(final_boundaries, counts)

        final_frames = []
        final_timestamps = []
        for i in indices:
            final_frames.append(frames[i])
            final_timestamps.append(timestamps[i])

        # visualize_frames(final_frames)

        return [Image.fromarray(frame) for frame in final_frames],final_timestamps


