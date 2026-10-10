"""视频抽帧、标注读取和 DataLoader 组批。"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset
from torch.utils.data import DataLoader

from teacher_plans import generate_plan_by_teacher


# def visualize_frames(frames):
#     if not frames:
#         raise ValueError("没有可显示的视频帧")
#
#     num_rows = math.ceil(len(frames) / 4)
#
#     fig, axes = plt.subplots(
#         num_rows, 4,
#         figsize=(16, 3 * num_rows),
#         squeeze=False,
#     )
#
#     # 包括没有图片的空白格子，也关闭坐标轴
#     for ax in axes.flat:
#         ax.axis("off")
#
#     for i, (frame, ax) in enumerate(zip(frames, axes.flat)):
#         ax.imshow(frame)
#         ax.set_title(f"Sample {i + 1}")
#
#     plt.tight_layout()
#     plt.show()


def get_frames(
        video_path:Path,
        num_frames:int
):
    """均匀抽帧，返回按时间排列的 RGB PIL 图像列表。"""
    if num_frames <= 0:
        raise ValueError("num_frames 必须大于 0")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"视频没有有效帧：{video_path}")

        frames_indices = np.linspace(
            0, total_frames - 1, min(total_frames,num_frames), dtype=int
        )

        for index in frames_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

            success, frame = cap.read()
            if not success:
                raise IOError(f'Cannot read frame {index} in  {video_path}')

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    finally:
        cap.release()

    # visualize_frames(frames)

    return [Image.fromarray(frame) for frame in frames]


def collate_samples(samples):
    """保留独立样本，避免默认堆叠长度不同的 plan。"""
    return samples


def load_vista_dataset(
        data_root,
        batch_size,
):
    """导入并处理数据集"""
    data_root = Path(data_root)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_root / "train_part1.json"),
        },
        split="train",
    )

    records = generate_plan_by_teacher(dataset, data_root / 'plan.json')
    dataset = dataset.add_column(
        "plan",
        [records[str(sample["id"])]["plan"] for sample in dataset],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 调试阶段方便检查顺序
        num_workers=0,
        collate_fn=collate_samples,
    )
