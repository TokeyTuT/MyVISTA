import matplotlib.pyplot as plt
import math

from sampler import UniformSampler, RandomSampler, SceneSampler, read_candidates
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

VIDEO_PATH = Path("vista_v2/data/practice_data/videos/train_part1/1-cvpr-2025.mp4")

def visualize_frames(frames,title=None):
    if not frames:
        raise ValueError("没有可显示的视频帧")

    num_rows = math.ceil(len(frames) / 4)
    fig, axes = plt.subplots(
        num_rows, 4,
        figsize=(16, 3 * num_rows),
        squeeze=False,
    )

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    # 包括没有图片的空白格子，也关闭坐标轴
    for ax in axes.flat:
        ax.axis("off")

    for i, (frame, ax) in enumerate(zip(frames, axes.flat)):
        ax.imshow(frame)
        ax.set_title(f"Sample {i + 1}")

    plt.tight_layout()
    plt.show()



def demo():

    num_frames = 8
    timestamps, frames = read_candidates(VIDEO_PATH, interval=1.0)
    uniform_sampler = UniformSampler(num_frames=num_frames)
    random_sampler = RandomSampler(num_frames=num_frames, random_seed=42)
    scene_sampler = SceneSampler(num_frames=num_frames, threshold=0.08)

    uniform_frames, uniform_timestamps = uniform_sampler.sample(frames, timestamps)
    visualize_frames(uniform_frames,'均匀采样')

    random_frames, random_timestamps = random_sampler.sample(frames, timestamps)
    visualize_frames(random_frames,'随机采样')

    scene_frames, scene_timestamps = scene_sampler.sample(frames, timestamps)
    visualize_frames(scene_frames,'按照图片变化程度采样')


if __name__ == '__main__':
    demo()



    