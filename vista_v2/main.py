import os
import sampler

from pathlib import Path
from openai import OpenAI
from getpass import getpass
from reasoning import reasoning


RANDOM_SEED = 42

def main():

    BASE_DIR = Path(__file__).resolve().parent
    DATA_ROOT = BASE_DIR / "data" / "practice_data"

    WORKSPACE = "ws-j18cah4tl98w954f"

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY") or getpass("请输入阿里百炼 API key:"),
        base_url=f"https://{WORKSPACE}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        timeout=120.0
    )
    model = "qwen3-vl-plus"

    num_frames = 32
    interval = 1.5
    max_workers = 3  # 同时处理的视频数；设为 1 即串行

    uniform_sampler = sampler.UniformSampler(num_frames)
    random_sampler = sampler.RandomSampler(num_frames,random_seed=RANDOM_SEED)
    scene_sampler = sampler.SceneSampler(num_frames)

    random_sampler_one = sampler.RandomSampler(1,random_seed=RANDOM_SEED)

    reasoning(
        DATA_ROOT,
        interval,
        scene_sampler,
        # uniform_sampler,
        # random_sampler,
        # random_sampler_one,
        model,
        client,
        max_workers=max_workers,
    )
if __name__ == '__main__':
    main()
