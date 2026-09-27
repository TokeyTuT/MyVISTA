"""程序入口：组装数据与模型，后续在这里接入训练流程。"""

from pathlib import Path

from data_utils import load_vista_dataset, get_frames
from pg_model import PlanGenerator
from sg_model import SummaryGenerator


def main():
    data_root = Path(__file__).resolve().parent / "data/practice_data"

    loader = load_vista_dataset(
        data_root,
        batch_size=1,
    )


if __name__ == "__main__":
    main()
