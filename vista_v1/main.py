from pathlib import Path

import torch
from train import train
from model_download import ensure_model_downloaded


def main():
    BASE_DIR = Path(__file__).resolve().parent

    DATA_ROOT = BASE_DIR / "data" / "practice_data"
    MODEL_DIR = BASE_DIR / "original_model"
    SAVE_DIR = BASE_DIR / "checkpoints"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    MODEL_DIR = ensure_model_downloaded(MODEL_DIR, source="auto")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    train(
        data_root=DATA_ROOT,
        model_dir=MODEL_DIR,
        save_dir=SAVE_DIR,
        device=device,
    )

if __name__ == "__main__":
    main()
