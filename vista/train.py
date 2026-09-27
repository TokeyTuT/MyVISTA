import torch
import gc
from torch.utils.data import DataLoader

from data_utils import get_frames,load_vista_dataset
from pathlib import Path
from pg_model import PlanGenerator
from sg_model import SummaryGenerator


def train_pg(
        model,
        loader,
        epochs,
        learning_rate,
        num_frames,
        data_root,
):

    print("PG Model 训练中")
    model.train()
    # 只把允许训练的参数交给 optimizer
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        for step,batch in enumerate(loader):
            # 当前 batch_size 暂时为 1
            sample = batch[0]

            frames = get_frames(
                data_root / sample['video_path'],
                num_frames
            )

            optimizer.zero_grad()
            outputs = model(frames,sample['plan'])

            loss = outputs.loss
            loss.backward()

            optimizer.step()

            print(
                f"PG | epoch {epoch + 1}/{epochs} "
                f"| step {step + 1}/{len(loader)} "
                f"| loss {loss.item():.4f}"
            )

def train_sg(
        model,
        loader,
        epochs,
        learning_rate,
        num_frames,
        data_root,
):
    print("SG Model 训练中")
    model.train()
    # 只把允许训练的参数交给 optimizer
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate,
    )

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            # 当前 batch_size 暂时为 1
            sample = batch[0]

            frames = get_frames(
                data_root / sample['video_path'],
                num_frames
            )

            optimizer.zero_grad()
            outputs = model(frames, sample['plan'],sample['abstract'])

            loss = outputs.loss
            loss.backward()

            optimizer.step()

            print(
                f"SG | epoch {epoch + 1}/{epochs} "
                f"| step {step + 1}/{len(loader)} "
                f"| loss {loss.item():.4f}"
            )


def train(
        data_root,
        model_dir,
        save_dir,
        device,
):
    data_root = Path(data_root)
    device = torch.device(device)
    print(f"training on {device}")

    loader = load_vista_dataset(
        data_root,
        batch_size=1,
    )


    num_frames = 10
    epochs = 5

    learning_rate = 1e-5

    # 第一阶段
    pg = PlanGenerator(
        model_dir,
        device=device,
    )

    train_pg(
        model=pg,
        loader=loader,
        epochs=epochs,
        learning_rate=learning_rate,
        num_frames=num_frames,
        data_root=data_root,
    )

    # train_pg 返回后，其局部 optimizer 已不再保留
    torch.save(pg.state_dict(), save_dir / "pg.pt")
    del pg
    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # 第二阶段
    sg = SummaryGenerator(
        model_dir,
        device
    )

    train_sg(
        model=sg,
        loader=loader,
        epochs=epochs,
        learning_rate=learning_rate,
        num_frames=num_frames,
        data_root=data_root,
    )

    torch.save(sg.state_dict(), save_dir / "sg.pt")

    print(f"训练完成，模型参数保存在: {save_dir}")


















