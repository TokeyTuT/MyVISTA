import torch
import gc
from torch.utils.data import DataLoader

from data_utils import get_frames,load_vista_dataset
from pathlib import Path
from pg_model import PlanGenerator
from sg_model import SummaryGenerator
from lora import save_lora


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
        foreach=False,
    )

    for epoch in range(epochs):
        for step,batch in enumerate(loader):
            # 当前 batch_size 暂时为 1
            sample = batch[0]

            frames = get_frames(
                data_root / sample['video_path'],
                num_frames
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(frames,sample['plan'])

            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("loss 非有限值，停止训练以避免保存损坏参数")
            loss.backward()
            # 更新前检查并裁剪梯度；发现 NaN/Inf 时直接报错。
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=1.0, error_if_nonfinite=True,
            )

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
        foreach=False,
    )

    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            # 当前 batch_size 暂时为 1
            sample = batch[0]

            frames = get_frames(
                data_root / sample['video_path'],
                num_frames
            )

            optimizer.zero_grad(set_to_none=True)
            outputs = model(frames, sample['plan'],sample['abstract'])

            loss = outputs.loss
            if not torch.isfinite(loss):
                raise RuntimeError("loss 非有限值，停止训练以避免保存损坏参数")
            loss.backward()
            # 更新前检查并裁剪梯度；发现 NaN/Inf 时直接报错。
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=1.0, error_if_nonfinite=True,
            )

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
    save_lora(pg, save_dir / "pg_lora.pt")
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

    save_lora(sg, save_dir / "sg_lora.pt")

    print(f"训练完成，模型参数保存在: {save_dir}")


















