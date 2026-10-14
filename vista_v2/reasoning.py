from pandas import Timestamp
import torch 
from pathlib import Path
from datasets import load_dataset
from preprocessingImage import image_to_data_url
from sampler import VideoSampler,read_candidates
from generator import generate_plan,generate_summary
from evaluate import evaluate_rouge

def reasoning(
        data_root,
        intervals,
        sampler:VideoSampler,
        model,
        client,
        device,
        max_tokens=1024
        ):

    data_root = Path(data_root)
    device = torch.device(device)

    dataset = load_dataset(
            "json",
            data_files={
                "train": str(data_root / "train_part1.json"),
            },
            split="train",
        )

    r1 = 0.0 # "rouge1",
    r2 = 0.0 # "rouge2"
    rlsum = 0.0 # "rougeLsum"

    evaluations = []
    for step,data in dataset:
        video_path = data_root / data['video_path']

        timestamps,frames = read_candidates(video_path,intervals)
        selected_frames,selected_timestamps = sampler(frames,timestamps)

        image_urls = [image_to_data_url(image) for image in selected_frames]
        plan = generate_plan(
            image_urls,model,client,max_tokens,timestamps=selected_timestamps
            )

        summary = generate_summary(
            image_urls,plan,model,client,max_tokens,timestamps=selected_timestamps
        )

        score = evaluate_rouge(summary,data['abstract'])

        print(f"第 {step + 1} 推理 R1：{score['rouge1']}\tR2：{score['rouge2']}\tRLSum：{score['rougeLsum']}")
        evaluations.append(score)

        r1 += score['rouge1']
        r2 += score['rouge2']
        rlsum += score['rougeLsum']

    
    r1 /= len(evaluations)
    r2 /= len(evaluations)
    rlsum /= len(evaluations)
    print(f"训练完成，最终分数: \nR1:{r1}\tR2:{r2}\tRLSum{rlsum}")

    return {
        "r1":r1,
        "r2":r2,
        "rlsum":rlsum
    },evaluations
