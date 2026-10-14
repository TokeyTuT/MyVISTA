import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
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
        max_tokens=1024,
        *, output_path=None,
        ):

    data_root = Path(data_root)

    dataset = load_dataset(
            "json",
            data_files={
                "train": str(data_root / "train_part1.json"),
            },
            split="train",
        )

    if len(dataset) == 0:
        raise ValueError("数据集为空，无法进行推理和评估")

    if output_path is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = Path(__file__).resolve().parent / "results" / f"{type(sampler).__name__}_{run_id}.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    r1 = 0.0 # "rouge1",
    r2 = 0.0 # "rouge2"
    rlsum = 0.0 # "rougeLsum"

    evaluations = []
    # 独占创建，防止覆盖已有实验结果。
    with output_path.open("x", encoding="utf-8") as output_file:
        for step,data in enumerate(dataset):
            print(f"正在推理第 {step + 1} 条", flush=True)
            timings = {}
            abstract = data['abstract']
            if not isinstance(abstract, str) or not abstract.strip():
                raise ValueError(f"第 {step + 1} 条样本的参考摘要为空")
            video_path = data_root / data['video_path']

            started = perf_counter()
            timestamps,frames = read_candidates(video_path,intervals)
            timings["read_candidates"] = perf_counter() - started
            print(f"  读取 {len(frames)} 张候选帧：{timings['read_candidates']:.2f} 秒", flush=True)

            started = perf_counter()
            selected_frames,selected_timestamps = sampler.sample(frames,timestamps)
            timings["sampling"] = perf_counter() - started
            # PIL 选中帧已独立保存，释放未选中的原尺寸候选图。
            del frames
            print(f"  选出 {len(selected_frames)} 帧：{timings['sampling']:.2f} 秒", flush=True)

            started = perf_counter()
            image_urls = [image_to_data_url(image) for image in selected_frames]
            timings["image_encoding"] = perf_counter() - started
            del selected_frames
            print(f"  图片编码：{timings['image_encoding']:.2f} 秒；开始生成计划…", flush=True)

            started = perf_counter()
            plan, plan_usage = generate_plan(
                image_urls,model,client,max_tokens,timestamps=selected_timestamps
                )

            timings["plan_api"] = perf_counter() - started
            print(f"  计划 API：{timings['plan_api']:.2f} 秒；开始生成摘要…", flush=True)

            started = perf_counter()
            summary, summary_usage = generate_summary(
                image_urls,plan,model,client,max_tokens,timestamps=selected_timestamps
            )

            timings["summary_api"] = perf_counter() - started
            print(f"  摘要 API：{timings['summary_api']:.2f} 秒", flush=True)
            del image_urls

            started = perf_counter()
            score = evaluate_rouge(summary,abstract)
            timings["evaluation"] = perf_counter() - started

            print(
                f"R1: {score['rouge1']['f1']:.4f} | "
                f"R2: {score['rouge2']['f1']:.4f} | "
                f"RLsum: {score['rougeLsum']['f1']:.4f}"
            )
            record = {
                "id": data.get("id", step),
                "video_path": str(video_path),
                "sampler": type(sampler).__name__,
                "sampler_config": vars(sampler),
                "interval": intervals,
                "model": model,
                "max_tokens": max_tokens,
                "timestamps": [float(t) for t in selected_timestamps],
                "plan": plan,
                "summary": summary,
                "abstract": abstract,
                "rouge": score,
                "timings_seconds": timings,
                "usage": {"plan": plan_usage, "summary": summary_usage},
            }
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_file.flush()  # 每完成一条就写入，后续失败也保留已有结果
            evaluations.append(score)

            r1 += score['rouge1']['f1']
            r2 += score['rouge2']['f1']
            rlsum += score['rougeLsum']['f1']

    r1 /= len(evaluations)
    r2 /= len(evaluations)
    rlsum /= len(evaluations)
    print(f"推理完成，最终平均 F1（0～1）: \nR1:{r1}\tR2:{r2}\tRLSum{rlsum}")

    print(f"逐条结果已保存到: {output_path}")

    return {
        "r1":r1,
        "r2":r2,
        "rlsum":rlsum
    },evaluations
