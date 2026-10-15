import json
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from datasets import load_dataset
from preprocessingImage import image_to_data_url
from sampler import VideoSampler,read_candidates
from generator import generate_plan,generate_summary
from evaluate import evaluate_rouge

def process_one_video(step, data, data_root, intervals, sampler, model, client,
                      max_tokens):
    """一个任务处理一条视频；内部按 PG → SG 顺序调用 API。"""
    print(f"正在推理第 {step + 1} 条", flush=True)
    timings = {}
    abstract = data['abstract']
    if not isinstance(abstract, str) or not abstract.strip():
        raise ValueError(f"第 {step + 1} 条样本的参考摘要为空")
    video_path = data_root / data['video_path']

    started = perf_counter()
    timestamps,frames = read_candidates(video_path,intervals)
    timings["read_candidates"] = perf_counter() - started
    print(f"[{step + 1}] 读取 {len(frames)} 张候选帧：{timings['read_candidates']:.2f} 秒", flush=True)

    started = perf_counter()
    selected_frames,selected_timestamps = sampler.sample(frames,timestamps)
    timings["sampling"] = perf_counter() - started
    # PIL 选中帧已独立保存，释放未选中的原尺寸候选图。
    del frames
    print(f"[{step + 1}] 选出 {len(selected_frames)} 帧：{timings['sampling']:.2f} 秒", flush=True)

    started = perf_counter()
    image_urls = [image_to_data_url(image) for image in selected_frames]
    timings["image_encoding"] = perf_counter() - started
    del selected_frames
    print(f"[{step + 1}] 图片编码：{timings['image_encoding']:.2f} 秒；开始生成计划…", flush=True)

    started = perf_counter()
    plan, plan_usage = generate_plan(
        image_urls,model,client,max_tokens,timestamps=selected_timestamps
        )

    timings["plan_api"] = perf_counter() - started
    print(f"[{step + 1}] 计划 API：{timings['plan_api']:.2f} 秒；开始生成摘要…", flush=True)

    started = perf_counter()
    summary, summary_usage = generate_summary(
        image_urls,plan,model,client,max_tokens,timestamps=selected_timestamps
    )

    timings["summary_api"] = perf_counter() - started
    print(f"[{step + 1}] 摘要 API：{timings['summary_api']:.2f} 秒", flush=True)
    del image_urls

    record = {
        "id": data.get("id", step),
        "video_path": str(video_path),
        "sampler": type(sampler).__name__,
        "sampler_config": vars(sampler).copy(),
        "interval": intervals,
        "model": model,
        "max_tokens": max_tokens,
        "timestamps": [float(t) for t in selected_timestamps],
        "plan": plan,
        "summary": summary,
        "abstract": abstract,
        "timings_seconds": timings,
        "usage": {"plan": plan_usage, "summary": summary_usage},
    }
    return record


def reasoning(
        data_root,
        intervals,
        sampler: VideoSampler,
        model,
        client,
        max_tokens=1024,
        *, output_path=None, max_workers=2,
        ):
    """最多同时处理 max_workers 条视频；设为 1 即串行。

    结果按完成顺序写入 JSONL；返回的 evaluations 按数据集原顺序排列。
    失败样本记录 status=error，平均分仅统计成功样本。
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError("max_workers 必须是正整数")
    data_root = Path(data_root)
    dataset = load_dataset(
        "json",
        data_files={"train": str(data_root / "train_part1.json")},
        split="train",
    )
    if len(dataset) == 0:
        raise ValueError("数据集为空，无法进行推理和评估")

    if output_path is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = Path(__file__).resolve().parent / "results" / f"{type(sampler).__name__}_{run_id}.jsonl"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores_by_index = {}
    failures = 0
    tasks = iter(enumerate(dataset))
    print(f"共 {len(dataset)} 条视频，最多同时处理 {max_workers} 条", flush=True)

    # 只有主线程负责评估和写文件，避免共享评估器及文件的并发访问。
    with output_path.open("x", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            pending = {}

            def submit_next():
                item = next(tasks, None)
                if item is None:
                    return
                step, data = item
                future = executor.submit(
                    process_one_video, step, data, data_root, intervals,
                    sampler, model, client, max_tokens,
                )
                pending[future] = (step, data)

            # 只保留一个并发窗口，避免一次创建整个数据集的任务。
            for _ in range(min(max_workers, len(dataset))):
                submit_next()

            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    step, data = pending.pop(future)
                    try:
                        record = future.result()
                        started = perf_counter()
                        score = evaluate_rouge(record["summary"], record["abstract"])
                        record["timings_seconds"]["evaluation"] = perf_counter() - started
                        record["rouge"] = score
                        record["status"] = "success"
                        scores_by_index[step] = score
                        print(
                            f"[{step + 1}] 完成 | R1: {score['rouge1']['f1']:.4f} | "
                            f"R2: {score['rouge2']['f1']:.4f} | "
                            f"RLsum: {score['rougeLsum']['f1']:.4f}",
                            flush=True,
                        )
                    except Exception as error:
                        failures += 1
                        record = {
                            "id": data.get("id", step),
                            "video_path": str(data_root / data["video_path"]),
                            "status": "error",
                            "error_type": type(error).__name__,
                            "error": str(error),
                        }
                        print(f"[{step + 1}] 失败：{error}", flush=True)

                    record["dataset_index"] = step
                    record["max_workers"] = max_workers
                    output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_file.flush()
                    submit_next()

    print(f"逐条结果已保存到: {output_path}")
    if not scores_by_index:
        raise RuntimeError(f"全部 {failures} 条视频失败，错误记录见 {output_path}")

    evaluations = [scores_by_index[i] for i in sorted(scores_by_index)]
    averages = {
        short: sum(score[metric]["f1"] for score in evaluations) / len(evaluations)
        for short, metric in [("r1", "rouge1"), ("r2", "rouge2"), ("rlsum", "rougeLsum")]
    }
    print(
        f"推理完成：成功 {len(evaluations)} 条，失败 {failures} 条。"
        f"成功样本平均 F1（0～1）：{averages}"
    )
    return averages, evaluations
