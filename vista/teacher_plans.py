"""调用教师模型生成 plan，并读写本地缓存。"""

import json
from pathlib import Path
from getpass import getpass

import pysbd
from openai import OpenAI

from prompts import QUESTION_PROMPT


def save_json_atomic(data, output_path):
    """先写临时文件，再替换正式文件。"""
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    temp_path.replace(output_path)


def generate_plan_by_teacher(
        dataset,
        output_path,
):
    """
    根据摘要内容生成 plan

    先检查每条样本的缓存，只为缺失或失效的样本调用 API，每完成一篇就保存。

    采用 pysbd 包进行分句
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    teacher_model = "deepseek-flash"
    seg = pysbd.Segmenter(language='en', clean=False)

    # 读取已有缓存
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            print(f'检测到本机上已有 teacher plans，直接从硬盘中读取')
            records = json.load(f)

        if not isinstance(records, dict):
            raise ValueError("plan 文件应当是以样本 id 为键的字典")
    else:
        records = {}
    # 第一遍：检查缓存，收集需要生成的样本
    pending_samples = []
    seen_ids = set()
    for sample in dataset:
        sample_id = str(sample["id"])
        if sample_id in seen_ids:
            raise ValueError(f"数据集中存在重复 id：{sample_id}")
        seen_ids.add(sample_id)

        abstract = sample["abstract"].strip()
        sentences = [s.strip() for s in seg.segment(abstract) if s.strip()]
        if not sentences:
            raise ValueError(f"样本 {sample_id} 的摘要没有有效句子")

        cached = records.get(sample_id, {})
        if not isinstance(cached, dict):
            cached = {}

        cached_plan = cached.get("plan")

        cache_valid = (
            cached.get("abstract") == abstract
            and cached.get("question_prompt") == QUESTION_PROMPT
            and cached.get("teacher_model") == teacher_model
            and cached.get("reference_sentences") == sentences
            and isinstance(cached_plan, list)
            and len(cached_plan) == len(sentences)
            and all(isinstance(q, str) and q.strip() for q in cached_plan)
        )

        if cache_valid:
            print(f"跳过已完成样本：{sample_id}")
            continue

        pending_samples.append((sample_id, abstract, sentences))

    if not pending_samples:
        print('所有 teacher plans 缓存均有效')
        return records

    print(f'有 {len(pending_samples)} 篇摘要需要生成 plan，请输入 DeepSeek API Key：')
    # 只有需要生成时才初始化客户端
    client = OpenAI(
        api_key=getpass(),
        base_url="https://api.deepseek.com",
    )

    # 第二遍：只生成待处理样本
    for idx, (sample_id, abstract, sentences) in enumerate(pending_samples):
        print(f'正在生成 {idx + 1}/{len(pending_samples)}：{sample_id}')
        plan = []
        for i, target in enumerate(sentences):
            context = " ".join(sentences[:i])  # 考虑前文
            prompt = QUESTION_PROMPT.format(
                context=context,
                target=target,
            )

            # 简单问题生成关闭思考，固定输出上限，不自动增加额度。
            response = client.chat.completions.create(
                model=teacher_model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=512,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if not response.choices:
                raise RuntimeError(f"样本 {sample_id} 第 {i + 1} 句未返回结果")
            choice = response.choices[0]
            question = (choice.message.content or "").strip()
            if choice.finish_reason != "stop" or not question:
                raise RuntimeError(
                    f"样本 {sample_id} 第 {i + 1} 句生成失败："
                    f"finish_reason={choice.finish_reason}, "
                    "max_tokens=512, thinking=disabled。已完成摘要的缓存仍保留。"
                )
            plan.append(question)

        records[sample_id] = {
            "id": sample_id,
            "abstract": abstract,
            "question_prompt": QUESTION_PROMPT,
            "teacher_model": teacher_model,
            "thinking": "disabled",
            "max_tokens": 512,
            "reference_sentences": sentences,
            "plan": plan,
        }

        save_json_atomic(records, output_path)
        print(f'已保存：{sample_id}')

    print('teacher plans 获取成功')
    return records
