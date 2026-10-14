import math
from Prompt import get_plan_generation_prompt,get_summary_generation_prompt

def call_vlm(
        image_urls,
        model,
        client,
        prompt,
        max_tokens=1024,
        *, timestamps=None,
):
    """
    调用视觉大模型
    生成摘要和计划的共同接口
    """
    if not image_urls:
        raise ValueError("至少需要一张图片")

    # 时间戳单位为秒，与图片逐一对应；未传时兼容原来的调用。
    if timestamps is not None:
        timestamps = [float(t) for t in timestamps]
        if len(timestamps) != len(image_urls):
            raise ValueError("timestamps 与 image_urls 数量必须一致")
        if any(not math.isfinite(t) or t < 0 for t in timestamps):
            raise ValueError("时间戳必须是非负有限数")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("时间戳必须严格递增，图片也应按相同顺序排列")

    content = [{"type": "text", "text": prompt}]

    for i, url in enumerate(image_urls):
        label = f"Frame {i + 1}:"
        if timestamps is not None:
            label = f"Frame {i + 1} (approximately {timestamps[i]:.3f} seconds from video start):"
        content.extend([
            {"type": "text", "text": label},
            {"type": "image_url", "image_url": {"url": url}},
        ])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": content},
        ],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"enable_thinking": False},
    )

    if not response.choices:
        raise RuntimeError("API 未返回生成结果")

    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    if choice.finish_reason != "stop" or not text:
        raise RuntimeError(
            f"生成未正常完成：finish_reason={choice.finish_reason}"
        )

    usage = response.usage.model_dump() if response.usage else None

    return text, usage


def generate_plan(
        image_urls,
        model,
        client,
        max_tokens=1024,
        *, timestamps=None,
        ):

    """
    根据图片信息生成 Plan (Question)
    """
    return call_vlm(
        image_urls,
        model,
        client,
        get_plan_generation_prompt(),
        max_tokens,
        timestamps=timestamps,
    )


def generate_summary(
        image_urls,
        plan,
        model,
        client,
        max_tokens=1024,
        *, timestamps=None,
        ):

    """
    根据图片信息生成 Summary (摘要)
    """


    return call_vlm(
        image_urls,
        model,
        client,
        get_summary_generation_prompt(plan),
        max_tokens,
        timestamps=timestamps,
    )



