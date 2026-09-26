import math
import json
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import pysbd

from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from openai import OpenAI
from getpass import getpass
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

QUESTION_PROMPT = """Generate a coherent and contextually relevant question based on the provided context and target sentence, ensuring that the target sentence can be treated as an answer to the generated question.

Output only one question in plain text.
Do not include Markdown formatting, labels, numbering, quotation marks, explanations, or answers.

Context: {context}
Target: {target}
Question Sentence:
"""

PG_PROMPT = (
    "Generate an ordered list of questions to guide a scientific abstract "
    "of the provided video. Output only the questions."
)


# def visualize_frames(frames):
#     if not frames:
#         raise ValueError("没有可显示的视频帧")
#
#     num_rows = math.ceil(len(frames) / 4)
#
#     fig, axes = plt.subplots(
#         num_rows, 4,
#         figsize=(16, 3 * num_rows),
#         squeeze=False,
#     )
#
#     # 包括没有图片的空白格子，也关闭坐标轴
#     for ax in axes.flat:
#         ax.axis("off")
#
#     for i, (frame, ax) in enumerate(zip(frames, axes.flat)):
#         ax.imshow(frame)
#         ax.set_title(f"Sample {i + 1}")
#
#     plt.tight_layout()
#     plt.show()


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



def get_frames(
        video_path:Path,
        num_frames:int
):
    """均匀抽帧，返回按时间排列的 RGB PIL 图像列表。"""
    if num_frames <= 0:
        raise ValueError("num_frames 必须大于 0")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")

    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise ValueError(f"视频没有有效帧：{video_path}")

        frames_indices = np.linspace(
            0, total_frames - 1, min(total_frames,num_frames), dtype=int
        )

        for index in frames_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

            success, frame = cap.read()
            if not success:
                raise IOError(f'Cannot read frame {index} in  {video_path}')

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    finally:
        cap.release()

    # visualize_frames(frames)

    return [Image.fromarray(frame) for frame in frames]


def collate_samples(samples):
    """保留独立样本，避免默认堆叠长度不同的 plan。"""
    return samples


def load_vista_dataset(
        data_root,
        batch_size,
):
    """导入并处理数据集"""
    data_root = Path(data_root)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_root / "train_part1.json"),
        },
        split="train",
    )

    records = generate_plan_by_teacher(dataset, data_root / 'plan.json')
    dataset = dataset.add_column(
        "plan",
        [records[str(sample["id"])]["plan"] for sample in dataset],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 调试阶段方便检查顺序
        num_workers=0,
        collate_fn=collate_samples,
    )


class PlanGenerator(nn.Module):
    """
    封装 Owl3，处理一条视频（目前不是多条视频同时训练）。

    训练路径：
        pg(video_frames, plan)
        → forward → prepare_training_inputs → _encode
        → 视觉特征 → 语言模型 → 返回包含 loss 的结果。

    推理路径：
        pg.generate_plan(video_frames)
        → _encode → model.generate → 返回生成的问题文本。

    注意：forward 只计算 loss；loss.backward() 和 optimizer.step()
    需要在外部训练循环中执行。本类目前还没有配置 LoRA。
    """

    def __init__(self, model_dir, num_frames, device):
        super().__init__()
        self.num_frames = num_frames
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, trust_remote_code=True, local_files_only=True,
        )

        self.model = AutoModel.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
            torch_dtype=torch.float32 if self.device.type == "cpu" else torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.processor = self.model.init_processor(self.tokenizer)

        # 冻结视觉编码器，不计算其参数梯度。
        # 语言模型和视觉投影层仍可训练；是否更新取决于外部 optimizer。
        self.model.vision_model.requires_grad_(False)

    def _encode(self, video_frames, plan_text=None):
        """
        输入：video_frames 是一条视频的 PIL 图像列表；plan_text 是答案文本。
        输出：包含 input_ids、pixel_values、media_offset 的张量字典。
        这里仅编码材料，不调用语言模型，也不计算 loss。
        """
        #   创建一次独立对话。<|video|> 标记视频在用户消息中的位置，
        #    真正的图像数据通过下面 videos 参数传入。
        #    plan_text=None 时 assistant 为空；训练编码时填入教师答案。
        #    processor 会修改 messages，所以不能重复使用同一个消息列表。
        messages = [
            {"role": "user", "content": "<|video|>\n" + PG_PROMPT}, # 这里的 <|video|> 相当于一个占位符，代表这里会插入视频内容。这是 owl3 的规定
            {"role": "assistant", "content": plan_text or ""},
        ]

        # 2. 暂存 processor 原来的格式处理模式，完成本次编码后恢复。
        #    这个 inference_mode 只控制是否保留末尾答案，
        #    它不是 torch.inference_mode()，也不控制梯度。
        previous_mode = self.processor.inference_mode
        try:
            # 3. 没有答案 → 保留提示词，等待模型生成；
            #    有答案 → 关闭推理格式处理，保留答案及结束标记。
            self.processor.inference_mode = plan_text is None

            # 4. videos=[video_frames] 表示这条对话包含一段视频，
            #    外层列表不是多条独立样本组成的 batch。
            #    单样本输出：input_ids (1,L)，pixel_values (T,3,H,W)，
            #    media_offset (1,L,2)。L 为 token 数，T 为帧数。
            inputs = self.processor(messages, images=None, videos=[video_frames])
        finally:
            # 5. 即使编码报错，也恢复 processor 的模式，避免影响下次调用。
            self.processor.inference_mode = previous_mode

        # 6. 将输入张量放到与模型相同的设备上。
        return inputs.to(self.device)

    def prepare_training_inputs(self, video_frames, plan):
        """将教师问题列表变成训练序列，并构建只监督答案部分的 labels。"""
        # 1. 将 ["问题A", "问题B"] 转成 "1. 问题A\n2. 问题B"。
        #    plan 应当是缓存中的非空问题字符串列表。
        plan_text = "\n".join(f"{i + 1}. {q.strip()}" for i, q in enumerate(plan))

        # 2. 编码提示词：确定答案之前到底有多少个 token。
        #    包括系统/用户消息、视频标记以及 assistant 的起始标记。
        prompt = self._encode(video_frames)

        # 3. 编码完整序列：同样的提示词 + 教师答案 + 结束标记。
        #    为了清楚起见这里编码两次，视频图像预处理也会执行两次。
        inputs = self._encode(video_frames, plan_text)

        # 4. input_ids 形状为 (1,L)，所以 shape[1] 是序列长度。
        prompt_ids = prompt["input_ids"]
        prefix_length = prompt_ids.shape[1]

        # 5. 确认两次编码的前缀完全相同，才能用这个长度划分答案位置。
        if not torch.equal(inputs["input_ids"][:, :prefix_length], prompt_ids):
            raise ValueError("提示词 token 前缀不匹配，无法安全构造 labels")

        # 6. 复制 token ID 作为监督目标。clone 避免修改原本的 input_ids。
        labels = inputs["input_ids"].clone()

        # 7. 提示词位置设为 -100，表示这些位置不参与交叉熵 loss。
        #    提示词仍在 input_ids 中，模型仍然看得见它。
        #    只有教师答案及结束标记保留真实 token ID，作为预测目标。
        labels[:, :prefix_length] = -100

        # 8. 确认至少存在一个参与 loss 的位置。
        if not (labels != -100).any():
            raise ValueError("训练样本没有答案 token")

        # 9. 在原输入字典中加入 labels，供语言模型计算监督损失。
        inputs["labels"] = labels
        return inputs

    def forward(self, video_frames, plan):
        """pg(video_frames, plan) 会进入这里；返回结果中包含 outputs.loss。"""
        # 1. 准备完整文本序列、视频帧张量、图文对应信息以及 labels。
        inputs = self.prepare_training_inputs(video_frames, plan)

        # 2. 从字典取出并移除像素张量，我们先单独处理视觉部分。
        #    余下的 input_ids、media_offset、labels 稍后传给语言模型。
        pixel_values = inputs.pop("pixel_values")

        # 3. 冻结的视觉编码器使用评估模式，关闭其中的训练态随机行为。
        #    eval() 本身不关闭梯度；下面的 no_grad() 才关闭梯度记录。
        self.model.vision_model.eval()
        with torch.no_grad():
            # 4. 将像素转换成视觉编码器的浮点类型，然后提取视觉特征。
            #    hidden_states[-2] 取倒数第二层输出，沿用本地 Owl3 实现。
            #    特征通常形如 (T, 每帧视觉token数, 视觉特征维度)。
            features = self.model.vision_model(
                pixel_values.to(dtype=next(self.model.vision_model.parameters()).dtype),
                output_hidden_states=True,
            ).hidden_states[-2]

        # 5. 视觉投影层将视觉特征映射到语言模型使用的维度。
        #    这一行在 no_grad 外，因此投影层可以计算梯度。
        #    本地原版 forward_image 使用 inference_mode；这里改用上面的
        #    no_grad，避免其输出成为后续可训练层无法保存的 inference tensor。
        image_embeds = self.model.vision2text_model(features)

        # 6. **inputs 将字典展开成具名参数，等价于传入 input_ids=... 等。
        #    语言模型结合视觉特征和文本，计算各位置的下一个 token 概率。
        #    它内部会将预测和 labels 错开一位，计算交叉熵；不要手动再错位。
        #    因果注意力不允许当前位置看到后面的答案，避免直接抄答案。
        #    use_cache=False：训练时不保存逐 token 生成使用的 KV 缓存。
        #    return_dict=True：返回可通过 outputs.loss、outputs.logits 访问的结果。
        return self.model.language_model(
            **inputs, image_embeds=image_embeds, use_cache=False, return_dict=True,
        )

    # 这个装饰器关闭整个方法的梯度记录，仅用于生成文本，不能用来训练。
    @torch.inference_mode()
    def generate_plan(self, video_frames, max_new_tokens=512):
        """不给教师答案，让模型根据视频和提示词自行生成 plan。"""
        # 1. 记录调用前的模式，避免训练中临时生成文本后一直停在 eval 模式。
        was_training = self.training
        self.eval()
        try:
            # 2. 不传 plan_text，因此输入只有视频和提示词，没有标准答案。
            inputs = self._encode(video_frames)

            # 3. 逐 token 生成，后续 token 使用模型自己生成的前文。
            #    max_new_tokens 限制新增 token 数；do_sample=False 不随机采样；
            #    decode_text=True 要求 Owl3 将生成的 token 解码成文本。
            return self.model.generate(
                **inputs,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                decode_text=True,
            )
        finally:
            # 4. 无论生成成功与否，都恢复调用前的 train/eval 模式。
            self.train(was_training)



def main():
    data_root = Path(__file__).resolve().parent / "data/practice_data"

    loader = load_vista_dataset(
        data_root,
        batch_size=1,
    )


if __name__ == '__main__':
    main()
