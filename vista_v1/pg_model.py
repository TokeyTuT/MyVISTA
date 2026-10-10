"""PG：根据视频生成问题计划。"""

import torch
from torch import nn
from lora import configure_lora
from transformers import AutoTokenizer, AutoModel

from prompts import PG_PROMPT


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
    需要在外部训练循环中执行。本类使用 LoRA 和视觉投影层微调。



    为了更好理解 Owl3 以及 PG Model 是如何工作的，我创建了一个学习笔记:
    见当前目录 Note/Owl3输入处理与 PG 训练原理.md
    """

    def __init__(self, model_dir, device):
        super().__init__()
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
        # 只训练语言模型中的 LoRA 参数和视觉投影层。
        configure_lora(self.model)

    def _encode(self, video_frames, plan_text=None):
        """
        输入：video_frames 是一条视频的 PIL 图像列表；plan_text 是答案文本。
        输出：包含 input_ids、pixel_values、media_offset 的张量字典。
        这里仅编码材料，不调用语言模型，也不计算 loss。
        """
        #    由于每次 processor 都会改变对话，所以每次需要创建一次独立对话。
        #    <|video|> 标记视频在用户消息中的位置，真正的图像数据通过下面 videos 参数传入。
        #    plan_text=None 时 assistant 为空；训练编码时填入教师答案。
        #    processor 会修改 messages，所以不能重复使用同一个消息列表。
        messages = [
            {
                "role": "user",
                "content": PG_PROMPT,
            },
            {
                "role": "assistant",
                "content": plan_text or "",
            },
        ]

        #    暂存 processor 原来的格式处理模式，完成本次编码后恢复。
        #    这个 inference_mode 只控制是否保留末尾答案，它不是 torch.inference_mode()，也不控制梯度。
        previous_mode = self.processor.inference_mode
        try:
            #    没有答案 → 保留提示词，等待模型生成；
            #    有答案 → 关闭推理格式处理，保留答案及结束标记。
            self.processor.inference_mode = plan_text is None

            #    videos=[video_frames] 表示这条对话包含一段视频，
            #    外层列表不是多条独立样本组成的 batch。
            #    单样本输出：input_ids (1,L)，pixel_values (T,3,H,W)，
            #    media_offset (1,L,2)。L 为 token 数，T 为帧数。
            inputs = self.processor(messages, images=None, videos=[video_frames])
        finally:
            #  即使编码报错，也恢复 processor 的模式，避免影响下次调用。
            self.processor.inference_mode = previous_mode

        #  将输入张量放到与模型相同的设备上。
        return inputs.to(self.device)

    def prepare_training_inputs(self, video_frames, plan):
        """将教师问题列表变成训练序列，并构建只监督答案部分的 labels。"""

        # 将 ["问题A", "问题B"] 转成 "1. 问题A\n2. 问题B"。
        plan_text = "\n".join(f"{i + 1}. {q.strip()}" for i, q in enumerate(plan))
        # 编码提示词：确定答案之前到底有多少个 token。包括系统/用户消息、视频标记以及 assistant 的起始标记。
        prompt = self._encode(video_frames) # 构成： 用户提示词 + 助手起始标记
        # 编码完整序列：同样的提示词 + 教师答案 + 结束标记。为了清楚起见这里编码两次，视频图像预处理也会执行两次。
        inputs = self._encode(video_frames, plan_text) # 构成： 用户提示词 + 助手起始标记 + 教师答案 + 结束标记

        #  input_ids 形状为 (1,L)，所以 shape[1] 是序列长度。
        prompt_ids = prompt["input_ids"]
        prefix_length = prompt_ids.shape[1]

        #  复制 token ID 作为监督目标。clone 避免修改原本的 input_ids。
        labels = inputs["input_ids"].clone()

        #    用户提示词位置设为 -100，表示这些位置不参与交叉熵 loss。
        #    只有教师答案及结束标记保留真实 token ID，作为预测目标
        # 这是一个很常见的 tricks
        labels[:, :prefix_length] = -100

        #  在原输入字典中加入 labels，供语言模型计算监督损失。
        inputs["labels"] = labels
        return inputs

    def forward(self, video_frames, plan):
        """pg(video_frames, plan) 会进入这里；返回结果中包含 outputs.loss。"""
        #  准备完整文本序列、视频帧张量、图文对应信息以及 labels。
        inputs = self.prepare_training_inputs(video_frames, plan)

        # 从字典取出并移除像素张量，这部分需要作为视觉输入部分
        pixel_values = inputs.pop("pixel_values")

        #  冻结的视觉编码器使用评估模式，关闭其中的训练态随机行为。
        self.model.vision_model.eval() # eval() 本身不关闭梯度；下面的 no_grad() 才关闭梯度记录。

        with torch.no_grad():
            features = self.model.vision_model(
                pixel_values.to(dtype=next(self.model.vision_model.parameters()).dtype),
                output_hidden_states=True,
            ).hidden_states[-2] # 其实可以取最后一层的输出，只是 Owl3 的原作者用了倒数第二层而已 😅

        #  视觉投影层将视觉特征映射到语言模型使用的维度。
        image_embeds = self.model.vision2text_model(features)

        #    将 inputs 字典展开成具名参数，等价于传入 input_ids=... 等。
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
        #  记录调用前的模式，避免训练中临时生成文本后一直停在 eval 模式。
        was_training = self.training
        self.eval()
        try:
            # 不传 plan_text，因此输入只有视频和提示词，没有标准答案。
            inputs = self._encode(video_frames)

            return self.model.generate(
                **inputs,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                decode_text=True,
            )
        finally:
            # 恢复之前的状态
            self.train(was_training)
