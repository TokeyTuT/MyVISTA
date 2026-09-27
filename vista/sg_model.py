"""SG：根据视频和问题计划生成摘要。"""

import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from prompts import SG_PROMPT


class SummaryGenerator(nn.Module):
    """
    摘要生成阶段，输入为 ((video,teacher_plan),abstract)
    """

    def __init__(self, model_dir, device,**kwargs):
        super().__init__(**kwargs)
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
        self.model.vision_model.requires_grad_(False)

    def _encode(self, video_frames,plan_text,abstract=None):
        """
        给视频和 plan 进行编码
        """

        prompt = SG_PROMPT.format(questions=plan_text)

        messages = [
            {"role": "user","content": prompt,},
            {"role": "assistant","content": abstract or "", },
        ]

        previous_mode = self.processor.inference_mode
        try:
            self.processor.inference_mode = abstract is None
            inputs = self.processor(messages, images=None, videos=[video_frames])
        finally:
            self.processor.inference_mode = previous_mode

        return inputs.to(self.device)

    def prepare_training_inputs(self, video_frames, plan,abstract=None):
        plan_text = "\n".join(f"{i + 1}. {q.strip()}" for i, q in enumerate(plan))
        prompt = self._encode(video_frames, plan_text)
        inputs = self._encode(video_frames, plan_text,abstract)

        prompt_ids = prompt["input_ids"]
        prefix_length = prompt_ids.shape[1]

        labels = inputs["input_ids"].clone()
        labels[:, :prefix_length] = -100

        inputs["labels"] = labels
        return inputs

    def forward(self,video_frames,plan,abstract):
        inputs = self.prepare_training_inputs(video_frames, plan, abstract)

        pixel_values = inputs.pop("pixel_values")

        # 冻结了视觉模型
        self.model.vision_model.eval()
        with torch.no_grad():
            features = self.model.vision_model(
                pixel_values.to(dtype=next(self.model.vision_model.parameters()).dtype),
                output_hidden_states=True,
            ).hidden_states[-2]

        image_embeds = self.model.vision2text_model(features)

        return self.model.language_model(
            **inputs,image_embeds=image_embeds,use_cache=False,return_dict=True
        )

    @torch.inference_mode()
    def generate_summary(self, video_frames, plan, max_new_tokens=512):
        """输入单条视频和问题列表，生成摘要；不提供参考摘要。"""
        # 和训练时使用相同的 plan 文本格式。
        plan_text = "\n".join(f"{i + 1}. {q.strip()}" for i, q in enumerate(plan))
        was_training = self.training
        self.eval()
        try:
            # abstract 默认为 None，保留助手开头，等待模型生成摘要。
            inputs = self._encode(video_frames, plan_text)
            return self.model.generate(
                **inputs,
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                decode_text=True,
            )
        finally:
            self.train(was_training)
