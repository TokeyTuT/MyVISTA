"""轻量 LoRA 实现（AI 辅助生成），不依赖 PEFT。"""
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Linear(base.in_features, rank, bias=False,
                                device=base.weight.device, dtype=torch.float32)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False,
                                device=base.weight.device, dtype=torch.float32)
        nn.init.normal_(self.lora_A.weight, std=0.02)
        nn.init.zeros_(self.lora_B.weight)

    @property
    def weight(self):
        # Owl3 的注意力实现会读取 q_proj.weight.dtype。
        return self.base.weight

    def forward(self, x):
        original = self.base(x)
        adjustment = self.lora_B(self.lora_A(x.float())) * self.scale
        return original + adjustment.to(original.dtype)


class Float32Projection(nn.Module):
    """投影层参数及计算使用 FP32，输出恢复输入精度，兼容训练和 generate。"""
    def __init__(self, base):
        super().__init__()
        self.base = base.float()
        self.base.requires_grad_(True)

    def forward(self, x):
        return self.base(x.float()).to(x.dtype)


def configure_lora(model):
    # 原始 Owl3 全部冻结，随后只打开 LoRA 和视觉投影层。
    model.requires_grad_(False)
    language = model.language_model
    targets = [(name, layer) for name, layer in language.named_modules()
               if isinstance(layer, nn.Linear)
               and name.split('.')[-1] in {'q_proj', 'v_proj'}]
    if not targets:
        raise ValueError('未找到 q_proj/v_proj，不能配置 LoRA')
    for name, layer in targets:
        parent_name, _, child_name = name.rpartition('.')
        parent = language.get_submodule(parent_name) if parent_name else language
        setattr(parent, child_name, LoRALinear(layer))
    model.vision2text_model = Float32Projection(model.vision2text_model)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'LoRA 已启用：{len(targets)} 个线性层，可训练参数 {count:,}')


def save_lora(model, path):
    # PG 和 SG 分别保存；不重复保存冻结的原始权重。
    state = {name: p.detach().cpu() for name, p in model.named_parameters()
             if p.requires_grad}
    torch.save({'rank': 8, 'alpha': 16, 'targets': ['q_proj', 'v_proj'],
                'state_dict': state}, path)


def load_lora(model, path):
    """先创建 PlanGenerator/SummaryGenerator，再加载对应阶段的文件。"""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if (checkpoint['rank'], checkpoint['alpha'], checkpoint['targets']) != (
            8, 16, ['q_proj', 'v_proj']):
        raise ValueError('checkpoint 的 LoRA 配置与当前实现不一致')
    expected = {name: p for name, p in model.named_parameters() if p.requires_grad}
    state = checkpoint['state_dict']
    if set(state) != set(expected) or any(
            state[name].shape != expected[name].shape for name in expected):
        raise ValueError('checkpoint 的参数名称或形状不匹配')
    model.load_state_dict(state, strict=False)
