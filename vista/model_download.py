"""检查 Owl3 本地文件；缺失时下载，返回可直接传给模型的目录。"""

import json
from pathlib import Path


REQUIRED_FILES = (
    "config.json", "tokenizer_config.json", "tokenizer.json",
    "configuration_mplugowl3.py", "configuration_hyper_qwen2.py",
    "modeling_mplugowl3.py", "modeling_hyper_qwen2.py",
    "processing_mplugowl3.py", "image_processing_mplugowl3.py", "x_sdpa.py",
)


def model_files_ready(model_dir):
    """检查配置、Owl3 自定义代码和权重，避免把空目录当成已下载。"""
    model_dir = Path(model_dir)
    try:
        for name in REQUIRED_FILES:
            path = model_dir / name
            if not path.is_file() or path.stat().st_size == 0:
                return False
            if path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))

        # 兼容单个权重文件和分片权重；分片必须全部存在。
        index = model_dir / "model.safetensors.index.json"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            weights = set(weight_map.values())
        else:
            weights = {"model.safetensors"}
        if not weights:
            return False

        # 只读取权重文件头，不把整个模型加载进内存。
        from safetensors import safe_open
        for name in weights:
            with safe_open(str(model_dir / name), framework="pt", device="cpu") as f:
                if not list(f.keys()):
                    return False
        return True
    except Exception:
        # 缺失、截断或不可解析的文件均不能作为完整模型加载。
        return False


def ensure_model_downloaded(model_dir, source="auto"):
    """auto 优先 Hugging Face，失败后尝试魔搭；也可指定单一来源。

    source 可选：auto、huggingface、modelscope。
    返回 Path；调用方应使用返回路径，而不是假定权重就在原目录。
    """
    if source not in ("auto", "huggingface", "modelscope"):
        raise ValueError("source 必须为 auto、huggingface 或 modelscope")
    model_dir = Path(model_dir).expanduser().resolve()
    sources = ["huggingface", "modelscope"] if source == "auto" else [source]

    # 先复用原有模型，或上次下载完成的模型，全程无需联网。
    for candidate in [model_dir] + [model_dir / name for name in sources]:
        if model_files_ready(candidate):
            print(f"本地 Owl3 模型文件已就绪：{candidate}")
            return candidate

    errors = []
    for name in sources:
        # 不同来源分开保存，避免失败后把两个仓库的文件混在一起。
        target = model_dir / name
        target.mkdir(parents=True, exist_ok=True)
        print(f"本地模型缺失或不完整，正在从 {name} 下载 Owl3：{target}")
        try:
            if name == "huggingface":
                from huggingface_hub import snapshot_download
                snapshot_download(
                    repo_id="mPLUG/mPLUG-Owl3-2B-241014",
                    local_dir=str(target),
                )
            else:
                from modelscope import snapshot_download
                snapshot_download(
                    model_id="iic/mPLUG-Owl3-2B-241014",
                    local_dir=str(target),
                )
            if not model_files_ready(target):
                raise RuntimeError("下载结束，但配置、代码或权重文件仍不完整")
            print(f"模型下载完成：{target}")
            return target
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"{name} 下载失败，原因：{type(exc).__name__}")

    raise RuntimeError("模型下载失败，请检查网络及下载依赖：\n" + "\n".join(errors))
