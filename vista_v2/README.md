## 总览

本次实现重点在视频采样的方法，目前实现了均匀采样、随机采样、按照视觉变化分段采样，后续可以通过接口扩展更多方法。后续模型推理直调用了 API，未做模型额外的训练和更新参数的逻辑。

本次实验中使用的的模型为 `qwen3-vl-plus`，是阿里云百炼提供的多模态大模型，支持图像和文本输入，输出文本。

使用的 Python 版本为 `3.10.21`，依赖见当前目录下 `requirements.txt`。

从仓库根目录安装依赖：

```bash
python -m pip install -r vista_v2/requirements.txt
```

依赖版本记录自当前 `vista_env` 环境，仅列出代码直接使用的包；间接依赖由 pip 自动安装。`matplotlib` 用于采样可视化，主推理流程不依赖它。Jupyter 编辑器或内核按需另行安装。

## 采样方法

所有采样器继承 `VideoSampler`，初始化时配置参数，使用统一接口：

```python
selected_frames, selected_timestamps = sampler.sample(frames, timestamps)
```

输入为按时间排列的 RGB 图像数组或 PIL 图像及其时间戳；输出为 PIL 图像列表和以秒为单位的时间戳列表，两者一一对应。候选帧不足 K 张时，返回全部可用候选，不复制帧补足。

| 方法 | 实现 | 主要参数 |
| --- | --- | --- |
| `UniformSampler` | 在候选帧序列中等间隔选择 | `num_frames` |
| `RandomSampler` | 无放回随机选择，再按时间排序 | `num_frames`、`random_seed` |
| `SceneSampler` | 根据相邻帧的视觉差异分段，再分配固定帧数 | `num_frames`、`threshold`、`min_scene_seconds` |

> 分段采样先将图片缩小到 160×90、转灰度并轻微模糊，再计算归一化后的相邻帧平均绝对像素差。这些处理仅用于计算变化，送入 API 的仍是选中的原尺寸图像。

设检测得到 M 个片段，目标为 K 帧：

- **M = K**：每段取一张中间位置的候选帧。
- **M > K**：保留变化最大的 K−1 个内部边界，合并其余相邻片段。
- **M < K**：每段先分配一帧，再将额度逐个分给“片段时长 ÷ 已分配帧数”最大的片段，直到满足预算或候选用尽。

上述代码在 `vista_v2` 目录下使用。三种方法从同一候选池中选取，便于控制输入条件。添加新方法时，继承 `VideoSampler` 并实现 `sample()` 即可。

## 代码结构

```text
vista_v2/
├── main.py                 # API 客户端、模型、采样及并发配置
├── reasoning.py            # 单视频处理、并发调度、评估和保存
├── sampler.py              # 候选帧读取、采样父类及三个子类
├── preprocessingImage.py   # JPEG/Base64 图片编码
├── generator.py            # PG、SG 及共同的 API 调用接口
├── Prompt.py               # 两阶段英文提示词
├── evaluate.py             # ROUGE 计算
├── sample_demo.py          # 采样可视化示例
├── lab.ipynb               # 采样方法探索 notebook
├── data/practice_data/     # 本地练习集，不随代码分发
└── results/                # 逐次实验结果，已被 Git 忽略
```

## 运行与参数配置

### API 配置

编辑 `main.py` 中的 `WORKSPACE`、`base_url` 和 `model`，使其与自己的阿里云百炼工作空间、地域和模型访问权限匹配。当前代码使用 OpenAI 兼容客户端，模型名称配置为 `qwen3-vl-plus`。

可通过环境变量提供密钥：

```bash
export DASHSCOPE_API_KEY="你的密钥"
```

也可以不设置环境变量，运行时由 `getpass()` 隐藏输入。不要将真实密钥写入源代码或提交到 Git。当前不会自动读取 `.env` 文件。

从仓库根目录运行：

```bash
python vista_v2/main.py
```

当前请求参数为 `temperature=0`、默认 `max_tokens=1024`，并传入 `enable_thinking=False`；请求超时配置为 120 秒。这是当前客户端配置，不代表单条视频总耗时上限。更换服务商或模型时，需要核实多图输入和这些参数是否被支持。

### 当前入口参数

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `model` | `qwen3-vl-plus` | PG、SG 使用同一模型 |
| `num_frames` | 32 | 每条视频目标帧数 |
| `interval` | 1.5 秒 | 候选帧读取间隔 |
| `RANDOM_SEED` | 42 | 随机采样种子 |
| 传入的采样器 | `scene_sampler` | 当前执行分段采样 |
| `max_workers` | 3 | `main.py` 当前同时处理 3 条视频 |

推理支持多并发，只需要在 `main.py` 中修改 `max_workers`，或在 `reasoning()` 调用时传入不同值。

入口每次只运行所选方法，并不会自动遍历所有采样器。不同采样实验应保持视频集合、候选间隔、帧数、模型及提示词一致。

## 并发、评估与结果

线程池最多保留 `max_workers` 个视频任务，完成后再补入新任务，避免一次提交整个数据集。每条视频的图片和计划独立；主线程统一执行评估和文件写入。

ROUGE 使用英文词干归一化，报告：

- ROUGE-1 F1：单词重合程度。
- ROUGE-2 F1：二元词组重合程度。
- ROUGE-Lsum F1：摘要层面的最长公共子序列指标；先用 `pysbd` 分句，每句一行。

逐条记录包含 Precision、Recall、F1，数值范围为 **0～1**；最终显示成功样本的平均 F1。ROUGE 衡量参考文本重合程度，不等于事实正确性，不能单独证明摘要忠实于视频。

默认结果文件为：

```text
results/<采样器类名>_<UTC时间戳>.jsonl
```


