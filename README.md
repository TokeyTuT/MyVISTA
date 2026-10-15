# VISTA 论文复现与实验

本项目用于学习和复现 VISTA 科研演讲视频摘要方法，采用两阶段生成流程：先由 PG 生成结构化计划，再由 SG 根据计划生成摘要。

## 项目结构

| 目录 / 文件 | 说明 |
| --- | --- |
| [VISTA-main/](VISTA-main/README.md) | 论文作者的原始代码，供参考与对照 |
| [vista_v1/](vista_v1/README.md) | 基于 mPLUG-Owl3-2B 的初版复现，包含 PG / SG 和 LoRA 微调实现 |
| [vista_v2/](vista_v2/README.md) | 基于多模态 API 的实验版本，重点比较均匀、随机和视觉变化分段采样，不进行模型训练 |
| [paper/](paper/) | 论文原文 |
| [data.zip](data.zip) | 测试用 mini 数据集，包含 6 个视频及对应标注 |
| [论文笔记.svg](论文笔记.svg) | 方法流程图与阅读笔记 |

环境配置、运行方法和实验参数见各子目录的 README。若关注视频采样与 API 推理，可从 `vista_v2` 开始；若关注本地模型微调，可查看 `vista_v1`。

本项目以学习和小规模实验为主，对模型和实验设置进行了调整，结果不直接等同于原论文结果。

## 测试数据

`data.zip` 用于小规模测试和跑通流程，不是完整的 VISTA 数据集。压缩包内包含 `data/practice_data/`，其中有视频文件和 `train_part1.json` 标注。

使用 `vista_v2` 时，在仓库根目录执行以下命令解压：

```bash
unzip data.zip 'data/*' -d vista_v2/
```

解压后数据位于 `vista_v2/data/practice_data/`，运行方法见 [vista_v2/README.md](vista_v2/README.md)。

## 相关资源

- [原论文：What Is That Talk About?](https://arxiv.org/abs/2502.08279)
- [作者仓库](https://github.com/dongqi-me/VISTA)
- [VISTA 数据集](https://huggingface.co/datasets/dongqi-me/VISTA)
