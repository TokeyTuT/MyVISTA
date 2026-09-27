# VISTA 数据集论文复现

原论文 [What Is That Talk About A Video-to-Text Summarization Dataset for Scientific Presentations](https://arxiv.org/abs/2502.08279)

原仓库 [VISTA](https://github.com/dongqi-me/VISTA)

数据集 [Dataset](https://huggingface.co/datasets/dongqi-me/VISTA)

## 说明
- `VISTA-main` 文件夹下存放的是论文源代码
- `vista` 文件夹下存放的是复现代码

因为本机配置有限，并且主要目的是为了学习。在实际实现中：
* 原文中最终采用的 mPLUG-Owl3-7B 在本项目中被换成了相同架构的 2B 模型,初次运行时程序自动会下载模型到 `./vista/original_model中`
* 原论文中用于预生成 Plan 的 GPT-o1 API 被替换成了 OpenAI 格式的 Deepseek-flash API。
* 初次运行时，程序会检查工作目录下是否存在预生成的 `plan.json` 如果存在会直接读取而不再重复调用 API 生成 plan
* 原论文作者尚未开放 VISTA 数据集访问的权限，本人从 YouTube 上抓取了 6 个 cvpr 
会议视频制作了一个 mini 数据集用于跑通实验做测试。 
* 为了实现简单，实验中 batch_size 使用大小为 1，也就是一次性只能训练 1 个样本
* 运行结束后， PG 和 SG 模型参数会自动保存在 `./vista/checkpoints` 中


## 实验环境
实验环境中使用的 Python 版本为 3.10 
依赖文件在：`./vista/requirement.txt` 
> 安装依赖：`pip install -r requirements.txt`
