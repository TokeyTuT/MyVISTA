# Owl3 输入处理与 PG 训练原理

这份笔记对应本项目本地的 mPLUG-Owl3-2B-241014 和 `PlanGenerator` 实现。尤其是 `processor.inference_mode` 的行为属于这份 processor 的实现，不能直接推广到所有模型。

## 1. 先记住各部分负责什么

| 部分 | 工作 |
|---|---|
| `get_frames()` | 从 MP4 抽取若干画面，返回按时间排列的 RGB PIL 图片列表 |
| tokenizer | 将文字和特殊标记转换成 token 编号，也能把编号解码成文字 |
| processor | 整理对话格式、调用 tokenizer、预处理图片，并组织图文对应信息 |
| Owl3 模型 | 从图像提取视觉特征，结合文字预测下一个 token |
| `generate()` | 反复预测并选择 token，直到结束，组成完整回答 |

**processor 准备材料，模型进行预测。processor 本身不会生成 plan。**

```text
MP4 → 抽帧 → PIL 图片列表 ──────┐
                              ↓
对话消息 messages ───────→ processor
                              ↓
          input_ids、pixel_values、media_offset
                              ↓
                            Owl3
                              ↓
                      下一个 token 的预测
```

此处输入只有抽出的视频画面，没有自动将视频声音传入模型。

## 2. messages 是什么？

```python
messages = [
    {"role": "user", "content": "<|video|>\n" + PG_PROMPT},
    {"role": "assistant", "content": plan_text or ""},
]
```

- `role`：谁说的。`user` 是提出任务的一方，`assistant` 是模型扮演的回答者。
- `content`：这条消息的具体内容。
- `<|video|>`：视频位置标记，不是视频文件本身。
- `\n`：换行。
- `PG_PROMPT`：例如“请根据视频生成摘要问题计划”。
- `plan_text or ""`：有教师 plan 就填入，没有则使用空字符串。

真正的视频通过另一个参数传入：

```python
inputs = processor(
    messages,
    images=None,
    videos=[video_frames],
)
```

`video_frames` 是一段视频的图片列表。外面的 `[...]` 表示这条对话包含的视频列表，**不是多条独立样本的 batch**。

这里所谓的“答案”，就是 DeepSeek 预生成的教师 plan。虽然 plan 是一组问题，但用户任务是“生成问题计划”，因此这组问题就是任务的标准回答。

## 3. processor 输出什么？

| 字段 | 单样本形状 | 含义 |
|---|---|---|
| `input_ids` | `(1, L)` | 文字、对话标记和媒体标记的 token 编号 |
| `pixel_values` | `(T, 3, H, W)` | 缩放、归一化等处理后的视频画面像素 |
| `media_offset` | `(1, L, 2)` | 本地 Owl3 使用的文本与视觉输入对应信息 |

`L` 是文本序列 token 数，`T` 是抽帧数。token 不一定等于一个单词。

此时 `pixel_values` 仍是像素张量，还不是模型理解出来的语义特征；`input_ids` 也只是编号，进入模型后才会通过 embedding 层转换成向量。

## 4. processor.inference_mode 到底控制什么？

**它控制对话编码格式，尤其是是否保留末尾 assistant 的答案及结束标记，不控制梯度，也不自动启动训练。**

假设原消息是：

```text
用户：[视频] 请生成问题计划。
助手：1. 这项研究解决什么问题？
```

### True：整理成等待回答的格式

编码内容可以示意为：

```text
[用户开始]
[视频] 请生成问题计划。
[用户结束]
[助手开始]
```

末尾助手答案及结束标记被去掉，只留下助手开头，供模型继续生成。这里讨论的是本项目末尾单条助手回复的情形。

### False：保留完整对话

```text
[用户开始]
[视频] 请生成问题计划。
[用户结束]
[助手开始]
1. 这项研究解决什么问题？
[助手结束]
[文本结束]
```

答案和结束标记都会编码成 token。后续训练代码可以据此构造监督标签。

上面使用可读名称示意；本地代码实际使用 `<|im_start|>`、`<|im_end|>`、`<|endoftext|>` 等标记。

### assistant 已经是空字符串，为什么还要 True？

空字符串只表示“没有答案文字”。processor 仍可能先为这条消息添加结束标记。

```text
空 content + False：
[助手开始][助手结束][文本结束]
→ 表示已经结束的空回答。

空 content + True：
[助手开始]
→ 表示等待模型接着回答。
```

因此通常配对使用：

| 目的 | assistant 内容 | processor.inference_mode |
|---|---|---|
| 让模型生成 plan | 空字符串 | `True` |
| 准备教师 plan 训练序列 | 教师 plan 文本 | `False` |

“不保留答案”只影响此次编码，不会删除原始 plan 或磁盘里的 `plan.json`。

### previous_mode 和 try/finally

```python
previous_mode = self.processor.inference_mode
try:
    self.processor.inference_mode = plan_text is None
    inputs = self.processor(messages, images=None, videos=[video_frames])
finally:
    self.processor.inference_mode = previous_mode
```

1. `previous_mode` 只是记住原设置。
2. `plan_text is None` 在没有答案时为 `True`，有答案文本时为 `False`。
3. 暂时切换格式，完成当前编码。
4. `finally` 确保即使发生异常，也恢复原设置。

注意：`None` 与 `""` 不相同，`"" is None` 为假。这里的接口约定是“不提供答案就使用默认的 None”；有效的训练 plan 应当非空。

processor 会修改消息内容，所以 `_encode()` 每次重新创建 `messages`，避免重复处理上次已经改写的列表。

## 5. 不要混淆三种模式

| 设置 | 作用 | 不会做什么 |
|---|---|---|
| `processor.inference_mode` | 控制末尾答案和结束标记是否保留 | 不计算 loss，不控制梯度 |
| `model.train()` / `model.eval()` | 切换 Dropout 等层的训练/评估行为 | 不自动开关梯度，不更新参数 |
| `torch.inference_mode()` | 关闭梯度记录等，用于推理计算 | 不等于 `model.eval()` |

`torch.no_grad()` 也关闭梯度记录。当前训练路径中，它用于冻结的视觉编码器；后续投影层位于该上下文之外，仍然可以计算梯度。

## 6. Owl3 如何生成 plan？

```text
pixel_values → 视觉编码器 → 视觉特征 → 视觉投影层 ──┐
                                                 ↓
input_ids → 文字 embedding ─────────────→ 多模态语言模型
                                                 ↓
                                     下一个 token 的预测分数
```

`generate()` 将预测分数用于选择下一个 token，再继续预测。按单词粗略示意：

```text
视频 + 指令                    → What
视频 + 指令 + What             → problem
视频 + 指令 + What problem     → does
……
```

遇到结束标记或达到生成长度上限就停止。tokenizer 将输出编号解码成文字。实际 token 切分不一定与上例单词边界一致。

## 7. 训练为什么不是先 generate，再比较文字？

PG 训练目标是：给定视频和指令，学习输出教师 plan。我们采用监督微调，直接比较每个位置的预测与教师 token。

假设教师答案是 `What problem is addressed?`，按单词示意：

| 可用上下文 | 正确的下一个词 |
|---|---|
| 视频 + 指令 | What |
| 视频 + 指令 + What | problem |
| 视频 + 指令 + What problem | is |

训练使用教师答案的前缀，这叫 **teacher forcing**。因果注意力阻止模型看见后面的答案；各位置通常能在一次前向计算中并行处理。

若正确词的预测概率为 0.2，该位置交叉熵为 `-ln(0.2)`；正确词概率提高到 0.8，损失就降低。

`generate()` 会离散地选择 token 并得到文本，不能直接用普通反向传播把最终字符串之间的差异传回模型。因此“生成完整 plan，再比较质量”主要用于评估；当前训练对 token 预测进行监督。

## 8. 当前 PG 类如何构建训练输入？

```text
prepare_training_inputs(video_frames, plan)
    ├─ 教师问题列表 → 带编号的 plan_text
    ├─ _encode(video_frames) → 计算提示词前缀长度
    ├─ _encode(video_frames, plan_text) → 编码完整问答
    └─ 构建 labels
```

假设前缀有 40 个 token，答案及结束标记有 20 个：

```text
input_ids：(1, 60)
[提示词等内容：40 个][教师答案与结束标记：20 个]

labels：(1, 60)
[-100：40 个        ][教师答案与结束标记的 token ID]
```

- `-100` 表示该位置不参与 loss，不表示模型看不到这部分输入。
- `labels` 是 `input_ids` 的副本，避免把模型输入本身改坏。
- 本地语言模型内部会将预测和标签错开一位，执行“预测下一个 token”的计算，不需要手动再次错位。
- 两次编码是为了清楚地确定前缀边界；当前实现也会重复预处理图片，后续可以再优化。

## 9. 一次训练与一次推理

训练步骤示意，前提是已经配置 optimizer：

```python
pg.train()
optimizer.zero_grad()
outputs = pg(video_frames, teacher_plan)  # 调用 forward，计算 loss
loss = outputs.loss
loss.backward()                          # 计算梯度
optimizer.step()                         # 更新参数
```

`forward()` 自己不会调用 `backward()` 或更新参数。当前实现冻结视觉编码器，语言模型与投影层可训练，尚未接入 LoRA。

推理：

```python
questions = pg.generate_plan(video_frames)
```

这时不提供教师答案，模型使用自己已经生成的前文继续写问题。

`batch_size=1` 表示一次取一条视频样本。DataLoader 分组不等于模型已支持多样本张量组批；当前 PG 接口按单条视频实现。

## 10. 最后复习

- **教师 plan 是 PG 任务的标准答案，即使它本身由问题组成。**
- **processor 整理输入，Owl3 进行预测，generate 组织连续生成。**
- **processor.inference_mode=True：停在助手开头，等待回答。**
- **processor.inference_mode=False：保留教师答案及结束标记，供训练使用。**
- **训练比较 token 预测与教师 token，评估时才让模型自行生成完整 plan。**
- **PG 的条件是视频和任务指令；摘要用于离线制作教师 plan，不应作为 PG 的用户输入。**
