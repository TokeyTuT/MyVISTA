"""
参照原论文，使用 ROUGE-1 F1	ROUGE-2 F1	ROUGE-Lsum F1 生成结果与样本中原论文摘要进行对比评估。
"""


import pysbd
from rouge_score import rouge_scorer


segmenter = pysbd.Segmenter(language="en", clean=False)

scorer = rouge_scorer.RougeScorer(
    ["rouge1", "rouge2", "rougeLsum"],
    use_stemmer=True,  # 英文词干归一化，例如 improves 与 improve
)


def prepare_summary(text):
    """将摘要分句，每句一行，供 ROUGE-Lsum 计算。"""
    if not isinstance(text, str):
        raise TypeError("摘要必须是字符串")

    # 先合并排版换行，再按句子重新分行
    text = " ".join(text.split())

    sentences = segmenter.segment(text)
    return "\n".join(s.strip() for s in sentences if s.strip())


def evaluate_rouge(prediction, abstract):
    """
    prediction: 模型生成的摘要
    abstract: 数据集中的参考摘要

    返回原始 0～1 尺度的 Precision、Recall 和 F1。
    """
    if not isinstance(abstract, str) or not abstract.strip():
        raise ValueError("参考摘要不能为空")

    scores = scorer.score(
        target=prepare_summary(abstract),
        prediction=prepare_summary(prediction),
    )

    return {
        name: {
            "precision": float(score.precision),
            "recall": float(score.recall),
            "f1": float(score.fmeasure),
        }
        for name, score in scores.items()
    }