"""教师、PG 和 SG 的提示词模板。"""

QUESTION_PROMPT = (
    "Generate a coherent and contextually relevant question based on the provided context and target sentence, "
    "ensuring that the target sentence can be treated as an answer to the generated question.\n"
    "Output only one question in plain text.Do not include Markdown formatting, labels, numbering, quotation marks, explanations, or answers."
    "Context: {context}Target: {target}Question Sentence:"
)

PG_PROMPT = (
    "Generate a list of questions for the provided video.\n"
    "Content: <|video|>\n"
    "Questions:"
)

SG_PROMPT = (
    "Generate a summary for the following video based on the plan questions.\n"
    "Content: <|video|>\n"
    "Plan Questions:\n{questions}\n"
    "Ensure that the generated summary sequentially answers the plan questions.\n"
    "Summary:"
)
