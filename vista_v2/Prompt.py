def get_plan_generation_prompt():
    return """
The images are sampled frames from ONE scientific presentation,
provided in chronological order.

Create an ordered plan of 3–6 questions for a scientific summary.
Focus on the research problem, proposed method, experiments,
and main findings, where supported by the frames.

Questions must be grounded in the visible content.
Do not assume results or details that are not shown.
Return only a numbered list of questions in English.
"""

def get_summary_generation_prompt(plan:str)->str:
    return f"""
The images are sampled frames from ONE scientific presentation,
provided in chronological order.

Write a coherent scientific abstract in English, approximately
150–200 words, using the following questions as an organizing plan.

Plan:
{plan}

Use only information supported by the frames.
The plan is guidance, not factual evidence.
Omit unsupported details rather than inventing answers.
Do not invent numerical results, dataset names, or claims.
Return only the abstract, without headings or a question list.
"""