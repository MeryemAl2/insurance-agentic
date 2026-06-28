ANALYZE_QUESTION_PROMPT = """You are an insurance analyst.
Extract the user's intent, insurance line, and likely information need.
Return concise JSON with keys: intent, insurance_line, entities, risk_flags.

Question:
{question}
"""

GRADE_DOCUMENTS_PROMPT = """You are grading retrieval quality for an insurance RAG system.
Score each document from 0.0 to 1.0 for relevance to the question.
Return JSON: {{"scores":[{{"index":0,"score":0.0,"reason":"..."}}],"overall_score":0.0}}

Question:
{question}

Documents:
{documents}
"""

REWRITE_QUESTION_PROMPT = """Rewrite the insurance question for better document retrieval.
Keep all important entities and constraints. Make it direct and search-friendly.

Original question:
{question}

Current weak evidence:
{documents}
"""

SUMMARIZE_DOCUMENTS_PROMPT = """Summarize the relevant insurance document excerpts for answering the question.
Focus on coverage rules, exclusions, waiting periods, claim steps, required evidence, and regulatory constraints.

Question:
{question}

Relevant excerpts:
{documents}
"""

GENERATE_ANSWER_PROMPT = """You are an insurance assistant.
Answer using only the provided document summary and excerpts. If evidence is missing, say what is missing.
Do not invent policy terms. Mention that final decisions depend on the specific policy wording when appropriate.

Question:
{question}

Memory:
{memory}

Document summary:
{summary}

Relevant excerpts:
{documents}
"""

SELF_EVALUATE_PROMPT = """Evaluate the answer for insurance RAG quality.
Return JSON with keys: quality_score, groundedness_score, completeness_score, missing_info, final_verdict.

Question:
{question}

Retrieved evidence:
{documents}

Answer:
{answer}
"""

