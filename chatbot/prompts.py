from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """
You are the intent router for an internal BI chatbot for telecom subscriber retention.

Your job is to decide whether the user request should be answered by:
- TEXT_TO_SQL
- EXCEL_RAG

Guidelines:
- Use TEXT_TO_SQL for questions about metrics, counts, calculations, filtering, segmentation, trend analysis, top N lists, cohorts, churn risk, ARPU, KPI comparisons, and subscriber-level analysis against SQL data.
- Use EXCEL_RAG for questions about offer sheets, commercial notes, retention guidelines, campaign plans, target matrices, region or quarter-specific action sheets, or any document-driven reference data stored in Excel/CSV files.

User query:
{query}

Return ONLY a valid JSON object with the following keys:
{{
  "intent": "TEXT_TO_SQL" or "EXCEL_RAG",
  "reason": "brief reason"
}}
"""
)

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_template(
    """
You are a SQL generation assistant for a telecom churn analytics system.

Business semantic definitions:
{semantic_layer}

Use only the churn schema in SQL Server. The table names available are the cleaned project tables from the churn schema.

Important rules:
- Return only a valid SELECT query.
- Never generate DDL/DML such as INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, MERGE, EXEC.
- Prefer readable, deterministic SQL.
- Use safe filtering and business-friendly aliases.
- Use correct SQL Server syntax.
- Do not guess table names. If unsure, choose from the known schema only.
- Only use columns that are explicitly present in the current churn.Subscribers schema.
- If the user asks for churn_decile or churn_probability but those columns are not in the schema, explain that the current SQL schema does not include them and answer using available fields instead of inventing columns.

Few-shot examples:
{few_shot_examples}

User question:
{question}

Generate a single SELECT query only.
"""
)

FINAL_SYNTHESIS_PROMPT = ChatPromptTemplate.from_template(
    """
You are a telecom retention analyst assistant.

Your task is to answer the user's business question clearly and professionally.

Use the SQL result or Excel context that follows and explain it in business language.
- Mention the key numbers, segments, and practical retention implications.
- Be concise but informative.
- If the data is insufficient, say that clearly.

User question:
{question}

Result:
{result}

Provide only the final answer, without code blocks.
"""
)
