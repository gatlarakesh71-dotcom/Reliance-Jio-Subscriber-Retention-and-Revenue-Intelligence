from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from config import FEW_SHOT_SQL_EXAMPLES, SEMANTIC_LAYER, get_llm
from database import (
    SQLGuardrailError,
    ensure_analytics_view,
    execute_read_query,
    get_real_schema_context,
    get_sql_examples,
    get_semantic_context,
    validate_schema_compatibility,
)
from excel_rag import ExcelRAG
from prompts import FINAL_SYNTHESIS_PROMPT, SQL_GENERATION_PROMPT
from router import IntentRouter
from config import EXCEL_DOCS_DIR

st.set_page_config(page_title="Jio Retention Copilot",
                   page_icon="📡", layout="wide")


def clean_sql_output(sql: str) -> str:
    sql = sql.strip()
    if sql.startswith("```"):
        sql = re.sub(r"^```(?:sql)?\s*", "", sql)
        sql = re.sub(r"\s*```$", "", sql)
    return sql.strip()


def render_dashboard() -> None:
    try:
        summary_df = execute_read_query(
            """
            SELECT
                COUNT(*) AS total_subscribers,
                CAST(AVG(CAST(churn_flag_90d AS FLOAT)) * 100 AS DECIMAL(10,2)) AS churn_rate_pct,
                AVG(arpu_last_month_inr) AS avg_arpu,
                SUM(CASE WHEN high_value_flag = 1 THEN 1 ELSE 0 END) AS high_value_count
            FROM churn.vw_subscribers_analytics;
            """
        )
        if summary_df.empty:
            return

        summary = summary_df.iloc[0]
        total_subscribers = int(summary["total_subscribers"])
        churn_rate_pct = float(summary["churn_rate_pct"])
        avg_arpu = float(summary["avg_arpu"])
        high_value_count = int(summary["high_value_count"])

        st.subheader("Executive overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Subscribers", f"{total_subscribers:,}")
        col2.metric("90-Day Churn Rate", f"{churn_rate_pct:.2f}%")
        col3.metric("Avg ARPU", f"₹{avg_arpu:,.0f}")
        col4.metric("High-Value Subscribers", f"{high_value_count:,}")

        circle_df = execute_read_query(
            """
            SELECT
                circle,
                CAST(AVG(CAST(churn_flag_90d AS FLOAT)) * 100 AS DECIMAL(10,2)) AS churn_rate_pct,
                SUM(CASE WHEN high_value_flag = 1 THEN 1 ELSE 0 END) AS high_value_subscribers
            FROM churn.vw_subscribers_analytics
            GROUP BY circle
            ORDER BY churn_rate_pct DESC;
            """
        )
        risk_df = execute_read_query(
            """
            SELECT
                churn_risk_label,
                COUNT(*) AS subscriber_count
            FROM churn.vw_subscribers_analytics
            GROUP BY churn_risk_label
            ORDER BY subscriber_count DESC;
            """
        )

        if not circle_df.empty:
            left_col, right_col = st.columns(2)
            left_col.subheader("Churn risk by circle")
            left_col.bar_chart(circle_df.set_index("circle")["churn_rate_pct"])
            right_col.subheader("High-value subscribers by circle")
            right_col.bar_chart(circle_df.set_index(
                "circle")["high_value_subscribers"])

        if not risk_df.empty:
            st.subheader("Risk segment mix")
            st.bar_chart(risk_df.set_index(
                "churn_risk_label")["subscriber_count"])

    except Exception as exc:
        st.caption(f"Dashboard summary unavailable: {exc}")


def generate_sql_query(question: str) -> str:
    validate_schema_compatibility(question)
    llm = get_llm()
    prompt = SQL_GENERATION_PROMPT.format(
        semantic_layer=get_semantic_context() + "\n\nCurrent SQL schema awareness:\n" +
        get_real_schema_context(),
        few_shot_examples=get_sql_examples(),
        question=question,
    )
    response = llm.invoke(prompt)
    sql = getattr(response, "content", str(response))
    sql = clean_sql_output(sql)
    if not sql.upper().startswith("SELECT"):
        raise ValueError("The model returned a non-SELECT SQL statement.")
    return sql


def run_sql_answer(question: str) -> Dict[str, Any]:
    sql = generate_sql_query(question)
    result_df = execute_read_query(sql)
    return {
        "sql": sql,
        "result": result_df,
        "records": result_df.to_dict(orient="records"),
    }


def answer_excel_question(question: str) -> Dict[str, Any]:
    rag = ExcelRAG(EXCEL_DOCS_DIR)
    context = rag.get_context(question, top_k=5)
    llm = get_llm()
    prompt = FINAL_SYNTHESIS_PROMPT.format(
        question=question,
        result=context,
    )
    answer = llm.invoke(prompt)
    return {
        "answer": getattr(answer, "content", str(answer)),
        "context": context,
        "source_files": [
            {"file": hit["file"], "row_index": hit["row_index"]} for hit in rag.search(question, top_k=5)
        ],
    }


def render_chatbot():
    try:
        ensure_analytics_view()
    except Exception as exc:
        st.warning(f"Analytics view initialization warning: {exc}")

    st.title("Jio Subscriber Retention Copilot")
    st.caption(
        "Internal BI Chatbot for churn analysis, retention strategy, and offer guidance")

    render_dashboard()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Ask about churn, subscribers, revenue, or offers...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        router = IntentRouter()
        route = router.route(prompt)
        intent = route["intent"]
        developer_log = []

        try:
            if intent == "TEXT_TO_SQL":
                try:
                    output = run_sql_answer(prompt)
                    sql = output["sql"]
                    result_df = output["result"]
                    developer_log.append(
                        {"intent": intent, "route_reason": route["reason"], "sql": sql})

                    synthesis_prompt = FINAL_SYNTHESIS_PROMPT.format(
                        question=prompt,
                        result=result_df.to_markdown(
                            index=False) if not result_df.empty else "No records returned.",
                    )
                    answer = get_llm().invoke(synthesis_prompt)
                    final_text = getattr(answer, "content", str(answer))

                    with st.chat_message("assistant"):
                        st.markdown(final_text)
                        with st.expander("Developer Log", expanded=False):
                            st.code(sql, language="sql")
                            st.dataframe(result_df, use_container_width=True)

                    st.session_state.messages.append(
                        {"role": "assistant", "content": final_text})

                except SQLGuardrailError as exc:
                    safe_message = f"I can only answer with read-only SQL. Details: {exc}"
                    with st.chat_message("assistant"):
                        st.warning(safe_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": safe_message})

                except Exception as exc:
                    error_message = f"I could not complete the SQL-based analysis. Error: {exc}"
                    with st.chat_message("assistant"):
                        st.error(error_message)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_message})

            else:
                excel_result = answer_excel_question(prompt)
                developer_log.append({
                    "intent": intent,
                    "route_reason": route["reason"],
                    "source_files": excel_result["source_files"],
                    "context": excel_result["context"],
                })

                with st.chat_message("assistant"):
                    st.markdown(excel_result["answer"])
                    with st.expander("Developer Log", expanded=False):
                        st.write(excel_result["source_files"])
                        st.code(excel_result["context"], language="text")

                st.session_state.messages.append(
                    {"role": "assistant", "content": excel_result["answer"]})

        except Exception as exc:
            fallback_message = f"I hit an issue while processing the request: {exc}"
            with st.chat_message("assistant"):
                st.error(fallback_message)
            st.session_state.messages.append(
                {"role": "assistant", "content": fallback_message})


if __name__ == "__main__":
    render_chatbot()
