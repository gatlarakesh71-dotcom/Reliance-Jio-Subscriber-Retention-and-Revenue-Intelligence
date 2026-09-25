from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from langchain_openai import AzureChatOpenAI, ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent
API_KEY_PATH = BASE_DIR / "API_Key.txt"
EXCEL_DOCS_DIR = BASE_DIR / "excel_docs"

SQL_SERVER = r"localhost\RAKESHSQLEXPRESS"
SQL_DATABASE = "JioChurnDB"
SQL_DRIVER = "{ODBC Driver 18 for SQL Server}"

MODEL_NAME = "gpt-4o"
TEMPERATURE = 1.0


def parse_api_file() -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not API_KEY_PATH.exists():
        return values

    for raw_line in API_KEY_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "API Key" in line:
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().replace("PASTE_YOUR_OPENAI_KEY_HERE", "").strip()
        if key and value:
            values[key] = value
    return values


def read_api_key() -> str:
    values = parse_api_file()
    if values.get("AZURE_OPENAI_API_KEY"):
        return values["AZURE_OPENAI_API_KEY"]
    if values.get("OPENAI_API_KEY"):
        return values["OPENAI_API_KEY"]
    if values.get("API_KEY"):
        return values["API_KEY"]
    raise ValueError(
        "No valid API key found in API_Key.txt. Add an OpenAI or Azure OpenAI key in the expected KEY=VALUE format."
    )


def get_llm():
    values = parse_api_file()

    if values.get("AZURE_OPENAI_ENDPOINT") and values.get("AZURE_OPENAI_API_KEY"):
        deployment = (
            values.get("AZURE_OPENAI_DEPLOYMENT_LARGE")
            or values.get("AZURE_OPENAI_DEPLOYMENT_MINI")
            or MODEL_NAME
        )
        return AzureChatOpenAI(
            azure_endpoint=values["AZURE_OPENAI_ENDPOINT"],
            api_key=values["AZURE_OPENAI_API_KEY"],
            api_version=values.get(
                "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_deployment=deployment,
            temperature=TEMPERATURE,
            timeout=120,
        )

    api_key = read_api_key()
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=TEMPERATURE,
        request_timeout=120,
    )


def get_sql_connection_string() -> str:
    return (
        f"DRIVER={SQL_DRIVER};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
        "Trusted_Connection=yes;"
        "TrustServerCertificate=yes;"
        "Encrypt=yes;"
    )


SEMANTIC_LAYER = {
    "Churn Risk": "A subscriber is classified as churn-risk if the churn propensity score is above the model decision threshold or if they fall in the top risk deciles.",
    "High-Value Subscriber": "A high-value subscriber is one whose revenue contribution is above the 75th percentile of monthly ARPU or whose annual contract contribution exceeds the business threshold used in the retention plan.",
    "ARPU": "Average Revenue Per User, calculated as the average revenue per active subscriber over the selected period.",
    "Retention Offer": "A commercial action designed to reduce churn risk by incentivizing continuity, recharge behavior, device upgrade, or service bundling.",
    "Risk Decile": "A rank-based segmentation from 1 (lowest risk) to 10 (highest risk), used to prioritize interventions.",
}

FEW_SHOT_SQL_EXAMPLES = [
    {
        "question": "How many subscribers are in the top 2 churn risk deciles?",
        "sql": "SELECT COUNT(*) AS subscriber_count FROM churn.Subscribers WHERE churn_decile IN (9, 10);",
    },
    {
        "question": "What is the average ARPU for high-risk subscribers in Bihar?",
        "sql": "SELECT AVG(arpu_last_month_inr) AS avg_arpu FROM churn.Subscribers WHERE churn_decile >= 8 AND circle = 'Bihar & Jharkhand';",
    },
    {
        "question": "List the top 10 subscribers with highest churn probability.",
        "sql": "SELECT TOP 10 subscriber_id, churn_probability, churn_decile FROM churn.Subscribers ORDER BY churn_probability DESC;",
    },
]
