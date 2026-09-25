from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import pyodbc

from config import SEMANTIC_LAYER, SQL_DATABASE, SQL_DRIVER, SQL_SERVER, get_sql_connection_string

BLOCKED_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "MERGE",
    "EXEC",
    "EXECUTE",
    "GRANT",
    "REVOKE",
    "COMMIT",
    "ROLLBACK",
]


class SQLGuardrailError(ValueError):
    pass


def validate_read_only_query(query: str) -> str:
    clean = " ".join(query.strip().split())
    normalized = re.sub(r"\s+", " ", clean).upper()

    for keyword in BLOCKED_KEYWORDS:
        if keyword in normalized:
            raise SQLGuardrailError(
                "The query was rejected because it contains a disallowed DDL/DML operation: "
                f"{keyword}. Only read-only SELECT queries are allowed."
            )

    if not normalized.startswith("SELECT"):
        raise SQLGuardrailError(
            "Only SELECT statements are permitted for the SQL agent.")

    return clean


def get_sql_connection(read_only: bool = True) -> pyodbc.Connection:
    try:
        conn = pyodbc.connect(
            get_sql_connection_string(),
            autocommit=True,
        )
        return conn
    except pyodbc.Error as exc:
        raise ConnectionError(
            f"Failed to connect to SQL Server: {exc}") from exc


def ensure_analytics_view() -> None:
    conn = get_sql_connection(read_only=False)
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "IF OBJECT_ID('churn.vw_subscribers_analytics', 'V') IS NOT NULL DROP VIEW churn.vw_subscribers_analytics;")
            cursor.execute("""
                CREATE VIEW churn.vw_subscribers_analytics AS
                SELECT
                    subscriber_id,
                    circle,
                    zone,
                    join_date,
                    tenure_months,
                    plan_type,
                    plan_price_inr,
                    plan_validity_days,
                    arpu_last_month_inr,
                    arpu_3m_avg_inr,
                    arpu_6m_avg_inr,
                    recharge_count_6m,
                    avg_recharge_gap_days,
                    days_since_last_recharge,
                    payment_failures_6m,
                    autopay_enabled,
                    data_gb_last_month,
                    data_gb_3m_avg,
                    voice_minutes_last_month,
                    sms_count_last_month,
                    device_brand,
                    is_5g_device,
                    is_5g_active,
                    avg_sinr_db,
                    drop_call_rate_pct,
                    site_congestion_score,
                    complaints_6m,
                    unresolved_complaints,
                    avg_resolution_days,
                    home_product,
                    num_services,
                    family_plan_flag,
                    app_logins_30d,
                    outgoing_to_competitor_pct,
                    roaming_user_flag,
                    offer_exposed_90d,
                    offer_redeemed_90d,
                    mnp_enquiry_flag,
                    churn_flag_30d,
                    churn_flag_90d,
                    CASE WHEN churn_flag_90d = 1 THEN 'High Risk' ELSE 'Low Risk' END AS churn_risk_label,
                    CASE WHEN arpu_last_month_inr >= 250 THEN 1 ELSE 0 END AS high_value_flag,
                    CASE WHEN churn_flag_90d = 1 AND arpu_last_month_inr >= 250 THEN 1 ELSE 0 END AS high_value_high_risk_flag
                FROM churn.Subscribers;
            """)
            conn.commit()
        print("Created churn.vw_subscribers_analytics")
    except Exception as exc:
        raise RuntimeError(f"Failed to create analytics view: {exc}") from exc
    finally:
        conn.close()


def execute_read_query(query: str) -> pd.DataFrame:
    safe_query = validate_read_only_query(query)
    conn = get_sql_connection(read_only=True)
    try:
        return pd.read_sql(safe_query, conn)
    except Exception as exc:
        raise RuntimeError(f"SQL execution failed: {exc}") from exc
    finally:
        conn.close()


def get_table_schema(table_name: str) -> List[Dict[str, Any]]:
    conn = get_sql_connection(read_only=True)
    try:
        query = f"SELECT TOP 0 * FROM {table_name};"
        df = pd.read_sql(query, conn)
        return df.dtypes.astype(str).reset_index().rename(columns={"index": "column_name", 0: "dtype"}).to_dict("records")
    except Exception:
        return []
    finally:
        conn.close()


def get_table_columns(table_name: str) -> List[str]:
    schema = get_table_schema(table_name)
    if not schema:
        return []
    return [str(row["column_name"]).lower() for row in schema]


def get_real_schema_context() -> str:
    ensure_analytics_view()
    table_name = "churn.vw_subscribers_analytics"
    columns = get_table_columns(table_name)
    if not columns:
        return "No schema metadata was available from the SQL server for churn.vw_subscribers_analytics."
    return "Available columns in churn.vw_subscribers_analytics: " + ", ".join(columns)


def validate_schema_compatibility(question: str, table_name: str = "churn.vw_subscribers_analytics") -> None:
    lowered = question.lower()
    columns = set(get_table_columns(table_name))

    if any(keyword in lowered for keyword in ["churn decile", "risk decile", "churn probability", "propensity score", "risk score"]):
        safe_fields = ["churn_flag_90d", "churn_risk_label", "high_value_flag",
                       "high_value_high_risk_flag", "arpu_last_month_inr"]
        missing = [field for field in safe_fields if field not in columns]
        if missing:
            raise ValueError(
                "The current SQL schema exposes churn risk through churn_flag_90d and churn_risk_label, not churn_decile or churn_probability. Use those real fields for an accurate answer."
            )


def get_semantic_context() -> str:
    return "\n".join(
        [f"- {name}: {definition}" for name,
            definition in SEMANTIC_LAYER.items()]
    )


def get_sql_examples() -> str:
    examples = []
    for item in [
        {"question": "How many high-risk subscribers are there?",
            "sql": "SELECT COUNT(*) AS high_risk_count FROM churn.vw_subscribers_analytics WHERE churn_flag_90d = 1;"},
        {"question": "What is the average ARPU by circle?",
            "sql": "SELECT circle, AVG(arpu_last_month_inr) AS avg_arpu FROM churn.vw_subscribers_analytics GROUP BY circle ORDER BY avg_arpu DESC;"},
        {"question": "Which circles have the highest share of high-value subscribers?",
            "sql": "SELECT circle, SUM(CASE WHEN high_value_flag = 1 THEN 1 ELSE 0 END) AS high_value_subscribers FROM churn.vw_subscribers_analytics GROUP BY circle ORDER BY high_value_subscribers DESC;"},
    ]:
        examples.append(f"Question: {item['question']}\nSQL: {item['sql']}")
    return "\n\n".join(examples)


def get_metadata_summary() -> Dict[str, Any]:
    conn = get_sql_connection(read_only=True)
    try:
        tables = pd.read_sql(
            "SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE' AND TABLE_SCHEMA = 'churn';",
            conn,
        )
        return {
            "database": SQL_DATABASE,
            "server": SQL_SERVER,
            "table_count": int(len(tables)),
            "tables": tables.to_dict("records"),
        }
    except Exception as exc:
        return {
            "database": SQL_DATABASE,
            "server": SQL_SERVER,
            "table_count": 0,
            "tables": [],
            "error": str(exc),
        }
    finally:
        conn.close()
