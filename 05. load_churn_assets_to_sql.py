"""
Load the Jio churn project assets into SQL Server.

Default target:
    localhost\\RAKESHSQLEXPRESS / JioChurnDB

Loaded objects:
    churn.Subscribers              cleaned subscribers.csv
    churn.ModelRegistry             XGBoost model, config, and metrics
    churn.SHAPFeatureImportance     global SHAP importance
    churn.SHAPLocalExplanations     top local SHAP drivers per subscriber
    churn.SHAPValues                wide SHAP value table
    churn.AssetFiles                model/report files as VARBINARY(MAX)

Run from the project folder:
    python "05. load_churn_assets_to_sql.py"

Use --replace to replace existing project tables. Without --replace, existing
subscriber/SHAP tables are preserved and the script stops before destructive work.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus

import pandas as pd
from sqlalchemy import create_engine, text


SCHEMA_SQL = """
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = N'churn')
    EXEC(N'CREATE SCHEMA churn');

IF OBJECT_ID(N'churn.ModelRegistry', N'U') IS NULL
CREATE TABLE churn.ModelRegistry (
    model_id INT IDENTITY(1,1) PRIMARY KEY,
    model_name NVARCHAR(100) NOT NULL,
    model_type NVARCHAR(50) NOT NULL,
    model_version NVARCHAR(100) NOT NULL,
    model_file_name NVARCHAR(260) NOT NULL,
    model_binary VARBINARY(MAX) NOT NULL,
    config_json NVARCHAR(MAX) NULL,
    metrics_json NVARCHAR(MAX) NULL,
    decision_threshold DECIMAL(18,10) NULL,
    created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
);

IF OBJECT_ID(N'churn.AssetFiles', N'U') IS NULL
CREATE TABLE churn.AssetFiles (
    asset_id BIGINT IDENTITY(1,1) PRIMARY KEY,
    model_id INT NULL,
    asset_name NVARCHAR(260) NOT NULL,
    asset_type NVARCHAR(30) NOT NULL,
    content VARBINARY(MAX) NOT NULL,
    created_at DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME(),
    CONSTRAINT FK_AssetFiles_Model FOREIGN KEY (model_id) REFERENCES churn.ModelRegistry(model_id)
);
"""

TABLES = [
    "Subscribers",
    "SHAPFeatureImportance",
    "SHAPLocalExplanations",
    "SHAPValues",
]


def make_engine(server: str, database: str):
    connection = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={server};DATABASE={database};"
        "Trusted_Connection=yes;TrustServerCertificate=yes;"
    )
    return create_engine(
        "mssql+pyodbc:///?odbc_connect=" + quote_plus(connection),
        fast_executemany=True,
    )


def ensure_database(server: str, database: str) -> None:
    master = make_engine(server, "master")
    with master.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        exists = connection.execute(
            text("SELECT 1 FROM sys.databases WHERE name = :name"),
            {"name": database},
        ).scalar()
        if not exists:
            safe_name = database.replace("]", "]]")
            connection.execute(text(f"CREATE DATABASE [{safe_name}]"))
            print(f"Created database: {database}")
        else:
            print(f"Using existing database: {database}")


def table_exists(engine, table: str) -> bool:
    with engine.connect() as connection:
        return bool(connection.execute(
            text("SELECT OBJECT_ID(:object_name, 'U')"),
            {"object_name": f"churn.{table}"},
        ).scalar())


def reset_table(engine, table: str, replace: bool) -> None:
    if table_exists(engine, table):
        if not replace:
            raise RuntimeError(
                f"churn.{table} already exists. Use --replace to replace project tables."
            )
        with engine.begin() as connection:
            connection.execute(text(f"DROP TABLE churn.[{table}]"))


def load_csv(engine, path: Path, table: str, replace: bool, chunksize: int = 500) -> int:
    reset_table(engine, table, replace)
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    print(f"Loading {path.name} into churn.{table} ({len(frame):,} rows)")
    frame.to_sql(table, engine, schema="churn", if_exists="replace", index=False,
                 chunksize=chunksize)
    print(f"Loaded {len(frame):,} rows into churn.{table}")
    return len(frame)


def load_model(engine, model_dir: Path, replace: bool) -> int:
    model_path = model_dir / "xgboost_churn_model.json"
    config_path = model_dir / "model_config.json"
    metrics_path = model_dir / "metrics_summary.csv"
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"Missing model artifacts in {model_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = pd.read_csv(
        metrics_path) if metrics_path.is_file() else pd.DataFrame()
    test_tuned = metrics[(metrics.get("split") == "Test") &
                         (metrics.get("threshold_type") == "tuned")]
    metric_values = test_tuned.iloc[0].to_dict(
    ) if not test_tuned.empty else {}
    model_version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    with engine.begin() as connection:
        if replace:
            connection.execute(text("DELETE FROM churn.AssetFiles"))
            connection.execute(text("DELETE FROM churn.ModelRegistry"))
        elif connection.execute(text("SELECT COUNT(*) FROM churn.ModelRegistry")).scalar():
            raise RuntimeError(
                "churn.ModelRegistry already contains a model. Use --replace.")
        result = connection.execute(text("""
            INSERT INTO churn.ModelRegistry
                (model_name, model_type, model_version, model_file_name,
                 model_binary, config_json, metrics_json, decision_threshold)
            OUTPUT INSERTED.model_id
            VALUES
                (:name, :type, :version, :file_name, :binary, :config, :metrics, :threshold)
        """), {
            "name": "Jio 90-day churn XGBoost",
            "type": "xgboost",
            "version": model_version,
            "file_name": model_path.name,
            "binary": model_path.read_bytes(),
            "config": json.dumps(config),
            "metrics": json.dumps(metric_values, default=str),
            "threshold": config.get("decision_threshold"),
        })
        model_id = result.scalar_one()

    print(f"Registered model_id={model_id}: {model_path.name}")
    return model_id


def load_assets(engine, model_dir: Path, shap_dir: Path, model_id: int, replace: bool) -> None:
    files = list(model_dir.glob("*.json")) + list(shap_dir.glob("*.png"))
    files += list(shap_dir.glob("*.txt"))
    with engine.begin() as connection:
        for path in files:
            connection.execute(text("""
                INSERT INTO churn.AssetFiles (model_id, asset_name, asset_type, content)
                VALUES (:model_id, :asset_name, :asset_type, :content)
            """), {
                "model_id": model_id,
                "asset_name": path.name,
                "asset_type": path.suffix.lstrip(".").lower() or "file",
                "content": path.read_bytes(),
            })
    print(f"Stored {len(files)} model/SHAP files in churn.AssetFiles")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load churn project assets into SQL Server")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--server", default=r"localhost\RAKESHSQLEXPRESS")
    parser.add_argument("--database", default="JioChurnDB")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    data_path = script_dir / "Cleaned_File" / "subscribers.csv"
    model_dir = script_dir / "Output" / "03. Xgboost"
    shap_dir = script_dir / "Output" / "04. SHAP"
    ensure_database(args.server, args.database)
    engine = make_engine(args.server, args.database)
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
        connection.exec_driver_sql(SCHEMA_SQL)
    print("SQL schema is ready")

    subscribers_count = load_csv(
        engine, data_path, "Subscribers", args.replace)
    load_csv(engine, shap_dir / "shap_feature_importance.csv",
             "SHAPFeatureImportance", args.replace)
    load_csv(engine, shap_dir / "shap_local_top15_by_subscriber.csv",
             "SHAPLocalExplanations", args.replace)
    load_csv(engine, shap_dir / "shap_values.csv", "SHAPValues", args.replace)
    model_id = load_model(engine, model_dir, args.replace)
    load_assets(engine, model_dir, shap_dir, model_id, args.replace)

    with engine.connect() as connection:
        counts = {
            table: connection.execute(
                text(f"SELECT COUNT(*) FROM churn.[{table}]")).scalar()
            for table in TABLES
        }
    print(f"SQL Server load complete: {subscribers_count:,} subscribers")
    print("Table counts:", counts)


if __name__ == "__main__":
    main()
