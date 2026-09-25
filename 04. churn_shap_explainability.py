"""
SHAP explainability for the saved XGBoost churn model.

This script explains the existing model produced by:
    03. churn_xgboost_model.py

It reproduces the original leakage-safe feature preparation and test split,
then writes global and local explanations to Output/04. SHAP.

Install once inside JIo_venv:
    python -m pip install shap

Run from the project folder:
    python "04. churn_shap_explainability.py"

Optional arguments:
    --data path/to/subscribers.csv
    --model-dir "Output/03. Xgboost"
    --output-dir "Output/04. SHAP"
    --max-samples 5000
"""

from __future__ import annotations
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")


SEED = 42
TARGET = "churn_flag_90d"
LEAKY_COLS = ["churn_reason", "churn_date", "churn_flag_30d"]
ID_COLS = ["subscriber_id"]
REDUNDANT_COLS = ["circle_code", "join_date"]
CATEGORICAL_COLS = ["circle", "zone",
                    "device_brand", "home_product", "plan_type"]


def find_path(cli_value: str | None, candidates: list[Path], label: str) -> Path:
    paths = [Path(cli_value)] if cli_value else []
    paths.extend(candidates)
    for path in paths:
        if path.is_file():
            return path
    tried = "\n  ".join(str(path) for path in paths)
    raise FileNotFoundError(f"{label} not found. Tried:\n  {tried}")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for column in df.select_dtypes(include=["object", "string"]).columns:
        values = set(df[column].dropna().unique())
        if values <= {"True", "False"}:
            df[column] = df[column] == "True"
    return df


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    drop_cols = [TARGET, *LEAKY_COLS, *ID_COLS, *REDUNDANT_COLS]
    X = df.drop(
        columns=[column for column in drop_cols if column in df.columns]).copy()
    y = df[TARGET].astype(int)
    ids = df["subscriber_id"].copy()

    leaked = [column for column in LEAKY_COLS +
              [TARGET] if column in X.columns]
    if leaked:
        raise ValueError(
            f"Leakage columns reached the SHAP feature frame: {leaked}")

    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(int)
    return X, y, ids


def encode_features(X_raw: pd.DataFrame, categorical_levels: dict[str, list[str]]) -> pd.DataFrame:
    parts = [X_raw.drop(columns=CATEGORICAL_COLS)]
    for column in CATEGORICAL_COLS:
        levels = categorical_levels[column]
        category = pd.Categorical(X_raw[column].astype(str), categories=levels)
        dummies = pd.get_dummies(category, prefix=column, dtype=int)
        dummies.index = X_raw.index
        parts.append(dummies)
    return pd.concat(parts, axis=1)


def reproduce_test_split(X: pd.DataFrame, y: pd.Series, ids: pd.Series):
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        X, y, ids, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )
    return X_test, y_test, id_test


def select_shap_values(explanation: shap.Explanation) -> np.ndarray:
    values = np.asarray(explanation.values)
    if values.ndim == 3:
        # SHAP versions that return one matrix per class: retain churn class.
        values = values[:, :, 1]
    if values.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")
    return values


def save_summary_plots(explanation: shap.Explanation, feature_names: list[str], output_dir: Path) -> None:
    shap.summary_plot(explanation.values, explanation.data, feature_names=feature_names,
                      max_display=20, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_bar.png",
                dpi=180, bbox_inches="tight")
    plt.close()

    shap.summary_plot(explanation.values, explanation.data, feature_names=feature_names,
                      max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_beeswarm.png",
                dpi=180, bbox_inches="tight")
    plt.close()


def save_local_plot(explanation: shap.Explanation, row_index: int,
                    output_dir: Path, title: str) -> None:
    row = explanation[row_index]
    shap.plots.waterfall(row, max_display=15, show=False)
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_waterfall_highest_risk.png",
                dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain the saved XGBoost churn model with SHAP")
    script_dir = Path(__file__).resolve().parent
    parser.add_argument("--data", default=None, help="path to subscribers.csv")
    parser.add_argument("--model-dir", default=None,
                        help="directory containing model JSON and config")
    parser.add_argument("--output-dir", default=None,
                        help="directory for SHAP outputs")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="maximum number of test rows to explain")
    args = parser.parse_args()

    data_path = find_path(
        args.data, [script_dir / "Cleaned_File" / "subscribers.csv"], "Data file")
    model_dir = Path(
        args.model_dir) if args.model_dir else script_dir / "Output" / "03. Xgboost"
    model_path = model_dir / "xgboost_churn_model.json"
    config_path = model_dir / "model_config.json"
    output_dir = Path(
        args.output_dir) if args.output_dir else script_dir / "Output" / "04. SHAP"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            f"Saved XGBoost artifacts are required in {model_dir}. "
            "Run 03. churn_xgboost_model.py first."
        )

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    df = load_data(data_path)
    X_raw, y, ids = build_feature_frame(df)
    X_encoded = encode_features(X_raw, config["categorical_levels"])
    feature_names = config["feature_names"]
    missing = [
        feature for feature in feature_names if feature not in X_encoded.columns]
    if missing:
        raise ValueError(f"Encoded data is missing model features: {missing}")
    X_encoded = X_encoded.reindex(
        columns=feature_names, fill_value=0).astype(float)

    X_test, y_test, id_test = reproduce_test_split(X_encoded, y, ids)
    if args.max_samples < len(X_test):
        sample_indices = X_test.sample(
            args.max_samples, random_state=SEED).index
        X_explain = X_test.loc[sample_indices]
        y_explain = y_test.loc[sample_indices]
        id_explain = id_test.loc[sample_indices]
    else:
        X_explain, y_explain, id_explain = X_test, y_test, id_test

    model = xgb.XGBClassifier()
    model.load_model(str(model_path))
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_explain)
    shap_values = select_shap_values(explanation)

    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        "mean_shap": shap_values.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_dir / "shap_feature_importance.csv", index=False)

    value_table = pd.DataFrame(shap_values, columns=feature_names)
    value_table.insert(0, "subscriber_id", id_explain.to_numpy())
    value_table.insert(1, "actual_churn_90d", y_explain.to_numpy())
    value_table.to_csv(output_dir / "shap_values.csv", index=False)

    probabilities = model.predict_proba(X_explain)[:, 1]
    local_rows = []
    for row_position, subscriber_id in enumerate(id_explain):
        top_features = np.argsort(np.abs(shap_values[row_position]))[::-1][:15]
        for rank, feature_position in enumerate(top_features, start=1):
            local_rows.append({
                "subscriber_id": subscriber_id,
                "actual_churn_90d": int(y_explain.iloc[row_position]),
                "predicted_churn_probability": float(probabilities[row_position]),
                "rank": rank,
                "feature": feature_names[feature_position],
                "feature_value": float(X_explain.iloc[row_position, feature_position]),
                "shap_value": float(shap_values[row_position, feature_position]),
                "direction": "increases churn risk" if shap_values[row_position, feature_position] > 0 else "reduces churn risk",
            })
    pd.DataFrame(local_rows).to_csv(
        output_dir / "shap_local_top15_by_subscriber.csv", index=False)

    highest_risk_position = int(np.argmax(probabilities))
    save_summary_plots(explanation, feature_names, output_dir)
    save_local_plot(
        explanation,
        highest_risk_position,
        output_dir,
        f"Highest-risk subscriber: {id_explain.iloc[highest_risk_position]}",
    )

    top_feature = importance.iloc[0]
    with (output_dir / "SHAP_RESULTS_SUMMARY.txt").open("w", encoding="utf-8") as handle:
        handle.write("SUBSCRIBER CHURN - SHAP EXPLAINABILITY\n")
        handle.write("=" * 48 + "\n")
        handle.write(
            f"Rows explained: {len(X_explain):,} of {len(X_test):,} test rows\n")
        handle.write(f"Model: {model_path}\n")
        handle.write(f"Leakage columns excluded: {', '.join(LEAKY_COLS)}\n")
        handle.write(f"Top global feature: {top_feature['feature']}\n")
        handle.write(
            f"Top feature mean absolute SHAP: {top_feature['mean_abs_shap']:.6f}\n")
        handle.write(
            f"Highest-risk subscriber: {id_explain.iloc[highest_risk_position]}\n")
        handle.write(
            f"Highest-risk predicted probability: {probabilities[highest_risk_position]:.6f}\n")

    print(f"SHAP analysis complete. Outputs saved to: {output_dir.resolve()}")
    print("Top 10 features by mean absolute SHAP:")
    print(importance.head(10).round(6).to_string(index=False))


if __name__ == "__main__":
    main()
