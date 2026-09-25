"""
Subscriber Churn - Logistic Regression Baseline
================================================
Target        : churn_flag_90d  (90-day churn)
Excluded cols : churn_reason, churn_date, churn_flag_30d   (outcome-linked -> leakage)
Split         : 70% train / 15% validation / 15% test  (stratified on the target)
Model         : Logistic Regression (L2, scikit-learn default) inside a scikit-learn Pipeline
Metrics       : Accuracy, Precision, Recall, F1-score, ROC-AUC  (+ PR-AUC and confusion matrix)

How the three splits are used
-----------------------------
  train (70%) : fit the model
  val   (15%) : choose the regularisation strength C and the decision threshold
  test  (15%) : touched once, for the final unbiased report

Run (VS Code terminal or the "Run Python File" button):

Optional arguments:
    --data   path/to/subscribers.csv     (default: looks in ./Cleaned_File/, then ./)
    --output path/to/output_folder       (default: ./output next to this script)
    --seed   42

Requirements:
    pip install pandas numpy scikit-learn matplotlib joblib
"""

from __future__ import annotations
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # write plots to files; no display window needed

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
TARGET = "churn_flag_90d"

# Outcome-linked columns: they encode the answer, so they must NEVER be features.
LEAKAGE_COLS = ["churn_reason", "churn_date", "churn_flag_30d"]

# Not useful as model inputs (kept out on purpose, see README notes in the chat):
#   subscriber_id -> unique identifier
#   join_date     -> raw date; already captured by tenure_months
#   circle_code   -> exact duplicate of `circle`
NON_FEATURE_COLS = ["subscriber_id", "join_date", "circle_code"]

TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15
# regularisation strengths tried on the validation set
C_GRID = [0.01, 0.1, 1.0, 10.0]
CLASS_WEIGHT = "balanced"  # ~5% churners -> reweight classes; set to None for plain LR

log = logging.getLogger("churn_lr")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def setup_logging(output_dir: Path) -> None:
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(output_dir / "run_log.txt", mode="w", encoding="utf-8")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def find_data_file(cli_path: str | None) -> Path:
    """Locate subscribers.csv. Tries the CLI path, then common locations."""
    script_dir = Path(__file__).resolve().parent
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path))
    candidates += [
        script_dir / "Cleaned_File" / "subscribers.csv",
        script_dir / "subscribers.csv",
        Path.cwd() / "Cleaned_File" / "subscribers.csv",
        Path.cwd() / "subscribers.csv",
    ]
    for path in candidates:
        if path.is_file():
            return path
    tried = "\n  ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"subscribers.csv not found. Looked in:\n  {tried}\n"
        "Pass the location explicitly with --data path/to/subscribers.csv"
    )


def to_binary(series: pd.Series) -> pd.Series:
    """Convert a True/False style column to 0/1 integers."""
    if series.dtype == bool:
        return series.astype(int)
    mapped = series.astype(str).str.strip().str.lower().map(
        {"true": 1, "false": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    )
    if mapped.isna().any():
        raise ValueError(
            f"Column '{series.name}' contains values that are not True/False/1/0.")
    return mapped.astype(int)


def load_data(path: Path) -> pd.DataFrame:
    # keep_default_na=False: home_product has a real category called "None" which
    # pandas would otherwise misread as missing. Only empty cells are treated as NaN.
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    log.info(
        f"Loaded {path}  ->  {df.shape[0]:,} rows x {df.shape[1]} columns")
    missing = [c for c in [TARGET, *LEAKAGE_COLS] if c not in df.columns]
    if missing:
        raise KeyError(f"Expected columns not found in the CSV: {missing}")
    return df


def prepare_features(df: pd.DataFrame):
    y = to_binary(df[TARGET])
    drop_cols = [TARGET, *LEAKAGE_COLS, *NON_FEATURE_COLS]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    # Hard safety check: nothing outcome-linked may reach the model.
    leaked = set(LEAKAGE_COLS + [TARGET]) & set(X.columns)
    assert not leaked, f"Leakage columns still present in features: {leaked}"

    # True/False columns -> 0/1
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)

    numeric_cols = X.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return X, y, numeric_cols, categorical_cols


def build_pipeline(numeric_cols, categorical_cols, C: float, seed: int) -> Pipeline:
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    model = LogisticRegression(
        C=C, solver="lbfgs", max_iter=3000,
        class_weight=CLASS_WEIGHT, random_state=seed,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def compute_metrics(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1_score": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def best_f1_threshold(y_true, proba) -> float:
    """Decision threshold that maximises F1 on the given (validation) data."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / \
        np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(f1))])


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def plot_roc(curves: dict, out: Path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    for name, (y_true, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        plt.plot(fpr, tpr, lw=2,
                 label=f"{name} (AUC = {roc_auc_score(y_true, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - Logistic Regression baseline")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_pr(curves: dict, base_rate: float, out: Path) -> None:
    plt.figure(figsize=(6.5, 5.5))
    for name, (y_true, proba) in curves.items():
        p, r, _ = precision_recall_curve(y_true, proba)
        plt.plot(
            r, p, lw=2, label=f"{name} (AP = {average_precision_score(y_true, proba):.3f})")
    plt.axhline(base_rate, color="k", ls="--", lw=1,
                label=f"Base rate = {base_rate:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve - Logistic Regression baseline")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_confusion(y_true, proba, thresholds: dict, out: Path) -> None:
    fig, axes = plt.subplots(
        1, len(thresholds), figsize=(5.5 * len(thresholds), 4.8))
    axes = np.atleast_1d(axes)
    for ax, (label, thr) in zip(axes, thresholds.items()):
        cm = confusion_matrix(
            y_true, (proba >= thr).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["Retained", "Churned"])
        ax.set_yticks([0, 1], ["Retained", "Churned"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Test set - {label}\n(threshold = {thr:.3f})")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_threshold_curve(y_val, proba_val, chosen: float, out: Path) -> None:
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    f1 = 2 * precision[:-1] * recall[:-1] / \
        np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.plot(thresholds, f1, label="F1", lw=2)
    plt.axvline(chosen, color="k", ls="--",
                label=f"Chosen threshold = {chosen:.3f}")
    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title("Threshold selection on the validation set")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


def plot_coefficients(coef_df: pd.DataFrame, out: Path, top_n: int = 15) -> None:
    top = pd.concat([coef_df.head(top_n), coef_df.tail(top_n)]
                    ).drop_duplicates("feature")
    top = top.sort_values("coefficient")
    colors = np.where(top["coefficient"] > 0, "#c0392b", "#1f7a8c")
    plt.figure(figsize=(9, 8))
    plt.barh(top["feature"], top["coefficient"], color=colors)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel(
        "Coefficient (log-odds)  |  red = raises churn risk, teal = lowers it")
    plt.title(f"Top {top_n} positive and negative drivers")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Logistic Regression baseline for 90-day churn")
    parser.add_argument("--data", default=None, help="Path to subscribers.csv")
    parser.add_argument("--output", default=None,
                        help="Output folder (default: ./output beside this script)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output) if args.output else script_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    log.info("=" * 70)
    log.info("SUBSCRIBER CHURN - LOGISTIC REGRESSION BASELINE")
    log.info("=" * 70)

    # 1) Load ------------------------------------------------------------------
    df = load_data(find_data_file(args.data))
    X, y, numeric_cols, categorical_cols = prepare_features(df)
    log.info(
        f"Target            : {TARGET}  (positive rate = {y.mean():.2%}, {int(y.sum()):,} churners)")
    log.info(f"Excluded (leakage): {LEAKAGE_COLS}")
    log.info(f"Excluded (non-feature): {NON_FEATURE_COLS}")
    log.info(
        f"Features used     : {X.shape[1]}  ({len(numeric_cols)} numeric/binary, {len(categorical_cols)} categorical)")
    log.info(f"Categorical cols  : {categorical_cols}")

    # 2) Stratified 70 / 15 / 15 split ----------------------------------------
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=VAL_SIZE + TEST_SIZE, stratify=y, random_state=args.seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=TEST_SIZE / (VAL_SIZE + TEST_SIZE), stratify=y_tmp, random_state=args.seed
    )
    n = len(df)
    split_info = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "rows": [len(X_train), len(X_val), len(X_test)],
        "share_of_data": [len(X_train) / n, len(X_val) / n, len(X_test) / n],
        "churners": [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())],
        "churn_rate": [y_train.mean(), y_val.mean(), y_test.mean()],
    })
    split_info.to_csv(output_dir / "split_summary.csv", index=False)
    log.info("\nData split (stratified on the target):")
    log.info(split_info.to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    # 3) Pick C on the validation set -----------------------------------------
    log.info(
        "\nTuning regularisation strength C on the validation set (metric: ROC-AUC)")
    tuning_rows, best_auc, best_C = [], -np.inf, None
    for C in C_GRID:
        pipe = build_pipeline(numeric_cols, categorical_cols,
                              C, args.seed).fit(X_train, y_train)
        auc = roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])
        tuning_rows.append({"C": C, "val_roc_auc": auc})
        log.info(f"  C = {C:<6}  validation ROC-AUC = {auc:.4f}")
        if auc > best_auc:
            best_auc, best_C = auc, C
    pd.DataFrame(tuning_rows).to_csv(
        output_dir / "hyperparameter_tuning.csv", index=False)
    log.info(f"Selected C = {best_C}")

    # 4) Final model (fit on the training split only) --------------------------
    pipeline = build_pipeline(
        numeric_cols, categorical_cols, best_C, args.seed).fit(X_train, y_train)
    proba = {
        "Train": (y_train, pipeline.predict_proba(X_train)[:, 1]),
        "Validation": (y_val, pipeline.predict_proba(X_val)[:, 1]),
        "Test": (y_test, pipeline.predict_proba(X_test)[:, 1]),
    }

    # 5) Decision threshold chosen on validation (max F1) ----------------------
    tuned_thr = best_f1_threshold(*proba["Validation"])
    log.info(
        f"Decision threshold maximising F1 on validation = {tuned_thr:.4f}")

    # 6) Metrics for every split, at both thresholds ---------------------------
    rows = []
    for split_name, (y_true, p) in proba.items():
        for thr_name, thr in (("default_0.50", 0.5), ("tuned_on_validation", tuned_thr)):
            rows.append({"split": split_name, "threshold_type": thr_name,
                        **compute_metrics(y_true, p, thr)})
    metrics_df = pd.DataFrame(rows)
    metrics_df.round(4).to_csv(output_dir / "metrics_summary.csv", index=False)

    show_cols = ["split", "threshold_type", "threshold", "accuracy",
                 "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    log.info("\n" + "=" * 70)
    log.info("METRICS  (Accuracy, Precision, Recall, F1-score, ROC-AUC)")
    log.info("=" * 70)
    log.info(metrics_df[show_cols].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

    # Test-set classification reports
    y_test_arr, p_test = proba["Test"]
    with open(output_dir / "classification_report_test.txt", "w", encoding="utf-8") as fh:
        for thr_name, thr in (("default 0.50", 0.5), ("tuned on validation", tuned_thr)):
            fh.write(
                f"=== Test set | threshold = {thr:.4f} ({thr_name}) ===\n")
            fh.write(classification_report(
                y_test_arr, (p_test >= thr).astype(int),
                target_names=["Retained", "Churned"], digits=4, zero_division=0,
            ))
            fh.write("\n")

    # 7) Coefficients ----------------------------------------------------------
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_.ravel()
    coef_df = pd.DataFrame({
        "feature": [f.split("__", 1)[-1] for f in feature_names],
        "coefficient": coefs,
        "odds_ratio": np.exp(coefs),
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)
    coef_df.to_csv(output_dir / "feature_coefficients.csv", index=False)

    # 8) Predictions, plots, model, summary ------------------------------------
    pd.DataFrame({
        "subscriber_id": df.loc[X_test.index, "subscriber_id"].values if "subscriber_id" in df.columns else X_test.index,
        "actual": y_test_arr.values,
        "churn_probability": p_test,
        "predicted_default_0.50": (p_test >= 0.5).astype(int),
        "predicted_tuned": (p_test >= tuned_thr).astype(int),
    }).to_csv(output_dir / "test_predictions.csv", index=False)

    plot_roc(proba, output_dir / "roc_curve.png")
    plot_pr({k: proba[k] for k in ("Validation", "Test")}, float(
        y.mean()), output_dir / "precision_recall_curve.png")
    plot_confusion(y_test_arr, p_test, {
                   "threshold 0.50": 0.5, "tuned threshold": tuned_thr}, output_dir / "confusion_matrix_test.png")
    plot_threshold_curve(y_val, proba["Validation"][1], tuned_thr,
                         output_dir / "threshold_selection_validation.png")
    plot_coefficients(coef_df, output_dir / "top_coefficients.png")

    joblib.dump(pipeline, output_dir / "logistic_regression_baseline.joblib")

    summary = {
        "target": TARGET,
        "excluded_leakage_columns": LEAKAGE_COLS,
        "excluded_non_feature_columns": NON_FEATURE_COLS,
        "n_features_used": int(X.shape[1]),
        "split": {"train": TRAIN_SIZE, "validation": VAL_SIZE, "test": TEST_SIZE, "seed": args.seed},
        "model": {"type": "LogisticRegression (L2)", "C": best_C, "class_weight": CLASS_WEIGHT},
        "tuned_threshold": tuned_thr,
        "test_metrics_default_0.50": {k: v for k, v in metrics_df.query("split == 'Test' and threshold_type == 'default_0.50'").iloc[0].items() if k not in ("split", "threshold_type")},
        "test_metrics_tuned_threshold": {k: v for k, v in metrics_df.query("split == 'Test' and threshold_type == 'tuned_on_validation'").iloc[0].items() if k not in ("split", "threshold_type")},
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: o.item()
                  if hasattr(o, "item") else str(o))

    log.info("\nTop 5 features raising churn risk:")
    log.info(coef_df.head(5)[["feature", "coefficient", "odds_ratio"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    log.info("\nTop 5 features lowering churn risk:")
    log.info(coef_df.tail(5)[["feature", "coefficient", "odds_ratio"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    log.info(f"\nAll results saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
