
"""
==============================================================================
 SUBSCRIBER CHURN  -  XGBoost 90-day churn classifier
==============================================================================
 Target            : churn_flag_90d   (5.01 % positive  ->  ~1 : 19 imbalance)
 Excluded columns  : churn_reason, churn_date, churn_flag_30d   (outcome-linked)
                     + subscriber_id (identifier), circle_code (duplicate of
                       circle), join_date (fully represented by tenure_months)
 Split             : 70 % train / 15 % validation / 15 % test  (stratified)

 The script looks for subscribers.csv in this order:
     1. --data <path>            (if you pass it)
     2. next to this script
     3. <script folder>/data/subscribers.csv
     4. current working directory
 Every result is written to  <script folder>/output/

 Design principles (so the numbers are honest, not artificial)
 -------------------------------------------------------------
  * Leakage columns are removed BEFORE modelling and asserted absent.
  * Split is stratified, so each subset keeps the same 5.01 % churn rate.
  * All tuning (early stopping, hyper-parameter choice, decision threshold)
    uses TRAIN + VALIDATION only. The TEST set is scored once, at the end.
  * Class imbalance is handled with scale_pos_weight (tuned on validation)
    and a validation-chosen decision threshold, NOT by duplicating rows.
  * Accuracy is reported but not trusted alone: predicting "nobody churns"
    already scores ~95 %. Precision / Recall / F1 / PR-AUC are the real signal.
  * Sanity checks are included: shuffled-label test (must be ~0.50 AUC),
    single-feature leakage scan, dummy + logistic-regression baselines, and
    an ablation without mnp_enquiry_flag.
==============================================================================
"""

from __future__ import annotations
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no GUI needed -> works in any terminal / VS Code

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
SEED = 42
TARGET = "churn_flag_90d"

# Outcome-linked columns that MUST NOT be used as features (as instructed).
LEAKY_COLS = ["churn_reason", "churn_date", "churn_flag_30d"]

# Non-predictive / redundant columns (documented, not outcome-related).
ID_COLS = ["subscriber_id"]                # unique identifier
# circle_code == circle ; join_date -> tenure_months
REDUNDANT_COLS = ["circle_code", "join_date"]

CATEGORICAL_COLS = ["circle", "zone",
                    "device_brand", "home_product", "plan_type"]

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

N_TRIALS_DEFAULT = 30      # random-search trials for hyper-parameters
N_BOOTSTRAP = 1000         # bootstrap resamples for test-set confidence intervals

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR = SCRIPT_DIR / "output"


# ----------------------------------------------------------------------------
# UTILITIES
# ----------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("churn")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(message)s")
    for h in (logging.StreamHandler(sys.stdout),
              logging.FileHandler(OUT_DIR / "run_log.txt", mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def banner(log: logging.Logger, title: str) -> None:
    log.info("\n" + "=" * 78)
    log.info(f" {title}")
    log.info("=" * 78)


def find_data_file(cli_path: str | None) -> Path:
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path))
    candidates += [
        SCRIPT_DIR / "Cleaned_File" / "subscribers.csv",
        SCRIPT_DIR / "subscribers.csv",
        SCRIPT_DIR / "data" / "subscribers.csv",
        Path.cwd() / "Cleaned_File" / "subscribers.csv",
        Path.cwd() / "subscribers.csv",
        Path("/mnt/user-data/uploads/subscribers.csv"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(
        "subscribers.csv not found in Cleaned_File or the usual locations. "
        "Put it in the script folder or run:  python churn_xgboost_model.py --data \"path/to/subscribers.csv\""
    )


def load_data(path: Path) -> pd.DataFrame:
    # keep_default_na=False  -> the text 'None' in home_product is a REAL category
    # (pandas would otherwise misread it as missing). Only empty cells become NaN.
    return pd.read_csv(path, keep_default_na=False, na_values=[""])


def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not serialisable: {type(obj)}")


# ----------------------------------------------------------------------------
# DATA PREPARATION
# ----------------------------------------------------------------------------
def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return raw feature frame (before encoding), target, and subscriber ids."""
    drop_cols = [TARGET] + LEAKY_COLS + ID_COLS + REDUNDANT_COLS
    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
    y = df[TARGET].astype(int)
    ids = df["subscriber_id"].copy()

    # Hard guarantee: no outcome-linked column may survive.
    leaked = [c for c in LEAKY_COLS + [TARGET] if c in X.columns]
    assert not leaked, f"LEAKAGE: outcome columns still present -> {leaked}"

    # bool -> int (XGBoost / sklearn friendly on every pandas version)
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X, y, ids


def encode_with_train_levels(X_raw: pd.DataFrame, train_levels: dict[str, list[str]]) -> pd.DataFrame:
    """One-hot encode categoricals using ONLY the levels seen in the training set."""
    parts = [X_raw.drop(columns=CATEGORICAL_COLS)]
    for col, levels in train_levels.items():
        cat = pd.Categorical(X_raw[col].astype(str), categories=levels)
        dummies = pd.get_dummies(cat, prefix=col, dtype=int)
        dummies.index = X_raw.index
        parts.append(dummies)
    return pd.concat(parts, axis=1)


def stratified_70_15_15(X, y, ids):
    X_tr, X_tmp, y_tr, y_tmp, id_tr, id_tmp = train_test_split(
        X, y, ids, test_size=(VAL_FRAC + TEST_FRAC), stratify=y, random_state=SEED
    )
    X_val, X_te, y_val, y_te, id_val, id_te = train_test_split(
        X_tmp, y_tmp, id_tmp, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=y_tmp, random_state=SEED
    )
    return (X_tr, y_tr, id_tr), (X_val, y_val, id_val), (X_te, y_te, id_te)


# ----------------------------------------------------------------------------
# METRICS
# ----------------------------------------------------------------------------
def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1_score": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "mcc": matthews_corrcoef(y_true, pred),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "log_loss": log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y_true, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Threshold that maximises F1 (picked on VALIDATION data only)."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def bootstrap_ci(y_true, proba, threshold, n_boot=N_BOOTSTRAP, seed=SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, pp = y_true[idx], proba[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        pr = (pp >= threshold).astype(int)
        draws["accuracy"].append(accuracy_score(yt, pr))
        draws["precision"].append(precision_score(yt, pr, zero_division=0))
        draws["recall"].append(recall_score(yt, pr, zero_division=0))
        draws["f1_score"].append(f1_score(yt, pr, zero_division=0))
        draws["roc_auc"].append(roc_auc_score(yt, pp))
        draws["pr_auc"].append(average_precision_score(yt, pp))
    rows = []
    for k in keys:
        a = np.array(draws[k])
        rows.append({"metric": k, "mean": a.mean(),
                     "ci_2.5%": np.percentile(a, 2.5), "ci_97.5%": np.percentile(a, 97.5)})
    return pd.DataFrame(rows)


def decile_table(y_true, proba) -> pd.DataFrame:
    d = pd.DataFrame({"y": np.asarray(y_true), "p": proba})
    d = d.sort_values("p", ascending=False).reset_index(drop=True)
    d["decile"] = (np.arange(len(d)) * 10 // len(d)) + 1  # 1 = riskiest 10 %
    base = d["y"].mean()
    g = d.groupby("decile").agg(customers=("y", "size"), churners=("y", "sum"),
                                min_score=("p", "min"), max_score=("p", "max")).reset_index()
    g["churn_rate"] = g["churners"] / g["customers"]
    g["lift"] = g["churn_rate"] / base
    g["cum_churners_captured_pct"] = g["churners"].cumsum() / \
        g["churners"].sum() * 100
    return g


# ----------------------------------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------------------------------
def make_xgb(params: dict, scale_pos_weight: float, n_jobs: int = -1) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        # early stopping uses the LAST metric (aucpr)
        eval_metric=["logloss", "aucpr"],
        tree_method="hist",
        # upper bound; early stopping decides the real number
        n_estimators=3000,
        early_stopping_rounds=50,
        random_state=SEED,
        n_jobs=n_jobs,
        scale_pos_weight=scale_pos_weight,
        **params,
    )


def sample_params(rng: np.random.Generator, imbalance_ratio: float) -> dict:
    spw_choices = [1.0, float(np.sqrt(imbalance_ratio)),
                   float(imbalance_ratio)]
    return {
        "learning_rate": float(rng.choice([0.03, 0.05, 0.08, 0.1])),
        "max_depth": int(rng.integers(3, 8)),
        "min_child_weight": float(rng.choice([1, 3, 5, 10, 20])),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "gamma": float(rng.choice([0, 0.5, 1, 2, 5])),
        "reg_lambda": float(rng.choice([1, 3, 5, 10, 20])),
        "reg_alpha": float(rng.choice([0, 0.1, 0.5, 1, 2])),
        "_spw": float(rng.choice(spw_choices)),
    }


def random_search(X_tr, y_tr, X_val, y_val, imbalance_ratio, n_trials, log) -> tuple[xgb.XGBClassifier, dict, pd.DataFrame]:
    """Pick hyper-parameters by validation PR-AUC (the right metric for imbalance)."""
    rng = np.random.default_rng(SEED)
    # trial 0 = sensible, conservative default (always evaluated)
    trials = [{"learning_rate": 0.05, "max_depth": 5, "min_child_weight": 5, "subsample": 0.8,
               "colsample_bytree": 0.8, "gamma": 0, "reg_lambda": 5, "reg_alpha": 0.1,
               "_spw": float(np.sqrt(imbalance_ratio))}]
    trials += [sample_params(rng, imbalance_ratio)
               for _ in range(n_trials - 1)]

    best_model, best_params, best_score = None, None, -np.inf
    rows = []
    for i, p in enumerate(trials, 1):
        t0 = time.time()
        params = {k: v for k, v in p.items() if k != "_spw"}
        model = make_xgb(params, p["_spw"])
        model.fit(X_tr, y_tr, eval_set=[
                  (X_tr, y_tr), (X_val, y_val)], verbose=False)
        val_proba = model.predict_proba(X_val)[:, 1]
        val_pr = average_precision_score(y_val, val_proba)
        val_roc = roc_auc_score(y_val, val_proba)
        rows.append({"trial": i, **p, "best_iteration": model.best_iteration,
                     "val_pr_auc": val_pr, "val_roc_auc": val_roc})
        flag = ""
        if val_pr > best_score:
            best_score, best_model, best_params, flag = val_pr, model, p, "  <-- best so far"
        log.info(f"  trial {i:>2}/{len(trials)} | depth={p['max_depth']} lr={p['learning_rate']:<5} "
                 f"spw={p['_spw']:<6.2f} | trees={model.best_iteration + 1:>4} | "
                 f"val PR-AUC={val_pr:.4f} ROC-AUC={val_roc:.4f} | {time.time() - t0:4.1f}s{flag}")
    return best_model, best_params, pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)


# ----------------------------------------------------------------------------
# PLOTS
# ----------------------------------------------------------------------------
PALETTE = {"navy": "#12233d", "teal": "#1f7a8c",
           "red": "#c0393b", "gold": "#d1a04b", "grey": "#9aa5b1"}


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT_DIR / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_val, p_val, y_te, p_te):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for y, p, lab, col in [(y_val, p_val, "Validation", PALETTE["gold"]), (y_te, p_te, "Test", PALETTE["teal"])]:
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=col, lw=2,
                   label=f"{lab} (AUC = {roc_auc_score(y, p):.3f})")
        pr, rc, _ = precision_recall_curve(y, p)
        ax[1].plot(rc, pr, color=col, lw=2,
                   label=f"{lab} (PR-AUC = {average_precision_score(y, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color=PALETTE["grey"],
               label="Random (AUC = 0.500)")
    ax[0].set(xlabel="False positive rate",
              ylabel="True positive rate", title="ROC curve")
    base = float(np.mean(y_te))
    ax[1].axhline(base, ls="--", color=PALETTE["grey"],
                  label=f"Random (precision = {base:.3f})")
    ax[1].set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall curve", ylim=(0, 1.02))
    for a in ax:
        a.legend(loc="best")
        a.grid(alpha=0.3)
    _save(fig, "roc_pr_curves.png")


def plot_confusions(cm_default, cm_tuned, thr_tuned):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.8))
    for a, cm, ttl in [(ax[0], cm_default, "Test - threshold 0.50"),
                       (ax[1], cm_tuned, f"Test - tuned threshold {thr_tuned:.3f}")]:
        a.imshow(cm, cmap="Blues")
        a.set_xticks([0, 1], ["Retained", "Churned"])
        a.set_yticks([0, 1], ["Retained", "Churned"])
        a.set(xlabel="Predicted", ylabel="Actual", title=ttl)
        for i in range(2):
            for j in range(2):
                a.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14, fontweight="bold",
                       color="white" if cm[i, j] > cm.max() / 2 else "black")
    _save(fig, "confusion_matrices_test.png")


def plot_threshold_sweep(y_val, p_val, thr):
    grid = np.linspace(0.02, 0.98, 97)
    prec = [precision_score(y_val, p_val >= t, zero_division=0) for t in grid]
    rec = [recall_score(y_val, p_val >= t, zero_division=0) for t in grid]
    f1 = [f1_score(y_val, p_val >= t, zero_division=0) for t in grid]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, prec, label="Precision", color=PALETTE["teal"], lw=2)
    ax.plot(grid, rec, label="Recall", color=PALETTE["red"], lw=2)
    ax.plot(grid, f1, label="F1", color=PALETTE["navy"], lw=2)
    ax.axvline(thr, ls="--", color=PALETTE["gold"],
               label=f"Chosen threshold = {thr:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Score",
           title="Threshold selection (validation set)")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "threshold_selection_validation.png")


def plot_lift(dt: pd.DataFrame):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    ax[0].bar(dt["decile"], dt["churn_rate"] * 100, color=PALETTE["red"])
    ax[0].axhline(dt["churners"].sum() / dt["customers"].sum() *
                  100, ls="--", color=PALETTE["grey"], label="Base rate")
    ax[0].set(xlabel="Risk decile (1 = riskiest 10 %)",
              ylabel="Actual churn %", title="Churn rate by risk decile (test)")
    ax[0].set_xticks(dt["decile"])
    ax[0].legend()
    ax[1].plot(dt["decile"], dt["cum_churners_captured_pct"],
               marker="o", color=PALETTE["navy"], lw=2, label="Model")
    ax[1].plot([0] + list(dt["decile"]), [0] + list(dt["decile"]
               * 10), "--", color=PALETTE["grey"], label="Random")
    ax[1].set(xlabel="Risk decile (cumulative)",
              ylabel="% of all churners captured", title="Cumulative gains (test)")
    ax[1].set_xticks(range(0, 11))
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    _save(fig, "lift_and_gains_test.png")


def plot_importance(fi: pd.DataFrame, top=20):
    top_df = fi.head(top).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 7))
    ax[0].barh(top_df["feature"], top_df["gain_importance"],
               color=PALETTE["teal"])
    ax[0].set(title="XGBoost gain importance (top 20)",
              xlabel="Normalised gain")
    pi = fi.sort_values("permutation_importance_pr_auc",
                        ascending=False).head(top).iloc[::-1]
    ax[1].barh(pi["feature"], pi["permutation_importance_pr_auc"],
               color=PALETTE["red"])
    ax[1].set(title="Permutation importance on validation (drop in PR-AUC)",
              xlabel="Mean PR-AUC decrease")
    _save(fig, "feature_importance.png")


def plot_learning_curve(model):
    ev = model.evals_result()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for k, lab, col in [("validation_0", "Train", PALETTE["teal"]), ("validation_1", "Validation", PALETTE["red"])]:
        ax[0].plot(ev[k]["logloss"], label=lab, color=col)
        ax[1].plot(ev[k]["aucpr"], label=lab, color=col)
    bi = model.best_iteration
    for a, ttl in zip(ax, ["Log-loss", "PR-AUC"]):
        a.axvline(
            bi, ls="--", color=PALETTE["gold"], label=f"Early stop @ {bi + 1}")
        a.set(xlabel="Boosting round", title=f"{ttl} by round")
        a.legend()
        a.grid(alpha=0.3)
    _save(fig, "learning_curve.png")


def plot_calibration(y_te, p_te, n_bins=10):
    d = pd.DataFrame({"y": np.asarray(y_te), "p": p_te})
    d["bin"] = pd.qcut(d["p"].rank(method="first"), n_bins, labels=False)
    g = d.groupby("bin").agg(pred=("p", "mean"), obs=("y", "mean"))
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["grey"],
            label="Perfect calibration")
    ax.plot(g["pred"], g["obs"], marker="o",
            color=PALETTE["navy"], label="Model")
    ax.set(xlabel="Mean predicted score", ylabel="Observed churn rate",
           title="Calibration (test)\nscores are risk RANKS when scale_pos_weight > 1")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "calibration_test.png")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="XGBoost 90-day churn model")
    ap.add_argument("--data", default=None, help="path to subscribers.csv")
    ap.add_argument("--trials", type=int, default=N_TRIALS_DEFAULT,
                    help="hyper-parameter search trials")
    args = ap.parse_args()

    log = setup_logging()
    t_start = time.time()
    banner(log, "1. LOAD DATA")
    data_path = find_data_file(args.data)
    df = load_data(data_path)
    log.info(f"File            : {data_path}")
    log.info(f"Shape           : {df.shape[0]:,} rows x {df.shape[1]} columns")
    assert df["subscriber_id"].is_unique, "duplicate subscriber_id found"
    pos_rate = df[TARGET].mean()
    log.info(f"Churn (90d)     : {int(df[TARGET].sum()):,} churned / {int((1 - df[TARGET]).sum()):,} retained "
             f"-> {pos_rate * 100:.2f} % positive  (1 : {(1 - pos_rate) / pos_rate:.1f})")

    # ------------------------------------------------------------------
    banner(log, "2. FEATURES - LEAKAGE CONTROL")
    X_raw, y, ids = build_feature_frame(df)
    log.info(f"Removed (outcome-linked) : {LEAKY_COLS}")
    log.info(f"Removed (identifier)     : {ID_COLS}")
    log.info(f"Removed (redundant)      : {REDUNDANT_COLS}")
    log.info(f"Raw features kept        : {X_raw.shape[1]}")

    # single-feature leakage scan
    scan = {}
    for c in X_raw.columns:
        s = X_raw[c]
        if c in CATEGORICAL_COLS:
            s = s.astype("category").cat.codes
        a = roc_auc_score(y, s.astype(float))
        scan[c] = max(a, 1 - a)
    scan = pd.Series(scan).sort_values(ascending=False)
    scan.rename("univariate_auc").to_csv(
        OUT_DIR / "leakage_scan_univariate_auc.csv")
    log.info(f"Single-feature AUC scan  : max = {scan.iloc[0]:.3f} ({scan.index[0]}) "
             f"-> {'no feature is a near-perfect predictor (OK)' if scan.iloc[0] < 0.95 else 'WARNING: possible leakage'}")

    # ------------------------------------------------------------------
    banner(log, "3. STRATIFIED 70 / 15 / 15 SPLIT")
    (Xr_tr, y_tr, id_tr), (Xr_val, y_val, id_val), (Xr_te,
                                                    y_te, id_te) = stratified_70_15_15(X_raw, y, ids)
    train_levels = {c: sorted(Xr_tr[c].astype(str).unique())
                    for c in CATEGORICAL_COLS}
    X_tr = encode_with_train_levels(Xr_tr, train_levels)
    X_val = encode_with_train_levels(Xr_val, train_levels)
    X_te = encode_with_train_levels(Xr_te, train_levels)
    feature_names = list(X_tr.columns)
    assert set(id_tr).isdisjoint(id_val) and set(id_tr).isdisjoint(id_te) and set(id_val).isdisjoint(id_te), \
        "subscribers overlap between splits"

    split_rows = []
    for name, yy in [("Train", y_tr), ("Validation", y_val), ("Test", y_te)]:
        split_rows.append({"split": name, "rows": len(yy), "share_%": len(yy) / len(y) * 100,
                           "churned": int(yy.sum()), "retained": int((1 - yy).sum()),
                           "churn_rate_%": yy.mean() * 100})
        log.info(f"  {name:<10}: {len(yy):>6,} rows ({len(yy) / len(y) * 100:4.1f} %) | "
                 f"churned {int(yy.sum()):>5,} | churn rate {yy.mean() * 100:.2f} %")
    pd.DataFrame(split_rows).to_csv(OUT_DIR / "split_summary.csv", index=False)
    log.info(
        f"Model input features (after one-hot encoding): {len(feature_names)}")

    imbalance_ratio = float((y_tr == 0).sum() / (y_tr == 1).sum())
    log.info(
        f"Imbalance ratio in TRAIN (retained : churned) = {imbalance_ratio:.2f} : 1")

    # ------------------------------------------------------------------
    banner(log, "4. BASELINES (to prove the model adds real value)")
    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    p_dummy = dummy.predict_proba(X_te)[:, 1]
    lr = make_pipeline(StandardScaler(), LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=SEED))
    lr.fit(X_tr, y_tr)
    p_lr_val, p_lr_te = lr.predict_proba(
        X_val)[:, 1], lr.predict_proba(X_te)[:, 1]
    thr_lr, _ = best_f1_threshold(y_val, p_lr_val)
    base_rows = []
    m = compute_metrics(y_te, p_dummy, 0.5)
    m["model"] = "Majority-class (predict nobody churns)"
    base_rows.append(m)
    m = compute_metrics(y_te, p_lr_te, thr_lr)
    m["model"] = "Logistic regression (balanced)"
    base_rows.append(m)
    for r in base_rows:
        log.info(f"  {r['model']:<42} acc={r['accuracy']:.4f} prec={r['precision']:.4f} "
                 f"rec={r['recall']:.4f} f1={r['f1_score']:.4f} roc_auc={r['roc_auc']:.4f}")
    log.info(
        "  -> Note how the do-nothing baseline scores ~95 % ACCURACY yet catches 0 churners.")

    # ------------------------------------------------------------------
    banner(
        log, f"5. XGBOOST - HYPER-PARAMETER SEARCH ({args.trials} trials, early stopping on validation PR-AUC)")
    model, best_p, trials_df = random_search(
        X_tr, y_tr, X_val, y_val, imbalance_ratio, args.trials, log)
    trials_df.to_csv(OUT_DIR / "hyperparameter_search.csv", index=False)
    best_params_clean = {k: v for k, v in best_p.items() if k != "_spw"}
    best_params_clean["scale_pos_weight"] = best_p["_spw"]
    best_params_clean["n_trees_after_early_stopping"] = int(
        model.best_iteration + 1)
    with open(OUT_DIR / "best_params.json", "w") as f:
        json.dump(best_params_clean, f, indent=2, default=json_safe)
    log.info(
        f"\nBest parameters: {json.dumps(best_params_clean, default=json_safe)}")

    # ------------------------------------------------------------------
    banner(log, "6. DECISION THRESHOLD (chosen on VALIDATION only)")
    p_tr = model.predict_proba(X_tr)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    thr, f1_at_thr = best_f1_threshold(y_val, p_val)
    log.info(
        f"F1-optimal threshold on validation = {thr:.4f} (validation F1 = {f1_at_thr:.4f})")
    log.info("Default 0.50 threshold is also reported for transparency.")
    plot_threshold_sweep(y_val, p_val, thr)

    # ------------------------------------------------------------------
    banner(log, "7. FINAL EVALUATION - TEST SET (touched once, never used for tuning)")
    p_te = model.predict_proba(X_te)[:, 1]
    rows = []
    for split, yy, pp in [("Train", y_tr, p_tr), ("Validation", y_val, p_val), ("Test", y_te, p_te)]:
        for tname, t in [("tuned", thr), ("0.50", 0.5)]:
            m = compute_metrics(yy, pp, t)
            rows.append({"split": split, "threshold_type": tname, **m})
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)

    def show(split, ttype):
        r = metrics_df[(metrics_df.split == split) & (
            metrics_df.threshold_type == ttype)].iloc[0]
        log.info(f"  {split:<10} @ {ttype:<5} (t={r['threshold']:.3f}) | "
                 f"Accuracy {r['accuracy']:.4f} | Precision {r['precision']:.4f} | Recall {r['recall']:.4f} | "
                 f"F1 {r['f1_score']:.4f} | ROC-AUC {r['roc_auc']:.4f} | PR-AUC {r['pr_auc']:.4f}")
    for s in ["Train", "Validation", "Test"]:
        show(s, "tuned")
    log.info("")
    for s in ["Validation", "Test"]:
        show(s, "0.50")

    te = metrics_df[(metrics_df.split == "Test") & (
        metrics_df.threshold_type == "tuned")].iloc[0]
    log.info(
        f"\n  TEST confusion matrix @ tuned threshold: TN={te.TN:,}  FP={te.FP:,}  FN={te.FN:,}  TP={te.TP:,}")
    log.info(
        f"  TEST balanced accuracy = {te.balanced_accuracy:.4f} | MCC = {te.mcc:.4f} | specificity = {te.specificity:.4f}")

    gap = metrics_df[(metrics_df.split == "Train") & (
        metrics_df.threshold_type == "tuned")].iloc[0]["roc_auc"] - te["roc_auc"]
    log.info(f"  Train-Test ROC-AUC gap = {gap:.4f}  "
             f"({'healthy' if gap < 0.05 else 'noticeable overfitting - review'})")

    pred_te = (p_te >= thr).astype(int)
    rep = classification_report(y_te, pred_te, target_names=[
                                "Retained", "Churned"], digits=4)
    with open(OUT_DIR / "classification_report_test.txt", "w") as f:
        f.write(f"Test set - threshold = {thr:.4f}\n\n{rep}")
    log.info("\n" + rep)

    ci = bootstrap_ci(y_te, p_te, thr)
    ci.to_csv(OUT_DIR / "test_metrics_bootstrap_ci.csv", index=False)
    log.info(
        f"95 % bootstrap confidence intervals on TEST ({N_BOOTSTRAP} resamples):")
    for _, r in ci.iterrows():
        log.info(
            f"  {r['metric']:<10} {r['mean']:.4f}   [{r['ci_2.5%']:.4f}, {r['ci_97.5%']:.4f}]")

    # model vs baselines table
    cmp_rows = []
    for r in base_rows:
        cmp_rows.append({k: r[k] for k in [
                        "model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]})
    cmp_rows.append({"model": "XGBoost (this work, tuned threshold)", **{k: te[k] for k in
                     ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]}})
    pd.DataFrame(cmp_rows).to_csv(
        OUT_DIR / "model_vs_baselines_test.csv", index=False)

    # deciles
    dt = decile_table(y_te, p_te)
    dt.to_csv(OUT_DIR / "decile_lift_table_test.csv", index=False)
    log.info("\nRisk-decile table (test):")
    log.info(dt[["decile", "customers", "churners", "churn_rate", "lift", "cum_churners_captured_pct"]]
             .round(3).to_string(index=False))
    log.info(f"\n  -> Contacting only the riskiest 10 % captures {dt.loc[0, 'cum_churners_captured_pct']:.1f} % of all churners "
             f"({dt.loc[0, 'lift']:.1f}x lift).")

    # predictions
    pd.DataFrame({"subscriber_id": id_te.values, "actual_churn_90d": y_te.values,
                  "churn_risk_score": p_te, "predicted_churn": pred_te,
                  "risk_decile": pd.Series(p_te).rank(ascending=False, method="first")
                  .pipe(lambda r: (np.ceil(r / len(r) * 10)).astype(int))}
                 ).sort_values("churn_risk_score", ascending=False).to_csv(OUT_DIR / "test_predictions.csv", index=False)

    # ------------------------------------------------------------------
    banner(log, "8. FEATURE IMPORTANCE")
    gain = pd.Series(model.get_booster().get_score(
        importance_type="total_gain"))
    gain = gain.reindex(feature_names).fillna(0.0)
    gain = gain / gain.sum()
    pi = permutation_importance(model, X_val, y_val, scoring="average_precision",
                                n_repeats=5, random_state=SEED, n_jobs=1)
    fi = pd.DataFrame({"feature": feature_names, "gain_importance": gain.values,
                       "permutation_importance_pr_auc": pi.importances_mean,
                       "permutation_std": pi.importances_std}).sort_values("gain_importance", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    log.info(fi.head(12)[["feature", "gain_importance",
             "permutation_importance_pr_auc"]].round(4).to_string(index=False))

    # ------------------------------------------------------------------
    banner(log, "9. SANITY CHECKS (proof the result is not artificial)")
    # (a) shuffled labels -> must collapse to ~0.5
    rng = np.random.default_rng(SEED)
    y_shuf = pd.Series(rng.permutation(y_tr.values), index=y_tr.index)
    sh = make_xgb({k: v for k, v in best_p.items()
                  if k != "_spw"}, best_p["_spw"])
    sh.fit(X_tr, y_shuf, eval_set=[
           (X_tr, y_shuf), (X_val, y_val)], verbose=False)
    auc_shuf = roc_auc_score(y_te, sh.predict_proba(X_te)[:, 1])
    log.info(f"  (a) Shuffled-label test ROC-AUC = {auc_shuf:.4f}  "
             f"({'~0.50 as expected: no hidden leakage' if abs(auc_shuf - 0.5) < 0.05 else 'CHECK: should be ~0.50'})")

    # (b) ablation: drop mnp_enquiry_flag (intent signal whose timing should be confirmed with the business)
    abl_rows = [{"variant": "Full model", **{k: te[k]
                                             for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]}}]
    if "mnp_enquiry_flag" in feature_names:
        cols = [c for c in feature_names if c != "mnp_enquiry_flag"]
        ab = make_xgb({k: v for k, v in best_p.items()
                      if k != "_spw"}, best_p["_spw"])
        ab.fit(X_tr[cols], y_tr, eval_set=[
               (X_tr[cols], y_tr), (X_val[cols], y_val)], verbose=False)
        pv, pt = ab.predict_proba(X_val[cols])[
            :, 1], ab.predict_proba(X_te[cols])[:, 1]
        t_ab, _ = best_f1_threshold(y_val, pv)
        m = compute_metrics(y_te, pt, t_ab)
        abl_rows.append({"variant": "Without mnp_enquiry_flag", **{k: m[k] for k in
                         ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]}})
        log.info(f"  (b) Without mnp_enquiry_flag  -> ROC-AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} | "
                 f"Precision {m['precision']:.4f} | Recall {m['recall']:.4f} | F1 {m['f1_score']:.4f}")
    pd.DataFrame(abl_rows).to_csv(OUT_DIR / "ablation_test.csv", index=False)
    with open(OUT_DIR / "sanity_checks.txt", "w") as f:
        f.write(f"Shuffled-label test ROC-AUC: {auc_shuf:.4f} (expected ~0.50)\n"
                f"Max single-feature AUC: {scan.iloc[0]:.4f} ({scan.index[0]})\n"
                f"Leaky columns excluded: {LEAKY_COLS}\n"
                f"Train-Test ROC-AUC gap: {gap:.4f}\n")

    # ------------------------------------------------------------------
    banner(log, "10. SAVING PLOTS & MODEL")
    plot_roc_pr(y_val, p_val, y_te, p_te)
    plot_confusions(confusion_matrix(y_te, (p_te >= 0.5).astype(
        int)), confusion_matrix(y_te, pred_te), thr)
    plot_lift(dt)
    plot_importance(fi)
    plot_learning_curve(model)
    plot_calibration(y_te, p_te)

    model.save_model(str(OUT_DIR / "xgboost_churn_model.json"))
    with open(OUT_DIR / "model_config.json", "w") as f:
        json.dump({"target": TARGET, "decision_threshold": thr, "feature_names": feature_names,
                   "categorical_levels": train_levels, "excluded_leaky_columns": LEAKY_COLS,
                   "best_params": best_params_clean, "seed": SEED,
                   "split": {"train": TRAIN_FRAC, "validation": VAL_FRAC, "test": TEST_FRAC}},
                  f, indent=2, default=json_safe)

    # executive summary
    with open(OUT_DIR / "RESULTS_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(
            "SUBSCRIBER CHURN - XGBoost 90-day model - TEST SET RESULTS\n" + "=" * 60 + "\n")
        f.write(
            f"Rows: {len(df):,} | churn rate {pos_rate * 100:.2f} % | split 70/15/15 (stratified)\n")
        f.write(f"Excluded leakage columns: {', '.join(LEAKY_COLS)}\n")
        f.write(
            f"Decision threshold (validation-tuned, F1-optimal): {thr:.4f}\n\n")
        for k in ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]:
            r = ci[ci.metric == k].iloc[0]
            f.write(
                f"{k:<10}: {te[k]:.4f}   (95 % CI {r['ci_2.5%']:.4f} - {r['ci_97.5%']:.4f})\n")
        f.write(
            f"\nConfusion matrix: TN={te.TN:,} FP={te.FP:,} FN={te.FN:,} TP={te.TP:,}\n")
        f.write(f"Top-10 % risk decile captures {dt.loc[0, 'cum_churners_captured_pct']:.1f} % of churners "
                f"({dt.loc[0, 'lift']:.1f}x lift)\n")
        f.write(
            f"Shuffled-label sanity ROC-AUC: {auc_shuf:.4f} (expected ~0.50)\n")

    log.info("Saved to: " + str(OUT_DIR))
    for p in sorted(OUT_DIR.iterdir()):
        log.info(f"   - {p.name}")
    log.info(f"\nDone in {time.time() - t_start:.0f} s.")


if __name__ == "__main__":
    main()
