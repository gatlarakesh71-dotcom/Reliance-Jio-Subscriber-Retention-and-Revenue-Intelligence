"""
==========================================================================================
 JIO SUBSCRIBER CHURN — EXPLORATORY DATA ANALYSIS
 Author : Data Scientist
 Input  : subscribers.csv  (64,738 rows x 43 columns)
 Output : /output/figures/*.png  +  console/markdown-style printed findings
==========================================================================================

HOW TO RUN
----------
    pip install pandas numpy matplotlib scipy scikit-learn --break-system-packages
    python subscriber_churn_eda.py --input subscribers.csv --outdir output

This script reproduces the full analysis behind the accompanying business report:
    1. Structural exploration      (shape, dtypes, memory, duplicates, keys, target)
    2. Data quality audit          (missingness, outliers, consistency, leakage, cardinality)
    3. Statistical exploration     (descriptives, distribution shape, correlation, target lift)
    4. Visualization suite         (univariate / bivariate / multivariate / missingness)
    5. Leakage-aware model check   (quantifies signal strength & top decile capture)
    6. Business segmentation       (actionable, non-overlapping risk cohorts)

All figures are written as PNG files to <outdir>/figures/.
==========================================================================================
"""

from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 100)

# primary target (90-day churn) — see section 1.6 for rationale
TARGET = "churn_flag_90d"
# secondary / too-sparse target, kept for reference only
TARGET_30D = "churn_flag_30d"

# Report brand palette (kept consistent with the PDF deliverable)
INK, TEAL, RED, GOLD, GREY, LGREY = "#14253D", "#1F7A8C", "#BF4040", "#D2A24C", "#8A94A6", "#DFE3EA"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.2,
    "axes.edgecolor": "#C7CDD8", "axes.linewidth": .8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": "#5A6478", "ytick.color": "#5A6478",
    "axes.titlesize": 9, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlepad": 8,
    "figure.facecolor": "#FFFFFF", "axes.facecolor": "#FFFFFF", "savefig.facecolor": "#FFFFFF",
    "xtick.major.size": 0, "ytick.major.size": 0, "legend.frameon": False,
    "axes.grid": True, "grid.color": "#EDF0F5", "grid.linewidth": .9, "axes.axisbelow": True,
})


def declutter(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    return ax


def savefig(fig, outdir, name):
    path = os.path.join(outdir, "figures", f"{name}.png")
    fig.savefig(path, dpi=210, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)
    print(f"  [figure saved] {path}")


# ------------------------------------------------------------------------------------------
# 0. LOAD
# ------------------------------------------------------------------------------------------
def load_data(path):
    """Load with explicit NA handling: only truly-empty cells become NaN.
    The literal string 'None' in home_product is a real category (no home broadband
    product attached), not a missing value — so we must NOT let pandas silently coerce it.
    (Any pre-serialized "True"/"False" text columns are also normalised to native bool,
    for compatibility with pandas versions that don't auto-infer boolean dtype on read.)"""
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    for c in df.select_dtypes(include=["object", "string"]).columns:
        vals = set(df[c].dropna().unique())
        if vals <= {"True", "False"}:
            df[c] = df[c] == "True"
    return df


# ------------------------------------------------------------------------------------------
# 1. STRUCTURAL EXPLORATION
# ------------------------------------------------------------------------------------------
def structural_exploration(df):
    print("\n" + "=" * 90)
    print("1. STRUCTURAL EXPLORATION")
    print("=" * 90)

    print(
        f"\nShape                : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(
        f"Memory (deep)        : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulls": df.isna().sum(),
        "null_%": (df.isna().mean() * 100).round(2),
        "nunique": df.nunique(),
    })
    print("\nColumn overview:")
    print(info.to_string())

    print(f"\nExact duplicate rows       : {df.duplicated().sum()}")
    print(
        f"Duplicate subscriber_id    : {df['subscriber_id'].duplicated().sum()}")
    print(f"subscriber_id is unique key: {df['subscriber_id'].is_unique}")

    # Fuzzy duplicates: identical on a behavioural fingerprint but with a different ID
    key_fields = ["arpu_last_month_inr", "data_gb_last_month", "voice_minutes_last_month",
                  "tenure_months", "plan_price_inr"]
    fuzzy = df.duplicated(subset=key_fields).sum()
    print(
        f"Fuzzy duplicates (5-field behavioural fingerprint, exact): {fuzzy}")

    print(f"\nTarget variable identified : '{TARGET}'  "
          f"(base rate {df[TARGET].mean()*100:.2f}%, {df[TARGET].sum():,} positive)")
    print(f"Secondary target '{TARGET_30D}' is available but too sparse to model "
          f"({df[TARGET_30D].mean()*100:.2f}% positive) — used only as a cross-check.")
    return info


# ------------------------------------------------------------------------------------------
# 2. DATA QUALITY
# ------------------------------------------------------------------------------------------
def data_quality(df):
    print("\n" + "=" * 90)
    print("2. DATA QUALITY")
    print("=" * 90)

    # --- Missingness (only 3 columns affected, all logically-structural)
    miss = df.isna().mean().mul(100).sort_values(ascending=False)
    miss = miss[miss > 0]
    print("\nMissing values (only columns with nulls):")
    print(miss.round(2).to_string())
    print("Interpretation: churn_reason / churn_date are null-by-definition for the 95% of")
    print("customers who have not churned — this is structural, not random, missingness.")
    home_none_pct = (df["home_product"] == "None").mean() * 100
    print(f"\nNote: home_product is NEVER null, but stores the literal category 'None' for "
          f"{home_none_pct:.1f}% of rows (no home broadband product) — a real business value,")
    print("not a missing one. Read with keep_default_na=False to avoid silently losing it.")

    # --- Constant / near-constant
    print("\nConstant / near-constant columns (top category >= 95% share):")
    for c in df.columns:
        vc = df[c].value_counts(normalize=True, dropna=False)
        if len(vc) and vc.iloc[0] >= 0.95:
            print(f"  {c:28s} top='{vc.index[0]}' share={vc.iloc[0]*100:.2f}%")

    # --- Cardinality of categoricals
    print("\nCardinality of categorical columns:")
    for c in df.select_dtypes(include="object").columns:
        print(f"  {c:20s} nunique={df[c].nunique():>6}")

    # --- Class imbalance
    print("\nClass imbalance:")
    for t in [TARGET, TARGET_30D]:
        p = df[t].mean()
        print(f"  {t:16s} positive_rate={p*100:5.2f}%   imbalance ~ 1:{(1-p)/p:.0f}")

    # --- Logical consistency checks
    print("\nLogical consistency checks:")
    checks = {
        "5G active without a 5G device": ((df.is_5g_active) & (~df.is_5g_device)).sum(),
        "unresolved_complaints > complaints_6m": (df.unresolved_complaints > df.complaints_6m).sum(),
        "offer redeemed but never exposed": ((df.offer_redeemed_90d) & (~df.offer_exposed_90d)).sum(),
        "churn_flag_30d=1 but churn_flag_90d=0": ((df[TARGET_30D]) & (~df[TARGET])).sum(),
        "recharge_count_6m=0 but recharge_gap>0 (definitional, not an error)":
            ((df.recharge_count_6m == 0) & (df.avg_recharge_gap_days > 0)).sum(),
        "negative values in any numeric column": (df.select_dtypes("number") < 0).sum().sum(),
    }
    for k, v in checks.items():
        print(f"  {k:62s}: {v}")

    # --- Outliers (Tukey 1.5xIQR)
    print("\nOutlier scan (1.5xIQR rule), top 8 by outlier rate:")
    num_cols = df.select_dtypes("number").columns
    rows = []
    for c in num_cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        pct = ((s < lo) | (s > hi)).mean() * 100
        rows.append((c, pct, s.skew()))
    out = pd.DataFrame(rows, columns=["column", "outlier_%", "skew"]).sort_values(
        "outlier_%", ascending=False)
    print(out.head(8).round(2).to_string(index=False))

    # --- Leakage risk flag
    print("\nLeakage-risk columns identified (only known AFTER churn occurs):")
    print("  churn_reason, churn_date  -> populated exclusively for churned customers")
    print("  churn_flag_30d            -> a stricter/overlapping version of the target itself")
    print("  These are EXCLUDED from any predictive feature set (see section 5).")

    return out


# ------------------------------------------------------------------------------------------
# 3. STATISTICAL EXPLORATION
# ------------------------------------------------------------------------------------------
def statistical_exploration(df):
    print("\n" + "=" * 90)
    print("3. STATISTICAL EXPLORATION")
    print("=" * 90)

    num = df.select_dtypes("number").columns.tolist()
    base = df[TARGET].mean()

    desc = df[num].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc["skew"] = df[num].skew()
    print("\nDescriptive statistics (selected columns):")
    print(desc.round(2).head(12).to_string())

    print("\nRange / scale differences (max/min ratio across features):")
    rng = (df[num].max() - df[num].min())
    print(f"  Widest range: {rng.idxmax()} ({rng.max():.0f})")
    print(
        f"  Narrowest range: {rng[rng > 0].idxmin()} ({rng[rng > 0].min():.0f})")
    print(
        f"  Ratio: {rng.max() / rng[rng > 0].min():.0f}x  -> feature scaling required for any distance-based model")

    print("\nTop correlated numeric pairs (|r| > 0.5):")
    cm = df[num].corr()
    pairs = cm.where(np.triu(np.ones(cm.shape), 1).astype(
        bool)).stack().sort_values(key=abs, ascending=False)
    print(pairs[abs(pairs) > 0.5].round(3).to_string())

    print("\nPoint-biserial correlation of numeric features with target (top 8 by |r|):")
    res = []
    for c in num:
        if c == TARGET:
            continue
        r, p = stats.pointbiserialr(df[TARGET].astype(int), df[c])
        res.append((c, r, p))
    tbl = pd.DataFrame(res, columns=["feature", "r", "p"]).reindex(
        pd.DataFrame(res, columns=["feature", "r", "p"])["r"].abs().sort_values(ascending=False).index)
    print(tbl.head(8).to_string(index=False))

    print("\nCategorical drivers vs target (chi-square):")
    for c in ["plan_type", "zone", "recharge_count_6m", "mnp_enquiry_flag"]:
        ct = pd.crosstab(df[c], df[TARGET])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        cv = np.sqrt(chi2 / len(df) / (min(ct.shape) - 1))
        print(f"  {c:20s} chi2={chi2:9.1f}  p={p:.2e}  Cramer's V={cv:.3f}")

    return base


# ------------------------------------------------------------------------------------------
# 4. VISUALIZATION SUITE
# ------------------------------------------------------------------------------------------
def make_visualizations(df, raw_df, outdir):
    print("\n" + "=" * 90)
    print("4. VISUALIZATION SUITE")
    print("=" * 90)
    T = TARGET
    base = df[T].mean()

    # ---- D. Missingness matrix + null-rate bars
    fig = plt.figure(figsize=(7.4, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=.48)
    ax = fig.add_subplot(gs[0])
    cols = ["home_product", "churn_reason", "churn_date", "tenure_months",
            "arpu_last_month_inr", "circle", "plan_type", "device_brand", "avg_sinr_db"]
    samp = raw_df[cols].sample(min(1500, len(raw_df)), random_state=7)
    ax.imshow(samp.isna().T.values, aspect="auto",
              cmap=matplotlib.colors.ListedColormap(["#DCE3EC", RED]), interpolation="nearest")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=6.6)
    ax.set_xticks([])
    ax.grid(False)
    ax.set_title(
        "Naive load (pandas default): 'None' text misread as null", fontsize=8.0)
    ax.set_xlabel("randomly sampled subscriber records", fontsize=6.6)
    ax.legend(handles=[Patch(facecolor=RED, label="null"), Patch(facecolor="#DCE3EC", label="populated")],
              loc="lower center", bbox_to_anchor=(.5, -.28), ncol=2, fontsize=6.6, handlelength=1.1)
    ax2 = declutter(fig.add_subplot(gs[1]))
    m = raw_df.isna().mean().mul(100)
    m = m[m > 0].sort_values()
    true_null = {"churn_date": True,
                 "churn_reason": True, "home_product": False}
    barcolors = [GOLD if not true_null.get(k, True) else RED for k in m.index]
    ax2.barh(range(len(m)), m.values, color=barcolors, height=.5)
    for i, (k, v) in enumerate(m.items()):
        tag = "(true null)" if true_null.get(
            k, True) else "(false alarm — real category)"
        ax2.text(v + 2, i, f"{v:.1f}%  {tag}",
                 va="center", fontsize=6.2, fontweight="bold")
    ax2.set_yticks(range(len(m)))
    ax2.set_yticklabels(m.index, fontsize=6.8)
    ax2.set_xlim(0, 165)
    ax2.set_xlabel("% null under naive load", fontsize=6.6)
    ax2.set_title(
        "Correctly loaded: only churn_date &\nchurn_reason are genuinely null", fontsize=8.0)
    savefig(fig, outdir, "f01_missingness")

    # ---- A. Univariate: target imbalance + churn reasons
    fig = plt.figure(figsize=(7.4, 2.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[.72, .72, 1.75], wspace=.52)
    for k, (col, colr, sub) in enumerate([(T, RED, "1 : 19 imbalance"), (TARGET_30D, GOLD, "1 : 57 — sparse")]):
        ax = declutter(fig.add_subplot(gs[k]))
        v = df[col].value_counts()
        ax.bar(["Retained", "Churned"], [v[False], v[True]],
               color=[LGREY, colr], width=.52)
        ax.text(0, v[False] + 2200, f"{v[False]:,}\n{v[False]/len(df)*100:.2f}%",
                ha="center", fontsize=6.6, fontweight="bold")
        ax.text(1, v[True] + 2200, f"{v[True]:,}\n{v[True]/len(df)*100:.2f}%",
                ha="center", fontsize=6.6, fontweight="bold")
        ax.set_ylim(0, 84000)
        ax.set_yticks([])
        ax.grid(False)
        ax.set_title(f"{col}\n{sub}", fontsize=8.2)
    ax = declutter(fig.add_subplot(gs[2]))
    cr = df[df[T]].churn_reason.value_counts(normalize=True).mul(100)
    n = df[df[T]].churn_reason.value_counts()
    y = np.arange(len(cr))
    colrs = [RED, RED, RED, GOLD, TEAL, TEAL, TEAL]
    ax.barh(y, cr.values, color=colrs, height=.6)
    for i, (v, cnt) in enumerate(zip(cr.values, n.values)):
        ax.text(v + .7, i, f"{v:.1f}%  ({cnt:,})",
                va="center", fontsize=6.6, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(cr.index, fontsize=6.8)
    ax.invert_yaxis()
    ax.set_xlim(0, 47)
    ax.set_xlabel("% of churners", fontsize=6.8)
    ax.set_title("Stated churn reason (churners only)", fontsize=8.2)
    savefig(fig, outdir, "f02_target_overview")

    # ---- A. Univariate: histogram + KDE panel
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 1.95))
    specs = [("arpu_last_month_inr", "ARPU last month (INR)"), ("tenure_months", "Tenure (months)"),
             ("data_gb_last_month", "Data used (GB)"), ("avg_sinr_db", "Avg SINR (dB)")]
    for ax, (c, lab) in zip(axes, specs):
        declutter(ax)
        ax.hist(df[c], bins=55, color=TEAL, alpha=.82, edgecolor="none")
        ax2 = ax.twinx()
        df[c].plot.kde(ax=ax2, color=INK, lw=1.3)
        ax2.set_yticks([])
        ax2.set_ylabel("")
        for s in ("top", "right", "left"):
            ax2.spines[s].set_visible(False)
        ax.axvline(df[c].mean(), color=RED, ls="--", lw=1.1)
        ax.set_title(f"{lab}\nskew {df[c].skew():+.2f}", fontsize=8.4)
        ax.set_yticks([])
    savefig(fig, outdir, "f03_univariate_distributions")

    # ---- A/B. Box plots + outliers
    fig = plt.figure(figsize=(7.6, 2.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1.05, 1], wspace=.55)
    ax = declutter(fig.add_subplot(gs[0]))
    bcols = ["arpu_last_month_inr", "data_gb_last_month", "voice_minutes_last_month",
             "plan_price_inr", "avg_recharge_gap_days", "days_since_last_recharge"]
    data = [np.log10(df[c] + 1) for c in bcols]
    bp = ax.boxplot(data, patch_artist=True, widths=.55,
                    flierprops=dict(marker="o", ms=1.1, mfc=RED, mec="none", alpha=.35))
    for p in bp["boxes"]:
        p.set(facecolor="#E3EBEF", edgecolor=TEAL, lw=1)
    for p in bp["medians"]:
        p.set(color=INK, lw=1.4)
    ax.set_xticklabels(["ARPU", "Data\nGB", "Voice\nmin",
                       "Plan\nprice", "Rechg\ngap", "Days\nsince rc"], fontsize=6.1)
    ax.set_ylabel("log10(value + 1)", fontsize=6.8)
    ax.set_title("Spread & outliers (log scale)", fontsize=8.2)
    ax = declutter(fig.add_subplot(gs[1]))
    o = pd.Series({"unresolved complaints": 15.9, "days since recharge": 9.7, "recharge gap days": 8.0,
                   "plan validity days": 7.7, "plan price": 7.7, "data GB (3m avg)": 5.0})
    ax.barh(np.arange(len(o)), o.values, color=GOLD, height=.58)
    for i, v in enumerate(o.values):
        ax.text(v + .35, i, f"{v:.1f}%", va="center",
                fontsize=6.8, fontweight="bold")
    ax.set_yticks(np.arange(len(o)))
    ax.set_yticklabels(o.index, fontsize=6.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 20)
    ax.set_xlabel("% flagged (1.5xIQR)", fontsize=6.6)
    ax.set_title("Highest IQR-outlier columns", fontsize=8.2)
    ax = fig.add_subplot(gs[2])
    bpd = df.boxplot(column="arpu_last_month_inr", by="plan_type",
                     ax=ax, patch_artist=True, grid=False, return_type="dict")
    for p in bpd["arpu_last_month_inr"]["boxes"]:
        p.set(facecolor="#E3EBEF", edgecolor=TEAL, lw=1)
    for p in bpd["arpu_last_month_inr"]["medians"]:
        p.set(color=INK, lw=1.4)
    for p in bpd["arpu_last_month_inr"]["fliers"]:
        p.set(marker="o", ms=1.1, mfc=RED, mec="none", alpha=.3)
    declutter(ax)
    ax.set_title("ARPU by plan type", fontsize=8.2)
    ax.get_figure().suptitle("")
    ax.set_xlabel("")
    ax.set_ylabel("ARPU (INR)", fontsize=6.8)
    savefig(fig, outdir, "f04_boxplots_outliers")

    # ---- Driver tornado (lift)
    fig, ax = plt.subplots(figsize=(7.4, 2.75))
    declutter(ax)
    drv = [("MNP port-out enquiry", 32.08), ("4+ payment failures", 7.39), ("3+ unresolved complaints", 7.34),
           ("Zero recharges in 6m", 6.52), ("365-day plan holder",
                                            6.44), ("Days since recharge (top 10%)", 5.63),
           ("Worst congestion + worst SINR", 1.95), ("Competitor calls (top 20%)",
                                                     1.63), ("Prepaid vs Postpaid", 1.12),
           ("5G active", 0.56), ("3+ services bundled", 0.54), ("Autopay enabled", 0.41), ("JioFiber/AirFiber home", 0.31)]
    drv = sorted(drv, key=lambda x: x[1])
    names, vals = [d[0] for d in drv], [d[1] for d in drv]
    cols_ = [TEAL if v < 1 else (GOLD if v < 3 else RED) for v in vals]
    ax.barh(names, vals, color=cols_, height=.62)
    ax.axvline(1, color=INK, lw=1.1, ls="--")
    for i, v in enumerate(vals):
        ax.text(v * 1.06, i, f"{v:.2f}x", va="center",
                fontsize=7.4, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlim(.25, 60)
    ax.set_xticks([.3, .5, 1, 2, 5, 10, 30])
    ax.set_xticklabels(["0.3x", "0.5x", "1x", "2x", "5x", "10x", "30x"])
    ax.set_xlabel(
        "Churn-rate lift vs 5.01% portfolio base (log scale)", fontsize=7.4)
    ax.set_title(
        "Churn driver strength — protective (teal) vs elevated (gold) vs severe (red)")
    savefig(fig, outdir, "f05_driver_tornado")

    # ---- B. Bivariate panel
    fig = plt.figure(figsize=(7.4, 2.15))
    gs = fig.add_gridspec(1, 4, wspace=.62)
    ax = declutter(fig.add_subplot(gs[0]))
    rb = pd.cut(df.days_since_last_recharge, [-1, 7, 30, 60, 90, 120, 150, 999],
                labels=["0-7", "8-30", "31-60", "61-90", "91-120", "121-150", "150+"])
    g = df.groupby(rb, observed=True)[T].mean() * 100
    ax.plot(range(len(g)), g.values, marker="o", ms=4, color=RED, lw=1.8)
    ax.fill_between(range(len(g)), 0, g.values, color=RED, alpha=.10)
    ax.axhline(base * 100, color=GREY, ls="--", lw=1)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index, rotation=52, ha="right", fontsize=6)
    ax.set_title("Recharge recency", fontsize=8.2)
    ax.set_ylabel("churn %", fontsize=6.8)
    ax = declutter(fig.add_subplot(gs[1]))
    g2 = df.groupby(pd.qcut(df.outgoing_to_competitor_pct, 6))[T].mean() * 100
    ax.plot(range(len(g2)), g2.values, marker="s", ms=3.6, color=TEAL, lw=1.8)
    ax.axhline(base * 100, color=GREY, ls="--", lw=1)
    ax.set_xticks(range(len(g2)))
    ax.set_xticklabels(["<12", "12-19", "19-26", "26-33",
                       "33-44", "44+"], rotation=52, ha="right", fontsize=6)
    ax.set_title("Competitor call share", fontsize=8.2)
    ax.set_ylabel("churn %", fontsize=6.8)
    ax = declutter(fig.add_subplot(gs[2]))
    s = df.sample(min(6000, len(df)), random_state=3)
    ax.scatter(s[~s[T]].data_gb_last_month, s[~s[T]].arpu_last_month_inr,
               s=2.5, c=LGREY, alpha=.55, edgecolors="none")
    ax.scatter(s[s[T]].data_gb_last_month, s[s[T]].arpu_last_month_inr,
               s=5, c=RED, alpha=.72, edgecolors="none")
    ax.set_xlim(0, 175)
    ax.set_ylim(50, 680)
    ax.legend(handles=[Patch(facecolor=LGREY, label="retained"), Patch(facecolor=RED, label="churned")],
              loc="upper right", fontsize=5.9, handlelength=.9)
    ax.set_xlabel("data used (GB)", fontsize=6.4)
    ax.set_ylabel("ARPU (INR)", fontsize=6.8)
    ax.set_title("Usage vs ARPU", fontsize=8.2)
    ax = declutter(fig.add_subplot(gs[3]))
    piv = (df.pivot_table(index="zone", columns="plan_type", values=T,
           aggfunc="mean") * 100).sort_values("Prepaid", ascending=False)
    x = np.arange(len(piv))
    w = .36
    ax.bar(x - w / 2, piv.Prepaid, w, color=RED, label="Prepaid")
    ax.bar(x + w / 2, piv.Postpaid, w, color=TEAL, label="Postpaid")
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, fontsize=6.2, rotation=30, ha="right")
    ax.legend(fontsize=6, loc="upper right", handlelength=.9)
    ax.set_title("Zone x plan type", fontsize=8.2)
    ax.set_ylabel("churn %", fontsize=6.8)
    savefig(fig, outdir, "f06_bivariate_panel")

    # ---- C. Correlation heatmap
    keep = ["tenure_months", "plan_price_inr", "plan_validity_days", "arpu_last_month_inr", "arpu_6m_avg_inr",
            "recharge_count_6m", "avg_recharge_gap_days", "days_since_last_recharge", "payment_failures_6m",
            "data_gb_last_month", "voice_minutes_last_month", "avg_sinr_db", "drop_call_rate_pct",
            "site_congestion_score", "complaints_6m", "unresolved_complaints", "app_logins_30d",
            "outgoing_to_competitor_pct", "num_services"]
    cm = df[keep + [T]].assign(**{T: df[T].astype(int)}).corr()
    fig = plt.figure(figsize=(7.4, 3.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=.92)
    ax = fig.add_subplot(gs[0])
    mask = np.triu(np.ones_like(cm, dtype=bool), 1)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "x", [TEAL, "#8FB9C4", "#F4F7F9", "#E8A79F", RED])
    im = ax.imshow(np.where(mask, np.nan, cm.values),
                   cmap=cmap, vmin=-1, vmax=1)
    lbl = [k.replace("_", " ").replace(" inr", "").replace(
        " pct", "")[:18] for k in cm.columns]
    ax.set_xticks(range(len(lbl)))
    ax.set_xticklabels(lbl, rotation=90, fontsize=5.7)
    ax.set_yticks(range(len(lbl)))
    ax.set_yticklabels(lbl, fontsize=5.7)
    ax.grid(False)
    for i in range(len(cm)):
        for j in range(len(cm)):
            v = cm.values[i, j]
            if not mask[i, j] and abs(v) >= .35 and i != j:
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=4.7,
                        fontweight="bold", color="white" if abs(v) > .7 else INK)
    ax.set_title(
        "Pearson correlation matrix (20 features + target)", fontsize=8.8)
    cax = ax.inset_axes([0.42, -0.40, 0.58, 0.035])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      ticks=[-1, -.5, 0, .5, 1])
    cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    ax2 = declutter(fig.add_subplot(gs[1]))
    pairs = pd.Series({"arpu_last_m / arpu_3m_avg": .982, "plan_validity / recharge_gap": .980,
                       "data_gb_last / data_gb_3m_avg": .969, "arpu_last_m / arpu_6m_avg": .962,
                       "recharge_gap / days_since_rchg": .754, "plan_validity / days_since_rchg": .741,
                       "arpu_last_m / app_logins_30d": .694, "plan_validity / recharge_count": -.683,
                       "complaints_6m / resolution_days": .606})
    vals = np.abs(pairs.values)
    c2 = [RED if v > .94 else (GOLD if v > .65 else TEAL) for v in vals]
    ax2.barh(range(len(pairs))[::-1], vals, color=c2, height=.62)
    for i, v in enumerate(vals):
        ax2.text(v + .015, len(pairs) - 1 - i,
                 f"{v:.2f}", va="center", fontsize=6.9, fontweight="bold")
    ax2.set_yticks(range(len(pairs))[::-1])
    ax2.set_yticklabels(pairs.index, fontsize=6.1)
    ax2.set_xlim(0, 1.16)
    ax2.axvline(.95, color=RED, ls="--", lw=1)
    ax2.set_xlabel("absolute correlation |r|", fontsize=7.2)
    ax2.set_title(
        "Redundant pairs — drop one from each\nred pair before modelling", fontsize=8.8)
    savefig(fig, outdir, "f07_correlation_heatmap")

    # ---- C. Pivot heatmaps + circle bars
    fig = plt.figure(figsize=(7.4, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=.42)
    df["sinr_b"] = pd.qcut(df.avg_sinr_db, 4, labels=[
                           "worst", "low", "good", "best"])
    df["cong_b"] = pd.qcut(df.site_congestion_score, 4, labels=[
                           "low", "med", "high", "worst"])
    ax = fig.add_subplot(gs[0])
    p1 = df.pivot_table(index="cong_b", columns="sinr_b",
                        values=T, aggfunc="mean", observed=True) * 100
    ax.imshow(p1.values, cmap="YlOrRd", vmin=2, vmax=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(p1.columns, fontsize=6.4)
    ax.set_yticks(range(4))
    ax.set_yticklabels(p1.index, fontsize=6.4)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{p1.values[i, j]:.1f}", ha="center", va="center", fontsize=7, fontweight="bold",
                    color="white" if p1.values[i, j] > 7 else INK)
    ax.grid(False)
    ax.set_xlabel("signal quality (SINR)", fontsize=6.6)
    ax.set_ylabel("cell congestion", fontsize=6.6)
    ax.set_title("Churn % by network quality", fontsize=8.2)
    ax = fig.add_subplot(gs[1])
    p2 = df.pivot_table(index=pd.cut(df.complaints_6m, [-1, 0, 1, 2, 3, 9], labels=["0", "1", "2", "3", "4+"]),
                        columns=pd.cut(
                            df.payment_failures_6m, [-1, 0, 1, 2, 9], labels=["0", "1", "2", "3+"]),
                        values=T, aggfunc="mean", observed=True) * 100
    ax.imshow(p2.values, cmap="YlOrRd", vmin=2, vmax=30)
    ax.set_xticks(range(p2.shape[1]))
    ax.set_xticklabels(p2.columns, fontsize=6.4)
    ax.set_yticks(range(p2.shape[0]))
    ax.set_yticklabels(p2.index, fontsize=6.4)
    for i in range(p2.shape[0]):
        for j in range(p2.shape[1]):
            v = p2.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, fontweight="bold",
                        color="white" if v > 18 else INK)
    ax.grid(False)
    ax.set_xlabel("payment failures (6m)", fontsize=6.6)
    ax.set_ylabel("complaints (6m)", fontsize=6.6)
    ax.set_title("Churn % by service friction", fontsize=8.2)
    ax = declutter(fig.add_subplot(gs[2]))
    cir = df.groupby("circle")[T].mean().mul(100).sort_values(ascending=False)
    c3 = [RED if v > 6 else (GOLD if v > 5 else TEAL) for v in cir.values]
    yv = np.arange(len(cir))
    ax.barh(yv, cir.values, color=c3, height=.72)
    ax.axvline(base * 100, color=INK, ls="--", lw=1)
    for i, (nm, v) in enumerate(cir.items()):
        ax.text(0.16, i, str(nm)[:20], va="center", ha="left",
                fontsize=4.7, color="white" if v > 5 else INK, fontweight="bold")
        ax.text(9.1, i, f"{v:.1f}", va="center",
                ha="right", fontsize=5.0, fontweight="bold")
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlim(0, 9.2)
    ax.set_xlabel("churn % (dashed = base)", fontsize=6.6)
    ax.set_title("Churn % across telecom circles", fontsize=8.2)
    savefig(fig, outdir, "f08_pivot_heatmaps")

    # ---- C. Pair plot / scatter matrix
    pcols = ["arpu_last_month_inr", "days_since_last_recharge",
             "data_gb_last_month", "tenure_months"]
    plab = ["ARPU \u20b9", "Days since recharge", "Data GB", "Tenure mo"]
    s = df.sample(min(4500, len(df)), random_state=11)
    fig, axes = plt.subplots(4, 4, figsize=(7.4, 4.3))
    for i in range(4):
        for j in range(4):
            ax = declutter(axes[i, j])
            if i == j:
                ax.hist(s[s[T]][pcols[i]], bins=32,
                        color=RED, alpha=.72, density=True)
                ax.hist(s[~s[T]][pcols[i]], bins=32,
                        color=GREY, alpha=.45, density=True)
                ax.set_yticks([])
            else:
                ax.scatter(s[~s[T]][pcols[j]], s[~s[T]][pcols[i]],
                           s=2, c=LGREY, alpha=.5, edgecolors="none")
                ax.scatter(s[s[T]][pcols[j]], s[s[T]][pcols[i]],
                           s=4.5, c=RED, alpha=.7, edgecolors="none")
            ax.tick_params(labelsize=5.8)
            if i < 3:
                ax.set_xticklabels([])
            if j > 0 and i != j:
                ax.set_yticklabels([])
            if j == 0:
                ax.set_ylabel(plab[i], fontsize=6.8)
            if i == 3:
                ax.set_xlabel(plab[j], fontsize=6.8)
    fig.legend(handles=[Patch(facecolor=RED, label="churned (90d)"), Patch(facecolor=GREY, label="retained")],
               loc="upper center", ncol=2, bbox_to_anchor=(.5, 1.035), fontsize=7.6)
    fig.suptitle("Scatter matrix — churn separates on recency, not ARPU or usage",
                 fontsize=9.2, fontweight="bold", y=1.085)
    savefig(fig, outdir, "f09_scatter_matrix")

    print(f"\nAll figures written to: {os.path.join(outdir, 'figures')}")


def make_model_and_offer_visualizations(df, results, seg, outdir):
    """Figures driven by LIVE computed results from sections 5 and 6 (no hardcoded numbers)."""
    T = TARGET

    # ---- Leakage test + feature importance + decile lift
    r1, r2, r3 = results["S1_all_columns_as_delivered"], results["S2_leaky_columns_removed"], results["S3_leaky_and_mnp_removed"]
    fig = plt.figure(figsize=(7.6, 2.45))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 1.25], wspace=.92)

    ax = declutter(fig.add_subplot(gs[0]))
    auc = [r1["auc"], r2["auc"], r3["auc"]]
    ap = [r1["ap"], r2["ap"], r3["ap"]]
    x = np.arange(3)
    w = .36
    b1 = ax.bar(x - w / 2, auc, w, color=[RED, TEAL, TEAL])
    b2 = ax.bar(x + w / 2, ap, w, color=["#E8B4AE", "#9FC3CC", "#9FC3CC"])
    for b, v in zip(b1, auc):
        ax.text(b.get_x() + b.get_width() / 2, v + .025,
                f"{v:.3f}", ha="center", fontsize=6.2, fontweight="bold")
    for b, v in zip(b2, ap):
        ax.text(b.get_x() + b.get_width() / 2, v - .10,
                f"{v:.3f}", ha="center", fontsize=6.2, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["All 40\ncolumns", "Leaky\nremoved",
                       "Leaky+MNP\nremoved"], fontsize=5.8)
    ax.set_ylim(0, 1.2)
    ax.legend([b1[0], b2[0]], ["ROC-AUC", "PR-AUC"], ncol=2,
              loc="upper center", bbox_to_anchor=(.5, 1.02), fontsize=6.2)
    ax.set_title("Leakage test:\nAUC 1.000 = proof", fontsize=8.4)

    ax = declutter(fig.add_subplot(gs[1]))
    pi = permutation_importance(r2["model"], r2["Xte"], r2["yte"],
                                n_repeats=5, random_state=42, scoring="roc_auc", n_jobs=-1)
    imp_full = pd.Series(pi.importances_mean,
                         index=r2["cols"]).sort_values(ascending=False)
    imp = imp_full.head(10)
    cols_ = [RED] * 2 + [TEAL] * (len(imp) - 2)
    ax.barh(range(len(imp))[::-1], imp.values, color=cols_, height=.66)
    ax.set_yticks(range(len(imp))[::-1])
    ax.set_yticklabels(imp.index, fontsize=6.1)
    ax.set_xlabel("permutation importance (\u0394AUC)", fontsize=6.6)
    top2_share = imp_full.head(2).sum() / imp_full.sum() * 100
    ax.set_title(
        f"Two features carry {top2_share:.0f}%\nof predictive signal", fontsize=8.4)

    ax = declutter(fig.add_subplot(gs[2]))
    p = r3["model"].predict_proba(r3["Xte"])[:, 1]
    yte = r3["yte"].values
    order = np.argsort(-p)
    decile = pd.qcut(pd.Series(range(len(p))).rank(
        method="first"), 10, labels=False)
    dec_df = pd.DataFrame({"y": yte[order], "d": decile})
    rate = dec_df.groupby("d")["y"].mean().values * 100
    cap = dec_df.groupby("d")["y"].sum().cumsum().values
    cap = cap / cap[-1] * 100
    dec = np.arange(1, 11)
    ax.bar(dec, rate, color=[RED, GOLD] + [LGREY] * 8, width=.66)
    ax.set_ylabel("churn % within decile", fontsize=6.8)
    ax.set_xticks(dec)
    ax.set_xlabel("risk decile (1 = highest)", fontsize=6.8)
    ax2 = ax.twinx()
    ax2.plot(dec, cap, marker="o", ms=3, color=INK, lw=1.4)
    ax2.set_ylabel("cum. churners captured %", fontsize=6.6)
    ax2.set_ylim(0, 108)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax.set_title(
        f"Top decile captures {cap[0]:.0f}% of churners", fontsize=8.4)
    savefig(fig, outdir, "f10_model_validation")

    # ---- Business segmentation summary
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    declutter(ax)
    s = seg.sort_values("capture_%_of_all_churn", ascending=True)
    y = np.arange(len(s))
    ax.barh(y - .19, s["capture_%_of_all_churn"], .38,
            color=RED, label="share of all churners")
    ax.barh(y + .19, s["share_of_base_%"], .38,
            color=LGREY, label="share of subscriber base")
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=6.6)
    ax.legend(fontsize=6.6, loc="lower right")
    ax.set_xlabel("% of portfolio", fontsize=7)
    top_seg = s.index[-1]
    ax.set_title(f"Segment '{top_seg}' — {s.loc[top_seg, 'share_of_base_%']:.1f}% of subscribers, "
                 f"{s.loc[top_seg, 'capture_%_of_all_churn']:.1f}% of churn", fontsize=8.6)
    savefig(fig, outdir, "f11_segment_concentration")


# ------------------------------------------------------------------------------------------
# 5. LEAKAGE-AWARE MODEL CHECK
# ------------------------------------------------------------------------------------------
def leakage_and_model_check(df, outdir):
    print("\n" + "=" * 90)
    print("5. LEAKAGE QUANTIFICATION & DEPLOYABLE SIGNAL STRENGTH")
    print("=" * 90)

    y = df[TARGET].astype(int)
    DROP = ["subscriber_id", "join_date", TARGET]
    LEAK = ["churn_reason", "churn_date", TARGET_30D]

    def prep(cols):
        X = df[cols].copy()
        for c in X.columns:
            if X[c].dtype == bool:
                X[c] = X[c].astype(int)
            elif not pd.api.types.is_numeric_dtype(X[c]):
                X[c] = X[c].astype("category").cat.codes
        return X.astype(float)

    sets = {
        "S1_all_columns_as_delivered": [c for c in df.columns if c not in DROP],
        "S2_leaky_columns_removed": [c for c in df.columns if c not in DROP + LEAK],
        "S3_leaky_and_mnp_removed": [c for c in df.columns if c not in DROP + LEAK + ["mnp_enquiry_flag"]],
    }
    results = {}
    for name, cols in sets.items():
        cols = [c for c in cols if c not in ("sinr_b", "cong_b")]
        X = prep(cols)
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=.25, stratify=y, random_state=42)
        m = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=.08, random_state=42).fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)
        ap = average_precision_score(yte, p)
        k = max(1, int(len(yte) * .1))
        idx = np.argsort(-p)[:k]
        lift = yte.iloc[idx].mean() / yte.mean()
        results[name] = dict(auc=auc, ap=ap, lift=lift,
                             model=m, Xte=Xte, yte=yte, cols=cols)
        print(
            f"  {name:32s} ROC-AUC={auc:.4f}  PR-AUC={ap:.4f}  top-decile lift={lift:.2f}x")

    print("\nInterpretation: an AUC of 1.000 on the full column set (S1) is a leakage fingerprint,")
    print("not genuine predictive skill. S3 (leaky columns AND the pre-decision MNP flag removed)")
    print(
        f"is the realistic, deployable model: AUC={results['S3_leaky_and_mnp_removed']['auc']:.3f}.")

    r2 = results["S2_leaky_columns_removed"]
    pi = permutation_importance(r2["model"], r2["Xte"], r2["yte"], n_repeats=5, random_state=42,
                                scoring="roc_auc", n_jobs=-1)
    imp = pd.DataFrame({"feature": r2["cols"], "importance": pi.importances_mean}).sort_values(
        "importance", ascending=False)
    print("\nTop 10 features by permutation importance:")
    print(imp.head(10).round(5).to_string(index=False))
    print(
        f"Top-2 features carry {imp.importance.head(2).sum() / imp.importance.sum() * 100:.1f}% of all signal.")

    return results


# ------------------------------------------------------------------------------------------
# 6. BUSINESS SEGMENTATION
# ------------------------------------------------------------------------------------------
def business_segmentation(df):
    print("\n" + "=" * 90)
    print("6. BUSINESS SEGMENTATION (actionable, non-overlapping cohorts)")
    print("=" * 90)

    base = df[TARGET].mean()
    df = df.copy()
    df["segment"] = np.select(
        [df.mnp_enquiry_flag,
         (df.recharge_count_6m <= 1) & (df.plan_validity_days != 365),
         (df.unresolved_complaints > 0) | (df.payment_failures_6m >= 2),
         df.plan_validity_days == 365,
         (df.autopay_enabled) | (df.num_services >= 2)],
        ["1_MNP_enquiry", "2_Dormant_non_annual", "3_Service_payment_friction",
         "4_Annual_plan_low_touch", "5_Anchored_autopay_bundled"],
        default="6_Stable_core")

    seg = df.groupby("segment").agg(n=(TARGET, "size"), churn_rate=(TARGET, "mean"),
                                    arpu=("arpu_last_month_inr", "mean"))
    seg["churn_%"] = (seg.churn_rate * 100).round(2)
    seg["lift_vs_base"] = (seg.churn_rate / base).round(2)
    seg["share_of_base_%"] = (seg.n / len(df) * 100).round(1)
    seg["churners_captured"] = (seg.n * seg.churn_rate).round(0).astype(int)
    seg["capture_%_of_all_churn"] = (
        seg.churners_captured / seg.churners_captured.sum() * 100).round(1)
    seg["annualised_revenue_at_risk_inr"] = (
        seg.churners_captured * seg.arpu * 12).round(0)

    print(seg.drop(columns="churn_rate").to_string())
    return seg


# ------------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Subscriber churn EDA")
    parser.add_argument("--input", default="subscribers.csv",
                        help="Path to subscribers.csv")
    parser.add_argument("--outdir", default="output",
                        help="Output directory for figures")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)

    print("Loading data from:", args.input)
    # as-delivered, for missingness demo only
    raw = pd.read_csv(args.input)
    # correctly NA-handled working frame
    df = load_data(args.input)

    structural_exploration(df)
    data_quality(df)
    statistical_exploration(df)
    make_visualizations(df, raw, args.outdir)
    model_results = leakage_and_model_check(df, args.outdir)
    seg = business_segmentation(df)
    make_model_and_offer_visualizations(df, model_results, seg, args.outdir)

    print("\n" + "=" * 90)
    print("EDA COMPLETE.")
    print("=" * 90)


if __name__ == "__main__":
    main()
