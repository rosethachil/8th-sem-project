"""
analyzer.py
===========
Phase 5 — Feature Analysis & Selection (NO model training)

Functions:
  analyze_features()        → basic stats for each feature column
  plot_label_distribution() → class balance bar chart
  plot_correlation()        → feature correlation heatmap (sampled)
  run_feature_selection()   → variance threshold + mutual info
  summarize_gaps()          → report dataset issues (NaNs, imbalance, etc.)

All outputs are saved to outputs/ and outputs/plots/.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for script mode
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

from config import (
    PLOTS_DIR, CSV_STATS, EMOTION_MAP, ensure_output_dirs
)


# ─────────────────────────────────────────────
# 1. BASIC STATS
# ─────────────────────────────────────────────
def analyze_features(df: pd.DataFrame, modality: str = "EEG") -> pd.DataFrame:
    """
    Compute per-feature statistics: mean, std, variance, min, max, skewness.
    Saves to feature_stats.csv.
    Returns the stats DataFrame.
    """
    ensure_output_dirs()
    # Keep only numeric feature columns (drop metadata)
    meta_cols = ["trial_id", "subject_id", "session_id", "label", "time_segment"]
    feat_cols  = [c for c in df.columns if c not in meta_cols]
    feat_df    = df[feat_cols].astype(float)

    print(f"\n  Feature analysis ({modality}): {len(feat_cols)} features, {len(feat_df):,} samples")

    stats = pd.DataFrame({
        "feature":  feat_cols,
        "mean":     feat_df.mean().values,
        "std":      feat_df.std().values,
        "variance": feat_df.var().values,
        "min":      feat_df.min().values,
        "max":      feat_df.max().values,
        "skewness": feat_df.apply(lambda col: skew(col.dropna())).values,
        "nan_count": feat_df.isna().sum().values,
    })
    stats = stats.round(6)
    stats.to_csv(CSV_STATS, index=False)
    print(f"  ✓ feature_stats.csv saved  ({len(stats)} features)")

    # Quick summary printout
    print(f"\n  Global stats across all features:")
    print(f"    Mean of means : {stats['mean'].mean():.4f}")
    print(f"    Mean variance : {stats['variance'].mean():.4f}")
    print(f"    Total NaNs    : {int(stats['nan_count'].sum())}")
    print(f"    Near-zero var : {(stats['variance'] < 1e-6).sum()} features")
    return stats


# ─────────────────────────────────────────────
# 2. LABEL DISTRIBUTION PLOT
# ─────────────────────────────────────────────
def plot_label_distribution(df: pd.DataFrame, title: str = "Label Distribution"):
    """Save class balance bar chart."""
    ensure_output_dirs()
    counts = df["label"].value_counts().sort_index()
    labels = [EMOTION_MAP.get(i, str(i)) for i in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts.values,
                  color=["#4e9af1", "#e05c5c", "#f5a623", "#6fcf97"], edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Emotion Class")
    ax.set_ylabel("Number of Samples (time-segment level)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "label_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ label_distribution.png saved")
    _print_balance_table(counts)


def _print_balance_table(counts):
    total = counts.sum()
    print(f"\n  Label distribution:")
    print(f"  {'Class':<10} {'Name':<10} {'Count':>8}  {'%':>6}")
    print("  " + "-" * 38)
    for cls, cnt in counts.items():
        print(f"  {cls:<10} {EMOTION_MAP.get(cls,'?'):<10} {cnt:>8,}  {cnt/total*100:>5.1f}%")


# ─────────────────────────────────────────────
# 3. CORRELATION HEATMAP (sampled)
# ─────────────────────────────────────────────
def plot_correlation(df: pd.DataFrame, n_features: int = 50, n_samples: int = 5000):
    """
    Plot correlation heatmap for the first n_features features,
    sampled from at most n_samples rows (for speed).
    """
    ensure_output_dirs()
    meta_cols = ["trial_id", "subject_id", "session_id", "label", "time_segment"]
    feat_cols  = [c for c in df.columns if c not in meta_cols][:n_features]

    sample = df[feat_cols]
    if len(sample) > n_samples:
        sample = sample.sample(n=n_samples, random_state=42)

    corr = sample.astype(float).corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, ax=ax,
        cmap="coolwarm", center=0,
        xticklabels=False, yticklabels=False,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(f"Feature Correlation (first {n_features} EEG features, {n_samples} samples)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ correlation_heatmap.png saved")

    # Report highly correlated pairs
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high  = (upper.abs() > 0.95).sum().sum()
    print(f"  Feature pairs with |corr| > 0.95 : {int(high)}")


# ─────────────────────────────────────────────
# 4. FEATURE SELECTION (non-model)
# ─────────────────────────────────────────────
def run_feature_selection(df: pd.DataFrame, variance_threshold: float = 1e-4,
                           top_k: int = 20) -> dict:
    """
    Apply:
      (a) Variance threshold — drop near-zero variance features
      (b) Mutual information with labels — rank feature informativeness

    Returns dict with results.
    NO model training involved.
    """
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

    meta_cols = ["trial_id", "subject_id", "session_id", "label", "time_segment"]
    feat_cols  = [c for c in df.columns if c not in meta_cols]
    X = df[feat_cols].astype(float).fillna(0).values
    y = df["label"].values

    # ── a) Variance threshold ─────────────────
    selector = VarianceThreshold(threshold=variance_threshold)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selector.fit(X)

    low_var_mask   = ~selector.get_support()
    low_var_feats  = [feat_cols[i] for i, keep in enumerate(selector.get_support()) if not keep]
    kept_feats     = [feat_cols[i] for i, keep in enumerate(selector.get_support()) if keep]

    print(f"\n  Variance Threshold (< {variance_threshold}):")
    print(f"    Dropped : {len(low_var_feats)} features")
    print(f"    Kept    : {len(kept_feats)} features")
    if low_var_feats:
        print(f"    Dropped example: {low_var_feats[:5]}")

    # ── b) Mutual information ─────────────────
    # Use subset for speed (max 10k rows)
    if len(X) > 10000:
        idx = np.random.choice(len(X), 10000, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y

    print(f"\n  Computing Mutual Information (across {len(X_sub):,} samples)...")
    mi_scores = mutual_info_classif(X_sub, y_sub, discrete_features=False, random_state=42)

    mi_df = pd.DataFrame({"feature": feat_cols, "mi_score": mi_scores})
    mi_df = mi_df.sort_values("mi_score", ascending=False).reset_index(drop=True)

    top_feats  = mi_df.head(top_k)
    low_feats  = mi_df.tail(top_k)

    print(f"\n  Top {top_k} most informative features (MI):")
    print(top_feats.to_string(index=False))
    print(f"\n  Least informative features (MI):")
    print(low_feats.to_string(index=False))

    # Save MI scores
    mi_path = os.path.join(os.path.dirname(CSV_STATS), "mutual_info_scores.csv")
    mi_df.to_csv(mi_path, index=False)
    print(f"\n  ✓ mutual_info_scores.csv saved")

    # Plot top features
    _plot_mi(top_feats, f"Top {top_k} Features by Mutual Information")

    return {
        "low_variance_features": low_var_feats,
        "kept_features":         kept_feats,
        "mi_scores":             mi_df,
        "top_features":          top_feats,
    }


def _plot_mi(mi_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(mi_df["feature"], mi_df["mi_score"], color="#4e9af1")
    ax.invert_yaxis()
    ax.set_xlabel("Mutual Information Score")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "mutual_info_top_features.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  ✓ mutual_info_top_features.png saved")


# ─────────────────────────────────────────────
# 5. DATASET GAP REPORT
# ─────────────────────────────────────────────
def summarize_gaps(df_eeg: pd.DataFrame, df_eye: pd.DataFrame,
                   df_merged: pd.DataFrame) -> dict:
    """
    Print and return a structured dataset gap report covering:
      - Class imbalance
      - Session variability
      - NaN counts
      - Sample counts per subject
    """
    print("\n" + "=" * 60)
    print("  DATASET GAP ANALYSIS")
    print("=" * 60)

    gaps = {}

    # NaN counts
    eeg_nans = df_eeg.select_dtypes(include=[float]).isna().sum().sum() if not df_eeg.empty else 0
    eye_nans  = df_eye.select_dtypes(include=[float]).isna().sum().sum() if not df_eye.empty else 0
    gaps["eeg_nan_count"] = int(eeg_nans)
    gaps["eye_nan_count"] = int(eye_nans)
    print(f"\n  NaN counts — EEG: {eeg_nans}   Eye: {eye_nans}")

    # Class balance
    if not df_eeg.empty and "label" in df_eeg.columns:
        counts = df_eeg["label"].value_counts().sort_index()
        total  = counts.sum()
        imbalance_ratio = counts.max() / counts.min()
        gaps["class_imbalance_ratio"] = round(float(imbalance_ratio), 3)
        print(f"\n  Class imbalance ratio (max/min): {imbalance_ratio:.3f}")
        print(f"  (< 1.2 = balanced, > 2.0 = imbalanced)")

    # Samples per subject
    if not df_eeg.empty:
        per_subj = df_eeg.groupby("subject_id").size()
        print(f"\n  Samples per subject — min: {per_subj.min():,}  max: {per_subj.max():,}  "
              f"std: {per_subj.std():.0f}")
        gaps["samples_per_subject"] = per_subj.to_dict()

    # Session variability
    if not df_eeg.empty:
        per_sess = df_eeg.groupby("session_id").size()
        print(f"\n  Samples per session:")
        for sess, cnt in per_sess.items():
            print(f"    Session {sess}: {cnt:,}")

    # Multimodal alignment loss
    if not df_eeg.empty and not df_merged.empty:
        merge_ratio = len(df_merged) / len(df_eeg)
        gaps["alignment_retention"] = round(merge_ratio, 4)
        print(f"\n  Multimodal merge retention: {merge_ratio*100:.1f}% of EEG rows matched eye")

    print("\n  Known risks:")
    print("  [!] LDS smoothing can over-smooth rapid emotion transitions")
    print("  [!] Averaging over frequency bands loses temporal dynamics")
    print("  [!] Variable n_windows per trial may cause length inconsistency")
    print("  [!] Eye features may have NaNs from blink artifacts")

    return gaps
