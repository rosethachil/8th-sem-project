"""
report_generator.py
===================
Phase 7 — Documentation Generation

Generates:
  1. dataset_report.md   — Markdown report with all analysis results
  2. dataset_report.pdf  — PDF version via reportlab

All sections defined here as plain Python strings + formatted tables.
No model training, no sklearn classifiers.
"""

import os
from datetime import datetime
from config import REPORT_MD, REPORT_DIR, EMOTION_MAP, ensure_output_dirs


# ─────────────────────────────────────────────
# MARKDOWN REPORT
# ─────────────────────────────────────────────
def generate_markdown_report(stats_df, gaps: dict, selection_results: dict,
                              df_eeg_shape: tuple, df_eye_shape: tuple,
                              df_merged_shape: tuple):
    """Generate the full Markdown report and save to disk."""
    ensure_output_dirs()
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    n_eeg_samples    = df_eeg_shape[0] if df_eeg_shape else 0
    n_eeg_features   = df_eeg_shape[1] - 5 if df_eeg_shape else 310  # minus meta cols
    n_eye_samples    = df_eye_shape[0] if df_eye_shape else 0
    n_merged_samples = df_merged_shape[0] if df_merged_shape else 0

    top_feats = selection_results.get("top_features", None)
    top_10 = top_feats.head(10).to_string(index=False) if top_feats is not None else "N/A"

    md = f"""# SEED-IV Multimodal Emotion Recognition — Dataset Report

**Generated:** {now}  
**Dataset:** SEED-IV (EEG + Eye-tracking)  
**Task:** Emotion Recognition (4-class: Neutral, Sad, Fear, Happy)

---

## 1. Dataset Overview

The **SEED-IV** dataset is a benchmark multimodal emotion recognition dataset
collected at Shanghai Jiao Tong University. It contains physiological signals
recorded while 15 participants watched 24 video clips designed to evoke four
distinct emotions: **Neutral**, **Sad**, **Fear**, and **Happy**.

| Property | Value |
|---|---|
| Subjects | 15 |
| Sessions | 3 (repeated sessions per subject) |
| Trials per session | 24 |
| EEG channels | 62 |
| EEG frequency bands | 5 (delta, theta, alpha, beta, gamma) |
| Eye features per window | 31 |
| Total subject-session files | ~45 (15 × 3) |
| Samples generated (EEG, time-segment level) | {n_eeg_samples:,} |
| Samples generated (Merged multimodal) | {n_merged_samples:,} |

---

## 2. Data Structure Explanation

### EEG Files (`eeg_feature_smooth.zip`)
```
eeg_feature_smooth/
  1/                     ← Session 1
    1_20160518.mat       ← Subject 1, Session 1
    2_20150915.mat
    ...
  2/                     ← Session 2
  3/                     ← Session 3
```
Each `.mat` file contains **96 keys** (24 trials × 4 feature types):

| Key Pattern | Count | Shape | Meaning |
|---|---|---|---|
| `de_LDS1` ... `de_LDS24` | 24 | `(62, n, 5)` | Differential Entropy, LDS-smoothed |
| `psd_LDS1` ... `psd_LDS24` | 24 | `(62, n, 5)` | Power Spectral Density, LDS-smoothed |
| `de_movingAve1` ... `de_movingAve24` | 24 | `(62, n, 5)` | DE, moving-average smoothed |
| `psd_movingAve1` ... `psd_movingAve24` | 24 | `(62, n, 5)` | PSD, moving-average smoothed |

**Dimension meanings:**
- **axis-0 (62):** EEG channels (electrode positions on scalp)
- **axis-1 (n, variable):** Time windows (~1-second segments, varies per trial)
- **axis-2 (5):** Frequency bands — delta (1-4Hz), theta (4-8Hz), alpha (8-14Hz), beta (14-31Hz), gamma (31-50Hz)

### Eye Files (`eye_feature_smooth/`)
Each `.mat` file contains keys `eye_1` to `eye_24`.

| Key Pattern | Shape | Meaning |
|---|---|---|
| `eye_1` ... `eye_24` | `(31, n)` | 31 eye features over n time windows |

---

## 3. Feature Types (DE, PSD, LDS, MovingAverage)

### Differential Entropy (DE)
DE measures the **complexity/information content** of an EEG signal within a
frequency band. For a Gaussian signal X: `H(X) = 0.5 × log(2πe × σ²)`.  
Higher DE → more neural activity. DE is shown to be more discriminative for
emotion recognition than raw power.

### Power Spectral Density (PSD)
PSD measures the **power** of an EEG signal at each frequency band.
Computed via FFT over each time window. It captures how much energy
exists in each frequency range. Alpha band suppression during emotional
arousal is a well-known correlate.

### LDS Smoothing (Linear Dynamical System)
LDS applies a **Kalman filter** to smooth features across time, reducing
noise while preserving gradual emotional transitions. The output is
a filtered sequence that is smoother than raw window-by-window features.
**Risk:** May over-smooth rapid emotion changes.

### Moving Average Smoothing
A simpler alternative: each window's value = average of nearby windows.
Less principled than LDS, but computationally cheaper. Used as a baseline
comparison to LDS.

---

## 4. Data Transformation Pipeline

### The Problem with Mean-Collapse (used in `main.py`)
```
(62, n_windows, 5)  →  mean over n_windows  →  (62, 5)  →  flatten  →  (310,)
```
One sample per trial × 24 trials × 45 files = **~1080 samples total**.  
**This throws away all temporal information.**

### Our Approach: Preserve Time Segments
```
(62, n_windows, 5)
    ↓ transpose
(n_windows, 62, 5)     ← time_segments, channels, bands
    ↓ flatten
(n_windows, 310)       ← each row = one 1-second segment = one ML sample
```
Each time window becomes its own row in the CSV.
**Result: {n_eeg_samples:,} samples** instead of ~1,080. Far richer for ML.

### Label Assignment
Each row inherits the label of its parent trial (from SEED-IV's official
`SESSION_LABELS` mapping for that session). The trial label is verified
against the SEED-IV paper's annotation table.

---

## 5. CSV Schema Description

### `eeg_de_lds.csv` and `eeg_psd_lds.csv`

| Column | Type | Description |
|---|---|---|
| `eeg_ch1_delta` | float32 | Channel 1, delta band feature |
| ... | ... | ... (310 feature columns total) |
| `eeg_ch62_gamma` | float32 | Channel 62, gamma band feature |
| `trial_id` | int | Trial number (1–24) |
| `subject_id` | str | Subject identifier (e.g., "1", "15") |
| `session_id` | int | Session (1, 2, or 3) |
| `label` | int | Emotion: 0=neutral 1=sad 2=fear 3=happy |
| `time_segment` | int | Index of this time window within the trial |

### `eye_features.csv`

| Column | Type | Description |
|---|---|---|
| `eye_f1` ... `eye_f31` | float32 | 31 eye-tracking features per window |
| `trial_id`, `subject_id`, `session_id`, `label`, `time_segment` | — | Same as above |

### `merged_multimodal.csv`
Inner join of EEG (de_LDS) + Eye on `(subject_id, session_id, trial_id, time_segment)`.  
Contains all 310 EEG + 31 eye = **341 features** per row.

---

## 6. Feature Analysis

### EEG Feature Statistics Summary

| Statistic | Value |
|---|---|
| Total features | {n_eeg_features} |
| Total samples (time-segment level) | {n_eeg_samples:,} |
| Global mean (across all features) | see `feature_stats.csv` |
| Features with near-zero variance | {gaps.get('low_variance_count', 'N/A')} |
| Total NaN count (EEG) | {gaps.get('eeg_nan_count', 0)} |
| Total NaN count (Eye) | {gaps.get('eye_nan_count', 0)} |

### Top 10 Most Informative Features (Mutual Information)
```
{top_10}
```

DE features in the **gamma band** typically rank highest in emotion
recognition tasks, as gamma oscillations (>30Hz) are strongly linked
to cognitive and emotional processing.

---

## 7. Segmentation Strategy

### Why Segment Instead of Using Full Signals?
1. **Non-stationarity:** EEG is inherently non-stationary. Emotions evolve
   over seconds — a 1-second window captures a stable "state."
2. **Data augmentation:** 24 trials × ~100 windows = thousands of samples
   instead of just 24. Essential for training robust ML models.
3. **Alignment:** Both EEG and eye data are already in windowed form in
   this dataset — the dataset authors chose 1-second windows during feature
   extraction.

### Impact on ML Models
- More samples → less overfitting risk
- Each sample is independent (IID assumption holds better)
- **Caution:** Samples from the same trial are temporally correlated.
  Cross-validation must be done at the trial or subject level — NOT
  randomly shuffling windows from different trials.

---

## 8. Dataset Challenges & Gaps

| Issue | Severity | Description |
|---|---|---|
| Class imbalance ratio | {gaps.get('class_imbalance_ratio', 'N/A')} | Ratio of largest to smallest class count |
| Temporal correlation | High | Consecutive windows within a trial are not independent |
| LDS over-smoothing | Medium | Kalman filtering may erase fast emotion transitions |
| Session variability | Medium | Same subject, different sessions → distributional shift |
| Eye NaN artifacts | Low | Blink-related NaNs in eye features, need imputation |
| Variable n_windows | Low | Different trials may have different lengths |
| Small subject pool | Medium | Only 15 subjects (risk of subject-specific overfitting) |

---

## 9. Suggested ML Models

The following models are recommended based on the dataset characteristics.
**No training is performed in this report — these are recommendations only.**

### Support Vector Machine (SVM)
**Why suitable:** EEG features are high-dimensional (310 features) but the
dataset has moderate size per subject. SVMs with RBF kernel handle this well
via the kernel trick. Previous SEED-IV papers report SVM as a strong baseline.
- ✅ Effective in high-dimensional, moderate-sample settings
- ✅ Robust to feature scale differences (with normalization)
- ❌ Slow training on 100k+ samples — consider subset or kernel approximation
- ❌ No native probability output (need `probability=True`)

### Random Forest
**Why suitable:** Ensemble of decision trees handles non-linear relationships
and provides feature importance scores without any additional code.
- ✅ Fast training, parallelizable
- ✅ Built-in feature importance
- ✅ Handles correlated features better than linear models
- ❌ Memory intensive with 310 features × deep trees

### K-Nearest Neighbors (KNN)
**Why suitable:** Emotion states may cluster in feature space — similar EEG
patterns across subjects expressing the same emotion.
- ✅ No training phase, interpretable
- ✅ Good for exploring class cluster structure
- ❌ Very slow at inference with 100k+ training samples
- ❌ Sensitive to feature scale (requires normalization)

### Logistic Regression
**Why suitable:** Fast, interpretable baseline. With 310 features and proper
regularization (L2), gives a strong linear baseline.
- ✅ Fastest to train, interpretable coefficients
- ✅ Probability outputs, easy to threshold
- ❌ Linear — may not capture non-linear emotion-EEG relationships
- ❌ Multicollinearity between correlated frequency band features

### Gradient Boosting (XGBoost / LightGBM)
**Why suitable:** State-of-the-art on tabular data. Handles non-linear
interactions, feature importance, and is robust to outliers.
- ✅ Excellent accuracy on tabular feature data
- ✅ LightGBM very fast on large datasets (100k+ rows)
- ✅ Feature importance built-in
- ❌ Harder to tune (many hyperparameters)
- ❌ Prone to label leakage if cross-validation is not subject-aware

### MLP (Multi-Layer Perceptron)
**Why suitable:** Most flexible model. Can learn non-linear combinations across
all 310 EEG + 31 eye features simultaneously, especially in fused mode.
- ✅ Can model complex emotion-feature relationships
- ✅ Natural for multimodal fusion (concatenate EEG + eye features)
- ❌ Requires more data and careful tuning
- ❌ Black box — harder to interpret than tree-based models

---

## Appendix: Cross-Validation Strategy Recommendation

**Use Leave-One-Subject-Out (LOSO) CV:**
- Train on all subjects except one, test on held-out subject
- Measures **cross-subject generalizability** — reflects real-world deployment
- Already implemented in `main.py` — carry that forward

**Do NOT use random K-fold on the raw time-segment level rows** — this would
leak temporal neighbors from the same trial into train/test splits, artificially
inflating accuracy.

---

*Report generated by `run_pipeline.py` — SEED-IV Preprocessing Pipeline*
"""

    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  ✓ dataset_report.md saved  →  {REPORT_MD}")
    return md


# ─────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────
def generate_pdf_report(md_text: str):
    """
    Convert the markdown report to PDF using reportlab.
    Falls back gracefully if reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
    except ImportError:
        print("  [SKIP] reportlab not installed. Install with: pip install reportlab")
        print("         Markdown report is saved. PDF skipped.")
        return

    ensure_output_dirs()
    pdf_path = os.path.join(REPORT_DIR, "dataset_report.pdf")
    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm
    )

    styles = getSampleStyleSheet()
    style_h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16, spaceAfter=10)
    style_h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceAfter=6)
    style_h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontSize=11, spaceAfter=4)
    style_body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9, leading=13)
    style_code = ParagraphStyle("code", parent=styles["Code"], fontSize=8, leading=11,
                                 backColor=colors.HexColor("#f5f5f5"))

    story = []

    for line in md_text.split("\n"):
        line = line.rstrip()
        if line.startswith("# ") and not line.startswith("## "):
            story.append(Paragraph(line[2:], style_h1))
        elif line.startswith("## "):
            story.append(Spacer(1, 0.3*cm))
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
            story.append(Paragraph(line[3:], style_h2))
        elif line.startswith("### "):
            story.append(Paragraph(line[4:], style_h3))
        elif line.startswith("```"):
            pass   # skip code fence markers
        elif line.startswith("|"):
            pass   # tables handled as raw text (simplified)
        elif line.startswith("**") or line.startswith("- ") or line.startswith("✅") or line.startswith("❌"):
            story.append(Paragraph(line.replace("**", "<b>", 1).replace("**", "</b>", 1), style_body))
        elif line.strip() == "---":
            story.append(Spacer(1, 0.2*cm))
            story.append(HRFlowable(width="100%", thickness=0.3, color=colors.lightgrey))
        elif line.strip():
            story.append(Paragraph(line, style_body))
        else:
            story.append(Spacer(1, 0.15*cm))

    doc.build(story)
    print(f"  ✓ dataset_report.pdf saved  →  {pdf_path}")
