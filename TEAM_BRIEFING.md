# 📊 Team Briefing — SEED-IV Preprocessing Pipeline

> **Who is this for?** Teammates who want to understand what the code does
> and what's in the output files — without reading the actual code.

---

## 🧠 The Big Picture

We are working on **emotion recognition from brain signals (EEG) and eye movements**.

The dataset is called **SEED-IV**. It has recordings from **15 people** who watched
video clips designed to make them feel 4 emotions:

| Label | Emotion |
|---|---|
| 0 | Neutral 😐 |
| 1 | Sad 😢 |
| 2 | Fear 😨 |
| 3 | Happy 😊 |

Each person did **3 sessions** of **24 video clips** each. After every clip,
we have a chunk of EEG data (brain signals) and eye-tracking data.

The `pipeline/` folder is our **data preparation system** — it takes all the raw
data files and converts them into clean tables (CSVs) ready for machine learning.

**No model is trained yet. This is purely the data preparation phase.**

---

## 📁 `pipeline/` Folder — What Each File Does

### `config.py` — Settings File

Think of this as the **control panel**. It holds all the settings in one place:
- Where the dataset files are stored
- Where to save the outputs
- The official emotion labels for each of the 24 video clips per session

> **Why it exists:** Instead of burying file paths inside every script, we put
> them all here. If you move a folder, you change *one line* instead of hunting
> through 5 files.

---

### `loader.py` — Data Reader

This file **reads the raw `.mat` files** and tells you what's inside them.

The raw EEG data looks like this inside each file:
```
Key: de_LDS1    Shape: (62, 127, 5)
     ↑               ↑   ↑    ↑
     trial 1     62 ch  127  5 bands
                  elec. windows
```

- **62** = 62 EEG electrodes placed on the person's scalp
- **127** = 127 time windows (each is ~1 second of data)
- **5** = 5 frequency bands: delta, theta, alpha, beta, gamma

It also checks that every person has both an EEG file and an eye file —
if one is missing, it skips that pair and warns you instead of crashing.

---

### `transformer.py` — The Shape Converter (Most Important Step)

This is the **core data transformation**. It converts 3D brain data into
a flat 2D table that machine learning algorithms can actually use.

**Before transformation (raw):**
```
Shape: (62, 127, 5)
= 62 electrodes × 127 time windows × 5 frequency bands
= a 3D cube of numbers
```

**After transformation (ML-ready):**
```
Shape: (127, 310)
= 127 rows (one row per 1-second window)
= 310 columns (62 electrodes × 5 bands, flattened into one long row)
```

**Why not just take the average?**
The old code (`main.py`) averaged all 127 windows into 1 number —
throwing away all the time information. Emotions change over time!
We keep every 1-second window as its own row, giving us **37,575 samples**
instead of just ~1,080. Much better for training a model.

Every row also gets:
- Which **trial** it came from (1–24)
- Which **subject** recorded it (1–15)
- Which **session** (1–3)
- The **emotion label** (0–3)
- Which **time window** within that trial

---

### `csv_exporter.py` — Saves Everything to CSV

Loops through all 45 subject-session file pairs, calls `transformer.py`
on each one, and saves the results to CSV files.

It also **merges** EEG + eye data by matching the same subject, session,
trial, and time window — giving one big combined table.

---

### `analyzer.py` — Feature Inspector

Once we have the data in CSV format, this file digs into it:

1. **Stats** — mean, variance, skewness of every feature → `feature_stats.csv`
2. **Label balance** — are there equal numbers of each emotion? → bar chart
3. **Correlation heatmap** — which features say the same thing? → heatmap image
4. **Feature ranking** — which features are most useful for predicting emotion?
   Uses a statistical technique called **Mutual Information** (no model training!)

---

### `report_generator.py` — Auto Documentation

Reads all the analysis results and writes a proper report in:
- **Markdown** (`.md`) — readable in any text editor or GitHub
- **PDF** — printable, shareable

---

## 📂 `outputs/` Folder — What Each File Contains

### 📄 `eeg_de_lds.csv` — **Most Important EEG File** (109 MB)

| | Value |
|---|---|
| Rows (samples) | **37,575** |
| Columns | **315** |
| Feature columns | 310 (named `eeg_ch1_delta` → `eeg_ch62_gamma`) |
| Metadata columns | `trial_id`, `subject_id`, `session_id`, `label`, `time_segment` |

**What "DE-LDS" means:**
- **DE** = Differential Entropy — measures how "complex" the brain signal is in each frequency band. Higher DE = more neural activity.
- **LDS** = Linear Dynamical System smoothing — a mathematical filter (like Kalman filter) applied to make the features smoother over time, reducing noise.

Each row = one 1-second snapshot of a person's brain during a particular emotion.

---

### 📄 `eeg_psd_lds.csv` — Second EEG Feature Set (108 MB)

Same structure as above but uses **PSD (Power Spectral Density)** instead of DE.

**What PSD means:** Measures the *power* (energy) of the brain signal in each frequency band. For example, alpha band (8–14 Hz) power is known to decrease when someone is emotionally aroused.

PSD and DE capture slightly different aspects of the same signal. You can experiment with which one gives better ML results.

---

### 📄 `eye_features.csv` — Eye Tracking Data (9.8 MB)

| | Value |
|---|---|
| Rows (samples) | **37,575** |
| Columns | **36** |
| Feature columns | 31 (named `eye_f1` → `eye_f31`) |
| Metadata columns | `trial_id`, `subject_id`, `session_id`, `label`, `time_segment` |

**What the 31 eye features capture:** Things like pupil diameter, blink rate, saccade (eye movement) speed and direction. These change with emotion — for example,
pupils typically dilate when scared or excited.

---

### 📄 `merged_multimodal.csv` — Combined EEG + Eye (119 MB)

| | Value |
|---|---|
| Rows (samples) | **37,575** |
| Columns | **346** |
| Features | 310 EEG + 31 eye = **341 features** + 5 metadata cols |

This is the **main file you'll use for multimodal ML experiments**. It has both
brain signals and eye movement data in one row, aligned to the same time window.

> Think of it as: "at this exact second, this is what the person's brain AND eyes were doing, and this is the emotion label."

---

### 📄 `feature_stats.csv` — Feature Quality Report

For every one of the 310 EEG features:

| Column | What it tells you |
|---|---|
| `mean` | Average value of this feature across all samples |
| `std` | How much it varies (low = flat/boring, high = informative) |
| `variance` | Same as std² — used to filter out useless features |
| `skewness` | Is it lopsided? High skewness may need log transform |
| `nan_count` | Missing values — **0 NaNs found** ✅ |

**Key finding:** 0 near-zero variance features, 0 NaN values — the EEG data is very clean.

---

### 📄 `mutual_info_scores.csv` — Feature Importance Ranking

Ranks all 310 EEG features by how much they help predict the emotion label.
Uses **Mutual Information** — a purely statistical method (no model training).

**Top 5 most informative features:**

| Rank | Feature | MI Score | Meaning |
|---|---|---|---|
| 1 | `eeg_ch46_beta` | 0.383 | Channel 46, beta band |
| 2 | `eeg_ch28_beta` | 0.356 | Channel 28, beta band |
| 3 | `eeg_ch28_gamma` | 0.356 | Channel 28, gamma band |
| 4 | `eeg_ch38_beta` | 0.353 | Channel 38, beta band |
| 5 | `eeg_ch55_beta` | 0.348 | Channel 55, beta band |

**Pattern:** Beta and gamma frequency bands dominate the top features.
This makes neuroscience sense — beta (14–31 Hz) and gamma (31–50 Hz)
oscillations are strongly linked to cognitive and emotional processing.

---

### 📄 `plots/label_distribution.png` — Class Balance Chart

Shows how many samples belong to each emotion class:

| Emotion | Samples | % |
|---|---|---|
| Neutral (0) | 8,820 | 23.5% |
| Sad (1) | 10,440 | 27.8% |
| Fear (2) | 10,155 | 27.0% |
| Happy (3) | 8,160 | 21.7% |

**Verdict:** Reasonably balanced. The max-to-min ratio is ~1.28 (< 2.0 = acceptable).
No need for oversampling or special class weights.

---

### 📄 `plots/correlation_heatmap.png` — Feature Correlation

Shows which features are saying the same thing.
Dark red = highly correlated (redundant). Dark blue = inversely correlated.

**What to look for:** Clusters of red blocks mean those channels/bands carry
similar information. A model like Random Forest handles this naturally;
a linear model (Logistic Regression) might struggle.

---

### 📄 `plots/mutual_info_top_features.png` — Top Features Bar Chart

Horizontal bar chart of the top 20 features ranked by informativeness.
Useful for deciding which features to prioritize if you want a smaller model.

---

### 📄 `report/dataset_report.md` + `dataset_report.pdf` — Full Technical Report

A complete auto-generated report covering:
- Dataset structure with shapes and dimensions explained
- What DE, PSD, LDS, Moving Average mean
- The transformation pipeline
- CSV schema
- Dataset gaps and risks
- Model recommendations (SVM, RF, KNN, XGBoost, MLP) with pros and cons

The PDF is shareable without needing to open code or markdown tools.

---

## 🔄 How to Reproduce the Outputs

If someone clones the repo, they just need to:

1. Add the `dataset/` folder (with the zip files) — not on GitHub due to size
2. Install dependencies:
   ```bash
   pip install scipy numpy pandas matplotlib seaborn scikit-learn reportlab
   ```
3. Run:
   ```bash
   python run_pipeline.py
   ```
   All CSVs, plots, and the report regenerate automatically in ~10 minutes.

---

## 🚀 What Comes Next (Not Done Yet)

The CSVs are ready. The next step (not in this commit) is to train ML models:

1. Load `merged_multimodal.csv`
2. Use **Leave-One-Subject-Out** cross-validation (train on 14 subjects, test on 1)
3. Try: SVM → Random Forest → XGBoost → MLP
4. Compare EEG-only vs Eye-only vs Combined (multimodal) accuracy

The existing `main.py` in the repo already has the LOSO CV framework — it just
needs to be pointed at our new, richer CSV data instead of the old mean-collapsed arrays.
