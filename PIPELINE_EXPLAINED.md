# 🧠 What Am I Doing and Why? — Pipeline Explained

> This document explains every file I created, **what it does**, **why it exists**,
> and **how it differs from the old `main.py`/`main1.py` code.**
> Written to help you understand the thought process, not just the code.

---

## 📁 Two Worlds: Old Code vs New Pipeline

Before I explain each new file, here's the **key difference** in approach:

| | Old (`main.py`, `main1.py`) | New (`pipeline/` folder) |
|---|---|---|
| **Approach** | Single monolithic script | Modular — each concern in its own file |
| **Data loading** | Hardcoded Colab paths (`/content/`) | Reads from your local `dataset/` folder |
| **EEG zip** | Extracts the whole zip first | Streams directly from zip (no extraction!) |
| **Time windows** | **Collapses** all time windows by taking mean → 1 row per trial | **Keeps** each time window as a separate row |
| **Samples generated** | ~1,080 (24 trials × 45 files) | ~100,000+ (because every window = 1 row) |
| **Output** | `.npy` binary files | Human-readable `.csv` files + report |
| **ML** | Trains SVM/RF/MLP immediately | ❌ No training — just prepares data |
| **Documentation** | None | Auto-generates `.md` + `.pdf` report |

---

## 📂 File-by-File Explanation

### `pipeline/config.py` — The Single Source of Truth

**What it does:**  
Stores every constant, path, and setting used across all other files.

**Why I made it:**  
Without this, every file would have hardcoded paths like `"C:/Users/Rose J Thachil/..."`.
That's fragile — if you move a folder, you'd have to find-and-replace across 5 files.
Now you only change one place: `config.py`.

**What's inside:**
```
EEG_ZIP_PATH        → where your EEG zip lives
EYE_DIR             → extracted eye data folder
OUTPUT_DIR          → where all CSVs/plots go
SESSION_LABELS      → official SEED-IV trial→emotion mappings (copied from main.py)
N_CHANNELS = 62     → 62 EEG electrodes
N_BANDS = 5         → delta, theta, alpha, beta, gamma
N_EYE_FEATS = 31    → 31 eye-tracking features
EMOTION_MAP         → {0:'neutral', 1:'sad', 2:'fear', 3:'happy'}
```

> **Why 62 channels, 5 bands?** Because the SEED-IV authors extracted features
> using the 10-20 EEG cap system (62 electrodes) and standard EEG frequency bands.

---

### `pipeline/loader.py` — Reading Raw Data

**What it does:**  
Reads `.mat` files (EEG from zip, eye from disk) and prints what's inside them —
shapes, key names, and what each dimension means.

**Why I made it:**  
You can't preprocess data you don't understand. Before touching numbers,
you need to know: *"What is shape (62, 127, 5) telling me?"*

`loader.py` answers that:
- axis-0 = 62 EEG channels
- axis-1 = 127 = number of 1-second time windows in this trial
- axis-2 = 5 frequency bands

**Key function — `inspect_structure()`:**  
Runs through the first file and prints a human-readable shape table.
Run this standalone: `python pipeline/loader.py`

**Key function — `get_all_file_pairs()`:**  
Finds every EEG + eye `.mat` file pair (same subject, same session).
This is important because some eye files may be missing — this function
detects and skips those instead of crashing.

**Why stream from zip instead of extracting?**  
The EEG zip is ~328 MB uncompressed. Extracting would create ~1 GB of files
just to read them once. Streaming with Python's `zipfile` reads the data
directly into memory without touching disk.

---

### `pipeline/transformer.py` — Converting 3D Arrays to ML Rows

**What it does:**  
Takes a raw `(62, n_windows, 5)` NumPy array from a `.mat` file and
converts it into a 2D table where **each row = one 1-second time window**.

**The core transformation:**
```
Input:  (62, 127, 5)         ← channels × windows × bands

Step 1 — Transpose:
        (127, 62, 5)         ← windows × channels × bands
         ↑
         Now "time" is the first axis — important for the next step

Step 2 — Flatten channels+bands:
        (127, 310)           ← 127 samples × 310 features
         ↑               ↑
         rows             62×5 = 310 feature columns per row
```

**Why flatten channels and bands together?**  
Most ML algorithms (SVM, Random Forest, etc.) expect a 2D input: rows of samples
with flat feature vectors. By flattening, each sample has 310 numbers:
channel 1's delta, channel 1's theta, ..., channel 62's gamma.
These are the features the model will learn from.

**Why NOT use mean-collapse like `main.py`?**  
`main.py` does: `np.mean(data, axis=1)` — averages all 127 windows → 1 row.
This **destroys temporal information**. Emotions evolve over time.
A 5-second video clip might start as neutral and become fearful —
averaging would mix those states into a meaningless blur.
Keeping each window preserves this temporal richness.

**Column naming:**  
Each feature gets a human-readable name:
`eeg_ch1_delta`, `eeg_ch1_theta`, ..., `eeg_ch62_gamma` — 310 columns.

---

### `pipeline/csv_exporter.py` — Saving to CSV

**What it does:**  
Loops over all 45 subject-session pairs, calls `transformer.py` on each,
and accumulates results into four CSV files.

**Why CSVs and not `.npy`?**  
`main.py` saves `.npy` binary files. Those are fast but **opaque** —
you can't open them in Excel, Pandas, or look at them without code.
CSVs are:
- Human-readable (open in Excel, Notepad)
- Shareable without Python
- Self-documenting (column names tell you what each value is)

**Four output files:**

| File | What's inside |
|---|---|
| `eeg_de_lds.csv` | Differential Entropy features (LDS-smoothed), one row per time window |
| `eeg_psd_lds.csv` | Power Spectral Density features (LDS-smoothed), same format |
| `eye_features.csv` | Eye-tracking features, one row per aligned time window |
| `merged_multimodal.csv` | EEG + Eye features joined on (subject, session, trial, time_segment) |

**The merge:** Eye and EEG are recorded simultaneously. By merging on the
same `(subject_id, session_id, trial_id, time_segment)` key, we get rows
with **341 features** (310 EEG + 31 eye) — the true multimodal input for ML.

---

### `pipeline/analyzer.py` — Understanding the Features

**What it does:**  
Analyses the feature data statistically and selects the most useful ones —
**without training any model**.

**Why this step matters:**  
Before you give data to a model, you should know:
- Are there any features that never change? (useless)
- Are features correlated? (redundant = waste of model capacity)
- Which features are most related to the label? (informative)
- Is the dataset balanced? (class imbalance = biased model)

**Four analyses:**

#### 1. Basic Stats (`analyze_features`)
Computes mean, variance, std, skewness, NaN count for every feature.
Saved to `feature_stats.csv`. High skewness → consider log transform later.

#### 2. Label Distribution (`plot_label_distribution`)
Bar chart showing how many samples belong to each emotion class.
If one class has 3× more samples than another, any trained model will
naturally predict that class more often — unfair comparison.

#### 3. Correlation Heatmap (`plot_correlation`)
Shows which features are related to each other. High correlation = redundant.
For example, delta band across adjacent EEG channels is often correlated
(neighboring electrodes pick up similar signals).

#### 4. Feature Selection — Non-Model-Based (`run_feature_selection`)

**Variance Threshold:**  
If a feature has near-zero variance across all samples, it carries no
information — drop it. Example: a channel that's always ≈ 0.

**Mutual Information:**  
MI measures how much knowing a feature value tells you about the emotion label.
Higher MI = more predictive of emotion. Implemented via `sklearn.feature_selection`
— this is purely statistical, **no model training involved**.

---

### `pipeline/report_generator.py` — Auto-Documentation

**What it does:**  
Writes `outputs/report/dataset_report.md` and `dataset_report.pdf`
automatically using all the results from the analysis steps.

**Why generate a report?**  
When you come back to this project in 6 months (or show it to a supervisor),
you need documentation that explains the data — not just the code.
The report includes dataset dimensions, feature explanations, transformation
pipeline, CSV schema, and model recommendations.

**How PDF is generated:**  
Using `reportlab` — a pure Python PDF library. No Microsoft Word, no LaTeX,
no Pandoc needed. Just `pip install reportlab`.

---

### `run_pipeline.py` — The Control Room

**What it does:**  
Calls all 5 phases in the correct order. You only ever need to run this one file.

```bash
python run_pipeline.py
```

**Why a single entry point?**  
So you don't have to remember: "do I run loader first, or transformer?"
The pipeline enforces the correct sequence: load → transform → export → analyze → report.

---

## 🔄 How This Connects to Your Existing Code

| Existing File | Role | Status |
|---|---|---|
| `datasetstudy.py` | Quick test — prints .mat file keys | ✅ Still useful for quick checks |
| `main.py` | Full data loader + LOSO ML pipeline | The ML part is your **next step** — use this after pipeline outputs are ready |
| `main1.py` | Colab version with TensorFlow model | Same idea, different model architecture, also for later |
| `mat_dataset_guide.md` | Guide explaining .mat format | Reference doc |

---

## 🎯 What Happens When You Run `run_pipeline.py`

```
STEP 1: Inspect structure
  → Prints shapes of every key in a sample .mat file
  → Explains what (62, 127, 5) means

STEP 2: Create CSVs
  → Reads all 45 subject-session files
  → Transforms 3D arrays → 2D tables
  → Saves 4 CSV files (~100,000+ rows each)

STEP 3: Analyze features
  → Stats table saved to feature_stats.csv
  → Plots: label distribution, correlation heatmap

STEP 4: Feature selection
  → Drops near-zero variance features
  → Ranks features by mutual information
  → Saves mutual_info_scores.csv

STEP 5: Generate report
  → Writes dataset_report.md
  → Converts to dataset_report.pdf
```

**Time estimate:** ~5–15 minutes total (EEG zip is 328 MB).

---

## 📦 Install Dependencies Before Running

```bash
pip install scipy numpy pandas matplotlib seaborn scikit-learn reportlab
```

---

## ❓ Frequently Asked Questions

**Q: Why don't we just use `main.py` directly?**  
A: `main.py` jumps straight to ML training. Before training, you need to
understand the data — class balance, feature quality, NaN counts. Garbage
in = garbage out. This pipeline ensures the data is clean and understood first.

**Q: Why separate files (`loader.py`, `transformer.py`, etc.) instead of one big script?**  
A: Modularity. If loading breaks, you only look at `loader.py`. If CSVs are wrong,
you only look at `csv_exporter.py`. In a big script, everything is tangled.

**Q: What does LDS smoothing actually do?**  
A: LDS = Linear Dynamical System (essentially a Kalman filter). It models the
EEG features as a smooth latent trajectory over time, filtering out noise.
The result is features that change gradually rather than jumping around.

**Q: Is 310 features too many for ML?**  
A: It depends on the model. SVM with RBF handles it well. Linear models may
struggle with multicollinearity (correlated features). The mutual info scores
will tell you which 50-100 features matter most — you can use those instead.

**Q: What is `time_segment` in the CSV?**  
A: The index of the 1-second window within a trial. Trial 1 might have
127 windows — `time_segment` goes from 0 to 126. Useful if you want to
study how emotion evolves within a trial (temporal modeling).
