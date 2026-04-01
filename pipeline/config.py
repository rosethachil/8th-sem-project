"""
config.py
=========
Central configuration for the SEED-IV preprocessing pipeline.
All paths, constants, and label maps live here so every other
module can import from one place.
"""

import os

# ─────────────────────────────────────────────
# ROOT PATHS  (edit these if you move files)
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

EEG_ZIP_PATH = os.path.join(DATASET_DIR, "eeg_feature_smooth.zip")
EYE_DIR      = os.path.join(DATASET_DIR, "extracted_eye", "eye_feature_smooth")

OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR    = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR   = os.path.join(OUTPUT_DIR, "report")

# ─────────────────────────────────────────────
# DATASET CONSTANTS
# ─────────────────────────────────────────────
N_TRIALS    = 24
N_CHANNELS  = 62    # EEG electrodes
N_BANDS     = 5     # delta, theta, alpha, beta, gamma
N_EYE_FEATS = 31    # eye-tracking features per window
N_SESSIONS  = 3
SESSIONS    = [1, 2, 3]

# Frequency band names (for column labels)
FREQ_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]

# EEG key prefixes inside .mat files
EEG_KEY_TYPES = ["de_LDS", "psd_LDS", "de_movingAve", "psd_movingAve"]

# Eye key prefix
EYE_KEY_PREFIX = "eye_"

# ─────────────────────────────────────────────
# SEED-IV OFFICIAL TRIAL LABELS (per session)
# 0=neutral  1=sad  2=fear  3=happy
# ─────────────────────────────────────────────
SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [0, 1, 2, 1, 2, 2, 2, 1, 3, 1, 3, 3, 1, 2, 1, 1, 1, 0, 2, 3, 0, 3, 0, 3],
}

EMOTION_MAP = {0: "neutral", 1: "sad", 2: "fear", 3: "happy"}

# ─────────────────────────────────────────────
# CSV OUTPUT PATHS
# ─────────────────────────────────────────────
CSV_EEG_DE_LDS   = os.path.join(OUTPUT_DIR, "eeg_de_lds.csv")
CSV_EEG_PSD_LDS  = os.path.join(OUTPUT_DIR, "eeg_psd_lds.csv")
CSV_EYE          = os.path.join(OUTPUT_DIR, "eye_features.csv")
CSV_MERGED       = os.path.join(OUTPUT_DIR, "merged_multimodal.csv")
CSV_STATS        = os.path.join(OUTPUT_DIR, "feature_stats.csv")
REPORT_MD        = os.path.join(REPORT_DIR, "dataset_report.md")
REPORT_PDF       = os.path.join(REPORT_DIR, "dataset_report.pdf")


def ensure_output_dirs():
    """Create output folders if they don't exist."""
    for d in [OUTPUT_DIR, PLOTS_DIR, REPORT_DIR]:
        os.makedirs(d, exist_ok=True)
