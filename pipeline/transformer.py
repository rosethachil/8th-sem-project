"""
transformer.py
==============
Phase 3 — Data Transformation

Core idea:
  main.py collapses time windows by mean → one row per trial (loses temporal info).
  We KEEP every time window as a separate sample row → many more samples for ML.

  EEG:  (62, n_windows, 5)
        → transpose  (n_windows, 62, 5)   [time, channels, bands]
        → flatten    (n_windows, 310)      [samples, features]

  Eye:  (31, n_windows)
        → transpose  (n_windows, 31)       [samples, features]

Functions:
  transform_eeg_trial()   → convert one EEG trial array
  transform_eye_trial()   → convert one eye trial array
  build_eeg_dataframe()   → process all trials in one EEG .mat → DataFrame
  build_eye_dataframe()   → process all trials in one eye .mat → DataFrame
"""

import numpy as np
import pandas as pd

from config import (
    N_TRIALS, N_CHANNELS, N_BANDS, N_EYE_FEATS,
    EEG_KEY_TYPES, EYE_KEY_PREFIX, FREQ_BANDS, SESSION_LABELS
)


# ─────────────────────────────────────────────
# COLUMN NAME HELPERS
# ─────────────────────────────────────────────
def eeg_feature_columns():
    """
    Returns list of 310 column names:
    eeg_ch1_delta, eeg_ch1_theta, ..., eeg_ch62_gamma
    """
    cols = []
    for ch in range(1, N_CHANNELS + 1):
        for band in FREQ_BANDS:
            cols.append(f"eeg_ch{ch}_{band}")
    return cols   # 62 × 5 = 310 columns


def eye_feature_columns():
    """Returns list of 31 column names: eye_f1, eye_f2, ..., eye_f31"""
    return [f"eye_f{i}" for i in range(1, N_EYE_FEATS + 1)]


# ─────────────────────────────────────────────
# TRANSFORM ONE TRIAL
# ─────────────────────────────────────────────
def transform_eeg_trial(arr: np.ndarray, key_type: str) -> np.ndarray:
    """
    Convert a single EEG trial array to (n_windows, 310).

    arr shape variants:
      (62, n_windows, 5)  ← most common: channels, time, bands
      (62, 5, n_windows)  ← rarely seen: channels, bands, time
    Returns (n_windows, 310) float32.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

    # Detect axis order
    if arr.shape[0] == N_CHANNELS and arr.shape[2] == N_BANDS:
        # (62, n_windows, 5) → transpose → (n_windows, 62, 5)
        arr = arr.transpose(1, 0, 2)
    elif arr.shape[0] == N_CHANNELS and arr.shape[1] == N_BANDS:
        # (62, 5, n_windows) → transpose → (n_windows, 62, 5)
        arr = arr.transpose(2, 0, 1)
    else:
        raise ValueError(f"Unrecognised EEG shape: {arr.shape}")

    # Now arr is (n_windows, 62, 5) → flatten last two dims
    n_windows = arr.shape[0]
    return arr.reshape(n_windows, N_CHANNELS * N_BANDS).astype(np.float32)


def transform_eye_trial(arr: np.ndarray) -> np.ndarray:
    """
    Convert a single eye trial array to (n_windows, 31).

    arr shape: (31, n_windows) → transpose → (n_windows, 31)
    Returns (n_windows, 31) float32.
    """
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D eye array, got shape {arr.shape}")

    if arr.shape[0] == N_EYE_FEATS:
        arr = arr.T     # (n_windows, 31)
    elif arr.shape[1] == N_EYE_FEATS:
        pass            # already (n_windows, 31)
    else:
        raise ValueError(f"Unrecognised eye shape: {arr.shape}")

    return arr.astype(np.float32)


# ─────────────────────────────────────────────
# BUILD DATAFRAME FROM ONE .MAT FILE
# ─────────────────────────────────────────────
def build_eeg_dataframe(mat: dict, key_type: str,
                         subject_id: str, session_id: int,
                         filename: str) -> pd.DataFrame:
    """
    Process all 24 trials of one key_type from an EEG mat dict.
    Returns a DataFrame with columns:
      [eeg_ch1_delta, ..., eeg_ch62_gamma, trial_id, subject_id, session_id, label, time_segment]
    """
    feat_cols = eeg_feature_columns()
    trial_labels = SESSION_LABELS[session_id]
    rows = []

    for trial in range(1, N_TRIALS + 1):
        key = f"{key_type}{trial}"
        if key not in mat:
            print(f"  [WARN] {filename} missing key {key}")
            continue

        arr = mat[key]
        try:
            feats = transform_eeg_trial(arr, key_type)   # (n_windows, 310)
        except ValueError as e:
            print(f"  [SKIP] {filename} {key}: {e}")
            continue

        n_windows = feats.shape[0]
        label = trial_labels[trial - 1]

        df_trial = pd.DataFrame(feats, columns=feat_cols)
        df_trial["trial_id"]     = trial
        df_trial["subject_id"]   = subject_id
        df_trial["session_id"]   = session_id
        df_trial["label"]        = label
        df_trial["time_segment"] = np.arange(n_windows)
        rows.append(df_trial)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_eye_dataframe(mat: dict,
                         subject_id: str, session_id: int,
                         filename: str) -> pd.DataFrame:
    """
    Process all 24 eye trials from an eye mat dict.
    Returns DataFrame with columns:
      [eye_f1, ..., eye_f31, trial_id, subject_id, session_id, label, time_segment]
    """
    feat_cols = eye_feature_columns()
    trial_labels = SESSION_LABELS[session_id]
    rows = []

    for trial in range(1, N_TRIALS + 1):
        key = f"{EYE_KEY_PREFIX}{trial}"
        if key not in mat:
            print(f"  [WARN] {filename} missing eye key {key}")
            continue

        arr = mat[key]
        try:
            feats = transform_eye_trial(arr)   # (n_windows, 31)
        except ValueError as e:
            print(f"  [SKIP] {filename} {key}: {e}")
            continue

        n_windows = feats.shape[0]
        label = trial_labels[trial - 1]

        df_trial = pd.DataFrame(feats, columns=feat_cols)
        df_trial["trial_id"]     = trial
        df_trial["subject_id"]   = subject_id
        df_trial["session_id"]   = session_id
        df_trial["label"]        = label
        df_trial["time_segment"] = np.arange(n_windows)
        rows.append(df_trial)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
