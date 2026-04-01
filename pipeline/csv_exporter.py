"""
csv_exporter.py
===============
Phase 4 — CSV Creation

Iterates over all matched (EEG, eye) file pairs, builds DataFrames via
transformer.py, and writes four CSVs:

  eeg_de_lds.csv       — DE-LDS EEG features (time-segment level)
  eeg_psd_lds.csv      — PSD-LDS EEG features
  eye_features.csv     — Eye-tracking features
  merged_multimodal.csv — DE-LDS EEG + Eye merged on aligned keys

Functions:
  create_csv()         — main export function
  merge_multimodal()   — join EEG + eye on (subject, session, trial, time_segment)
"""

import zipfile
import pandas as pd

from config import (
    EEG_ZIP_PATH,
    CSV_EEG_DE_LDS, CSV_EEG_PSD_LDS, CSV_EYE, CSV_MERGED,
    ensure_output_dirs
)
from loader import get_all_file_pairs, load_eeg_mat, load_eye_mat
from transformer import build_eeg_dataframe, build_eye_dataframe


# ─────────────────────────────────────────────
# MAIN EXPORT
# ─────────────────────────────────────────────
def create_csv(verbose=True):
    """
    Iterate over all subject-session pairs, transform data, write CSVs.
    Returns DataFrames: (df_de_lds, df_psd_lds, df_eye)
    """
    ensure_output_dirs()
    pairs = get_all_file_pairs(verbose=verbose)

    all_de_lds   = []
    all_psd_lds  = []
    all_eye      = []

    total = len(pairs)
    print(f"\n  Processing {total} subject-session pairs...")

    with zipfile.ZipFile(EEG_ZIP_PATH, 'r') as zf:
        for i, pair in enumerate(pairs, 1):
            sid  = pair["subject_id"]
            sess = pair["session_id"]
            fname = pair["filename"]

            if verbose:
                print(f"  [{i:>3}/{total}] subject={sid:>2}  session={sess}  {fname}")

            # ── EEG ──────────────────────────────────
            eeg_mat = load_eeg_mat(zf, pair["eeg_zip_info"])

            df_de  = build_eeg_dataframe(eeg_mat, "de_LDS",  sid, sess, fname)
            df_psd = build_eeg_dataframe(eeg_mat, "psd_LDS", sid, sess, fname)

            if not df_de.empty:
                all_de_lds.append(df_de)
            if not df_psd.empty:
                all_psd_lds.append(df_psd)

            # ── Eye ──────────────────────────────────
            eye_mat = load_eye_mat(pair["eye_path"])
            df_eye  = build_eye_dataframe(eye_mat, sid, sess, fname)

            if not df_eye.empty:
                all_eye.append(df_eye)

    # ── Concatenate & save ────────────────────
    print("\n  Writing CSVs...")

    df_de_lds  = _save_csv(all_de_lds,  CSV_EEG_DE_LDS,  "eeg_de_lds.csv")
    df_psd_lds = _save_csv(all_psd_lds, CSV_EEG_PSD_LDS, "eeg_psd_lds.csv")
    df_eye_out = _save_csv(all_eye,     CSV_EYE,          "eye_features.csv")

    return df_de_lds, df_psd_lds, df_eye_out


def _save_csv(frames, path, name):
    if not frames:
        print(f"  [WARN] No data for {name}")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(path, index=False)
    print(f"  ✓ {name:<30} {df.shape[0]:>8,} rows × {df.shape[1]} cols  →  {path}")
    return df


# ─────────────────────────────────────────────
# MERGE MULTIMODAL
# ─────────────────────────────────────────────
def merge_multimodal(df_eeg: pd.DataFrame, df_eye: pd.DataFrame) -> pd.DataFrame:
    """
    Inner join EEG (de_LDS) and eye DataFrames on:
      (subject_id, session_id, trial_id, time_segment)

    Only rows where BOTH modalities have a matching time window are kept.
    This is the clean multimodal dataset.
    """
    if df_eeg.empty or df_eye.empty:
        print("  [WARN] Cannot merge: one or both DataFrames are empty.")
        return pd.DataFrame()

    merge_keys = ["subject_id", "session_id", "trial_id", "time_segment"]

    # Drop duplicate label/metadata cols from eye before merge
    eye_drop = ["label"] if "label" in df_eye.columns else []
    df_eye_clean = df_eye.drop(columns=eye_drop)

    df_merged = pd.merge(df_eeg, df_eye_clean, on=merge_keys, how="inner")
    df_merged.to_csv(CSV_MERGED, index=False)
    print(f"  ✓ merged_multimodal.csv        {df_merged.shape[0]:>8,} rows × {df_merged.shape[1]} cols  →  {CSV_MERGED}")
    return df_merged


if __name__ == "__main__":
    df_de, df_psd, df_eye = create_csv(verbose=True)
    df_merged = merge_multimodal(df_de, df_eye)
    print("\n  CSV export complete.")
