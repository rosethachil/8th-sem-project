"""
loader.py
=========
Phase 2 — Data Loading & Structure Analysis

Functions:
  inspect_structure()   → prints shapes + keys for a sample file
  load_eeg_from_zip()   → streams .mat files from zip, returns raw mat dicts
  load_eye_files()      → loads .mat files from extracted eye directory
  get_all_file_pairs()  → pairs EEG (inside zip) and eye file paths by subject+session
"""

import os
import io
import zipfile
import scipy.io as sio
import numpy as np

from config import (
    EEG_ZIP_PATH, EYE_DIR, SESSIONS, N_TRIALS,
    EEG_KEY_TYPES, EYE_KEY_PREFIX, N_CHANNELS, N_BANDS, N_EYE_FEATS
)


# ─────────────────────────────────────────────
# HELPER: list .mat entries inside the zip per session
# ─────────────────────────────────────────────
def _zip_entries_for_session(zf: zipfile.ZipFile, session_id: int):
    """Return ZipInfo objects for all .mat files inside session folder."""
    prefix = f"eeg_feature_smooth/{session_id}/"
    return [
        info for info in zf.infolist()
        if info.filename.startswith(prefix) and info.filename.endswith(".mat")
    ]


def _subject_id_from_filename(filename: str) -> str:
    """Extract subject id from e.g. '1_20160518.mat' -> '1'."""
    return os.path.basename(filename).split("_")[0]


# ─────────────────────────────────────────────
# STRUCTURE INSPECTION  (print shapes for one file)
# ─────────────────────────────────────────────
def inspect_structure(verbose=True):
    """
    Open first available EEG and eye .mat file and print:
      - All keys in the file
      - Shape of each trial key (de_LDS1..24, psd_LDS1..24, etc.)
    Explains what each dimension means.
    """
    print("\n" + "=" * 65)
    print("  STRUCTURE INSPECTION — First EEG file (Session 1, Subject 1)")
    print("=" * 65)

    with zipfile.ZipFile(EEG_ZIP_PATH, 'r') as zf:
        entries = _zip_entries_for_session(zf, 1)
        entries.sort(key=lambda x: x.filename)
        first_entry = entries[0]
        subject = _subject_id_from_filename(first_entry.filename)

        print(f"\n  File: {first_entry.filename}")
        with zf.open(first_entry) as f:
            mat = sio.loadmat(io.BytesIO(f.read()))

        all_keys = [k for k in mat.keys() if not k.startswith("__")]
        print(f"\n  All keys ({len(all_keys)}): {all_keys[:8]} ...")

        print(f"\n  {'Key':<20} {'Shape':<25} Meaning")
        print("  " + "-" * 65)

        shapes_seen = {}
        for key_type in EEG_KEY_TYPES:
            for trial in range(1, N_TRIALS + 1):
                key = f"{key_type}{trial}"
                if key in mat:
                    arr = mat[key]
                    shapes_seen[key_type] = arr.shape
                    if trial == 1:   # print only first trial per type
                        meaning = _explain_shape(arr.shape, key_type)
                        print(f"  {key:<20} {str(arr.shape):<25} {meaning}")

    print("\n  → EEG dimension meanings:")
    print("      axis-0 : 62 EEG channels (electrodes on scalp)")
    print("      axis-1 : variable time windows (~1 sec each)")
    print("      axis-2 : 5 frequency bands (delta/theta/alpha/beta/gamma)")

    # Eye structure
    print("\n" + "=" * 65)
    print("  STRUCTURE INSPECTION — First Eye file (Session 1, Subject 1)")
    print("=" * 65)

    eye_sess_dir = os.path.join(EYE_DIR, "1")
    eye_files = sorted(f for f in os.listdir(eye_sess_dir) if f.endswith(".mat"))
    first_eye = os.path.join(eye_sess_dir, eye_files[0])

    mat_eye = sio.loadmat(first_eye)
    eye_keys = [k for k in mat_eye.keys() if not k.startswith("__")]
    print(f"\n  File: {eye_files[0]}")
    print(f"  All keys ({len(eye_keys)}): {eye_keys}")

    print(f"\n  {'Key':<15} {'Shape':<25} Meaning")
    print("  " + "-" * 55)
    for trial in range(1, N_TRIALS + 1):
        key = f"{EYE_KEY_PREFIX}{trial}"
        if key in mat_eye and trial == 1:
            arr = mat_eye[key]
            print(f"  {key:<15} {str(arr.shape):<25} {N_EYE_FEATS} eye features × time windows")

    print("\n  → Eye dimension meanings:")
    print("      axis-0 : 31 eye-tracking features (pupil, saccade, blink, etc.)")
    print("      axis-1 : variable time windows (aligned with EEG)")
    print()


def _explain_shape(shape, key_type):
    if len(shape) == 3:
        return f"{shape[0]} ch × {shape[1]} windows × {shape[2]} bands"
    return str(shape)


# ─────────────────────────────────────────────
# LOAD ONE EEG .MAT  (returns raw dict)
# ─────────────────────────────────────────────
def load_eeg_mat(zf: zipfile.ZipFile, zip_info: zipfile.ZipInfo) -> dict:
    """
    Load a single EEG .mat file from inside the zip.
    Returns the raw scipy mat dict.
    """
    with zf.open(zip_info) as f:
        return sio.loadmat(io.BytesIO(f.read()))


# ─────────────────────────────────────────────
# LOAD ONE EYE .MAT  (returns raw dict)
# ─────────────────────────────────────────────
def load_eye_mat(filepath: str) -> dict:
    """Load a single eye .mat file from disk."""
    return sio.loadmat(filepath)


# ─────────────────────────────────────────────
# GET ALL (session, subject, eeg_zip_info, eye_path) PAIRS
# ─────────────────────────────────────────────
def get_all_file_pairs(verbose=True):
    """
    Returns a list of dicts:
      {session_id, subject_id, eeg_zip_info, eye_path}
    Only includes pairs where both EEG and eye files exist.
    """
    pairs = []
    missing = []

    with zipfile.ZipFile(EEG_ZIP_PATH, 'r') as zf:
        for session_id in SESSIONS:
            entries = _zip_entries_for_session(zf, session_id)
            entries.sort(key=lambda x: x.filename)

            for entry in entries:
                fname    = os.path.basename(entry.filename)
                subject  = _subject_id_from_filename(fname)
                eye_path = os.path.join(EYE_DIR, str(session_id), fname)

                if os.path.exists(eye_path):
                    pairs.append({
                        "session_id":   session_id,
                        "subject_id":   subject,
                        "eeg_zip_info": entry,
                        "eye_path":     eye_path,
                        "filename":     fname,
                    })
                else:
                    msg = f"[MISS] session={session_id} subject={subject} eye file not found"
                    missing.append(msg)
                    if verbose:
                        print(f"  {msg}")

    if verbose:
        print(f"\n  Loaded {len(pairs)} matched subject-session pairs "
              f"({len(missing)} missing eye files)")
    return pairs


if __name__ == "__main__":
    inspect_structure()
    pairs = get_all_file_pairs(verbose=True)
    print(f"\n  Total pairs: {len(pairs)}")
