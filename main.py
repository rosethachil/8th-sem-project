"""
SEED-IV Data Loader  (fixed)
==============================
Confirmed structure from inspection:

EEG file keys: de_LDS1 ... de_LDS24  (one key per trial)
  Each key shape: (62, n_windows, 5)
    62         = EEG channels
    n_windows  = variable time windows per trial
    5          = freq bands (delta, theta, alpha, beta, gamma)
  → mean over n_windows → (62, 5) → flatten → (310,) per trial

Eye file keys: eye_1 ... eye_24  (one key per trial)
  Each key shape: (31, n_windows)
    31         = eye features
    n_windows  = variable time windows per trial
  → mean over n_windows → (31,) per trial

Output:
  X_eeg  : (N, 310)   62ch × 5 bands, mean over time windows
  X_eye  : (N, 31)    31 eye features, mean over time windows
  y      : (N,)       0=neutral 1=sad 2=fear 3=happy
  meta   : list of dicts {subject_id, session_id, trial_id, label}
"""

import os
import json
import scipy.io as sio
import numpy as np

# ──────────────────────────────────────────────
# CONFIGURE YOUR PATHS
# ──────────────────────────────────────────────
EEG_ROOT = "/content/drive/MyDrive/datasetseed4/eeg_feature_smooth"
EYE_ROOT = "/content/drive/MyDrive/datasetseed4/eye_feature_smooth"

# SEED-IV official trial labels per session (24 trials each)
# 0=neutral  1=sad  2=fear  3=happy
SESSION_LABELS = {
    1: [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    2: [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    3: [0,1,2,1,2,2,2,1,3,1,3,3,1,2,1,1,1,0,2,3,0,3,0,3],
}

EEG_KEY_PREFIX = "de_LDS"   # or "de_movingAve" — change if you prefer
N_TRIALS       = 24
N_CHANNELS     = 62
N_BANDS        = 5          # delta theta alpha beta gamma
N_EYE_FEATS   = 31


# ──────────────────────────────────────────────
# LOAD ONE EEG FILE
# ──────────────────────────────────────────────
def load_eeg_file(filepath):
    """
    Returns np.array of shape (24, 310)
    For each trial key de_LDS{i}: shape (62, n_windows, 5)
    → mean over n_windows → (62, 5) → flatten → (310,)
    """
    mat = sio.loadmat(filepath)
    features = []

    for trial in range(1, N_TRIALS + 1):
        key = f"{EEG_KEY_PREFIX}{trial}"
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {os.path.basename(filepath)}. "
                           f"Available: {[k for k in mat if not k.startswith('__')]}")

        data = mat[key]           # (62, n_windows, 5)
        assert data.shape[0] == N_CHANNELS, \
            f"Expected 62 channels, got {data.shape[0]} in {key}"
        assert data.shape[2] == N_BANDS, \
            f"Expected 5 bands, got {data.shape[2]} in {key}"

        # mean over time windows axis → (62, 5)
        trial_feat = np.mean(data, axis=1)   # (62, 5)
        features.append(trial_feat.flatten())  # (310,)

    return np.array(features)   # (24, 310)


# ──────────────────────────────────────────────
# LOAD ONE EYE FILE
# ──────────────────────────────────────────────
def load_eye_file(filepath):
    """
    Returns np.array of shape (24, 31)
    For each trial key eye_{i}: shape (31, n_windows)
    → mean over n_windows → (31,)
    """
    mat = sio.loadmat(filepath)
    features = []

    for trial in range(1, N_TRIALS + 1):
        key = f"eye_{trial}"
        if key not in mat:
            raise KeyError(f"Key '{key}' not found in {os.path.basename(filepath)}. "
                           f"Available: {[k for k in mat if not k.startswith('__')]}")

        data = mat[key]          # (31, n_windows)
        assert data.shape[0] == N_EYE_FEATS, \
            f"Expected 31 eye features, got {data.shape[0]} in {key}"

        # mean over time windows axis → (31,)
        trial_feat = np.mean(data, axis=1)   # (31,)
        features.append(trial_feat)

    return np.array(features)   # (24, 31)


# ──────────────────────────────────────────────
# LOAD ALL SUBJECTS × ALL SESSIONS
# ──────────────────────────────────────────────
def load_all_data(eeg_root, eye_root, verbose=True):
    all_eeg, all_eye, all_labels, all_meta = [], [], [], []
    errors = []

    sessions = sorted(
        [d for d in os.listdir(eeg_root) if os.path.isdir(os.path.join(eeg_root, d))],
        key=int
    )

    for session_folder in sessions:
        session_id    = int(session_folder)
        eeg_sess_dir  = os.path.join(eeg_root, session_folder)
        eye_sess_dir  = os.path.join(eye_root,  session_folder)
        trial_labels  = SESSION_LABELS[session_id]

        eeg_files = sorted(
            [f for f in os.listdir(eeg_sess_dir) if f.endswith(".mat")]
        )

        for fname in eeg_files:
            subject_id = fname.split("_")[0]
            eeg_path   = os.path.join(eeg_sess_dir, fname)
            eye_path   = os.path.join(eye_sess_dir, fname)   # same filename

            if not os.path.exists(eye_path):
                msg = f"  [MISS] No eye file: session={session_id} file={fname}"
                print(msg); errors.append(msg)
                continue

            try:
                eeg_feat = load_eeg_file(eeg_path)   # (24, 310)
                eye_feat = load_eye_file(eye_path)    # (24, 31)
                labels   = np.array(trial_labels)     # (24,)

                all_eeg.append(eeg_feat)
                all_eye.append(eye_feat)
                all_labels.append(labels)

                for t in range(N_TRIALS):
                    all_meta.append({
                        "subject_id": subject_id,
                        "session_id": session_id,
                        "trial_id":   t + 1,
                        "label":      trial_labels[t],
                    })

                if verbose:
                    print(f"  OK  subject={subject_id:>2}  session={session_id}  "
                          f"eeg={eeg_feat.shape}  eye={eye_feat.shape}")

            except Exception as e:
                msg = f"  [ERR] subject={subject_id} session={session_id} → {e}"
                print(msg); errors.append(msg)

    if not all_eeg:
        raise RuntimeError("No data loaded. Check your paths and key names.")

    X_eeg = np.vstack(all_eeg)          # (N, 310)
    X_eye = np.vstack(all_eye)          # (N, 31)
    y     = np.concatenate(all_labels)  # (N,)

    return X_eeg, X_eye, y, all_meta, errors


# ──────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────
if __name__ == "__main__":

    print("=" * 55)
    print("  SEED-IV Loader  (fixed)")
    print("=" * 55)

    X_eeg, X_eye, y, meta, errors = load_all_data(EEG_ROOT, EYE_ROOT, verbose=True)

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  RESULT")
    print("=" * 55)
    print(f"  X_eeg  : {X_eeg.shape}   (samples × EEG features)")
    print(f"  X_eye  : {X_eye.shape}    (samples × eye features)")
    print(f"  y      : {y.shape}         (labels)")

    print("\n  Label distribution:")
    emotion_map = {0:"neutral", 1:"sad", 2:"fear", 3:"happy"}
    for cls in sorted(np.unique(y)):
        cnt = int(np.sum(y == cls))
        print(f"    {cls}  {emotion_map[cls]:>7}  →  {cnt} samples  "
              f"({cnt/len(y)*100:.1f}%)")

    print("\n  EEG stats:")
    print(f"    min={X_eeg.min():.4f}  max={X_eeg.max():.4f}  "
          f"mean={X_eeg.mean():.4f}  std={X_eeg.std():.4f}  "
          f"NaNs={np.isnan(X_eeg).sum()}")

    print("\n  Eye stats:")
    print(f"    min={X_eye.min():.4f}  max={X_eye.max():.4f}  "
          f"mean={X_eye.mean():.4f}  std={X_eye.std():.4f}  "
          f"NaNs={np.isnan(X_eye).sum()}")

    if errors:
        print(f"\n  Warnings / errors ({len(errors)}):")
        for e in errors:
            print(f"    {e}")

    # ── Save ──
    np.save("X_eeg.npy", X_eeg)
    np.save("X_eye.npy", X_eye)
    np.save("y.npy",     y)
    with open("meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n  Saved: X_eeg.npy  X_eye.npy  y.npy  meta.json")
    print("  → Ready for baseline ML pipeline.")
    print("=" * 55)

"""
SEED-IV Baseline ML Pipeline
==============================
Inputs : X_eeg.npy (1080,310)  X_eye.npy (1080,31)  y.npy (1080,)  meta.json
Models : SVM · Random Forest · MLP
Modes  : EEG-only · Eye-only · Fused (EEG + Eye)
CV     : Leave-One-Subject-Out (LOSO) — 15 subjects, 72 trials each

Outputs:
  results_baseline.csv   — per-fold accuracy for every model × modality
  results_summary.csv    — mean ± std accuracy
  confusion_matrices.png — 3×3 grid of confusion matrices
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm           import SVC
from sklearn.ensemble      import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing  import StandardScaler
from sklearn.impute         import SimpleImputer
from sklearn.metrics        import accuracy_score, confusion_matrix, classification_report

# ──────────────────────────────────────────────
# 1. LOAD DATA
# ──────────────────────────────────────────────
print("=" * 60)
print("  SEED-IV Baseline ML Pipeline")
print("=" * 60)

X_eeg = np.load("X_eeg.npy")          # (1080, 310)
X_eye = np.load("X_eye.npy")          # (1080, 31)
y     = np.load("y.npy")              # (1080,)

with open("meta.json") as f:
    meta = json.load(f)

subject_ids = np.array([m["subject_id"] for m in meta])   # string labels "1".."15"
unique_subjects = sorted(set(subject_ids))

print(f"\n  X_eeg : {X_eeg.shape}")
print(f"  X_eye : {X_eye.shape}")
print(f"  y     : {y.shape}")
print(f"  Subjects: {len(unique_subjects)}  →  {unique_subjects}")

EMOTION_MAP  = {0:"neutral", 1:"sad", 2:"fear", 3:"happy"}
EMOTION_NAMES = [EMOTION_MAP[i] for i in range(4)]

# ──────────────────────────────────────────────
# 2. HANDLE NaNs IN EYE FEATURES
#    8 NaNs across 1080×31 — median imputation
# ──────────────────────────────────────────────
print(f"\n  Eye NaNs before imputation : {np.isnan(X_eye).sum()}")
imputer = SimpleImputer(strategy="median")
X_eye   = imputer.fit_transform(X_eye)
print(f"  Eye NaNs after  imputation : {np.isnan(X_eye).sum()}")

# ──────────────────────────────────────────────
# 3. BUILD FEATURE SETS
# ──────────────────────────────────────────────
X_fused = np.concatenate([X_eeg, X_eye], axis=1)   # (1080, 341)

feature_sets = {
    "EEG only" : X_eeg,
    "Eye only" : X_eye,
    "Fused"    : X_fused,
}

# ──────────────────────────────────────────────
# 4. DEFINE MODELS
# ──────────────────────────────────────────────
def get_models():
    return {
        "SVM": SVC(
            kernel="rbf", C=10, gamma="scale",
            decision_function_shape="ovr", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None,
            n_jobs=-1, random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu", solver="adam",
            max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15
        ),
    }

# ──────────────────────────────────────────────
# 5. LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION
# ──────────────────────────────────────────────
print("\n  Running LOSO cross-validation...")
print(f"  {'Modality':<12} {'Model':<16} {'Fold accs (per subject)'}")
print("  " + "─" * 80)

results   = []   # rows: {modality, model, subject, accuracy}
all_cms   = {}   # (modality, model) → summed confusion matrix

for feat_name, X in feature_sets.items():
    for model_name, clf in get_models().items():
        fold_accs = []
        cm_total  = np.zeros((4, 4), dtype=int)

        for subj in unique_subjects:
            test_mask  = subject_ids == subj
            train_mask = ~test_mask

            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]

            # Scale per fold (fit on train only)
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            fold_accs.append(acc)
            cm_total += confusion_matrix(y_test, y_pred, labels=[0,1,2,3])

            results.append({
                "modality"  : feat_name,
                "model"     : model_name,
                "subject"   : subj,
                "accuracy"  : round(acc * 100, 2),
                "n_test"    : int(test_mask.sum()),
            })

        all_cms[(feat_name, model_name)] = cm_total
        mean_acc = np.mean(fold_accs) * 100
        std_acc  = np.std(fold_accs)  * 100
        print(f"  {feat_name:<12} {model_name:<16}  "
              f"mean={mean_acc:.2f}%  std={std_acc:.2f}%")

# ──────────────────────────────────────────────
# 6. SAVE RESULTS
# ──────────────────────────────────────────────
df = pd.DataFrame(results)
df.to_csv("results_baseline.csv", index=False)

# Summary table
summary = (
    df.groupby(["modality", "model"])["accuracy"]
    .agg(["mean", "std", "min", "max"])
    .round(2)
    .reset_index()
)
summary.columns = ["modality", "model", "mean_acc", "std_acc", "min_acc", "max_acc"]
summary = summary.sort_values("mean_acc", ascending=False).reset_index(drop=True)
summary.to_csv("results_summary.csv", index=False)

print("\n" + "=" * 60)
print("  SUMMARY (sorted by mean accuracy)")
print("=" * 60)
print(f"  {'Modality':<12} {'Model':<16} {'Mean%':>7} {'Std%':>7} {'Min%':>7} {'Max%':>7}")
print("  " + "─" * 60)
for _, row in summary.iterrows():
    print(f"  {row.modality:<12} {row.model:<16} "
          f"{row.mean_acc:>7.2f} {row.std_acc:>7.2f} "
          f"{row.min_acc:>7.2f} {row.max_acc:>7.2f}")

# ──────────────────────────────────────────────
# 7. CONFUSION MATRIX PLOTS  (3 modalities × 3 models)
# ──────────────────────────────────────────────
modalities = list(feature_sets.keys())
model_names = list(get_models().keys())

fig = plt.figure(figsize=(15, 13))
fig.suptitle("SEED-IV Confusion Matrices — LOSO Cross-Validation", fontsize=14, y=1.01)
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

for r, feat_name in enumerate(modalities):
    for c, model_name in enumerate(model_names):
        ax  = fig.add_subplot(gs[r, c])
        cm  = all_cms[(feat_name, model_name)]
        # Normalize per row (true label)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest",
                       cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{feat_name}\n{model_name}", fontsize=9, pad=4)
        ax.set_xticks(range(4)); ax.set_xticklabels(EMOTION_NAMES, fontsize=7, rotation=30)
        ax.set_yticks(range(4)); ax.set_yticklabels(EMOTION_NAMES, fontsize=7)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)

        # Annotate cells
        for i in range(4):
            for j in range(4):
                val  = cm_norm[i, j]
                color = "white" if val > 0.55 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

plt.colorbar(im, ax=fig.get_axes(), shrink=0.6, label="Recall")
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n  Saved: confusion_matrices.png")

# ──────────────────────────────────────────────
# 8. PER-CLASS REPORT (best model overall)
# ──────────────────────────────────────────────
best_row = summary.iloc[0]
print(f"\n  Best combo: [{best_row.modality}] [{best_row.model}]  "
      f"mean={best_row.mean_acc:.2f}%")

# Re-run best model to get full classification report
best_X    = feature_sets[best_row.modality]
best_name = best_row.model

y_true_all, y_pred_all = [], []
for subj in unique_subjects:
    test_mask  = subject_ids == subj
    train_mask = ~test_mask
    X_tr, X_te = best_X[train_mask], best_X[test_mask]
    y_tr, y_te = y[train_mask], y[test_mask]
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)
    clf    = get_models()[best_name]
    clf.fit(X_tr, y_tr)
    y_true_all.extend(y_te)
    y_pred_all.extend(clf.predict(X_te))

print("\n  Per-class report for best model:")
print(classification_report(
    y_true_all, y_pred_all,
    target_names=EMOTION_NAMES, digits=3
))

print("=" * 60)
print("  Done. Files saved:")
print("    results_baseline.csv")
print("    results_summary.csv")
print("    confusion_matrices.png")
print("=" * 60)