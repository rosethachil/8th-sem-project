# =========================
# 1. IMPORTS
# =========================
import os
import zipfile
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Multiply, Concatenate, Dropout
from tensorflow.keras.models import Model

# =========================
# 2. UNZIP
# =========================
def unzip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

unzip('/content/eeg_feature_smooth.zip', '/content/eeg')
unzip('/content/eye_feature_smooth.zip', '/content/eye')

# =========================
# 3. LOAD EEG (STRUCTURED)
# =========================
def load_eeg(folder):
    X, y = [], []
    labels = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2]
    FIXED_SIZE = 310

    for root, _, files in os.walk(folder):
        for file_idx, file in enumerate(sorted(files)):
            if file.endswith('.mat'):
                data = sio.loadmat(os.path.join(root, file))

                for i in range(1, 25):
                    key = f'de_movingAve{i}'
                    if key in data:
                        trial = np.mean(data[key], axis=0).flatten()

                        if len(trial) < FIXED_SIZE:
                            trial = np.pad(trial, (0, FIXED_SIZE - len(trial)))
                        else:
                            trial = trial[:FIXED_SIZE]

                        X.append(trial)
                        y.append(labels[file_idx % len(labels)])

    return np.array(X), np.array(y)

# =========================
# 4. LOAD EYE (STRUCTURED)
# =========================
def load_eye(folder):
    X = []
    FIXED_SIZE = 200

    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if file.endswith('.mat'):
                data = sio.loadmat(os.path.join(root, file))

                for key in data:
                    if not key.startswith("__"):
                        arr = np.mean(data[key], axis=0).flatten()

                        if len(arr) < FIXED_SIZE:
                            arr = np.pad(arr, (0, FIXED_SIZE - len(arr)))
                        else:
                            arr = arr[:FIXED_SIZE]

                        X.append(arr)
                        break

    return np.array(X)

# =========================
# 5. LOAD DATA
# =========================
X_eeg, y = load_eeg('/content/eeg')
X_eye = load_eye('/content/eye')

print("EEG:", X_eeg.shape)
print("Eye:", X_eye.shape)

# =========================
# 🔥 6. PROPER ALIGNMENT (NOT RANDOM)
# =========================
# Match first N trials consistently
N = min(len(X_eye), len(X_eeg))

X_eeg = X_eeg[:N]
X_eye = X_eye[:N]
y = y[:N]

print("Aligned:", X_eeg.shape, X_eye.shape)

# =========================
# 7. NORMALIZE + PCA
# =========================
X_eeg = StandardScaler().fit_transform(X_eeg)
X_eye = StandardScaler().fit_transform(X_eye)

X_eeg = PCA(n_components=min(20, N)).fit_transform(X_eeg)
X_eye = PCA(n_components=min(10, N)).fit_transform(X_eye)

# =========================
# 🔥 8. MODEL FUNCTION
# =========================
def create_model(input_eeg, input_eye):

    eeg_input = Input(shape=(input_eeg,))
    eeg_feat = Dense(16, activation='relu')(eeg_input)

    eye_input = Input(shape=(input_eye,))
    eye_feat = Dense(16, activation='relu')(eye_input)

    # Attention
    attention = Dense(16, activation='sigmoid')(Concatenate()([eeg_feat, eye_feat]))

    eeg_att = Multiply()([eeg_feat, attention])
    eye_att = Multiply()([eye_feat, attention])

    fusion = Concatenate()([eeg_att, eye_att])
    fusion = Dense(8, activation='relu')(fusion)
    fusion = Dropout(0.2)(fusion)

    output = Dense(4, activation='softmax')(fusion)

    model = Model(inputs=[eeg_input, eye_input], outputs=output)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# =========================
# 🔥 9. K-FOLD TRAINING
# =========================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []

for fold, (train_idx, test_idx) in enumerate(kf.split(X_eeg)):
    print(f"\n===== Fold {fold+1} =====")

    X_eeg_train, X_eeg_test = X_eeg[train_idx], X_eeg[test_idx]
    X_eye_train, X_eye_test = X_eye[train_idx], X_eye[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = create_model(X_eeg.shape[1], X_eye.shape[1])

    model.fit(
        [X_eeg_train, X_eye_train],
        y_train,
        epochs=30,
        batch_size=8,
        verbose=0
    )

    loss, acc = model.evaluate([X_eeg_test, X_eye_test], y_test, verbose=0)
    print("Accuracy:", acc)
    acc_scores.append(acc)

    # Confusion Matrix
    y_pred = np.argmax(model.predict([X_eeg_test, X_eye_test]), axis=1)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# =========================
# FINAL RESULT
# =========================
print("\nFinal Average Accuracy:", np.mean(acc_scores))

# =========================

# FINAL DATASET-SPECIFIC PIPELINE

# =========================

import os
import zipfile
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Multiply, Concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
import shap
import warnings
warnings.filterwarnings('ignore')

# =========================

# UNZIP

# =========================

def unzip(zip_path, extract_to):
  if not os.path.exists(extract_to):
    with zipfile.ZipFile('/content/eye_feature_smooth (1).zip', 'r') as zip_ref:
      zip_ref.extractall('/content/eye')

unzip('/content/eeg_feature_smooth (1).zip', '/content/eeg')
unzip('/content/eye_feature_smooth (1).zip', '/content/eye')
# =========================

# LABELS

# =========================

EMOTION_NAMES = ['Neutral', 'Sad', 'Fear', 'Happy']
SESSION_LABELS = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2]

# =========================

# PARSE FILENAME

# =========================

def parse_filename(filename):
  import re
  nums = re.findall(r'\d+', filename)
  if len(nums) >= 2:
    return int(nums[0]), int(nums[1])
    return None, None

# =========================

# EEG LOADING

# =========================

def load_eeg(folder):
    eeg_dict = {}

    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if file.endswith('.mat'):

                mat = sio.loadmat(os.path.join(root, file))
                subject, session = parse_filename(file)

                trials = []
                for i in range(1, 25):
                    key = f'de_movingAve{i}'
                    if key in mat:
                        feature = np.mean(mat[key], axis=1).flatten()
                        trials.append(feature)

                if (subject, session) not in eeg_dict:
                    eeg_dict[(subject, session)] = []

                eeg_dict[(subject, session)].extend(trials)

    return eeg_dict


# =========================

# EYE LOADING

# =========================

def load_eye(folder):
    eye_dict = {}
    MAX_SIZE = 200

    for root, _, files in os.walk(folder):
        for file in sorted(files):
            if file.endswith('.mat'):

                mat = sio.loadmat(os.path.join(root, file))
                subject, session = parse_filename(file)

                trials = []
                keys = [k for k in mat.keys() if not k.startswith('__')]

                for key in sorted(keys):
                    arr = mat[key]

                    if isinstance(arr, np.ndarray):
                        feature = arr.flatten()

                        # FIX SIZE
                        if len(feature) < MAX_SIZE:
                            feature = np.pad(feature, (0, MAX_SIZE - len(feature)))
                        else:
                            feature = feature[:MAX_SIZE]

                        trials.append(feature)

                trials = np.array(trials)

                if (subject, session) not in eye_dict:
                    eye_dict[(subject, session)] = trials
                else:
                    eye_dict[(subject, session)] = np.vstack([eye_dict[(subject, session)], trials])

    return eye_dict

# =========================

# ALIGNMENT

# =========================

def align_data(eeg_dict, eye_dict):
    X_eeg, X_eye, y = [], [], []

    for key in eeg_dict:
        if key not in eye_dict:
            continue

        eeg_trials = eeg_dict[key]
        eye_trials = eye_dict[key]

        n = min(len(eeg_trials), len(eye_trials), len(SESSION_LABELS))

        for i in range(n):
            X_eeg.append(eeg_trials[i])
            X_eye.append(eye_trials[i])
            y.append(SESSION_LABELS[i])

    return np.array(X_eeg), np.array(X_eye), np.array(y)

# =========================

# LOAD DATA

# =========================

eeg_dict = load_eeg('/content/eeg')
eye_dict = load_eye('/content/eye/eye_feature_smooth')

X_eeg, X_eye, y = align_data(eeg_dict, eye_dict)

print("Aligned Shapes:", X_eeg.shape, X_eye.shape)

# =========================

# PREPROCESS

# =========================

scaler_eeg = StandardScaler()
scaler_eye = StandardScaler()

X_eeg = scaler_eeg.fit_transform(X_eeg)
X_eye = scaler_eye.fit_transform(X_eye)

X_eeg = PCA(n_components=50).fit_transform(X_eeg)
X_eye = PCA(n_components=30).fit_transform(X_eye)
# =========================

# MODEL

# =========================

def create_model(eeg_dim, eye_dim):


  eeg_input = Input(shape=(eeg_dim,))
  eye_input = Input(shape=(eye_dim,))

  eeg_feat = Dense(32, activation='relu')(eeg_input)
  eye_feat = Dense(32, activation='relu')(eye_input)

  concat = Concatenate()([eeg_feat, eye_feat])

  attention = Dense(64, activation='sigmoid')(concat)

  eeg_att = Lambda(lambda x: x[:, :32])(attention)
  eye_att = Lambda(lambda x: x[:, 32:])(attention)

  eeg_feat = Multiply()([eeg_feat, eeg_att])
  eye_feat = Multiply()([eye_feat, eye_att])

  fusion = Concatenate()([eeg_feat, eye_feat])
  fusion = Dense(16, activation='relu')(fusion)
  fusion = Dropout(0.3)(fusion)

  output = Dense(4, activation='softmax')(fusion)

  model = Model([eeg_input, eye_input], output)
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  attention_model = Model([eeg_input, eye_input], [eeg_att, eye_att])

  return model, attention_model


# =========================

# TRAINING

# =========================

kf = KFold(n_splits=5, shuffle=True, random_state=42)

acc_scores = []
all_true, all_pred = [], []

for train_idx, test_idx in kf.split(X_eeg):

  model, att_model = create_model(X_eeg.shape[1], X_eye.shape[1])

  model.fit(
      [X_eeg[train_idx], X_eye[train_idx]],
      y[train_idx],
      epochs=30,
      batch_size=16,
      verbose=0
  )

  _, acc = model.evaluate([X_eeg[test_idx], X_eye[test_idx]], y[test_idx], verbose=0)
  acc_scores.append(acc)

  pred = np.argmax(model.predict([X_eeg[test_idx], X_eye[test_idx]]), axis=1)

  all_true.extend(y[test_idx])
  all_pred.extend(pred)


# =========================

# RESULTS

# =========================

print("\nFinal Accuracy:", np.mean(acc_scores))

cm = confusion_matrix(all_true, all_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n", classification_report(all_true, all_pred))

# =========================

# MODALITY IMPORTANCE

# =========================

eeg_att, eye_att = att_model.predict([X_eeg, X_eye])

print("\nModality Importance:")
print("EEG Contribution:", np.mean(eeg_att))
print("Eye Contribution:", np.mean(eye_att))

# =========================

# SHAP

# =========================

def model_wrapper(X):
  n = X_eeg.shape[1]
  return model.predict([X[:, :n], X[:, n:]])

  X_combined = np.hstack([X_eeg, X_eye])

  explainer = shap.KernelExplainer(model_wrapper, X_combined[:50])
  shap_values = explainer.shap_values(X_combined[:20])

  shap.summary_plot(shap_values[0], X_combined[:20])
