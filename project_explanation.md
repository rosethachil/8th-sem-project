# SEED-IV Project Technical Explanation

This document provides a comprehensive technical overview of your local SEED-IV project, detailing the codebase logic, dataset structure, improvement suggestions, and local execution strategies.

---

## 1. Code Understanding: `main.py` vs `main1.py`

### `main.py`: The Baseline ML Pipeline
**Purpose**: This script is built to reliably extract data from the SEED-IV dataset and run traditional baseline Machine Learning algorithms. 
**Key Functions & Flow**:
1. **Data Loading (`load_all_data`, `load_eeg_file`, `load_eye_file`)**: 
   - It iterates over session folders (1, 2, 3) and pairs EEG and Eye `.mat` files. 
   - It extracts features trial-by-trial (24 trials per session) and correctly assigns the Ground Truth label based on a hardcoded mapping (`SESSION_LABELS`).
   - For EEG, it collapses time windows by taking a `mean(axis=1)` to output a flattened `(310,)` feature vector per trial.
   - It saves this processed structured data as NumPy arrays (`X_eeg.npy`, `X_eye.npy`, `y.npy`).
2. **ML Pipeline (`get_models()`)**:
   - Uses traditional classical models from `scikit-learn` (SVM, Random Forest, MLP).
   - Validates using a rigorous **Leave-One-Subject-Out (LOSO) cross-validation**, meaning it evaluates generalization on an unseen subject.
   - Evaluates three feature modalities: "EEG only", "Eye only", and "Fused" (concatenated features).
   - **Data Leakage Prevention**: It strictly standardizes (`StandardScaler`) on the train split only.

### `main1.py`: Multimodal Deep Learning (TensorFlow)
**Purpose**: This script defines Neural Network architectures using TensorFlow/Keras to perform Multimodal Fusion with Attention mechanisms. It contains two distinct iterations (or "halves") of a similar pipeline.
**Key Functions & Flow**:
1. **Data Unzipping & Loading**: 
   - In the first iteration, data arrays are padded to fixed maximum values (310 for EEG, 200 for Eye) uniformly if they fall short of a fixed size boundary.
   - The first iteration suffers from naive array alignment `min(len(X_eye), len(X_eeg))` slicing. This risks assigning an EEG trial to an mismatched Eye trial. The second iteration effectively solves this problem by utilizing `parse_filename` and dictionary keys `(subject, session)`.
2. **Preprocessing**:
   - Performs Dimensionality Reduction via **PCA** (e.g. `PCA(n_components=50)`) coupled with Standard Scaling.
3. **Model Architecture (`create_model()`)**:
   - Takes multi-input structure: `eeg_input` and `eye_input`.
   - Feature branches pass through separate `Dense` layers.
   - Uses a **cross-attention mechanism**: concatenates the features, applies a `sigmoid` activation mapping to output an attention distribution, which then re-multiplies (`Multiply()()`) the isolated branch features to emphasize important modalities before feeding into a final fusion layer and a softmax output classifier.
4. **Training Strategy**: Uses 5-Fold Cross Validation instead of subject-wise Leave-One-Subject-Out.

---

## 2. Dataset Explanation (.mat format)

The SEED-IV (SJTU Emotion EEG Dataset) measures 4 emotion states across 15 subjects. The data structure you work with is stored in `.mat` files which are binary data containers native to MATLAB.

- **Structure**: It functions much like a Python dictionary. The keys of the structure represent variable names defined in MATLAB, and the values are multidimensional nested NumPy Arrays.
- **EEG Typical Structure**: Biological EEG time-series arrays typically store a 3D matrix. In SEED-IV, `de_LDS1` usually represents `(Channels, Time Windows, Frequency Bands)`. 
  - `62` physical EEG sensors.
  - `5` freq bands: delta, theta, alpha, beta, gamma.
- **Eye Data Structure**: Features like pupil diameter, dispersion, blink frequency, and saccade magnitude mapped to 31 distinct dimensions.
- **Tools**: `scipy.io.loadmat` can unpack `.mat` contents up to MATLAB version 7.2. For MATLAB version 7.3+, the `h5py` library stringifier is natively required instead.

---

## 3. Improvement Suggestions & Local Refactoring

Code initially targeted for Google Colab needs to be streamlined.

### 1) Remove Colab-Specific Assumptions
- **Hardcoded Absolute Paths**: Paths like `/content/drive/MyDrive/...` break locally. 
  - *Fix*: Adopt dynamic dataset paths relative to your local folder using `os.path.join()`. In `main1.py`, remove `/content/` paths entirely and replace with local paths like `dataset/eeg_feature_smooth.zip`. 
- **Zip Management**: `unzip('/content/eeg_feature_smooth.zip')` assumes the file is in root. Since you have these inside your `dataset` folder, adapt logic to extract to an `extracted/` folder.
- **GPU Usage**: Colab assumes dynamic memory allocation on its single GPU.
  - *Fix*: Limit TensorFlow memory growth locally to prevent crashes:
    ```python
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
    ```

### 2) Code Quality & Reusability Issues
- `main1.py` essentially contains two completely duplicated execution cycles concatenated back to back. You should delete the first half and keep the robust "FINAL DATASET-SPECIFIC PIPELINE" part.
- Structure it modularly: Move `load_eeg`, `load_eye`, `create_model` into separate files (e.g., `data_loader.py`, `models.py`) instead of leaving them scattered. Avoid repeating imports and suppressing warnings dynamically midway through execution.

### 3) Critical Data Leakage in `main1.py`
In `main1.py` around Line 354, the PCA and Standardization are fitted onto the **entire dataset entirely prior to Splitting into K-Folds**.
`X_eeg = scaler_eeg.fit_transform(X_eeg)`
- *Why is this bad for ML?*: The scaling metrics (Mean, Variance) and PCA Covariance components incorporate dataset characteristics from your Test Split. Your model evaluates overly optimistically because test-set knowledge has "leaked" into the training phase. 
- *Fix*: `StandardScaler` and `PCA.fit()` should solely occur isolated within the `for train_idx, test_idx` loop block on `X_train`, and then `transform()` onto `X_test`. `main.py` actually does this correctly for standard scaling.

---

## 4. How to Preprocess This Dataset locally

1. **Alignment Technique**: Use the improved implementation from the bottom half of `main1.py` that utilizes dictionary tuple keys `(subject_id, session_id)` rather than array splicing so your Eye and EEG feature rows always perfectly map to the same human emotional trial without error.
2. **Temporal Dimension Processing**: Right now, both scripts flatten the temporal properties by taking `np.mean(mat[key], axis=1)`. Emotion is a temporal cognitive process. Consider reshaping the matrices without the time mean-reduction, allowing recurrent state architectures like an LSTM or Temporal Convolution layer to analyze sequence trends.
