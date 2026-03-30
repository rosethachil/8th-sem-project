# Guide to Understanding & Using `.mat` Datasets

Working with `.mat` (MATLAB) files is very common in physiological research (like EEG and Eye-tracking for emotion recognition). While they might seem like black boxes at first, they are actually very structured and straightforward to use in Python.

Here is a simple, step-by-step guide answering your questions.

---

## 1. Understanding the Dataset

### What is a `.mat` file?
A `.mat` file is simply a binary "container" created by MATLAB to save variables. Think of it exactly like a **Python Dictionary**. 

* **Keys**: The names of the variables as they were saved in MATLAB (e.g., `'de_LDS1'` or `'eye_1'`).
* **Values**: The actual data, which are almost always multidimensional matrices (arrays of numbers).

### What kind of data structures do they store?
In physiological datasets like SEED-IV, these files usually store tightly packed **NumPy arrays** (tensors).
* **EEG Arrays**: Usually 3-Dimensional. For example, a shape of `(62, 300, 5)` means 62 EEG channels/sensors, 300 time-windows, and 5 brainwave frequency bands (Delta, Theta, Alpha, Beta, Gamma).
* **Eye-Tracking Arrays**: Usually 2-Dimensional. A shape of `(31, 300)` means 31 distinct eye features (pupil size, blink rate, saccade length) recorded over 300 time-windows.

---

## 2. Practical Steps (Code Required)

To open these files, we use the `scipy.io` Python library. Here is a script designed to load a file, peek inside, and understand its dimensions.

```python
import scipy.io as sio
import numpy as np

def explore_mat_file(filepath):
    """Loads a .mat file and prints its internal structure."""
    print(f"Loading: {filepath}")
    
    # 1. Load the .mat file into a dictionary-like object
    try:
        mat_data = sio.loadmat(filepath)
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    # 2. Extract keys (ignore MATLAB metadata which starts with '__')
    clean_keys = [key for key in mat_data.keys() if not key.startswith('__')]
    
    print(f"Total variables found: {len(clean_keys)}\n")
    
    # 3. Print shapes and sample data
    for key in clean_keys[:3]: # Let's just look at the first 3 variables
        data_array = mat_data[key]
        
        print(f"Variable Name: '{key}'")
        print(f"  - Shape: {data_array.shape}  (e.g., Channels x Time x Freq)")
        print(f"  - Data Type: {data_array.dtype}")
        
        # Print a tiny slice of the data just to see what the numbers look like
        if data_array.ndim >= 2:
            print(f"  - Sample Value (first channel/feature, first time step): \n    {data_array[0, 0]}")
        print("-" * 40)

# Example Usage:
# explore_mat_file('dataset/extracted_eye/1_20131027.mat')
```

---

## 3. Converting to Readable Formats

You can easily convert `.mat` matrices into CSVs or Excel files, but you must flatten them first because CSVs are strictly 2-Dimensional (Rows and Columns).

### Python Code for Conversion to Pandas / CSV:
```python
import pandas as pd
import scipy.io as sio
import numpy as np

mat_data = sio.loadmat('example_eeg_file.mat')
eeg_trial_1 = mat_data['de_LDS1'] # Shape is likely 3D (62, 200, 5)

# 1. Take the average across time to squash it into 2D (62 channels, 5 frequencies)
eeg_mean_time = np.mean(eeg_trial_1, axis=1) 

# 2. Flatten it completely to a 1D vector of 310 features (62 x 5)
flattened_features = eeg_mean_time.flatten() 

# 3. Create a Pandas DataFrame (1 row, 310 columns)
df = pd.DataFrame([flattened_features])

# 4. Save to CSV
df.to_csv("converted_trial.csv", index=False)
```

### 🛑 When should you NOT convert?
**Never convert to CSV if you plan on using Deep Learning (like CNNs or LSTMs).** 
Why? CSVs force your data to be flat (2D). If you flatten a 3D EEG array, the algorithm loses the "spatial" map of where the electrodes are on the scalp, and it loses the "temporal" sequence of how the brainwaves changed over time. 

Instead, extract the NumPy arrays out of the `.mat` file and save them as Python `.npy` binaries (using `np.save('eeg_data.npy', data)`). 

---

## 4. Exploration & Preprocessing

Before giving data to an AI model, follow these three preprocessing steps:

### A. Feature Understanding
If this is an emotion recognition task, your features (like `de_root` or `de_movingAve`) usually represent **Differential Entropy (DE)**. DE is heavily proven to correlate with human emotional states (especially in the Beta and Gamma frequency bands). 
* *Tip*: Don't treat Eye tracking and EEG features equally. Your neural network should have a dual-branch architecture, one for EEG and one for Eye-tracking.

### B. Normalization / Scaling
Physiological data wildly fluctuates between individuals. Person A's "angry" brainwaves might be drastically quiet compared to Person B's "relaxed" brainwaves because of skull thickness or electrode placement.
* **Bad**: `StandardScaler.fit(Entire_Dataset)`
* **Good**: Scale data *per subject* or *per session*. 
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Correct way: isolate one person's trials, scale them, move to the next person.
subject_1_eeg = scaler.fit_transform(subject_1_eeg)
```

### C. Handling Missing Data (NaNS)
In eye-tracking datasets, missing data (NaN) almost always means the subject blinked or the camera lost their pupil temporarily. 
* **Do NOT drop the row!** If you drop a time-window row in the eye-tracker, it will no longer align chronologically with your EEG data for that exact same millisecond.
* **Fix**: Use median imputation or "forward-fill" to guess what the eye was doing during the blink.
```python
from sklearn.impute import SimpleImputer
# Replace missing eye-tracking frames with the median value of that feature
imputer = SimpleImputer(strategy='median')
clean_eye_data = imputer.fit_transform(raw_eye_data)
```
