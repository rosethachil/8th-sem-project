import os
import zipfile
from scipy.io import loadmat

zip_path = r"C:/Users/Rose J Thachil/Documents/8th sem/dataset/eeg_feature_smooth.zip"
internal_folder_prefix = "eeg_feature_smooth/1/"

with zipfile.ZipFile(zip_path, 'r') as zf:
    for file_info in zf.infolist():
        if file_info.filename.startswith(internal_folder_prefix) and file_info.filename.endswith(".mat"):
            with zf.open(file_info) as f:
                data = loadmat(f)
                filename = os.path.basename(file_info.filename)
                print(filename, data.keys())