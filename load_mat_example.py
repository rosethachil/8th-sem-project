import os
import zipfile
import scipy.io as sio
import numpy as np

def extract_and_preview_mat(zip_filepath, extract_dir):
    """
    Extracts a zip file containing .mat files and previews the structure of the first .mat file found.
    """
    print(f"Extracting {zip_filepath}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        extracted_files = zip_ref.namelist()
    
    # Find the first .mat file to preview
    mat_file = None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith('.mat'):
                mat_file = os.path.join(root, f)
                break
        if mat_file:
            break

    if not mat_file:
        print("No .mat files found in the extracted directory.")
        return

    print(f"\nPreviewing contents of: {mat_file}")
    
    # Load .mat file
    try:
        mat_contents = sio.loadmat(mat_file)
    except Exception as e:
        print(f"Error loading {mat_file}: {e}")
        return

    # Filter out MATLAB metadata variables (they start with '__')
    keys = [k for k in mat_contents.keys() if not k.startswith('__')]
    
    print(f"\nFound {len(keys)} variables. Top 5 keys preview:")
    
    for key in keys[:5]:
        data = mat_contents[key]
        print(f"  - Key Name: '{key}'")
        print(f"    - Type: {type(data)}")
        if isinstance(data, np.ndarray):
            print(f"    - Shape: {data.shape}")
            print(f"    - Data Type: {data.dtype}")
        print("-" * 30)

if __name__ == "__main__":
    # Example usage: Using the dataset zip files available in your local folder
    dataset_zip = os.path.join("dataset", "eye_feature_smooth.zip")
    extract_folder = os.path.join("dataset", "extracted_eye")
    
    if os.path.exists(dataset_zip):
        extract_and_preview_mat(dataset_zip, extract_folder)
    else:
        print(f"Dataset zip file not found at {dataset_zip}")
