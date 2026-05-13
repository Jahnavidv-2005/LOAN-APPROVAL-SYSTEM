import kagglehub
import shutil
import os
import glob

# Download latest version
path = kagglehub.dataset_download("architsharma01/loan-approval-prediction-dataset")
print("Path to dataset files:", path)

# Find the csv file in the downloaded directory
csv_files = glob.glob(os.path.join(path, "*.csv"))
if not csv_files:
    print("No CSV files found in the dataset.")
else:
    source_file = csv_files[0]
    # Destination path
    dest_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'loan_data.csv')
    
    # Copy file
    shutil.copy2(source_file, dest_file)
    print(f"Successfully copied {source_file} to {dest_file}")
