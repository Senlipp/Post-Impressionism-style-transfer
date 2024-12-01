import os
from tqdm import tqdm

# Directory containing the files to be renamed
directory = "datasets\\Post_Impressionism"

# Define the desired filename format
file_format = "{:04d}"  # This creates filenames like 0001, 0002, etc.

# Get a sorted list of files in the directory
files = sorted(os.listdir(directory))

# Rename files in sequential order
for idx, file_name in enumerate(tqdm(files, desc="Renaming files")):
    # Construct the full original file path
    old_path = os.path.join(directory, file_name)
    
    # Skip directories, process only files
    if not os.path.isfile(old_path):
        continue

    # Extract the file extension
    _, extension = os.path.splitext(file_name)
    
    # Construct the new file name with extension
    new_name = file_format.format(idx + 1) + extension
    new_path = os.path.join(directory, new_name)
    
    # Rename the file
    os.rename(old_path, new_path)

print(f"Renamed {len(files)} files in {directory}.")
