import os
import cv2
import numpy as np
from tqdm import tqdm
import random
import shutil

# Directories for content and style images
content_dir = "datasets\\un_processed_content"
style_dir = "datasets\\Post_Impressionism"
output_dir = "datasets"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f"{output_dir}\\content", exist_ok=True)
os.makedirs(f"{output_dir}\\style", exist_ok=True)

# Image resizing parameters
target_size = (224, 224)

# Random brightness, contrast, and saturation adjustments
def random_adjustments(image):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Random brightness adjustment
    brightness_factor = random.uniform(0.8, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)

    # Random saturation adjustment
    saturation_factor = random.uniform(0.8, 1.3)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_factor, 0, 255)

    # Convert back to BGR
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# Data augmentation for style images
def augment_style_image(image):
    # Random cropping
    crop_size = random.uniform(0.8, 1.0)
    h, w = image.shape[:2]
    new_h, new_w = int(h * crop_size), int(w * crop_size)
    top = random.randint(0, h - new_h)
    left = random.randint(0, w - new_w)
    cropped = image[top:top+new_h, left:left+new_w]

    # Resize back to target size
    resized = cv2.resize(cropped, target_size)

    # Random horizontal flip
    if random.random() > 0.5:
        resized = cv2.flip(resized, 1)

    # Apply random adjustments
    return random_adjustments(resized)

# Preprocessing pipeline
def preprocess_images(input_dir, output_dir, is_style=False):
    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_dir, file_name)
            image = cv2.imread(file_path)

            if image is None:
                print(f"Warning: Unable to load image {file_name}. Skipping.")
                continue

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(image, (5, 5), 1.0)

            # Augment style images
            if is_style:
                augmented = augment_style_image(blurred)
                cv2.imwrite(os.path.join(output_dir, f"aug_{file_name}"), augmented)

            # Resize to target size
            resized = cv2.resize(blurred, target_size)
            cv2.imwrite(os.path.join(output_dir, file_name), resized)

# Process content images
# preprocess_images(content_dir, f"{output_dir}/content")

# Process and augment style images
# preprocess_images(style_dir, f"{output_dir}/style", is_style=True)

# Split the dataset into train, validation, and test sets
def split_dataset(content_dir, style_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    content_files = sorted(os.listdir(content_dir))
    style_files = sorted(os.listdir(style_dir))
    assert len(content_files) == len(style_files), "Content and style directories must have the same number of images."

    total_files = len(content_files)
    indices = list(range(total_files))
    random.shuffle(indices)

    train_end = int(train_ratio * total_files)
    val_end = int((train_ratio + val_ratio) * total_files)

    subsets = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:]
    }

    for subset, indices in subsets.items():
        os.makedirs(f"{output_dir}/{subset}/content", exist_ok=True)
        os.makedirs(f"{output_dir}/{subset}/style", exist_ok=True)

        for idx in indices:
            shutil.copy(f"{content_dir}/{content_files[idx]}", f"{output_dir}/{subset}/content/{content_files[idx]}")
            shutil.copy(f"{style_dir}/{style_files[idx]}", f"{output_dir}/{subset}/style/{style_files[idx]}")

    print("Data split completed!")

# Split the dataset
split_dataset(f"{output_dir}/content", f"{output_dir}/style", output_dir)