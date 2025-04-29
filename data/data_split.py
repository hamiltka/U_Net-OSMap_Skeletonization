import os
import random
from PIL import Image
import numpy as np

# Image directories
image_dir = '/content/data/thinning/Oxford/images'
skeleton_dir = '/content/data/thinning/Oxford/skeletons'
output_dir = '/content/data/thinning/Oxford_split' # Split image directory

# Image size
img_size = (256, 256)

# Set seed for reproducibility
random.seed(42)

# List all files
image_files = sorted(os.listdir(image_dir))
skeleton_files = sorted(os.listdir(skeleton_dir))

# Correctly pair images and skeletons
paired_files = list(zip(image_files, skeleton_files))

# Shuffle the paired files
random.shuffle(paired_files)

# Split the dataset (70%/15%/15%)
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

n_total = len(paired_files)
n_train = int(train_ratio * n_total)
n_val = int(val_ratio * n_total)
n_test = n_total - n_train - n_val

train_files = paired_files[:n_train]
val_files = paired_files[n_train:n_train + n_val]
test_files = paired_files[n_train + n_val:]

# Process and save data
def process_and_save(img_path, skel_path, output_dir, split, img_name, img_size):
    # Open images
    img = Image.open(img_path).convert('L').resize(img_size)
    skel = Image.open(skel_path).convert('L').resize(img_size)

    # Convert to numpy arrays
    img_array = np.array(img, dtype=np.float32)
    skel_array = np.array(skel, dtype=np.float32)

    # Normalize images
    img_array /= 255.0
    skel_array /= 255.0

    # Create directories if they don't exist
    os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, split, 'skeletons'), exist_ok=True)

    # Save the processed images
    Image.fromarray((img_array * 255).astype(np.uint8)).save(os.path.join(output_dir, split, 'images', img_name))
    Image.fromarray((skel_array * 255).astype(np.uint8)).save(os.path.join(output_dir, split, 'skeletons', img_name))

# Loop through each split and process the images
for split, files in zip(['train', 'val', 'test'], [train_files, val_files, test_files]):
    for img_name, skel_name in files:
        img_path = os.path.join(image_dir, img_name)
        skel_path = os.path.join(skeleton_dir, skel_name)
        process_and_save(img_path, skel_path, output_dir, split, img_name, img_size)

print("Data splitting and preprocessing complete!")
