import torch
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.ndimage import distance_transform_edt

def get_distance_map(mask):
    # mask: numpy array, 1 for skeleton, 0 for background
    return distance_transform_edt(1 - mask)

class RoadSkeletonDataset(Dataset):
    def __init__(self, image_dir, skeleton_dir, transform=None):
        self.image_dir = image_dir
        self.skeleton_dir = skeleton_dir
        self.image_paths = sorted(os.listdir(image_dir))
        self.skeleton_paths = sorted(os.listdir(skeleton_dir))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        skel_name = self.skeleton_paths[idx]

        img_path = os.path.join(self.image_dir, img_name)
        skel_path = os.path.join(self.skeleton_dir, skel_name)

        img = Image.open(img_path).convert('L')
        skel = Image.open(skel_path).convert('L')

        img = np.array(img, dtype=np.float32) / 255.0
        skel = np.array(skel, dtype=np.float32) / 255.0
        skel = (skel > 0.5).astype(np.float32)  # binarize

        dist_map = get_distance_map(skel)
        dist_map = dist_map / (dist_map.max() + 1e-8)

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        skel = torch.tensor(skel, dtype=torch.float32).unsqueeze(0)
        dist_map = torch.tensor(dist_map, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img, skel, dist_map = self.transform(img, skel, dist_map)
        return img, skel, dist_map

def get_dataloaders(output_dir, batch_size=8, transform=None):
    train_dataset = RoadSkeletonDataset(
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'train', 'skeletons'),
        transform=transform
    )
    val_dataset = RoadSkeletonDataset(
        os.path.join(output_dir, 'val', 'images'),
        os.path.join(output_dir, 'val', 'skeletons'),
        transform=transform
    )
    test_dataset = RoadSkeletonDataset(
        os.path.join(output_dir, 'test', 'images'),
        os.path.join(output_dir, 'test', 'skeletons'),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    output_dir = '/content/data/thinning/Oxford_split'
    train_loader, val_loader, test_loader = get_dataloaders(output_dir)
    for images, skel_masks, dist_maps in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch masks shape: {skel_masks.shape}")
        print(f"Batch dist_maps shape: {dist_maps.shape}")
        break
