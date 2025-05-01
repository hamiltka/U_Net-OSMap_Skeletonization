import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os

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

        img = Image.open(img_path)
        skel = Image.open(skel_path)

        img = np.array(img, dtype=np.float32) / 255.0
        skel = np.array(skel, dtype=np.float32) / 255.0

        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        skel = torch.tensor(skel, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            img, skel = self.transform(img, skel)
        return img, skel

def get_dataloaders(output_dir, batch_size=32):
    train_dataset = RoadSkeletonDataset(
        os.path.join(output_dir, 'train', 'images'),
        os.path.join(output_dir, 'train', 'skeletons')
    )
    val_dataset = RoadSkeletonDataset(
        os.path.join(output_dir, 'val', 'images'),
        os.path.join(output_dir, 'val', 'skeletons')
    )
    test_dataset = RoadSkeletonDataset(
        os.path.join(output_dir, 'test', 'images'),
        os.path.join(output_dir, 'test', 'skeletons')
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("DataLoaders created!")
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    output_dir = '/content/data/thinning/Oxford_split'
    train_loader, val_loader, test_loader = get_dataloaders(output_dir, batch_size=32)
    # Example: Iterate through one batch
    for images, masks in train_loader:
        print(f"Batch images shape: {images.shape}")
        print(f"Batch masks shape: {masks.shape}")
        break