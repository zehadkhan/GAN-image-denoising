import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from scipy.io import loadmat
from config_phase2 import POSE_ANGLES, POSE_TOLERANCE, IMAGE_SIZE


def yaw_to_class(yaw_deg):
    """Discretize yaw angle to one of 5 pose classes."""
    diffs = [abs(yaw_deg - a) for a in POSE_ANGLES]
    return int(np.argmin(diffs))


def angle_to_onehot(angle_class, num_classes=5):
    v = torch.zeros(num_classes)
    v[angle_class] = 1.0
    return v


class PoseDataset(Dataset):
    """
    Loads 300W-LP dataset.
    Directory structure:
      root/
        AFW/  HELEN/  IBUG/  LFPW/
          img1.jpg  img1.mat  img2.jpg  img2.mat ...

    Each .mat file has Pose_Para field: [pitch, yaw, roll, tx, ty, tz, scale]
    We use yaw (index 1) for pose class.
    """

    def __init__(self, root_dir, split='train', image_size=IMAGE_SIZE):
        self.samples = []  # (img_path, pose_class)
        self.image_size = image_size

        subsets = sorted(os.listdir(root_dir))
        for subset in subsets:
            subset_dir = os.path.join(root_dir, subset)
            if not os.path.isdir(subset_dir):
                continue
            for fname in os.listdir(subset_dir):
                if not fname.lower().endswith('.jpg'):
                    continue
                mat_path = os.path.join(subset_dir, fname.replace('.jpg', '.mat'))
                img_path = os.path.join(subset_dir, fname)
                if not os.path.exists(mat_path):
                    continue
                try:
                    mat = loadmat(mat_path)
                    pose_para = mat['Pose_Para'][0]
                    yaw_deg = float(pose_para[1]) * (180.0 / np.pi)
                    pose_class = yaw_to_class(yaw_deg)
                    self.samples.append((img_path, pose_class))
                except Exception:
                    continue

        # Train/val split
        n = len(self.samples)
        n_train = int(n * 0.9)
        self.samples = self.samples[:n_train] if split == 'train' else self.samples[n_train:]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, pose_class = self.samples[idx]
        img = self.transform(Image.open(img_path).convert('RGB'))

        # Random target pose (different from source for training variety)
        import random
        target_class = random.randint(0, 4)

        source_pose = angle_to_onehot(pose_class)
        target_pose = angle_to_onehot(target_class)

        return img, source_pose, target_pose, pose_class, target_class


def get_pose_dataloaders(data_dir, batch_size=16, image_size=IMAGE_SIZE, num_workers=4):
    train_ds = PoseDataset(data_dir, 'train', image_size)
    val_ds = PoseDataset(data_dir, 'val', image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    print(f"Pose dataset — Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader
