import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CelebADenoiseDataset(Dataset):
    def __init__(self, root_dir, split='train', image_size=128, noise_std=0.10):
        self.noise_std = noise_std
        all_images = sorted([
            os.path.join(root_dir, f)
            for f in os.listdir(root_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        n_train = int(len(all_images) * 0.8333)
        self.image_paths = all_images[:n_train] if split == 'train' else all_images[n_train:]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        clean = self.transform(Image.open(self.image_paths[idx]).convert('RGB'))
        noisy = torch.clamp(clean + torch.randn_like(clean) * self.noise_std, 0.0, 1.0)
        return noisy, clean


def get_dataloaders(data_dir, batch_size=16, image_size=128, noise_std=0.10, num_workers=4):
    train_ds = CelebADenoiseDataset(data_dir, 'train', image_size, noise_std)
    val_ds = CelebADenoiseDataset(data_dir, 'val', image_size, noise_std)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
