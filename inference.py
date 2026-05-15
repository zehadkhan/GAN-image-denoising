import os
import sys
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import config
from models.generator import UNetGenerator


def denoise(image_path, checkpoint_path=None, output_path='denoised.png', add_noise=False):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    ckpt_path = checkpoint_path or os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')

    G = UNetGenerator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt['generator'] if 'generator' in ckpt else ckpt)
    G.eval()

    tf = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    img = tf(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    if add_noise:
        noisy = torch.clamp(img + torch.randn_like(img) * config.NOISE_STD, 0, 1)
    else:
        noisy = img

    with torch.no_grad():
        out = G(noisy)

    if add_noise:
        save_image(torch.cat([noisy, out, img], dim=3), output_path)
        print(f"Saved [noisy | denoised | original] → {output_path}")
    else:
        save_image(out, output_path)
        print(f"Saved denoised image → {output_path}")
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default='denoised.png')
    parser.add_argument('--add-noise', action='store_true')
    args = parser.parse_args()
    denoise(args.image, args.checkpoint, args.output, args.add_noise)
