import os
import sys
import torch
from torch.cuda.amp import autocast
from torchvision.utils import save_image

import config
from data.dataset import get_dataloaders
from models.generator import UNetGenerator
from utils.metrics import psnr, ssim


def evaluate(checkpoint_path=None):
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    amp_enabled = torch.cuda.is_available()

    ckpt_path = checkpoint_path or os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    G = UNetGenerator().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt['generator'] if 'generator' in ckpt else ckpt)
    if 'epoch' in ckpt:
        print(f"Checkpoint: epoch {ckpt['epoch']}, best PSNR={ckpt.get('best_psnr', '?'):.2f}")
    G.eval()

    _, val_loader = get_dataloaders(config.DATA_DIR, config.BATCH_SIZE,
                                    config.IMAGE_SIZE, config.NOISE_STD, config.NUM_WORKERS)

    os.makedirs('eval_results', exist_ok=True)
    total_psnr = total_ssim = 0.0

    with torch.no_grad():
        for i, (noisy, clean) in enumerate(val_loader):
            noisy, clean = noisy.to(device), clean.to(device)
            with autocast(enabled=amp_enabled):
                gen = G(noisy)
            total_psnr += psnr(gen, clean)
            total_ssim += ssim(gen, clean)
            if i < 5:
                save_image(torch.cat([noisy[:4], gen[:4], clean[:4]], 0),
                           f'eval_results/batch_{i:02d}.png', nrow=4)

    n = len(val_loader)
    avg_psnr, avg_ssim = total_psnr / n, total_ssim / n
    print(f"\nEvaluation  →  PSNR: {avg_psnr:.4f} dB  |  SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim


if __name__ == '__main__':
    evaluate(sys.argv[1] if len(sys.argv) > 1 else None)
