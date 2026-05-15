import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

import config
from data.dataset import get_dataloaders
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from models.losses import GeneratorLoss, GANLoss
from utils.metrics import psnr, ssim


def train():
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    amp_enabled = torch.cuda.is_available()

    for d in (config.CHECKPOINT_DIR, config.RESULTS_DIR, config.LOG_DIR):
        os.makedirs(d, exist_ok=True)

    print(f"Device: {device} | AMP: {amp_enabled}")

    train_loader, val_loader = get_dataloaders(
        config.DATA_DIR, config.BATCH_SIZE, config.IMAGE_SIZE,
        config.NOISE_STD, config.NUM_WORKERS,
    )
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    G = UNetGenerator().to(device)
    D = PatchGANDiscriminator().to(device)

    opt_g = optim.Adam(G.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))
    opt_d = optim.Adam(D.parameters(), lr=config.LEARNING_RATE, betas=(config.BETA1, config.BETA2))

    g_loss_fn = GeneratorLoss(config.LAMBDA_L1, config.LAMBDA_PERCEPTUAL,
                              config.LAMBDA_SSIM, config.LAMBDA_ADV).to(device)
    d_loss_fn = GANLoss().to(device)

    scaler_g = GradScaler(enabled=amp_enabled)
    scaler_d = GradScaler(enabled=amp_enabled)

    best_psnr = 0.0
    log_path = os.path.join(config.LOG_DIR, 'train_log.csv')
    log = open(log_path, 'w')
    log.write('epoch,mode,g_loss,d_loss,val_psnr,val_ssim\n')

    for epoch in range(1, config.NUM_EPOCHS + 1):
        G.train(); D.train()
        warmup = epoch <= config.WARMUP_EPOCHS
        g_total = d_total = 0.0
        t0 = time.time()

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            # Discriminator step (skipped during warmup)
            if not warmup:
                opt_d.zero_grad()
                with autocast(enabled=amp_enabled):
                    fake_d = G(noisy).detach()
                    loss_d = 0.5 * (d_loss_fn(D(noisy, clean), True) +
                                    d_loss_fn(D(noisy, fake_d), False))
                scaler_d.scale(loss_d).backward()
                scaler_d.step(opt_d); scaler_d.update()
                d_total += loss_d.item()

            # Generator step
            opt_g.zero_grad()
            with autocast(enabled=amp_enabled):
                fake = G(noisy)
                disc_fake = D(noisy, fake) if not warmup else None
                loss_g, _ = g_loss_fn(fake, clean, disc_fake, warmup=warmup)
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g); scaler_g.update()
            g_total += loss_g.item()

        n = len(train_loader)
        g_avg, d_avg = g_total / n, d_total / n

        val_psnr, val_ssim = validate(G, val_loader, device, epoch, amp_enabled)
        mode = 'WARMUP' if warmup else 'FULL  '
        print(f"[{mode}] Epoch {epoch:03d}/{config.NUM_EPOCHS} | "
              f"G={g_avg:.4f} D={d_avg:.4f} | PSNR={val_psnr:.2f} SSIM={val_ssim:.4f} | "
              f"{time.time()-t0:.1f}s")

        log.write(f'{epoch},{mode.strip()},{g_avg:.4f},{d_avg:.4f},{val_psnr:.4f},{val_ssim:.4f}\n')
        log.flush()

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({'epoch': epoch, 'generator': G.state_dict(),
                        'discriminator': D.state_dict(), 'best_psnr': best_psnr},
                       os.path.join(config.CHECKPOINT_DIR, 'best_model.pth'))

        if epoch % config.SAVE_INTERVAL == 0:
            torch.save(G.state_dict(),
                       os.path.join(config.CHECKPOINT_DIR, f'generator_epoch{epoch:03d}.pth'))

    log.close()
    print(f"\nDone. Best PSNR: {best_psnr:.2f} dB")


@torch.no_grad()
def validate(G, loader, device, epoch, amp_enabled):
    G.eval()
    total_psnr = total_ssim = 0.0
    for i, (noisy, clean) in enumerate(loader):
        noisy, clean = noisy.to(device), clean.to(device)
        with autocast(enabled=amp_enabled):
            gen = G(noisy)
        total_psnr += psnr(gen, clean)
        total_ssim += ssim(gen, clean)
        if i == 0 and epoch % config.SAVE_INTERVAL == 0:
            d = os.path.join(config.RESULTS_DIR, f'epoch_{epoch:03d}')
            os.makedirs(d, exist_ok=True)
            save_image(torch.cat([noisy[:4], gen[:4], clean[:4]], 0),
                       os.path.join(d, 'comparison.png'), nrow=4)
    n = len(loader)
    return total_psnr / n, total_ssim / n


if __name__ == '__main__':
    train()
