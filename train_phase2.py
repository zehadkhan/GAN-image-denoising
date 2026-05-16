import os
import time
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image

import config_phase2 as cfg
from data.pose_dataset import get_pose_dataloaders
from models.multi_angle_generator import MultiAngleGenerator
from models.pose_discriminator import PoseDiscriminator
from models.multi_angle_losses import MultiAngleLoss


def train_phase2():
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    amp_enabled = torch.cuda.is_available()

    for d in (cfg.CHECKPOINT_DIR_PHASE2, cfg.RESULTS_DIR_PHASE2, cfg.LOG_DIR_PHASE2):
        os.makedirs(d, exist_ok=True)

    print(f"Phase 2 Training | Device: {device}")

    train_loader, val_loader = get_pose_dataloaders(
        cfg.DATA_DIR_PHASE2, cfg.BATCH_SIZE, cfg.IMAGE_SIZE, cfg.NUM_WORKERS
    )

    G = MultiAngleGenerator(num_pose_classes=cfg.NUM_POSE_CLASSES).to(device)
    D = PoseDiscriminator(num_pose_classes=cfg.NUM_POSE_CLASSES).to(device)

    # Load Phase 1 encoder weights if available (transfer learning)
    if os.path.exists(cfg.PHASE1_CHECKPOINT):
        ckpt = torch.load(cfg.PHASE1_CHECKPOINT, map_location=device)
        phase1_state = ckpt.get('generator', ckpt)
        gen_state = G.state_dict()
        matched = {k: v for k, v in phase1_state.items()
                   if k in gen_state and gen_state[k].shape == v.shape}
        gen_state.update(matched)
        G.load_state_dict(gen_state)
        print(f"Loaded {len(matched)} layers from Phase 1 checkpoint")

    opt_g = optim.Adam(G.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA1, cfg.BETA2))
    opt_d = optim.Adam(D.parameters(), lr=cfg.LEARNING_RATE, betas=(cfg.BETA1, cfg.BETA2))

    loss_fn = MultiAngleLoss(cfg.LAMBDA_RECON, cfg.LAMBDA_IDENTITY,
                             cfg.LAMBDA_ADV, cfg.LAMBDA_POSE).to(device)

    scaler_g = GradScaler(enabled=amp_enabled)
    scaler_d = GradScaler(enabled=amp_enabled)

    best_loss = float('inf')
    log = open(os.path.join(cfg.LOG_DIR_PHASE2, 'train_log.csv'), 'w')
    log.write('epoch,g_loss,d_loss\n')

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        G.train(); D.train()
        g_total = d_total = 0.0
        t0 = time.time()

        for source, src_pose, tgt_pose, src_cls, tgt_cls in train_loader:
            source = source.to(device)
            src_pose = src_pose.to(device)
            tgt_pose = tgt_pose.to(device)
            src_cls = src_cls.to(device)
            tgt_cls = tgt_cls.to(device)
            same_pose_mask = (src_cls == tgt_cls)

            # Discriminator step
            opt_d.zero_grad()
            with autocast(enabled=amp_enabled):
                fake = G(source, tgt_pose).detach()
                real_patch, real_pose_logits = D(source, source)
                fake_patch, fake_pose_logits = D(source, fake)
                loss_d = loss_fn.discriminator_loss(
                    real_patch, fake_patch, real_pose_logits, fake_pose_logits, src_cls
                )
            scaler_d.scale(loss_d).backward()
            scaler_d.step(opt_d); scaler_d.update()

            # Generator step
            opt_g.zero_grad()
            with autocast(enabled=amp_enabled):
                fake = G(source, tgt_pose)
                fake_patch, fake_pose_logits = D(source, fake)
                loss_g, components = loss_fn.generator_loss(
                    fake, source, fake_patch, fake_pose_logits, tgt_cls, same_pose_mask
                )
            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g); scaler_g.update()

            g_total += loss_g.item()
            d_total += loss_d.item()

        n = len(train_loader)
        g_avg, d_avg = g_total / n, d_total / n
        print(f"Epoch {epoch:03d}/{cfg.NUM_EPOCHS} | G={g_avg:.4f} D={d_avg:.4f} | {time.time()-t0:.1f}s")

        log.write(f'{epoch},{g_avg:.4f},{d_avg:.4f}\n'); log.flush()

        if g_avg < best_loss:
            best_loss = g_avg
            torch.save({'epoch': epoch, 'generator': G.state_dict(),
                        'discriminator': D.state_dict(), 'best_loss': best_loss},
                       os.path.join(cfg.CHECKPOINT_DIR_PHASE2, 'best_model_phase2.pth'))

        if epoch % cfg.SAVE_INTERVAL == 0:
            save_samples(G, val_loader, device, epoch, amp_enabled)

    log.close()
    print(f"\nPhase 2 training done. Best G loss: {best_loss:.4f}")


@torch.no_grad()
def save_samples(G, loader, device, epoch, amp_enabled):
    G.eval()
    import config_phase2 as cfg
    source, src_pose, tgt_pose, _, _ = next(iter(loader))
    source = source[:4].to(device)

    results_dir = os.path.join(cfg.RESULTS_DIR_PHASE2, f'epoch_{epoch:03d}')
    os.makedirs(results_dir, exist_ok=True)

    all_views = [source]
    import torch
    for angle_idx in range(cfg.NUM_POSE_CLASSES):
        tgt = torch.zeros(4, cfg.NUM_POSE_CLASSES).to(device)
        tgt[:, angle_idx] = 1.0
        with autocast(enabled=amp_enabled):
            gen = G(source, tgt)
        all_views.append(gen)

    # Save: [original, -90deg, -45deg, 0deg, +45deg, +90deg]
    grid = torch.cat(all_views, dim=0)
    save_image(grid, os.path.join(results_dir, 'multiangle.png'), nrow=4)
    G.train()


if __name__ == '__main__':
    train_phase2()
