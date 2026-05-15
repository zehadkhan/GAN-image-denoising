import math
import torch
import torch.nn.functional as F


def psnr(generated, target, max_val=1.0):
    mse = F.mse_loss(generated.detach(), target.detach()).item()
    return float('inf') if mse == 0 else 20 * math.log10(max_val) - 10 * math.log10(mse)


def ssim(x, y, window_size=11, sigma=1.5):
    x, y = x.detach(), y.detach()
    C1, C2 = 0.01**2, 0.03**2
    coords = torch.arange(window_size, dtype=x.dtype, device=x.device) - window_size // 2
    g = torch.exp(-(coords**2) / (2*sigma**2)); g = g / g.sum()
    win = (g.unsqueeze(1)*g.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(x.size(1), 1, -1, -1)
    pad = window_size // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=x.size(1))
    mu_y = F.conv2d(y, win, padding=pad, groups=x.size(1))
    mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
    sx = F.conv2d(x*x, win, padding=pad, groups=x.size(1)) - mu_x2
    sy = F.conv2d(y*y, win, padding=pad, groups=x.size(1)) - mu_y2
    sxy = F.conv2d(x*y, win, padding=pad, groups=x.size(1)) - mu_xy
    return ((2*mu_xy+C1)*(2*sxy+C2) / ((mu_x2+mu_y2+C1)*(sx+sy+C2))).mean().item()
