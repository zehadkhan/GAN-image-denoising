import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PerceptualLoss(nn.Module):
    """VGG16-based perceptual loss using relu1_2, relu2_2, relu3_3 feature layers."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4])    # relu1_2
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9])   # relu2_2
        self.slice3 = nn.Sequential(*list(vgg.children())[9:16])  # relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, generated, target):
        loss = 0.0
        g, t = generated, target
        g1 = self.slice1(g); t1 = self.slice1(t); loss += self.criterion(g1, t1)
        g2 = self.slice2(g1); t2 = self.slice2(t1); loss += self.criterion(g2, t2)
        g3 = self.slice3(g2); t3 = self.slice3(t2); loss += self.criterion(g3, t3)
        return loss / 3.0


class SSIMLoss(nn.Module):
    """Structural dissimilarity loss: 1 - SSIM."""

    def __init__(self, window_size=11, sigma=1.5):
        super().__init__()
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g.unsqueeze(1) * g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        self.register_buffer('window', window)
        self.window_size = window_size

    def _ssim(self, x, y):
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window_size // 2
        win = self.window.expand(x.size(1), 1, -1, -1)
        mu_x = F.conv2d(x, win, padding=pad, groups=x.size(1))
        mu_y = F.conv2d(y, win, padding=pad, groups=x.size(1))
        mu_x2, mu_y2, mu_xy = mu_x**2, mu_y**2, mu_x*mu_y
        sx = F.conv2d(x*x, win, padding=pad, groups=x.size(1)) - mu_x2
        sy = F.conv2d(y*y, win, padding=pad, groups=x.size(1)) - mu_y2
        sxy = F.conv2d(x*y, win, padding=pad, groups=x.size(1)) - mu_xy
        num = (2*mu_xy + C1) * (2*sxy + C2)
        den = (mu_x2 + mu_y2 + C1) * (sx + sy + C2)
        return (num / den).mean()

    def forward(self, generated, target):
        return 1.0 - self._ssim(generated, target)


class GANLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, pred, is_real):
        target = torch.ones_like(pred) if is_real else torch.zeros_like(pred)
        return self.criterion(pred, target)


class GeneratorLoss(nn.Module):
    """Combined generator loss: λ1·L1 + λ2·Perceptual + λ3·SSIM + λ4·Adversarial."""

    def __init__(self, lambda_l1=100.0, lambda_perceptual=10.0, lambda_ssim=10.0, lambda_adv=1.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.perceptual = PerceptualLoss()
        self.ssim = SSIMLoss()
        self.adv = GANLoss()
        self.lambda_l1 = lambda_l1
        self.lambda_perceptual = lambda_perceptual
        self.lambda_ssim = lambda_ssim
        self.lambda_adv = lambda_adv

    def forward(self, generated, target, disc_fake=None, warmup=False):
        l1 = self.l1(generated, target)
        if warmup:
            return self.lambda_l1 * l1, {'l1': l1.item(), 'perceptual': 0.0, 'ssim': 0.0, 'adv': 0.0}
        perc = self.perceptual(generated, target)
        ssim = self.ssim(generated, target)
        adv = self.adv(disc_fake, is_real=True)
        total = self.lambda_l1*l1 + self.lambda_perceptual*perc + self.lambda_ssim*ssim + self.lambda_adv*adv
        return total, {'l1': l1.item(), 'perceptual': perc.item(), 'ssim': ssim.item(), 'adv': adv.item()}
