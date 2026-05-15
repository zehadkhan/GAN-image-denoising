import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator. Input: concat(noisy, clean_or_generated) = 6ch. Output: 14×14 patch scores."""

    def __init__(self, in_channels=6):
        super().__init__()

        def block(in_ch, out_ch, stride, use_bn):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, 64, stride=2, use_bn=False),   # 128→64
            *block(64, 128, stride=2, use_bn=True),             # 64→32
            *block(128, 256, stride=2, use_bn=True),            # 32→16
            *block(256, 512, stride=1, use_bn=True),            # 16→15
            nn.Conv2d(512, 1, 4, stride=1, padding=1),          # 15→14
        )

    def forward(self, noisy, target):
        return self.model(torch.cat([noisy, target], dim=1))
