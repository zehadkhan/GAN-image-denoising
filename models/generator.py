import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        return torch.cat([self.block(x), skip], dim=1)


class UNetGenerator(nn.Module):
    """U-Net generator for 128×128 image denoising with skip connections."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder: 128→64→32→16→8→4→2
        self.enc1 = DownBlock(in_channels, 64, use_bn=False)  # 128→64
        self.enc2 = DownBlock(64, 128)                         # 64→32
        self.enc3 = DownBlock(128, 256)                        # 32→16
        self.enc4 = DownBlock(256, 512)                        # 16→8
        self.enc5 = DownBlock(512, 512)                        # 8→4
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )                                                       # 4→2

        # Decoder (in_ch = prev_out + skip_ch from concat)
        self.dec1 = UpBlock(512, 512, use_dropout=True)    # 2→4,  out: 512+512=1024
        self.dec2 = UpBlock(1024, 512, use_dropout=True)   # 4→8,  out: 512+512=1024
        self.dec3 = UpBlock(1024, 256, use_dropout=True)   # 8→16, out: 256+256=512
        self.dec4 = UpBlock(512, 128)                       # 16→32, out: 128+128=256
        self.dec5 = UpBlock(256, 64)                        # 32→64, out: 64+64=128
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )                                                    # 64→128

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)
        d = self.dec1(b, e5)
        d = self.dec2(d, e4)
        d = self.dec3(d, e3)
        d = self.dec4(d, e2)
        d = self.dec5(d, e1)
        return self.out_conv(d)
