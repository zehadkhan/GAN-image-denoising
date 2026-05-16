import torch
import torch.nn as nn


class PoseDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class PoseUpBlock(nn.Module):
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


class MultiAngleGenerator(nn.Module):
    """
    Pose-conditioned U-Net generator.
    Input:  source face (3x128x128) + target pose one-hot (5,)
    Output: face at target pose (3x128x128)

    Pose code is injected at the bottleneck by tiling and concatenating.
    """

    def __init__(self, in_channels=3, out_channels=3, num_pose_classes=5):
        super().__init__()
        self.num_pose_classes = num_pose_classes

        # Encoder: 128->64->32->16->8->4->2
        self.enc1 = PoseDownBlock(in_channels, 64, use_bn=False)
        self.enc2 = PoseDownBlock(64, 128)
        self.enc3 = PoseDownBlock(128, 256)
        self.enc4 = PoseDownBlock(256, 512)
        self.enc5 = PoseDownBlock(512, 512)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Pose projection: map pose one-hot -> 512 channel feature map
        self.pose_proj = nn.Sequential(
            nn.Linear(num_pose_classes, 512),
            nn.ReLU(inplace=True),
        )

        # Decoder input is bottleneck(512) + pose(512) = 1024
        self.dec1 = PoseUpBlock(1024, 512, use_dropout=True)   # 2->4,  +e5=1024
        self.dec2 = PoseUpBlock(1024, 512, use_dropout=True)   # 4->8,  +e4=1024
        self.dec3 = PoseUpBlock(1024, 256, use_dropout=True)   # 8->16, +e3=512
        self.dec4 = PoseUpBlock(512, 128)                       # 16->32,+e2=256
        self.dec5 = PoseUpBlock(256, 64)                        # 32->64,+e1=128
        self.out_conv = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, target_pose):
        # Encode
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        b = self.bottleneck(e5)           # B x 512 x 2 x 2

        # Inject pose at bottleneck
        p = self.pose_proj(target_pose)   # B x 512
        p = p.unsqueeze(-1).unsqueeze(-1).expand_as(b)  # B x 512 x 2 x 2
        b = torch.cat([b, p], dim=1)      # B x 1024 x 2 x 2

        # Decode
        d = self.dec1(b, e5)
        d = self.dec2(d, e4)
        d = self.dec3(d, e3)
        d = self.dec4(d, e2)
        d = self.dec5(d, e1)
        return self.out_conv(d)
