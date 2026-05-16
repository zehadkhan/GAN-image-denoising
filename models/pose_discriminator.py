import torch
import torch.nn as nn


class PoseDiscriminator(nn.Module):
    """
    Multi-task discriminator:
      - PatchGAN real/fake classification
      - Pose angle classification (5 classes)

    Input: source_face (3ch) + generated/real_face (3ch) = 6ch
    """

    def __init__(self, in_channels=6, num_pose_classes=5):
        super().__init__()

        def block(in_ch, out_ch, stride, use_bn):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=stride, padding=1, bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.shared = nn.Sequential(
            *block(in_channels, 64, stride=2, use_bn=False),   # 128->64
            *block(64, 128, stride=2, use_bn=True),             # 64->32
            *block(128, 256, stride=2, use_bn=True),            # 32->16
            *block(256, 512, stride=1, use_bn=True),            # 16->15
        )

        # Real/fake patch output
        self.patch_head = nn.Conv2d(512, 1, 4, stride=1, padding=1)   # 15->14

        # Pose classification head (global average pool -> FC)
        self.pose_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_pose_classes),
        )

    def forward(self, source, target):
        x = self.shared(torch.cat([source, target], dim=1))
        real_fake = self.patch_head(x)
        pose_logits = self.pose_head(x)
        return real_fake, pose_logits
