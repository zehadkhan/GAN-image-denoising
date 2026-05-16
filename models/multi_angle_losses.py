import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class IdentityLoss(nn.Module):
    """VGG16 identity preservation loss — same person, different angle should share identity features."""

    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.features = nn.Sequential(*list(vgg.children())[:16])  # up to relu3_3
        for p in self.parameters():
            p.requires_grad = False
        self.criterion = nn.L1Loss()

    def forward(self, generated, source):
        return self.criterion(self.features(generated), self.features(source))


class MultiAngleLoss(nn.Module):
    """
    Combined Phase 2 loss:
      - Adversarial (real/fake)
      - Reconstruction L1 (when source_pose == target_pose)
      - Identity (VGG features, person identity preserved)
      - Pose classification (generated image has correct target pose)
    """

    def __init__(self, lambda_recon=10.0, lambda_identity=5.0,
                 lambda_adv=1.0, lambda_pose=2.0):
        super().__init__()
        self.identity = IdentityLoss()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.pose_criterion = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()

        self.lambda_recon = lambda_recon
        self.lambda_identity = lambda_identity
        self.lambda_adv = lambda_adv
        self.lambda_pose = lambda_pose

    def generator_loss(self, generated, source, fake_patch, fake_pose_logits,
                       target_pose_class, same_pose_mask):
        # Adversarial: fool discriminator
        adv = self.adv_criterion(fake_patch, torch.ones_like(fake_patch))

        # Identity: preserve person features
        identity = self.identity(generated, source)

        # Reconstruction: when source and target angle are same, output = input
        if same_pose_mask.any():
            recon = self.l1(generated[same_pose_mask], source[same_pose_mask])
        else:
            recon = torch.tensor(0.0, device=generated.device)

        # Pose classification: generated image should be at target angle
        pose = self.pose_criterion(fake_pose_logits, target_pose_class)

        total = (self.lambda_adv * adv +
                 self.lambda_identity * identity +
                 self.lambda_recon * recon +
                 self.lambda_pose * pose)

        return total, {
            'adv': adv.item(),
            'identity': identity.item(),
            'recon': recon.item() if isinstance(recon, torch.Tensor) else 0.0,
            'pose': pose.item(),
        }

    def discriminator_loss(self, real_patch, fake_patch, real_pose_logits,
                           fake_pose_logits, real_pose_class):
        # Real/fake
        d_real = self.adv_criterion(real_patch, torch.ones_like(real_patch))
        d_fake = self.adv_criterion(fake_patch, torch.zeros_like(fake_patch))

        # Pose classification on real images only
        pose_cls = self.pose_criterion(real_pose_logits, real_pose_class)

        return 0.5 * (d_real + d_fake) + self.lambda_pose * pose_cls
