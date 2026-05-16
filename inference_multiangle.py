import os
import argparse
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

import config_phase2 as cfg
from models.multi_angle_generator import MultiAngleGenerator


ANGLE_LABELS = ['-90deg', '-45deg', '0deg_front', '+45deg', '+90deg']
ANGLE_DISPLAY = ['-90°', '-45°', '0° (front)', '+45°', '+90°']


def generate_all_angles(image_path, checkpoint_path=None, output_dir='multiangle_output'):
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else 'cpu')
    ckpt_path = checkpoint_path or os.path.join(cfg.CHECKPOINT_DIR_PHASE2, 'best_model_phase2.pth')

    G = MultiAngleGenerator(num_pose_classes=cfg.NUM_POSE_CLASSES).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(ckpt['generator'] if 'generator' in ckpt else ckpt)
    G.eval()

    tf = transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    img = tf(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)

    os.makedirs(output_dir, exist_ok=True)
    all_views = [img]

    with torch.no_grad():
        for angle_idx in range(cfg.NUM_POSE_CLASSES):
            pose = torch.zeros(1, cfg.NUM_POSE_CLASSES).to(device)
            pose[0, angle_idx] = 1.0
            gen = G(img, pose)
            all_views.append(gen)
            save_image(gen, os.path.join(output_dir, f'angle_{ANGLE_LABELS[angle_idx]}.png'))
            print(f"Generated: {ANGLE_DISPLAY[angle_idx]}")

    # Save all views in one grid: original + 5 angles (concatenate along width)
    grid = torch.cat(all_views, dim=3)
    save_image(grid, os.path.join(output_dir, 'all_angles.png'))
    print(f"\nAll angles saved to {output_dir}/all_angles.png")
    print(f"Layout: [Original | -90° | -45° | 0° | +45° | +90°]")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to clean face image')
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--output', default='multiangle_output')
    args = parser.parse_args()
    generate_all_angles(args.image, args.checkpoint, args.output)
