# GAN-based Image Denoising & Multi-Angle View Generation

A deep learning project implementing two-phase facial image restoration using Generative Adversarial Networks (GANs).

**Bangladesh University of Business and Technology (BUBT)**
Department of Computer Science & Engineering

**Supervisor:** Sondip Poul Singh — **Course Teacher:** Md. Shahiduzzaman

---

## Overview

```
Noisy Face Image
      ↓
Phase 1: GAN Denoising  (U-Net + PatchGAN)
      ↓
Clean Face Image
      ↓
Phase 2: Multi-Angle Generation  (Pose-conditioned cGAN)
      ↓
Five Viewpoints: -90° | -45° | 0° | +45° | +90°
```

### Phase 1 — Image Denoising (Complete)
Restores noisy/corrupted facial images using a U-Net generator paired with a PatchGAN discriminator. Trained on the CelebA dataset with Gaussian noise (σ = 0.10).

### Phase 2 — Multi-Angle View Generation (Complete)
Takes a single clean face image and synthesizes it at five different horizontal angles. Uses a pose-conditioned U-Net with Phase 1 encoder weights as initialization.

---

## Project Structure

```
gan/
├── config.py                        # Phase 1 hyperparameters
├── config_phase2.py                 # Phase 2 hyperparameters
├── train.py                         # Phase 1 training
├── train_phase2.py                  # Phase 2 training
├── evaluate.py                      # Phase 1 evaluation
├── inference.py                     # Denoise a single image
├── inference_multiangle.py          # Generate multiple angles
├── kaggle_notebook.ipynb            # Full Kaggle training notebook
├── requirements.txt
│
├── data/
│   ├── dataset.py                   # CelebA loader
│   └── pose_dataset.py              # 300W-LP / pose dataset loader
│
├── models/
│   ├── generator.py                 # U-Net Generator
│   ├── discriminator.py             # PatchGAN Discriminator
│   ├── losses.py                    # L1 + Perceptual + SSIM + Adversarial
│   ├── multi_angle_generator.py     # Pose-conditioned U-Net
│   ├── pose_discriminator.py        # PatchGAN + Pose Classifier
│   └── multi_angle_losses.py        # Identity + Pose + Adversarial
│
└── utils/
    └── metrics.py                   # PSNR + SSIM
```

---

## Architecture

### Phase 1 — U-Net + PatchGAN

| Component | Detail |
|---|---|
| Generator | U-Net with 6-level encoder-decoder + skip connections |
| Discriminator | PatchGAN with spectral normalization (14×14 patch output) |
| Loss | λ100·L1 + λ10·Perceptual (VGG16) + λ10·SSIM + λ1·Adversarial |
| Optimizer | Adam (LR=1e-4, β1=0.5) + CosineAnnealingLR |
| Training | 5-epoch L1 warmup → full multi-loss training |

### Phase 2 — Pose-conditioned cGAN

| Component | Detail |
|---|---|
| Generator | U-Net + pose one-hot embedding injected at bottleneck |
| Discriminator | PatchGAN + global pose classification head (5 classes) |
| Loss | Adversarial + VGG Identity + L1 Reconstruction + Pose CrossEntropy |
| Poses | -90°, -45°, 0°, +45°, +90° |
| Init | Phase 1 encoder weights transferred via weight matching |

---

## Datasets

### Phase 1 — CelebA

| Property | Value |
|---|---|
| Source | [CelebA on Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) |
| Images used | 2,400 (2,000 train / 400 val) |
| Resolution | 128 × 128 |
| Noise | Gaussian σ = 0.10 (added on-the-fly) |

Place images in: `data/celeba/`

---

### Phase 2 — Multi-Angle Pose Datasets

Choose any one of the following datasets. All are supported by `data/pose_dataset.py`.

#### Option 1 — 300W-LP ⭐ Recommended
> Synthesized large-pose face dataset with .mat pose annotations

| | |
|---|---|
| Kaggle | [maulidio16/300w-lp](https://www.kaggle.com/datasets/maulidio16/300w-lp) |
| Images | ~61,000 across 4 subsets (AFW, HELEN, IBUG, LFPW) |
| Pose range | -90° to +90° yaw |
| Annotations | `.mat` files with `Pose_Para` field (pitch, yaw, roll) |
| Free | Yes |

Place in: `data/300w_lp/` — structure:
```
data/300w_lp/
  AFW/   img.jpg + img.mat
  HELEN/ img.jpg + img.mat
  IBUG/  img.jpg + img.mat
  LFPW/  img.jpg + img.mat
```

---

#### Option 2 — Multi-PIE
> Real multi-view dataset with 15 camera angles and 19 lighting conditions

| | |
|---|---|
| Source | [CMU Multi-PIE](https://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html) |
| People | 337 subjects |
| Images | 750,000+ |
| Angles | 15 viewpoints (-90° to +90°) |
| Free | No (requires academic license) |

Best quality results — recommended for academic publication.

---

#### Option 3 — CFP (Celebrities in Frontal-Profile)
> Front vs. profile face verification dataset

| | |
|---|---|
| Kaggle | Search: `cfp-dataset` |
| Images | 7,000 (500 people × 14 images) |
| Angles | Frontal + Profile (side view) |
| Free | Yes |

Good for binary angle generation (front ↔ side).

---

#### Option 4 — CASIA-WebFace with Pose Labels
> Large-scale face recognition dataset with pose estimation

| | |
|---|---|
| Source | [CASIA-WebFace](https://github.com/happynear/AMSoftmax) |
| Images | 494,414 images |
| People | 10,575 subjects |
| Free | Yes (research use) |

Requires running a pose estimator (e.g. HopeNet) to extract yaw angles.

---

#### Option 5 — FaceScape
> High-quality 3D face dataset with multi-angle renders

| | |
|---|---|
| Source | [facescape.nju.edu.cn](https://facescape.nju.edu.cn) |
| Images | 400 subjects × multiple angles |
| Quality | Very high (3D rendered) |
| Free | Yes (research license) |

Best geometric accuracy for multi-angle generation.

---

#### Dataset Comparison

| Dataset | Size | Quality | Free | Angles | Best for |
|---|---|---|---|---|---|
| **300W-LP** | Medium | Good | ✅ | Full range | General use |
| **Multi-PIE** | Large | Excellent | ❌ | 15 views | Research paper |
| **CFP** | Small | Good | ✅ | Front+Profile | Quick experiments |
| **CASIA-WebFace** | Very Large | Good | ✅ | Estimated | Scale |
| **FaceScape** | Medium | Excellent | ✅ | Full range | Geometry |

---

## Installation

```bash
git clone https://github.com/zehadkhan/GAN-image-denoising.git
cd GAN-image-denoising
pip install -r requirements.txt
```

---

## Training

### Option A — Kaggle (Recommended)

1. Go to [kaggle.com](https://kaggle.com) → Create → New Notebook
2. Import from GitHub: `zehadkhan/GAN-image-denoising` → select `kaggle_notebook.ipynb`
3. Add datasets: `jessicali9530/celeba-dataset` + `maulidio16/300w-lp`
4. Settings → Accelerator → **GPU T4 x2**
5. Run All → ~4-6 hours
6. Download `models.zip` from Output sidebar

---

### Option B — Local

**Phase 1 — Denoising:**
```bash
# Place CelebA images in data/celeba/
python train.py
```

**Phase 2 — Multi-Angle:**
```bash
# Place 300W-LP in data/300w_lp/
python train_phase2.py
```

---

## Inference

**Denoise a single image:**
```bash
python inference.py photo.jpg --add-noise --output denoised.png
```

**Generate all 5 angles from one face:**
```bash
python inference_multiangle.py face.jpg --output multiangle_output/
```

Output layout:
```
Original | -90° | -45° | 0° | +45° | +90°
```

**Evaluate on validation set:**
```bash
python evaluate.py
# Output: PSNR: XX.XX dB | SSIM: 0.XXXX
```

---

## Evaluation Metrics

| Metric | Description | Target |
|---|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio — pixel fidelity | Higher is better (dB) |
| **SSIM** | Structural Similarity Index — structural quality | Closer to 1.0 |

---

## Training Configuration

Key settings in `config.py`:

```python
IMAGE_SIZE    = 128       # Input/output resolution
NOISE_STD     = 0.10      # Gaussian noise level
BATCH_SIZE    = 32
NUM_EPOCHS    = 100
WARMUP_EPOCHS = 5         # L1-only warmup before GAN loss
LEARNING_RATE = 1e-4
LAMBDA_L1     = 100.0
LAMBDA_PERCEPTUAL = 10.0
LAMBDA_SSIM   = 10.0
LAMBDA_ADV    = 1.0
```

---

## References

1. Goodfellow et al. — Generative Adversarial Nets. *NeurIPS 2014*
2. Ronneberger et al. — U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*
3. Isola et al. — Image-to-Image Translation with Conditional Adversarial Networks (Pix2Pix). *CVPR 2017*
4. Zhang et al. — Beyond a Gaussian Denoiser: Residual Learning of Deep CNN. *IEEE TIP 2017*
5. Johnson et al. — Perceptual Losses for Real-Time Style Transfer. *ECCV 2016*
6. Wang et al. — Image Quality Assessment: SSIM. *IEEE TIP 2004*
7. Liu et al. — Deep Learning Face Attributes in the Wild (CelebA). *ICCV 2015*
