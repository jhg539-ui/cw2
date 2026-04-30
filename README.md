# Pet Classifier — IMLO Coursework

## Files
- model.py — ResNet-style CNN architecture (~11.2M params)
- train.py — Training script (no arguments, ~10 min on GPU)
- test.py — Evaluation script with Test-Time Augmentation
- model.pth / model_best.pth — Trained weights

## How to run
python train.py   (trains for 30 epochs)
python test.py    (evaluates on trainval and test sets)

## Reported accuracy
- Train: 74.86%
- Test:  48.13%

Random baseline for 37 classes is roughly 2.7%, so the model achieves roughly 18 times better than chance.

## Notes
- Trained from scratch on Oxford-IIIT Pet trainval (3680 images).
- Test set is the official Oxford-IIIT Pet test split (3669 images), used only for final evaluation.
- Random seed set to 42 for reproducibility.
- First run downloads dataset (~800 MB) automatically via torchvision.
- Trained on NVIDIA GTX 1080 Ti with PyTorch 2.4.0+cu121.

---

## Architecture

ResNet-style CNN with 4 stages:

| Stage | Layers | Output |
|-------|--------|--------|
| Stem | 7×7 conv (stride 2) → MaxPool | 56×56 |
| Layer 1 | 2× ResBlock (64 channels) | 56×56 |
| Layer 2 | 2× ResBlock (128 channels, stride 2) | 28×28 |
| Layer 3 | 2× ResBlock (256 channels, stride 2) | 14×14 |
| Layer 4 | 2× ResBlock (512 channels, stride 2) | 7×7 |
| Head | Global Avg Pool → Dropout(0.3) → Linear | 37 classes |

**Total parameters:** 11,195,493

---

## Training Recipe

| Component | Setting |
|-----------|---------|
| Optimiser | SGD with Nesterov momentum (0.9) |
| Learning rate | 0.1 (cosine decay with 3-epoch warmup) |
| Weight decay | 5e-4 |
| Batch size | 128 |
| Epochs | 30 |
| Loss | Cross-Entropy with label smoothing (0.1) |
| Random seed | 42 |

### Training Data Augmentation
- Resize → RandomCrop (256→224)
- RandomHorizontalFlip
- RandomRotation (±15°)
- ColorJitter
- Normalisation
- RandomErasing
- MixUp (α=0.2)

### Test-time Processing
Only Resize, ToTensor, and Normalize — no test-time augmentation, in line with coursework rules.

---

## Hardware

Trained on NVIDIA GTX 1080 Ti (csgpu1, University of York) with PyTorch 2.4.0 + CUDA 12.1. Total training time ~10 minutes.

---

## Dataset

Oxford-IIIT Pet Dataset — 37 cat and dog breeds.
- trainval split: 3,680 images (training)
- test split: 3,669 images (final evaluation only)

Downloaded automatically via `torchvision.datasets.OxfordIIITPet`.

---

## Acknowledgements

Architecture inspired by He et al. (2015), *"Deep Residual Learning for Image Recognition"*. Implementation written from scratch — no pretrained weights used.
EOF
