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
- Test:  50.67% (with TTA)

## Notes
- Trained from scratch on Oxford-IIIT Pet trainval (3680 images).
- Test set is the official Oxford-IIIT Pet test split (3669 images), used only for final evaluation.
- Random seed set to 42 for reproducibility.
- First run downloads dataset (~800 MB) automatically via torchvision.
- Trained on NVIDIA GTX 1080 Ti with PyTorch 2.4.0+cu121.

## Architecture summary
ResNet-style with 4 stages, each containing 2 ResBlocks (channels: 64, 128, 256, 512).
Global average pooling + dropout + linear classifier. Trained with SGD + Nesterov momentum,
cosine LR schedule with linear warmup, MixUp augmentation, and label smoothing.

## Test-Time Augmentation
Test accuracy is reported using TTA: predictions are averaged across 5 augmented versions
of each image (original, horizontal flip, center crop variations) for more robust evaluation.
