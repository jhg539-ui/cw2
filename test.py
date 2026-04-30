import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import PetClassifier
from tqdm import tqdm

import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)

# Reproducibility
import random
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
random.seed(42)
import os


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Test-Time Augmentation: predict on multiple augmented versions and average
    tta_transforms = [
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), base_normalize,
        ]),
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), base_normalize,
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(), base_normalize,
        ]),
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(), base_normalize,
        ]),
        transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.CenterCrop(224),
            transforms.ToTensor(), base_normalize,
        ]),
    ]

    model = PetClassifier(num_classes=37).to(device)
    model_path = "model_best.pth" if os.path.exists("model_best.pth") else "model.pth"
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def evaluate_with_tta(split):
        all_preds = []
        all_labels = None
        for i, tfm in enumerate(tta_transforms):
            ds = datasets.OxfordIIITPet(root="./data", split=split, transform=tfm, download=True)
            loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
            preds = []
            labels_list = []
            with torch.no_grad():
                for images, labels in tqdm(loader, desc=f"{split} TTA {i+1}/{len(tta_transforms)}"):
                    images = images.to(device, non_blocking=True)
                    probs  = F.softmax(model(images), dim=1)
                    preds.append(probs.cpu())
                    labels_list.append(labels)
            all_preds.append(torch.cat(preds, dim=0))
            if all_labels is None:
                all_labels = torch.cat(labels_list, dim=0)
        avg_probs = torch.stack(all_preds, dim=0).mean(dim=0)
        final_preds = avg_probs.argmax(dim=1)
        return 100.0 * (final_preds == all_labels).float().mean().item()

    print("\nEvaluating trainval set with TTA...")
    train_acc = evaluate_with_tta("trainval")

    print("\nEvaluating test set with TTA...")
    test_acc = evaluate_with_tta("test")

    print("\n" + "="*50)
    print(f"Train Accuracy : {train_acc:.2f}%")
    print(f"Test  Accuracy : {test_acc:.2f}%")
    print("="*50)
