import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import PetClassifier
from tqdm import tqdm
import os


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")
    trainval_dataset = datasets.OxfordIIITPet(root="./data", split="trainval", transform=eval_transforms, download=True)
    test_dataset     = datasets.OxfordIIITPet(root="./data", split="test",     transform=eval_transforms, download=True)

    trainval_loader = DataLoader(trainval_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    test_loader     = DataLoader(test_dataset,     batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model      = PetClassifier(num_classes=37).to(device)
    model_path = "model_best.pth" if os.path.exists("model_best.pth") else "model.pth"
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    train_correct = 0
    train_total   = 0
    with torch.no_grad():
        for images, labels in tqdm(trainval_loader, desc="Evaluating trainval"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            _, predicted = model(images).max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
    train_acc = 100.0 * train_correct / train_total

    test_correct = 0
    test_total   = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating test"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            _, predicted = model(images).max(1)
            test_total   += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_acc = 100.0 * test_correct / test_total

    print("\n" + "="*50)
    print(f"Train Accuracy : {train_acc:.2f}%")
    print(f"Test  Accuracy : {test_acc:.2f}%")
    print("="*50)
