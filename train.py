import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import PetClassifier
from tqdm import tqdm


def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[index], y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def lr_lambda(epoch, warmup_epochs, num_epochs):
    if epoch < warmup_epochs:
        return (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    NUM_EPOCHS    = 30
    BATCH_SIZE    = 128
    NUM_CLASSES   = 37
    LR            = 0.1
    WARMUP_EPOCHS = 3
    MIXUP_ALPHA   = 0.2
    NUM_WORKERS   = 4

    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")
    trainval_aug  = datasets.OxfordIIITPet(root="./data", split="trainval", transform=train_transforms, download=True)
    trainval_eval = datasets.OxfordIIITPet(root="./data", split="trainval", transform=eval_transforms, download=False)
    test_dataset  = datasets.OxfordIIITPet(root="./data", split="test",     transform=eval_transforms, download=True)

    print(f"Trainval: {len(trainval_aug)} | Test: {len(test_dataset)}")

    train_loader      = DataLoader(trainval_aug,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
    train_eval_loader = DataLoader(trainval_eval, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader       = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model     = PetClassifier(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lambda e: lr_lambda(e, WARMUP_EPOCHS, NUM_EPOCHS)
    )

    print("Starting training...")
    best_test_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total   = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            images, labels_a, labels_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total   += labels_a.size(0)
            correct += (lam * predicted.eq(labels_a).sum().item()
                        + (1 - lam) * predicted.eq(labels_b).sum().item())

        train_acc  = 100.0 * correct / total
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        test_correct = 0
        test_total   = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                _, predicted = model(images).max(1)
                test_total   += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        test_acc = 100.0 * test_correct / test_total

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), "model_best.pth")

        print(f"Epoch {epoch+1:02d}/{NUM_EPOCHS}: Loss={running_loss/len(train_loader):.4f}, "
              f"Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%, LR={current_lr:.6f}")

    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load("model_best.pth", map_location=device))
    model.eval()

    train_correct = 0
    train_total   = 0
    with torch.no_grad():
        for images, labels in tqdm(train_eval_loader, desc="Final train eval"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            _, predicted = model(images).max(1)
            train_total   += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
    final_train_acc = 100.0 * train_correct / train_total

    test_correct = 0
    test_total   = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Final test eval"):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            _, predicted = model(images).max(1)
            test_total   += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    final_test_acc = 100.0 * test_correct / test_total

    torch.save(model.state_dict(), "model.pth")
    print("\n" + "="*50)
    print(f"Final Train Accuracy : {final_train_acc:.2f}%")
    print(f"Final Test  Accuracy : {final_test_acc:.2f}%")
    print(f"Best  Test  Accuracy : {best_test_acc:.2f}%")
    print("="*50)
