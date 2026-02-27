"""
Train a ResNet18 binary classifier: waymo vs not_waymo.

Expects an ImageFolder structure:
    training/dataset/
    ├── waymo/       (images of Waymo vehicles)
    └── not_waymo/   (images of other vehicles)

Usage:
    python training/train_classifier.py
    python training/train_classifier.py --data training/dataset --output models/waymo_classifier.pth --epochs 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms


def build_transforms(is_train: bool) -> transforms.Compose:
    """Build data augmentation pipeline. Heavy augmentation for training robustness."""
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1
                ),
                transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0)),
                transforms.RandomGrayscale(p=0.1),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                # Random erasing simulates occlusion
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def build_model(num_classes: int = 2) -> nn.Module:
    """Build ResNet18 with custom classifier head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the best available device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device_str


def train(args):
    device = torch.device(_resolve_device(args.device))
    print(f"Using device: {device}")

    # Load dataset with train/val split
    full_dataset = datasets.ImageFolder(
        args.data, transform=build_transforms(is_train=True)
    )

    # Print class distribution
    class_names = full_dataset.classes
    class_counts = [0] * len(class_names)
    for _, label in full_dataset.samples:
        class_counts[label] += 1

    print(f"Classes: {class_names}")
    for name, count in zip(class_names, class_counts):
        print(f"  {name}: {count} images")

    if any(c == 0 for c in class_counts):
        print("\nERROR: One or more classes have zero images.")
        print("Ensure both training/dataset/waymo/ and training/dataset/not_waymo/ have images.")
        return

    # 80/20 train/val split
    total = len(full_dataset)
    val_size = max(1, int(total * 0.2))
    train_size = total - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Apply val transforms to val set
    val_dataset.dataset = datasets.ImageFolder(
        args.data, transform=build_transforms(is_train=False)
    )

    # Weighted sampling to handle class imbalance
    train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
    train_class_counts = [0] * len(class_names)
    for label in train_labels:
        train_class_counts[label] += 1

    class_weights = [
        1.0 / c if c > 0 else 0.0 for c in train_class_counts
    ]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    # Build model
    model = build_model(num_classes=len(class_names))
    model = model.to(device)

    # Loss with class weights for additional imbalance handling
    weight_tensor = torch.tensor(
        [1.0 / c if c > 0 else 0.0 for c in class_counts], dtype=torch.float32
    ).to(device)
    weight_tensor = weight_tensor / weight_tensor.sum()
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_epoch = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"Train: {train_size}, Val: {val_size}")
    print()

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()

        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
        avg_train_loss = train_loss / train_total if train_total > 0 else 0

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        val_tp = 0  # true positives (waymo correctly identified)
        val_fp = 0  # false positives (not_waymo classified as waymo)
        val_fn = 0  # false negatives (waymo classified as not_waymo)

        waymo_idx = class_names.index("waymo") if "waymo" in class_names else 1

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

                # Precision/recall tracking
                for pred, true in zip(predicted, labels):
                    if pred == waymo_idx and true == waymo_idx:
                        val_tp += 1
                    elif pred == waymo_idx and true != waymo_idx:
                        val_fp += 1
                    elif pred != waymo_idx and true == waymo_idx:
                        val_fn += 1

        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
        precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0
        recall = val_tp / (val_tp + val_fn) if (val_tp + val_fn) > 0 else 0

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.1f}% | "
            f"Val Acc: {val_acc:.1f}%, Precision: {precision:.2f}, Recall: {recall:.2f}"
        )

        # Save best model by validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(output_path))
            print(f"  -> Saved best model (val_acc={val_acc:.1f}%)")

    print(f"\nTraining complete!")
    print(f"Best model: epoch {best_epoch}, val_acc={best_val_acc:.1f}%")
    print(f"Saved to: {args.output}")
    print(f"\nClass mapping: {dict(enumerate(class_names))}")
    print(f"(Index 1 should be 'waymo' for the classifier to work correctly)")


def main():
    parser = argparse.ArgumentParser(description="Train Waymo/not_waymo classifier")
    parser.add_argument("--data", default="training/dataset", help="Dataset root with waymo/ and not_waymo/ subdirs")
    parser.add_argument("--output", default="models/waymo_classifier.pth", help="Output model path")
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="auto", help="Compute device (auto/mps/cuda/cpu)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
