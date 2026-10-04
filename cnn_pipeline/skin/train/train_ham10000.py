"""
HAM10000 Model Training & Evaluation Pipeline

Trains a ResNet18 classifier on the HAM10000 skin lesion dataset using:
- Lesion-level train/validation splitting (preventing data leakage across duplicate lesion IDs)
- Class-weighted Cross-Entropy loss for handling severe class imbalance
- Dermatoscopic data augmentations
- Checkpoint saving to weights/model_ham10000_best.pth
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .ham10000_inference import CLASS_INDEX_TO_NAME, CLINICAL_LABEL_MAP, IMG_SIZE

# Inverse mapping: name to class index
NAME_TO_CLASS_INDEX = {v: k for k, v in CLASS_INDEX_TO_NAME.items()}


class HAM10000Dataset(Dataset):
    """
    PyTorch Dataset for HAM10000 skin lesion dermoscopy images.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['image_id']}.jpg"
        img_path = os.path.join(self.img_dir, img_name)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert('RGB')
        label = NAME_TO_CLASS_INDEX[row['dx']]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """
    Returns train and validation transforms.
    """
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def prepare_data(data_dir, test_size=0.2, random_state=42):
    """
    Loads metadata and performs a lesion_id-aware train/test split.
    """
    meta_path = os.path.join(data_dir, "HAM10000_metadata.csv")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata not found at {meta_path}. Run download_ham10000.py first.")

    df = pd.read_csv(meta_path)
    img_dir = os.path.join(data_dir, "images")

    # Filter to only images that exist locally
    existing_images = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()
    df = df[df['image_id'].apply(lambda x: f"{x}.jpg" in existing_images)].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(f"No valid images found in {img_dir}. Please run download_ham10000.py first.")

    print(f"[*] Total available images for training/validation: {len(df)}")
    print(f"[*] Class distribution:\n{df['dx'].value_counts()}")

    # Group by lesion_id to prevent data leakage
    lesion_df = df.groupby('lesion_id').first().reset_index()
    
    # Check if we can stratify
    class_counts = lesion_df['dx'].value_counts()
    can_stratify = (class_counts >= 2).all()
    stratify = lesion_df['dx'] if can_stratify else None

    train_lesions, val_lesions = train_test_split(
        lesion_df['lesion_id'],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    train_df = df[df['lesion_id'].isin(train_lesions)].reset_index(drop=True)
    val_df = df[df['lesion_id'].isin(val_lesions)].reset_index(drop=True)

    print(f"[*] Train set: {len(train_df)} samples, Val set: {len(val_df)} samples")
    return train_df, val_df, img_dir


def compute_class_weights(train_df, device):
    """
    Calculates balanced class weights inversely proportional to class frequencies.
    """
    counts = train_df['dx'].value_counts()
    total = len(train_df)
    n_classes = len(CLASS_INDEX_TO_NAME)
    
    weights = np.ones(n_classes, dtype=np.float32)
    for class_name, count in counts.items():
        if class_name in NAME_TO_CLASS_INDEX:
            idx = NAME_TO_CLASS_INDEX[class_name]
            weights[idx] = total / (n_classes * count)

    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    return weights_tensor


def build_model(pretrained=True):
    """
    Builds ResNet18 model configured for HAM10000 7 classes.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_INDEX_TO_NAME))
    return model


def train_model(data_dir=None, output_weights=None, epochs=10, batch_size=32, lr=1e-4, pretrained=True):
    """
    Runs the complete HAM10000 training and evaluation cycle.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if data_dir is None:
        data_dir = os.path.join(base_dir, "data")
    if output_weights is None:
        output_weights = os.path.join(base_dir, "weights", "model_ham10000_best.pth")

    os.makedirs(os.path.dirname(output_weights), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on device: {device}")

    train_df, val_df, img_dir = prepare_data(data_dir)
    train_transform, val_transform = get_transforms()

    train_dataset = HAM10000Dataset(train_df, img_dir, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, img_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(pretrained=pretrained).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_f1 = 0.0
    best_val_acc = 0.0

    print("\n" + "="*50)
    print(f"Starting HAM10000 training for {epochs} epochs...")
    print("="*50)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save(model.state_dict(), output_weights)
            print(f"  --> Saved new best checkpoint to {output_weights} (F1: {best_val_f1:.4f})")

    # If not saved (e.g. 0 F1), save last state
    if not os.path.exists(output_weights):
        torch.save(model.state_dict(), output_weights)
        print(f"[*] Saved model checkpoint to {output_weights}")

    print("\n" + "="*50)
    print(f"[OK] Training complete. Best Val F1: {best_val_f1:.4f}, Best Val Acc: {best_val_acc:.4f}")
    print("="*50)
    return output_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HAM10000 Skin Lesion ResNet18 Classifier")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing HAM10000 data")
    parser.add_argument("--output", type=str, default=None, help="Output path for best model weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()
    train_model(
        data_dir=args.data_dir,
        output_weights=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
