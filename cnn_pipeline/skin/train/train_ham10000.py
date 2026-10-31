"""
HAM10000 Model Training & Evaluation Pipeline

Trains a ResNet18 classifier on the HAM10000 skin lesion dataset using:
- Persisted lesion-level train/validation/test splitting
- Class-weighted Cross-Entropy loss for handling severe class imbalance
- Dermatoscopic data augmentations
- Structured upgraded checkpoints; the original baseline is protected
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

if __package__ and '.' in __package__:
    from ..core.common import (BASE_DIR, CLASS_INDEX_TO_NAME, CLASS_NAMES, PREPROCESSING, MODEL_VERSION,
                          build_resnet18, seed_everything, seed_worker, save_json, timestamp,
                          file_sha256, protect_baseline, runtime_environment)
    from ..core.data_pipeline import HAM10000Dataset, load_metadata, split_data, save_splits, load_splits
    from ..core.augmentation import get_transforms
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from core.common import (BASE_DIR, CLASS_INDEX_TO_NAME, CLASS_NAMES, PREPROCESSING, MODEL_VERSION,
                        build_resnet18, seed_everything, seed_worker, save_json, timestamp,
                        file_sha256, protect_baseline, runtime_environment)
    from core.data_pipeline import HAM10000Dataset, load_metadata, split_data, save_splits, load_splits
    from core.augmentation import get_transforms

# Inverse mapping: name to class index
NAME_TO_CLASS_INDEX = {v: k for k, v in CLASS_INDEX_TO_NAME.items()}


def prepare_data(data_dir, test_size=0.15, random_state=42, validation_size=0.15):
    """Return train, validation, held-out test frames and image directory."""
    df, img_dir = load_metadata(data_dir)
    splits, _ = split_data(df, validation_size, test_size, random_state)
    return splits['train'], splits['validation'], splits['test'], str(img_dir)


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
    return build_resnet18(pretrained)


def train_model(data_dir=None, output_weights=None, epochs=10, batch_size=32, lr=1e-4, pretrained=True,
                *, seed=42, validation_size=.15, test_size=.15, augmentation='custom',
                morphology_probability=.12, elastic_probability=.15, results_dir=None,
                splits_path=None, device=None, num_workers=0):
    """
    Runs the complete HAM10000 training and evaluation cycle.
    """
    base_dir = str(BASE_DIR)
    if data_dir is None:
        data_dir = os.path.join(base_dir, "data")
    if output_weights is None:
        output_weights = os.path.join(base_dir, "weights", "model_ham10000_upgraded.pth")

    protect_baseline(output_weights)
    if Path(output_weights).exists():
        raise FileExistsError(f'Checkpoint already exists: {output_weights}. Choose a new --output.')
    if epochs < 1 or batch_size < 1 or not np.isfinite(lr) or lr <= 0 or num_workers < 0:
        raise ValueError('epochs/batch-size/lr must be positive and num-workers nonnegative.')
    train_transform, val_transform = get_transforms(augmentation, morphology_probability, elastic_probability)
    seed_everything(seed)
    experiment = Path(results_dir) if results_dir else BASE_DIR / 'results' / timestamp()
    if (experiment / 'config.json').exists():
        raise FileExistsError('Experiment directory already contains a training run; choose a new --results-dir.')
    experiment.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(experiment / 'training.log')], force=True)
    Path(output_weights).parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"[*] Training on device: {device}")

    df, img_dir = load_metadata(data_dir)
    if splits_path:
        splits, img_dir, manifest = load_splits(splits_path, data_dir)
        seed, validation_size, test_size = manifest['seed'], manifest['validation_size'], manifest['test_size']
        seed_everything(seed)
        notes = manifest['limitations']
    else:
        splits, notes = split_data(df, validation_size, test_size, seed)
    split_path = experiment / 'splits.json'
    split_hash = save_splits(split_path, df, splits, seed, validation_size, test_size, notes)
    train_df, val_df = splits['train'], splits['validation']
    if set(train_df.dx) != set(CLASS_NAMES):
        raise ValueError('Training split lacks diagnostic classes; obtain more lesions or a suitable predeclared split.')
    config = {'model_version': MODEL_VERSION, 'timestamp': timestamp(), 'data_dir': str(Path(data_dir).resolve()),
              'output_weights': str(Path(output_weights).resolve()), 'epochs': epochs, 'batch_size': batch_size,
              'lr': lr, 'weight_decay': 1e-4, 'pretrained': pretrained, 'seed': seed, 'device': str(device),
              'num_workers': num_workers, 'validation_size': validation_size, 'test_size': test_size,
              'augmentation': augmentation, 'morphology_probability': morphology_probability,
              'elastic_probability': elastic_probability, 'preprocessing': PREPROCESSING,
              'split_sha256': split_hash, 'splits_path': str(split_path.resolve()),
              **runtime_environment()}
    save_json(experiment / 'config.json', config)
    logging.info('Augmentation policy=%s; morphology p=%s; elastic p=%s (training only)',
                 augmentation, morphology_probability, elastic_probability)
    logging.info('Splits: %s', {name: (len(frame), frame.lesion_id.nunique()) for name, frame in splits.items()})

    train_dataset = HAM10000Dataset(train_df, img_dir, transform=train_transform)
    val_dataset = HAM10000Dataset(val_df, img_dir, transform=val_transform)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              generator=generator, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            generator=torch.Generator().manual_seed(seed), worker_init_fn=seed_worker)

    class_weights = compute_class_weights(train_df, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(pretrained=pretrained).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_f1 = -1.0
    best_val_acc = 0.0
    history = []

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
        val_f1 = float(f1_score(all_labels, all_preds, labels=list(CLASS_INDEX_TO_NAME), average='macro', zero_division=0))

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_accuracy': train_acc,
                        'validation_loss': val_loss, 'validation_accuracy': val_acc,
                        'validation_macro_f1': val_f1, 'seconds': elapsed})
        save_json(experiment / 'metrics/training_history.json', history)
        logging.info('Epoch %d: train loss %.4f; validation loss %.4f; accuracy %.4f; macro-F1 %.4f',
                     epoch, train_loss, val_loss, val_acc, val_f1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1:.4f}")

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc > best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(), 'epoch': epoch,
                        'best_validation_metrics': {'macro_f1': best_val_f1, 'accuracy': best_val_acc, 'loss': val_loss},
                        'class_index_to_name': CLASS_INDEX_TO_NAME, 'preprocessing': PREPROCESSING,
                        'training_config': config, 'seed': seed, 'model_version': MODEL_VERSION,
                        'split_sha256': split_hash, 'calibration_temperature': None,
                        'test_used_for_training': False}, output_weights)
            print(f"  --> Saved new best checkpoint to {output_weights} (F1: {best_val_f1:.4f})")

    save_json(experiment / 'reports/training_summary.json',
              {'checkpoint': str(Path(output_weights).resolve()), 'checkpoint_sha256': file_sha256(output_weights),
               'best_validation_macro_f1': best_val_f1, 'best_validation_accuracy': best_val_acc,
               'test_evaluated': False, 'limitations': notes})

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
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--validation-size', type=float, default=.15)
    parser.add_argument('--test-size', type=float, default=.15)
    parser.add_argument('--augmentation', choices=['none', 'baseline', 'custom', 'randaugment', 'autoaugment', 'trivialaugment'], default='custom')
    parser.add_argument('--morphology-probability', type=float, default=.12)
    parser.add_argument('--elastic-probability', type=float, default=.15)
    parser.add_argument('--results-dir')
    parser.add_argument('--splits', help='Reuse an existing split manifest for paired experiments')
    parser.add_argument('--device', choices=['cpu', 'cuda'])
    parser.add_argument('--num-workers', type=int, default=0)

    args = parser.parse_args()
    try:
        train_model(
            data_dir=args.data_dir,
            output_weights=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr, pretrained=args.pretrained, seed=args.seed, validation_size=args.validation_size,
            test_size=args.test_size, augmentation=args.augmentation, morphology_probability=args.morphology_probability,
            elastic_probability=args.elastic_probability, results_dir=args.results_dir, splits_path=args.splits,
            device=args.device, num_workers=args.num_workers)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.error(str(error))
