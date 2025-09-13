#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
training.py - EMII
Train/Evaluate ResNet-18 auf dem Brain Tumor MRI Data (4 Klassen).
Funktionen:
- Daten-Transforms (Train/ Test)
- Modell + Klassen-Gewichte
- Training mit Early Stopping + Cosine LR
- Evaluation (Report + Confusion Matrix)
- Optional: Temperatur-Skalierung (Kalibrierung)
- Artefakte werden gespeichert: weights (.pth), labels.json, temperature.pt

Startbeispiel:
    python training.py \
        --train-root "/brain-tumor-mri-dataset/Training" \
        --test-root  "/brain-tumor-mri-dataset/Testing" \
        --epochs 30 --batch-size 32 --lr 2e-4 --use-class-weights
CPU/ GPU
    nutzt automatisch CUDA falls verfügbar
"""
import os, json, argparse, random
import time
from pathlib import Path
from typing import Tuple, List
import contextlib
from functools import partial

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader 
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
os.makedirs("./models", exist_ok=True)

# --------------------
# Utilities
# --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    """Get best available device with proper configuration"""
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    
    if use_cuda:
        device = torch.device("cuda")
        # Enable TF32 for better performance on A100/RTX 30xx+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    elif use_mps:
        device = torch.device("mps")
        # Recommended env var for MPS
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    else:
        device = torch.device("cpu")
        # Set MKL threads to CPU count for better CPU performance
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(os.cpu_count())
    
    return device

# Update DataLoader in build_dataloaders():
def build_dataloaders(
    train_root: str,
    test_root: str,
    batch_size: int = 32,
    workers: int = 4
) -> Tuple[DataLoader, DataLoader, List[str]]:
    """ImageFolder expects /root/<class_name>/*"""
    # Data transforms
    train_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(train_root, transform=train_tf)
    test_ds = datasets.ImageFolder(test_root, transform=test_tf)
    
    # Optimize DataLoader for device type
    device = get_device()
    
    # Don't pin memory for MPS as it doesn't help
    pin_memory = (device.type == 'cuda')
    persistent_workers = workers > 0

    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=max(2 * batch_size, 64),
        shuffle=False,
        num_workers=workers, 
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=2 if persistent_workers else None
    )

    return train_loader, test_loader, train_ds.classes

def build_model(num_classes: int = 4, device: torch.device = torch.device("cpu")) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    m.to(device)
    return m

def compute_class_weights(train_loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    """Inverse-Frequency-Gewichte aus ImageFolder-Samples."""
    ds = train_loader.dataset
    targets = np.array([t for (_, t) in ds.samples], dtype=np.int64)
    counts  = np.bincount(targets, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             criterion: nn.Module) -> Tuple[float, float]:
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        pred   = torch.softmax(logits, dim=1).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
        loss_sum += float(loss.item()) * y.size(0)
    return loss_sum / total, correct / total

def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    class_weights: torch.Tensor = None,
    patience: int = 5,
    save_path: str = "./models/brain_tumor_resnet18.pth",
    update_ui_callback=None
):
    # Setup optimizer, loss and scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs // 2))
    best_acc, bad_epochs = 0.0, 0

    # Create directory if it doesn't exist
    Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
    
    # Setup AMP based on device
    amp_device_type = device.type
    use_amp = amp_device_type in ("cuda", "mps")
    #scaler = torch.amp.GradScaler(device_type=amp_device_type, enabled=use_amp)

    import pkg_resources
    pytorch_version = pkg_resources.get_distribution("torch").version
    is_torch_2_plus = int(pytorch_version.split('.')[0]) >= 2

    if is_torch_2_plus:
        scaler = torch.amp.GradScaler(device_type=amp_device_type, enabled=use_amp)
    else:
        # For older PyTorch versions
        if amp_device_type == "cuda":
            scaler = torch.amp.GradScaler(enabled=use_amp)
        else:
            # Fallback for MPS or CPU
            print("Warning: Mixed precision requires PyTorch 2.0+ for non-CUDA devices.")
            scaler = torch.amp.GradScaler(enabled=False)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    # Main training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Use tqdm with disable option
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", 
                   disable=update_ui_callback is not None)
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Track data loading time
            t0 = time.perf_counter()
            
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Reset gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass with scaler for mixed precision
            scaler.scale(loss).backward()
            
            # Gradient clipping helps stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            
            # Update weights with scaler
            scaler.step(optimizer)
            scaler.update()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix(loss=float(loss.item()), 
                             acc=f"{100.*correct/total:.2f}%")
            
            # Update UI if callback provided
            if update_ui_callback:
                batch_loss = loss.item()
                batch_acc = 100. * correct / total
                
                update_ui_callback(batch_idx/len(train_loader), {
                    'epoch': epoch,
                    'total_epochs': epochs,
                    'batch': batch_idx+1, 
                    'total_batches': len(train_loader),
                    'lr': scheduler.get_last_lr()[0],
                    'loss': batch_loss,
                    'acc': batch_acc
                })
            
        # Step the scheduler after each epoch
        scheduler.step()
        
        # Calculate epoch metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print metrics
        print(f"Epoch {epoch}/{epochs} - "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.2f}% | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc*100:.2f}%")
        
        # Update UI with history
        if update_ui_callback:
            update_ui_callback(1.0, {
                'epoch': epoch,
                'total_epochs': epochs,
                'history': history
            })
        
        # Save best model and check early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"[SAVE] {save_path}")
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping.")
                break

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

# export labels
def export_labels(classes: List[str], path: str = "./models/labels.json"):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, ensure_ascii=False, indent=2)

def full_evaluation_report(model: nn.Module, loader: DataLoader, device: torch.device,
                           class_names: List[str]):
    model.eval()
    all_y, all_p = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred   = logits.argmax(dim=1).cpu().numpy()
            all_p.append(pred)
            all_y.append(y.numpy())
    import numpy as np
    all_p = np.concatenate(all_p)
    all_y = np.concatenate(all_y)
    print("\nClassification report:\n")
    print(classification_report(all_y, all_p, target_names=class_names))
    print("\nConfusion matrix:\n")
    print(confusion_matrix(all_y, all_p))

def fit_temperature(model: nn.Module, loader: DataLoader, device: torch.device,
                    max_iter: int = 50) -> torch.Tensor:
    """Einfache Logit-Temperatur-Kalibrierung (Guo et al., 2017)."""
    model.eval()
    logits_list, y_list = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits_list.append(model(x).cpu())
            y_list.append(y)
    logits = torch.cat(logits_list, dim=0).to(device)
    y      = torch.cat(y_list,    dim=0).to(device)

    T = torch.nn.Parameter(torch.ones(1, device=device))
    opt = optim.LBFGS([T], lr=0.1, max_iter=max_iter)

    def nll_closure():
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(logits / T, y)
        loss.backward()
        return loss

    opt.step(nll_closure)
    print(f"Learned temperature T = {float(T.item()):.4f}")
    return T.detach()
    
def save_temperature(T: torch.Tensor, path: str = "./models/temperature.pt"):
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    torch.save(T.cpu(), path)


# -----------------
# Main
# -----------------

def main():
    print("Started training.py")
    try:
        print("inside of try block")
        parser = argparse.ArgumentParser(description="Train ResNet-18 on Brain Tumor MRI (4 classes)")
        parser.add_argument("--train-root", type=str, required=True, help="Pfad zu Training/")
        parser.add_argument("--test-root",  type=str, required=True, help="Pfad zu Testing/")
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--use-class-weights", action="store_true")
        parser.add_argument("--patience", type=int, default=5)
        parser.add_argument("--save-path", type=str, default="./models/brain_tumor_resnet18.pth")
        parser.add_argument("--fit-temperature", action="store_true",
                            help="Kalibriere T auf dem Test-Loader und speichere ./models/temperature.pt")
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()

        set_seed(args.seed)
        # enable Apple Silicon GPU (MPS backend)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import torch

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print("Device:", device)

        train_loader, test_loader, classes = build_dataloaders(
            args.train_root, args.test_root,
            batch_size=args.batch_size, workers=args.workers
        )
        print("Classes:", classes)
        print("")

        model = build_model(num_classes=len(classes), device=device)

        class_weights = None
        if args.use_class_weights:
            class_weights = compute_class_weights(train_loader, num_classes=len(classes), device=device)
            print("Class weights:", class_weights.detach().cpu().numpy())

        # Train + Early stopping
        train_loop(model, train_loader, test_loader, device,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                class_weights=class_weights, patience=args.patience,
                save_path=args.save_path)

        # Artefakte: Labels
        export_labels(classes, path="./models/labels.json")

        # Final Evaluation
        full_evaluation_report(model, test_loader, device, class_names=classes)

        # Temperatur-Kalibrierung (optional)
        if args.fit_temperature:
            T = fit_temperature(model, test_loader, device, max_iter=50)
            save_temperature(T, path="./models/temperature.pt")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    print("Started training.py")
    try:
        parser = argparse.ArgumentParser(description="Train ResNet-18 on Brain Tumor MRI (4 classes)")
        parser.add_argument("--train-root", type=str, required=True, help="Path to Training/")
        parser.add_argument("--test-root",  type=str, required=True, help="Path to Testing/")
        parser.add_argument("--epochs", type=int, default=30)
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--workers", type=int, default=4)
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--use-class-weights", action="store_true")
        parser.add_argument("--patience", type=int, default=5)
        parser.add_argument("--save-path", type=str, default="./models/brain_tumor_resnet18.pth")
        parser.add_argument("--fit-temperature", action="store_true",
                            help="Calibrate T on test set and save ./models/temperature.pt")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--gui", action="store_true", help="Launch with GUI monitor")
        args = parser.parse_args()

        if args.gui:
            from training_gui import launch_gui_monitor
            launch_gui_monitor(args)
        else:
            main()
    except Exception as e:
        print(f"Error: {e}")

