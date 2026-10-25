from pathlib import Path
import argparse
import random
from runtime_utils import ROOT, select_device, loss_weight, rng_state, restore_rng, clean_validation
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from medmnist import PneumoniaMNIST

# 1. Architecture: Author Custom Small-Image ResNet-18
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        # Author 3x3 stride-1 stem (no maxpool)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.linear(out)

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_weight = 0.0
    all_targets, all_probs = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.squeeze(1).to(device)
            logits = model(x)
            loss = criterion(logits, y)
            weight = loss_weight(criterion, y)
            total_loss += loss.item() * weight
            total_weight += weight
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_targets.extend(y.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy())
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    preds = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_targets, preds)
    auc = roc_auc_score(all_targets, all_probs)
    return total_loss / total_weight, acc, auc, all_targets, all_probs

def main():
    parser = argparse.ArgumentParser(description="Exploratory PneumoniaMNIST learning experiment")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output-dir', type=str, required=True, help='New run directory; historical checkpoints are preserved')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--evaluate-test', action='store_true', help='Explicit exploratory test evaluation after training')
    parser.add_argument('--official-validation', action='store_true', help='Keep known train/validation duplicates for historical protocol only')
    args = parser.parse_args()
    if args.epochs < 1: parser.error('--epochs must be positive')
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output directory is nonempty; choose a new directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print(f"Starting MedMNIST ResNet-18 Replication on device: {device}")
    
    root = str(ROOT / 'data')
    os.makedirs(root, exist_ok=True)
    os.makedirs(output / 'checkpoints', exist_ok=True)
    artifact_dir = str(output)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_ds = PneumoniaMNIST(split='train', transform=transform, root=root, download=True)
    val_ds   = PneumoniaMNIST(split='val',   transform=transform, root=root, download=True)
    test_ds  = PneumoniaMNIST(split='test',  transform=transform, root=root, download=True)

    if not args.official_validation:
        val_ds = clean_validation(train_ds, val_ds)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=128, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    model = ResNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    best_epoch = 0
    start_epoch = 1
    num_epochs = args.epochs

    checkpoint_best_path = str(output / 'checkpoints' / 'best_resnet18_val_auc.pth')
    checkpoint_latest_path = str(output / 'checkpoints' / 'latest_checkpoint.pth')
    history_path = str(output / 'checkpoints' / 'resnet18_replication_history.json')

    history = {
        'epoch': [], 'lr': [],
        'train_loss': [], 'val_loss': [],
        'val_acc': [], 'val_auc': []
    }

    # Resume if latest checkpoint exists
    if args.resume and not os.path.exists(checkpoint_latest_path):
        raise FileNotFoundError('No latest checkpoint found in requested run directory')
    if args.resume and os.path.exists(checkpoint_latest_path):
        print(f"Resuming from saved checkpoint: {checkpoint_latest_path}")
        chk = torch.load(checkpoint_latest_path, map_location=device)
        model.load_state_dict(chk['model'])
        optimizer.load_state_dict(chk['optimizer'])
        scheduler.load_state_dict(chk['scheduler'])
        if 'rng' not in chk:
            raise ValueError('Legacy checkpoint lacks RNG state; exact resume unavailable. Use a new run directory.')
        if chk.get('official_validation') != args.official_validation:
            raise ValueError('Resume validation protocol differs from saved checkpoint')
        restore_rng(chk['rng'])
        start_epoch = chk['epoch'] + 1
        best_val_auc = chk['best_val_auc']
        best_epoch = chk['best_epoch']
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
        print(f"Resumed at epoch {start_epoch} (Current Best Val AUC: {best_val_auc:.4f} at epoch {best_epoch})")

    start_time = time.time()

    print(f"{'Epoch':<6} | {'LR':<8} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val AUC':<9} | {'Status'}")
    print("-" * 75)

    for epoch in range(start_epoch, num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_weight = 0.0
        current_lr = optimizer.param_groups[0]['lr']

        for x, y in train_loader:
            x, y = x.to(device), y.squeeze(1).to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            weight = loss_weight(criterion, y)
            train_loss += loss.item() * weight
            train_weight += weight

        train_loss /= train_weight
        val_loss, val_acc, val_auc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history['epoch'].append(epoch)
        history['lr'].append(current_lr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)

        is_best = False
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            is_best = True
            torch.save(model.state_dict(), checkpoint_best_path)

        # Save latest checkpoint after every single epoch for crash resilience
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val_auc': best_val_auc,
            'best_epoch': best_epoch,
            'rng': rng_state(),
            'official_validation': args.official_validation
        }, checkpoint_latest_path)

        # Update history file
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        # Write to human-readable log file
        with open(output / 'training_progress.log', 'a') as f:
            status_str = f"BEST (Val AUC: {best_val_auc:.4f})" if is_best else ""
            f.write(f"Epoch {epoch:3d}/{num_epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | Val AUC: {val_auc:.4f} | {status_str}\n")

        status_str = f"BEST (Val AUC: {best_val_auc:.4f})" if is_best else ""
        print(f"{epoch:4d}/{num_epochs} | {current_lr:8.6f} | {train_loss:10.4f} | {val_loss:10.4f} | {val_acc*100:6.2f}%   | {val_auc:7.4f}   | {status_str}", flush=True)

    elapsed_time = time.time() - start_time
    print(f"\nTraining Complete in {elapsed_time/60:.2f} minutes!")
    print(f"Selected Best Checkpoint from Epoch {best_epoch} with Validation AUC: {best_val_auc:.4f}")

    if not args.evaluate_test:
        print('Training finished. Test evaluation withheld; freeze decisions before using --evaluate-test.')
        return

    # 5. FINAL TEST EVALUATION (UNLOCKING THE VAULT ONCE)
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK TEST EVALUATION (Frozen Best Checkpoint)")
    print("=" * 60)
    best_model = ResNet18().to(device)
    best_model.load_state_dict(torch.load(checkpoint_best_path, map_location=device))

    test_loss, test_acc, test_auc, test_targets, test_probs = evaluate(best_model, test_loader, criterion, device)
    test_preds = (test_probs >= 0.5).astype(int)

    # Compute Confusion Matrix, Sensitivity, Specificity
    cm = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print(f"Test Loss        : {test_loss:.4f}")
    print(f"Test Accuracy    : {test_acc*100:.2f}%")
    print(f"Test AUC-ROC     : {test_auc:.4f}")
    print(f"Sensitivity      : {sensitivity*100:.2f}% (Pneumonia Detection Recall)")
    print(f"Specificity      : {specificity*100:.2f}% (Normal Correct Rejection)")
    print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    # 6. Generate 100-Epoch Training Curve Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    eps = history['epoch']

    ax1.plot(eps, history['train_loss'], color='#0284c7', lw=1.8, label='Train Loss')
    ax1.plot(eps, history['val_loss'], color='#dc2626', lw=1.8, label='Val Loss')
    ax1.axvline(50, color='#64748b', linestyle='--', label='LR Drop (Ep 50)')
    ax1.axvline(75, color='#94a3b8', linestyle=':', label='LR Drop (Ep 75)')
    ax1.set_title('ResNet-18: Cross-Entropy Loss', fontweight='bold', fontsize=11)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.legend(frameon=True, fontsize=8.5)
    ax1.grid(True, linestyle='--', alpha=0.4)

    ax2.plot(eps, history['val_auc'], color='#7c3aed', lw=2, label='Validation AUC')
    ax2.axvline(best_epoch, color='#16a34a', linestyle='-', lw=2, label=f'Best Checkpoint (Ep {best_epoch}: {best_val_auc:.4f})')
    ax2.axhline(test_auc, color='#f59e0b', linestyle='--', lw=2, label=f'Final Test AUC: {test_auc:.4f}')
    ax2.set_title('ResNet-18: Validation AUC Progression', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('AUC-ROC', fontsize=10)
    ax2.legend(frameon=True, fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plot_path = os.path.join(artifact_dir, 'resnet18_100epochs_replication.png')
    plt.savefig(plot_path)
    print(f"\nTraining curves plot saved to {plot_path}")

if __name__ == '__main__':
    main()
