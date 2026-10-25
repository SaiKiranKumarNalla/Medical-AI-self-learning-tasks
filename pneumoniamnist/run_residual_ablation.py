from pathlib import Path
import argparse
import random
from runtime_utils import ROOT, select_device, loss_weight, rng_state, restore_rng, clean_validation
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import numpy as np
from medmnist import PneumoniaMNIST

# PlainBlock WITHOUT residual addition (out += self.shortcut(x) REMOVED)
class PlainBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(PlainBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # RESIDUAL ADDITION INTENTIONALLY REMOVED FOR ABLATION!
        # out += self.shortcut(x) <-- SEVERED
        return F.relu(out)

class PlainNet18(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(PlainNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(PlainBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(PlainBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(PlainBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(PlainBlock, 512, 2, stride=2)
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
    if args.resume: parser.error('Resume currently supported by train_replication.py only')
    if args.epochs < 1: parser.error('--epochs must be positive')
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error('Output directory is nonempty; choose a new directory or --resume')
    output.mkdir(parents=True, exist_ok=True)
    device = select_device()
    print(f"Starting Residual Ablation (PlainNet-18 without skip connections) on device: {device}")
    
    root = str(ROOT / 'data')
    os.makedirs(root, exist_ok=True)
    os.makedirs(output / 'checkpoints', exist_ok=True)

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

    model = PlainNet18().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val_auc = 0.0
    best_epoch = 0
    checkpoint_path = str(output / 'checkpoints' / 'best_plainnet18.pth')

    patience = 10
    epochs_no_improve = 0
    start_time = time.time()

    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val AUC':<9} | {'Status'}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_weight = 0.0
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

        is_best = False
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_no_improve = 0
            is_best = True
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1

        status_str = f"BEST (Val AUC: {best_val_auc:.4f})" if is_best else f"No improve: {epochs_no_improve}/{patience}"
        print(f"{epoch:4d}/100 | {train_loss:10.4f} | {val_loss:10.4f} | {val_acc*100:6.2f}%   | {val_auc:7.4f}   | {status_str}", flush=True)

        if epochs_no_improve >= patience:
            print(f"\n[EARLY STOPPING TRIGGERED] PlainNet-18 Val AUC stopped improving at Epoch {epoch}!")
            break

    elapsed = time.time() - start_time
    print(f"Training ended in {elapsed/60:.2f} minutes.")
    print(f"Selected Best Checkpoint from Epoch {best_epoch} with Val AUC: {best_val_auc:.4f}")

    if not args.evaluate_test:
        print('Training finished. Test evaluation withheld; freeze decisions before using --evaluate-test.')
        return

    # Final Test Evaluation
    best_model = PlainNet18().to(device)
    best_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_acc, test_auc, test_targets, test_probs = evaluate(best_model, test_loader, criterion, device)
    test_preds = (test_probs >= 0.5).astype(int)

    cm = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("\n" + "=" * 60)
    print("TEST EVALUATION: PLAIN-18 (RESIDUAL ABLATION)")
    print("=" * 60)
    print(f"Test Accuracy    : {test_acc*100:.2f}%")
    print(f"Test AUC-ROC     : {test_auc:.4f}")
    print(f"Sensitivity      : {sensitivity*100:.2f}% (Pneumonia Recall: {tp}/{tp+fn})")
    print(f"Specificity      : {specificity*100:.2f}% (Normal Rejection: {tn}/{tn+fp})")
    print(f"Confusion Matrix : TN={tn}, FP={fp}, FN={fn}, TP={tp}")

if __name__ == '__main__':
    main()
