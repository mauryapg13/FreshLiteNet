import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, accuracy_score, f1_score

from dataset_3class import get_stratified_splits, get_train_transforms, get_val_transforms, TomatoDataset
from models.freshlitenet import FreshLiteNet
from models.baselines import get_baseline_model

# Color palette for logging or terminal outputs
PRIMARY_COLOR = "\033[38;2;71;52;114m"   # #473472
SECONDARY_COLOR = "\033[38;2;83;98;158m" # #53629E
ACCENT_COLOR = "\033[38;2;135;186;195m" # #87BAC3
RESET_COLOR = "\033[0m"

def label_to_coral_target(label, num_classes, device):
    """
    Converts a standard integer label [B] into a binary CORAL target [B, num_classes - 1].
    target[i, j] = 1.0 if label[i] > j else 0.0
    """
    thresholds = torch.arange(num_classes - 1, device=device).unsqueeze(0) # [1, K-1]
    target = (label.unsqueeze(1) > thresholds).float() # [B, K-1]
    return target

def coral_target_to_label(probs, threshold=0.5):
    """
    Converts CORAL probabilities [B, num_classes - 1] to predicted class indices [B].
    y_pred = sum_j (probs_j > threshold)
    """
    return (probs > threshold).sum(dim=1)

def train_one_epoch(model, dataloader, optimizer, criterion, device, mode, num_classes, scaler=None):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training (only on CUDA)
        if scaler is not None and device.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(images)
                if mode == 'ordinal':
                    coral_targets = label_to_coral_target(targets, num_classes, device)
                    loss = criterion(logits, coral_targets)
                else:
                    loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            if mode == 'ordinal':
                coral_targets = label_to_coral_target(targets, num_classes, device)
                loss = criterion(logits, coral_targets)
            else:
                loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
        running_loss += loss.item() * images.size(0)
        
        # Predictions
        if mode == 'ordinal':
            probs = torch.sigmoid(logits)
            preds = coral_target_to_label(probs)
        else:
            preds = torch.argmax(logits, dim=1)
            
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_mae = mean_absolute_error(all_targets, all_preds)
    epoch_qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    
    return epoch_loss, epoch_acc, epoch_mae, epoch_qwk

def validate(model, dataloader, criterion, device, mode, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images)
            
            if mode == 'ordinal':
                coral_targets = label_to_coral_target(targets, num_classes, device)
                loss = criterion(logits, coral_targets)
                probs = torch.sigmoid(logits)
                preds = coral_target_to_label(probs)
            else:
                loss = criterion(logits, targets)
                preds = torch.argmax(logits, dim=1)
                
            running_loss += loss.item() * images.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    val_loss = running_loss / len(dataloader.dataset)
    val_acc = accuracy_score(all_targets, all_preds)
    val_mae = mean_absolute_error(all_targets, all_preds)
    val_qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    val_f1 = f1_score(all_targets, all_preds, average='macro')
    
    return val_loss, val_acc, val_mae, val_qwk, val_f1

def train_model(model_name, mode, epochs=100, batch_size=32, lr=1e-4, device=None, 
                save_dir="checkpoints", data_splits=None, run_id=""):
    """
    Main training function. Fits either FreshLiteNet or a baseline model.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
    print(f"{PRIMARY_COLOR}Training {model_name} in {mode} mode on device: {device}{RESET_COLOR}")
    
    # 1. Dataset & Dataloaders
    if data_splits is None:
        data_splits = get_stratified_splits(limit_per_class=120)
        
    train_paths, train_labels = data_splits['train']
    val_paths, val_labels = data_splits['val']
    
    train_dataset = TomatoDataset(train_paths, train_labels, transform=get_train_transforms())
    val_dataset = TomatoDataset(val_paths, val_labels, transform=get_val_transforms())
    
    num_workers = 2 if device.type == 'cuda' else 0
    pin_memory = True if device.type == 'cuda' else False
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    num_classes = 3
    
    # 2. Instantiating Model
    if model_name.lower() == 'freshlitenet':
        model = FreshLiteNet(num_classes=num_classes, mode=mode, pretrained=True)
    else:
        # Baselines are standard classification
        model = get_baseline_model(model_name, num_classes=num_classes, pretrained=True)
        assert mode == 'classification', "Baselines must be trained in classification mode."
        
    model = model.to(device)
    
    # 3. Loss & Optimizer & Scheduler
    if mode == 'ordinal':
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision scaler (enable for CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # 4. Training Loop
    best_val_loss = float('inf')
    best_val_qwk = -1.0
    patience = 15
    patience_counter = 0
    
    history = {
        'train_loss': [], 'train_acc': [], 'train_mae': [], 'train_qwk': [],
        'val_loss': [], 'val_acc': [], 'val_mae': [], 'val_qwk': [], 'val_f1': []
    }
    
    model_filename = f"{model_name}_{mode}_3class{'_' + run_id if run_id else ''}.pth"
    save_path = os.path.join(save_dir, model_filename)
    
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, train_mae, train_qwk = train_one_epoch(
            model, train_loader, optimizer, criterion, device, mode, num_classes, scaler
        )
        
        val_loss, val_acc, val_mae, val_qwk, val_f1 = validate(
            model, val_loader, criterion, device, mode, num_classes
        )
        
        scheduler.step()
        
        # Save metrics history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_mae'].append(train_mae)
        history['train_qwk'].append(train_qwk)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_mae'].append(val_mae)
        history['val_qwk'].append(val_qwk)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} QWK: {train_qwk:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} QWK: {val_qwk:.4f} F1: {val_f1:.4f}")
        
        # Checkpoint saving & Early stopping (based on val loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_qwk = val_qwk
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_qwk': val_qwk,
                'history': history,
                'mode': mode,
                'model_name': model_name
            }, save_path)
            patience_counter = 0
            print(f"  {SECONDARY_COLOR}--> Saved best model checkpoint to {save_path}{RESET_COLOR}")
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"{ACCENT_COLOR}Early stopping triggered after {epoch} epochs.{RESET_COLOR}")
            break
            
    # Load best weights back
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['history']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3-model comparison pipeline.")
    parser.add_argument('--epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--limit_per_class', type=int, default=120, help='Images per class')
    args = parser.parse_args()
    
    limit = args.limit_per_class if args.limit_per_class > 0 else None
    
    print(f"Generating dataset splits with limit_per_class={limit if limit else 'ALL'}...")
    splits = get_stratified_splits(limit_per_class=limit)
    
    models_to_train = [
        ('resnet50', 'classification'),
        ('mobilenetv2', 'classification'),
        ('freshlitenet', 'ordinal')
    ]
    
    for model_name, mode in models_to_train:
        print(f"\n{'='*50}\nStarting training for {model_name.upper()} ({mode})\n{'='*50}")
        train_model(model_name, mode, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, data_splits=splits)
        
    print("\nAll models trained successfully!")
