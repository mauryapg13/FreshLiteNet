import os
import torch
from torch.utils.data import DataLoader
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_absolute_error, cohen_kappa_score, f1_score, confusion_matrix

from dataset import get_stratified_splits, get_val_transforms, TomatoDataset
from models.freshlitenet import FreshLiteNet
from models.baselines import get_baseline_model
from train import label_to_coral_target, coral_target_to_label

PRIMARY_COLOR = "\033[38;2;71;52;114m"
ACCENT_COLOR = "\033[38;2;135;186;195m"
RESET_COLOR = "\033[0m"

def get_model(model_name, mode, num_classes=10):
    if model_name.lower() == 'freshlitenet':
        model = FreshLiteNet(num_classes=num_classes, mode=mode, pretrained=False)
    else:
        model = get_baseline_model(model_name, num_classes=num_classes, pretrained=False)
    return model

def evaluate_model(model, dataloader, device, mode, num_classes=10):
    model.eval()
    all_preds = []
    all_targets = []
    
    # For inference time
    start_time = time.time()
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            logits = model(images)
            
            if mode == 'ordinal':
                probs = torch.sigmoid(logits)
                preds = coral_target_to_label(probs)
            else:
                preds = torch.argmax(logits, dim=1)
                
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    end_time = time.time()
    
    # Calculate metrics
    acc = accuracy_score(all_targets, all_preds)
    
    # Calculate relaxed accuracy (within +/- 1 stage)
    correct_plus_minus_1 = sum(1 for t, p in zip(all_targets, all_preds) if abs(t - p) <= 1)
    acc_plus_minus_1 = correct_plus_minus_1 / len(all_targets) if all_targets else 0.0
    
    mae = mean_absolute_error(all_targets, all_preds)
    qwk = cohen_kappa_score(all_targets, all_preds, weights='quadratic')
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    total_time = end_time - start_time
    time_per_image_ms = (total_time / len(dataloader.dataset)) * 1000
    
    return acc, acc_plus_minus_1, mae, qwk, f1, time_per_image_ms, all_targets, all_preds

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"{PRIMARY_COLOR}Running Evaluation on Device: {device}{RESET_COLOR}\n")
    
    # Get test set (using the same splits as training)
    splits = get_stratified_splits(limit_per_class=120)
    test_paths, test_labels = splits['test']
    
    print(f"Test set size: {len(test_paths)} images\n")
    
    test_dataset = TomatoDataset(test_paths, test_labels, transform=get_val_transforms())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    models_to_evaluate = [
        ('resnet50', 'classification'),
        ('mobilenetv2', 'classification'),
        ('freshlitenet', 'ordinal')
    ]
    
    results = []
    
    for model_name, mode in models_to_evaluate:
        checkpoint_path = f"checkpoints/{model_name}_{mode}.pth"
        if not os.path.exists(checkpoint_path):
            print(f"{ACCENT_COLOR}Checkpoint not found for {model_name}. Skipping.{RESET_COLOR}")
            continue
            
        print(f"Evaluating {model_name}...")
        
        # Load model and weights
        model = get_model(model_name, mode, num_classes=10)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        # Calculate param count
        params = count_parameters(model) / 1e6 # In Millions
        
        # Evaluate
        acc, acc_pm1, mae, qwk, f1, inf_time, all_targets, all_preds = evaluate_model(model, test_loader, device, mode, num_classes=10)
        
        # Plot Confusion Matrix
        cm = confusion_matrix(all_targets, all_preds, labels=range(10))
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.title(f"{model_name.capitalize()} Confusion Matrix")
        plt.xlabel("Predicted Stage")
        plt.ylabel("True Stage")
        os.makedirs("figures", exist_ok=True)
        plt.savefig(f"figures/cm_{model_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        results.append({
            "Model": model_name.capitalize(),
            "Accuracy": f"{acc:.4f}",
            "Acc (±1)": f"{acc_pm1:.4f}",
            "F1-Score": f"{f1:.4f}",
            "QWK": f"{qwk:.4f}",
            "MAE": f"{mae:.4f}",
            "Params (M)": f"{params:.2f}",
            "Inference (ms/img)": f"{inf_time:.2f}"
        })
        
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print("FINAL 3-MODEL COMPARISON RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80 + "\n")
        
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/comparison_metrics.csv", index=False)
        print("Results saved to results/comparison_metrics.csv")
    else:
        print("No models were evaluated (checkpoints not found).")

if __name__ == "__main__":
    main()
