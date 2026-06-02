import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import random

from models.freshlitenet import FreshLiteNet
from dataset import get_val_transforms, get_stratified_splits

class OrdinalWrapper(torch.nn.Module):
    """
    Wraps the ordinal model to output a single continuous scalar representing
    the freshness severity. We sum the logits so Grad-CAM can backpropagate 
    from the total predicted freshness index.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        logits = self.model(x)
        # Summing logits acts as a proxy for the continuous ordinal prediction score
        return logits.sum(dim=1).unsqueeze(1)

def generate_gradcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Model
    model = FreshLiteNet(num_classes=10, mode='ordinal', pretrained=False)
    checkpoint_path = "checkpoints/freshlitenet_ordinal.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    wrapped_model = OrdinalWrapper(model)
    
    # Target Layer: We use the MultiScaleFreshnessModule
    target_layers = [model.multiscale]
    
    # Initialize GradCAM
    cam = GradCAM(model=wrapped_model, target_layers=target_layers)
    
    # Get some test images
    splits = get_stratified_splits()
    test_paths, test_labels = splits['test']
    
    # Select 4 random images across different classes
    random.seed(1337)
    selected_indices = random.sample(range(len(test_paths)), 4)
    
    transforms = get_val_transforms()
    
    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    for i, idx in enumerate(selected_indices):
        img_path = test_paths[idx]
        label = test_labels[idx]
        
        # Original Image
        rgb_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = cv2.resize(rgb_img, (224, 224))
        
        # Float image for overlay
        float_img = np.float32(rgb_img) / 255.0
        
        # Transform for model
        input_tensor = transforms(image=rgb_img)['image'].unsqueeze(0).to(device)
        
        # Get actual prediction
        pred = model.predict(input_tensor).item()
        
        # Generate CAM
        # Since we wrapped it to output a single scalar at dim 1 (index 0), target is 0
        targets = [ClassifierOutputTarget(0)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        # Overlay
        visualization = show_cam_on_image(float_img, grayscale_cam, use_rgb=True)
        
        # Plot Original
        axes[0, i].imshow(rgb_img)
        axes[0, i].set_title(f"True: Stage {label}\nPred: Stage {pred}")
        axes[0, i].axis('off')
        
        # Plot GradCAM
        axes[1, i].imshow(visualization)
        axes[1, i].set_title(f"Grad-CAM Heatmap")
        axes[1, i].axis('off')
        
    plt.tight_layout()
    save_path = "figures/gradcam_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved Grad-CAM visualizations to {save_path}")

if __name__ == "__main__":
    generate_gradcam()
