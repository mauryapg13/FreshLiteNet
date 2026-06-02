import torch
import torch.nn as nn
import torchvision.models as models

def get_baseline_model(name, num_classes=10, pretrained=True):
    """
    Loads a baseline model with its classifier head adapted to the given number of classes.
    """
    name = name.lower()
    
    if name == 'mobilenetv2':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    elif name == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unknown baseline model name: {name}")
        
    return model

if __name__ == "__main__":
    # Test baselines
    print("Testing Baselines compilation...")
    x = torch.randn(2, 3, 224, 224)
    for name in ['mobilenetv2', 'resnet50']:
        model = get_baseline_model(name, num_classes=10, pretrained=False)
        out = model(x)
        print(f"Baseline {name} output shape: {out.shape} (Expected: [2, 10])")
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Total trainable parameters: {params / 1e6:.2f}M")
