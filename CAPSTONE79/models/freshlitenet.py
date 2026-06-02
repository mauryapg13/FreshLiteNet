import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution block to minimize parameter count.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.Hardswish(inplace=True)
        
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.Hardswish(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

class MultiScaleFreshnessModule(nn.Module):
    """
    Extracts multi-scale features for freshness defects (e.g., small spots vs. large rot).
    Uses parallel branches: 1x1, 3x3 depthwise separable, and 5x5 depthwise separable.
    """
    def __init__(self, in_channels, branch_channels=256):
        super().__init__()
        # Branch A: 1x1 Conv
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.Hardswish(inplace=True)
        )
        
        # Branch B: 3x3 depthwise separable Conv (padding=1)
        self.branch3x3 = DepthwiseSeparableConv(in_channels, branch_channels, kernel_size=3, padding=1)
        
        # Branch C: 5x5 depthwise separable Conv (padding=2)
        self.branch5x5 = DepthwiseSeparableConv(in_channels, branch_channels, kernel_size=5, padding=2)

    def forward(self, x):
        out1 = self.branch1x1(x)
        out2 = self.branch3x3(x)
        out3 = self.branch5x5(x)
        # Concatenate along the channel dimension
        return torch.cat([out1, out2, out3], dim=1)

class ECAModule(nn.Module):
    """
    Efficient Channel Attention (ECA) module.
    Performs local 1D convolution over global pooled features to capture cross-channel relations.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # Determine kernel size k adaptively
        t = int(abs((np.log2(channels) + b) / gamma))
        k = t if t % 2 != 0 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # Shape: [B, C, 1, 1]
        y = y.squeeze(-1).transpose(-1, -2)  # Shape: [B, 1, C]
        y = self.conv(y)  # Shape: [B, 1, C]
        y = y.transpose(-1, -2).unsqueeze(-1)  # Shape: [B, C, 1, 1]
        y = self.sigmoid(y)
        return x * y

class FreshLiteNet(nn.Module):
    """
    FreshLiteNet: An Ordinal Multi-Scale Attention Network for Lightweight Tomato Freshness Grading.
    Supports standard classification (mode='classification') and ordinal regression (mode='ordinal').
    """
    def __init__(self, num_classes=10, mode='ordinal', pretrained=True, use_attention=True):
        super().__init__()
        self.num_classes = num_classes
        self.mode = mode
        self.use_attention = use_attention
        
        # 1. Load MobileNetV3-Large Backbone
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        
        # Extract feature extractor part
        self.backbone_features = backbone.features
        
        # Feature channels from MobileNetV3 Large last layer is 960
        in_channels = 960
        branch_channels = 256
        concat_channels = branch_channels * 3  # 768 channels
        
        # 2. Multi-Scale Feature Module
        self.multiscale = MultiScaleFreshnessModule(in_channels, branch_channels)
        
        # 3. Efficient Channel Attention
        self.attention = ECAModule(concat_channels)
        
        # 4. Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 5. FC and Projection
        self.fc = nn.Sequential(
            nn.Linear(concat_channels, 128),
            nn.LayerNorm(128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2)
        )
        
        # 6. Classification / Ordinal Heads
        if self.mode == 'ordinal':
            # For ordinal regression, output K - 1 logits
            self.head = nn.Linear(128, num_classes - 1)
        else:
            # Standard classification, output K logits
            self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # MobileNetV3 features -> shape [B, 960, 7, 7] for [B, 3, 224, 224] input
        x = self.backbone_features(x)
        
        # Multi-scale features -> shape [B, 768, 7, 7]
        x = self.multiscale(x)
        
        # ECA attention -> shape [B, 768, 7, 7]
        if self.use_attention:
            x = self.attention(x)
        
        # Global pooling -> shape [B, 768, 1, 1]
        x = self.gap(x)
        x = torch.flatten(x, 1)  # shape [B, 768]
        
        # 128-dimensional embedding -> shape [B, 128]
        feat = self.fc(x)
        
        # Out logits -> shape [B, K-1] (ordinal) or [B, K] (classification)
        logits = self.head(feat)
        
        return logits

    def predict(self, x):
        """
        Performs inference and maps logits to predicted class indices (0 to num_classes - 1).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if self.mode == 'ordinal':
                # Convert logits to probabilities
                probs = torch.sigmoid(logits)
                # Count how many thresholds are exceeded
                preds = (probs > 0.5).sum(dim=1)
            else:
                preds = torch.argmax(logits, dim=1)
        return preds

if __name__ == "__main__":
    # Test compilation and forward pass
    print("Testing FreshLiteNet models...")
    x = torch.randn(2, 3, 224, 224)
    
    # Test Ordinal mode
    model_ord = FreshLiteNet(num_classes=10, mode='ordinal', pretrained=False)
    out_ord = model_ord(x)
    preds_ord = model_ord.predict(x)
    print(f"Ordinal Model output shape: {out_ord.shape} (Expected: [2, 9])")
    print(f"Ordinal Predictions shape: {preds_ord.shape} (Expected: [2]), Values: {preds_ord.tolist()}")
    
    # Test Classification mode
    model_clf = FreshLiteNet(num_classes=10, mode='classification', pretrained=False)
    out_clf = model_clf(x)
    preds_clf = model_clf.predict(x)
    print(f"Classification Model output shape: {out_clf.shape} (Expected: [2, 10])")
    print(f"Classification Predictions shape: {preds_clf.shape} (Expected: [2]), Values: {preds_clf.tolist()}")
    
    # Count parameters
    params_ord = sum(p.numel() for p in model_ord.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {params_ord / 1e6:.2f}M")
