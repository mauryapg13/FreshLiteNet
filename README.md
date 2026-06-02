# FreshLiteNet: Efficient Ordinal Regression for Tomato Freshness Grading

This repository contains the official PyTorch implementation, experimental setup, and evaluation code for **FreshLiteNet**, a lightweight and highly efficient neural network architecture designed specifically for **tomato freshness grading** on edge devices.

FreshLiteNet approaches freshness grading not as a standard classification task, but as an **ordinal regression** problem, predicting the freshness level of a tomato across progressive decay stages.

---

## 🍅 1. Overview & Motivation

Agricultural produce grading on the edge requires models that are highly accurate yet computationally inexpensive. While heavy models like ResNet-50 perform well, they are not suitable for low-power edge deployment. Conversely, lightweight models like MobileNetV2 often struggle with the fine-grained, multi-scale visual differences inherent in early-stage rotting or bruising.

**FreshLiteNet** achieves a superior trade-off between accuracy and computational cost. It employs a custom lightweight backbone paired with multi-scale feature extraction, efficient channel attention, and an ordinal regression loss mechanism (CORAL). This results in state-of-the-art performance for fine-grained freshness grading.

---

## 🔬 2. Architecture Details

The **FreshLiteNet** architecture (implemented in `models/freshlitenet.py`) consists of the following key components:

### 2.1 MobileNetV3-Large Backbone
We utilize the feature extractor from **MobileNetV3-Large** (pretrained on ImageNet) to extract foundational visual features. This provides a highly efficient starting point (960 output channels).

### 2.2 Multi-Scale Freshness Module (MSFM)
Rotting and bruising can manifest at various scales—from tiny localized spots to large systemic decay. To capture this, we introduce the MSFM, which processes the backbone features through parallel parallel branches:
- **1x1 Convolution**
- **3x3 Depthwise Separable Convolution**
- **5x5 Depthwise Separable Convolution**

These branches are concatenated to form a 768-dimensional multi-scale feature map, extracting both local textures and global structural changes without a massive parameter explosion.

### 2.3 Efficient Channel Attention (ECA)
We apply an **ECA Module** to the multi-scale features. ECA uses a fast 1D convolution over globally pooled features to capture cross-channel interactions, emphasizing channels most relevant to freshness indicators (like specific color shifts or texture breakdown) while suppressing background noise.

### 2.4 Ordinal Regression Head (CORAL)
Instead of standard Softmax classification, FreshLiteNet uses the **COnsistent RAnk Logits (CORAL)** framework for ordinal regression. For $K$ freshness classes, the network outputs $K-1$ binary logits. The loss function is a Binary Cross-Entropy (BCE) over these cumulative targets, ensuring the model inherently understands the ordered severity of decay (e.g., predicting Stage 5 instead of Stage 2 for a Stage 3 tomato is penalized heavily).

---

## 📊 3. Dataset & Augmentation

The model is evaluated on the **FGrade Tomato Dataset**. We formulated two distinct experimental setups to test generalizability and fine-grained discrimination:

1. **10-Class Fine-Grained Grading (`dataset.py`)**: The dataset is divided into 10 highly granular freshness stages (0 = freshest, 9 = most decayed).
2. **3-Class Coarse Grading (`dataset_3class.py`)**: A simplified configuration mapping the dataset into 3 broad categories (e.g., Fresh, Semi-Fresh, Spoiled).

### Augmentation Pipeline
To ensure robustness in real-world agricultural environments, we employ an aggressive data augmentation pipeline using Albumentations:
- **Spatial:** Random Resized Crops (0.8x-1.0x), Rotations, and Flips to handle variable orientations.
- **Lighting:** Random Brightness, Contrast, and Color Jitter to simulate diverse warehouse/field lighting.
- **Camera Artifacts:** Gaussian Noise, Motion Blur, and JPEG Compression to simulate low-cost edge camera sensors.

---

## 🚀 4. Experimental Setup & Training

### Training Methodology
- **Optimizer:** AdamW with weight decay $1e-4$.
- **Learning Rate Schedule:** Cosine Annealing LR (initial LR = $1e-4$).
- **Mixed Precision:** Automatic Mixed Precision (AMP) is used for memory efficiency and speed.
- **Epochs & Early Stopping:** Models are trained for up to 100 epochs, with early stopping triggered if validation loss does not improve for 15 epochs.
- **Hardware:** Training scripts automatically detect and utilize CUDA or MPS (Apple Silicon).

### Baselines Evaluated
We compare FreshLiteNet against two established baselines:
- **ResNet-50:** A heavy, highly accurate standard baseline.
- **MobileNetV2:** A lightweight, efficient baseline for edge devices.

---

## 📈 5. Results & Edge Benchmarking

Models were evaluated using metrics tailored for ordinal classification, including standard **Accuracy**, **Accuracy (±1)** (accuracy allowing a tolerance of 1 ordinal step), **Macro F1-Score**, **Quadratic Weighted Kappa (QWK)**, and **Mean Absolute Error (MAE)**.

### Experiment 1: 10-Class Fine-Grained Grading
FreshLiteNet significantly outperforms MobileNetV2. It even achieves higher off-by-one accuracy and QWK than the 23M parameter ResNet-50, all while maintaining a remarkably small footprint (~3.85M parameters).

| Model | Accuracy | Acc (±1) | F1-Score | QWK | MAE | Params (M) | Inference (ms/img) |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 35.00% | 73.33% | 0.3467 | 0.8603 | 1.0778 | 23.53 | 7.60 ms |
| MobileNetV2 | 33.89% | 68.33% | 0.3116 | 0.8155 | 1.2722 | 2.24 | **2.63 ms** |
| **FreshLiteNet (Ours)** | **41.67%** | **86.11%** | **0.4203** | **0.9160** | **0.7833** | 3.85 | 2.99 ms |

### Experiment 2: 3-Class Coarse Grading
On the simplified 3-class task, FreshLiteNet achieves the highest accuracy and QWK across the board, demonstrating strong feature extraction capabilities.

| Model | Accuracy | F1-Score | QWK | MAE | Params (M) | Inference (ms/img) |
|---|---|---|---|---|---|---|
| ResNet-50 | 90.56% | 0.9078 | 0.9091 | 0.1000 | 23.51 | 8.32 ms |
| MobileNetV2 | 86.67% | 0.8702 | 0.8938 | 0.1333 | **2.23** | **3.07 ms** |
| **FreshLiteNet (Ours)** | **92.78%** | **0.9298** | **0.9401** | **0.0722** | 3.85 | 3.09 ms |

---

## 🔍 6. Explainability (Grad-CAM)
To ensure the network is learning meaningful physical features rather than dataset artifacts, we employed Grad-CAM analysis. The visualizations confirmed that FreshLiteNet consistently focuses its attention on the exact regions of the tomato where bruising, wrinkling, or rotting occurs, unlike baseline models which often attend to background features.

---

## 💻 7. Usage Guide

### Requirements
```bash
pip install torch torchvision numpy pandas scikit-learn albumentations opencv-python tqdm
```

### Training the Models
To train the models from scratch, simply run the corresponding training script. The script will automatically train all three models (FreshLiteNet, ResNet-50, MobileNetV2) and save the checkpoints in the `checkpoints/` directory.

**For 10-Class Task:**
```bash
python train.py --epochs 100 --batch_size 32 --lr 1e-4
```

**For 3-Class Task:**
```bash
python train_3class.py --epochs 100 --batch_size 32 --lr 1e-4
```

### Evaluating the Models
To evaluate the trained models on the test set and generate the comparison metrics (`results/`) and confusion matrices (`figures/`):

**For 10-Class Task:**
```bash
python evaluate.py
```

**For 3-Class Task:**
```bash
python evaluate_3class.py
```

---

## 📜 8. License
This project is licensed under the MIT License.
