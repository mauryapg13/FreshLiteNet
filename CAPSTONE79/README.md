# FreshLiteNet: Efficient Ordinal Regression for Tomato Freshness Grading

This repository contains the official PyTorch implementation, experimental setup, and evaluation code for **FreshLiteNet**, a lightweight and highly efficient neural network architecture designed specifically for **tomato freshness grading** on edge devices.

FreshLiteNet approaches freshness grading as an **ordinal regression** problem, predicting the freshness level of a tomato across different progressive stages.

## Overview

Agricultural produce grading on the edge requires models that are both highly accurate and computationally efficient. While large models like ResNet-50 perform well, they are too heavy for edge deployment. Lightweight models like MobileNetV2 are efficient but often struggle with fine-grained visual differences in freshness stages. 

**FreshLiteNet** achieves a superior trade-off between accuracy and computational cost by employing a custom lightweight backbone paired with ordinal regression loss mechanisms, resulting in state-of-the-art performance for this task.

## Repository Structure

```text
CAPSTONE79/
├── models/
│   ├── freshlitenet.py       # Proposed FreshLiteNet architecture
│   ├── baselines.py          # ResNet-50 and MobileNetV2 baselines
│   └── __init__.py
├── dataset.py                # Dataloader for 6-class freshness grading
├── dataset_3class.py         # Dataloader for 3-class freshness grading
├── train.py                  # Training pipeline for 6-class task
├── train_3class.py           # Training pipeline for 3-class task
├── evaluate.py               # Evaluation script for 6-class task
├── evaluate_3class.py        # Evaluation script for 3-class task
├── results/                  # Performance metrics (CSV)
└── figures/                  # Generated confusion matrices
```

## Experimental Setup

The model is evaluated on the **FGrade Tomato Dataset** in two separate experimental configurations:

1. **6-Class Ordinal Grading:** The dataset is divided into 6 fine-grained freshness stages. This is a highly challenging task.
2. **3-Class Grading:** A simplified configuration where the dataset is grouped into 3 broad freshness categories (e.g., Fresh, Semi-Fresh, Spoiled).

### Models Evaluated
- **ResNet-50:** A heavy, highly accurate standard baseline.
- **MobileNetV2:** A lightweight, efficient baseline for edge devices.
- **FreshLiteNet (Ours):** The proposed lightweight architecture.

### Metrics
We evaluated the models using several metrics tailored for ordinal classification:
- **Accuracy:** Standard top-1 accuracy.
- **Accuracy (±1):** Top-1 accuracy allowing a tolerance of ±1 ordinal class (used for the 6-class task).
- **F1-Score:** Macro F1-score to handle class imbalances.
- **QWK (Quadratic Weighted Kappa):** Measures agreement between predictions and ground truth, penalizing larger ordinal errors heavily.
- **MAE (Mean Absolute Error):** Measures the average absolute distance between the predicted ordinal class and the ground truth.
- **Inference Time / Parameters:** To measure edge deployment feasibility.

## Results

### 1. 6-Class Fine-Grained Grading

FreshLiteNet significantly outperforms MobileNetV2 and achieves higher off-by-one accuracy and QWK than the heavy ResNet-50, all while being ~6x smaller than ResNet-50.

| Model | Accuracy | Acc (±1) | F1-Score | QWK | MAE | Params (M) | Inference (ms/img) |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 35.00% | 73.33% | 0.3467 | 0.8603 | 1.0778 | 23.53 | 7.60 ms |
| MobileNetV2 | 33.89% | 68.33% | 0.3116 | 0.8155 | 1.2722 | 2.24 | **2.63 ms** |
| **FreshLiteNet (Ours)** | **41.67%** | **86.11%** | **0.4203** | **0.9160** | **0.7833** | 3.85 | 2.99 ms |

### 2. 3-Class Coarse Grading

On the simplified 3-class task, FreshLiteNet achieves the highest accuracy across the board, demonstrating strong feature extraction capabilities even with fewer parameters.

| Model | Accuracy | F1-Score | QWK | MAE | Params (M) | Inference (ms/img) |
|---|---|---|---|---|---|---|
| ResNet-50 | 90.56% | 0.9078 | 0.9091 | 0.1000 | 23.51 | 8.32 ms |
| MobileNetV2 | 86.67% | 0.8702 | 0.8938 | 0.1333 | **2.23** | **3.07 ms** |
| **FreshLiteNet (Ours)** | **92.78%** | **0.9298** | **0.9401** | **0.0722** | 3.85 | 3.09 ms |

## Usage Guide

### 1. Requirements

Ensure you have PyTorch and standard scientific computing libraries installed:
```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn tqdm
```

### 2. Training the Models

To train the models from scratch, simply run the corresponding training script. The script will automatically train all three models (FreshLiteNet, ResNet-50, MobileNetV2) and save the checkpoints in the `checkpoints/` directory.

**For 6-Class Task:**
```bash
python train.py
```

**For 3-Class Task:**
```bash
python train_3class.py
```

### 3. Evaluating the Models

To evaluate the trained models on the test set and generate the comparison metrics and confusion matrices, run:

**For 6-Class Task:**
```bash
python evaluate.py
```

**For 3-Class Task:**
```bash
python evaluate_3class.py
```

The evaluation scripts will output CSV files in the `results/` folder and confusion matrices in the `figures/` folder.

## Future Work
- Explore further model pruning and quantization (e.g., INT8) for deployment on extremely low-power microcontrollers (TinyML).
- Expand the dataset to include other types of fruits and vegetables to test the generalizability of the FreshLiteNet architecture.

## License
MIT License
