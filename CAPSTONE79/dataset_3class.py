import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_image_paths_and_labels(data_dir="data", limit_per_class=None):
    """
    Scans the data directories and returns image paths and labels.
    Both Training_set and Testing_set folders contain subfolders 0 to 9.
    If limit_per_class is set, limits the number of images per class.
    """
    image_paths = []
    labels = []
    
    class_files = {i: [] for i in range(10)}
    
    for split in ['Training_set', 'Testing_set']:
        split_dir = os.path.join(data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for class_idx in range(10):
            class_dir = os.path.join(split_dir, str(class_idx))
            if not os.path.isdir(class_dir):
                continue
            
            # Match standard image file formats
            files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                files.extend(glob.glob(os.path.join(class_dir, ext)))
                
            for f in files:
                class_files[class_idx].append(os.path.abspath(f))
                
    import random
    random.seed(42) # fixed seed for reproducible subsampling
    for class_idx, files in class_files.items():
        # Map original class to Grade A(0), B(1), C(2)
        if class_idx in [0, 1, 2]:
            mapped_label = 0
        elif class_idx in [3, 4, 5, 6]:
            mapped_label = 1
        else:
            mapped_label = 2
            
        if limit_per_class is not None:
            random.shuffle(files)
            files = files[:limit_per_class]
        
        for f in files:
            image_paths.append(f)
            labels.append(mapped_label)
                
    return image_paths, labels

def get_stratified_splits(data_dir="data", random_state=42, limit_per_class=None):
    """
    Combines the dataset and returns stratified 70% Train, 15% Val, 15% Test splits.
    """
    image_paths, labels = get_image_paths_and_labels(data_dir, limit_per_class=limit_per_class)
    
    if len(image_paths) == 0:
        raise ValueError(f"No image files found in {data_dir} directory.")
        
    # Split: 70% Train, 30% Temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.30, random_state=random_state, stratify=labels
    )
    
    # Split Temp: 50% Val, 50% Test (which is 15% and 15% of total)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=random_state, stratify=temp_labels
    )
    
    return {
        'train': (train_paths, train_labels),
        'val': (val_paths, val_labels),
        'test': (test_paths, test_labels)
    }

def get_train_transforms():
    """
    Augmentations designed for real-world tomato freshness grading:
    - Random Rotation / Horizontal Flips: Handle variable tomato orientations.
    - Random Brightness / Contrast: Simulate various lighting environments.
    - Gaussian Noise / Motion Blur: Simulate low-cost cameras and hand shakes.
    - JPEG Compression: Simulate quality loss during transmission.
    - Random Resized Crop: Adapt to variable distance of tomato from the camera.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.ImageCompression(quality_range=(70, 95), p=1.0),
        ], p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms():
    """
    Validation and test set transformations: standard resize and normalization.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

class TomatoDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, cache=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache = cache
        self.cached_images = []
        
        if self.cache:
            print(f"Pre-loading and caching {len(image_paths)} images in RAM...")
            for i, path in enumerate(image_paths):
                image = cv2.imread(path)
                if image is None:
                    raise FileNotFoundError(f"Failed to read image at: {path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (224, 224))
                self.cached_images.append(image)
                if (i + 1) % 1000 == 0 or (i + 1) == len(image_paths):
                    print(f"  Cached {i + 1}/{len(image_paths)} images...")
            print(f"Finished caching {len(image_paths)} images in memory.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.cache:
            image = self.cached_images[idx]
        else:
            path = self.image_paths[idx]
            image = cv2.imread(path)
            if image is None:
                raise FileNotFoundError(f"Failed to read image at: {path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (224, 224))

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, label

if __name__ == "__main__":
    # Small test module
    print("Testing Dataset pipeline...")
    try:
        splits = get_stratified_splits()
        print(f"Stratification complete:")
        for name, (paths, labels) in splits.items():
            print(f"  {name.capitalize()} split size: {len(paths)}")
            # Print per-class counts
            unique, counts = np.unique(labels, return_counts=True)
            print(f"  {name.capitalize()} class distribution: {dict(zip(unique, counts))}")
            
        # Try loading a sample
        train_paths, train_labels = splits['train']
        dataset = TomatoDataset(train_paths, train_labels, transform=get_train_transforms())
        img, lbl = dataset[0]
        print(f"Dataset item loaded successfully. Image shape: {img.shape}, Label: {lbl}")
    except Exception as e:
        print(f"Error during dataset testing: {e}")
