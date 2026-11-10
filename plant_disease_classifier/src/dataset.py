"""
Dataset and DataLoader for Plant Disease Classification.
Supports ImageFolder structure with albumentations augmentations.
"""

import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


class PlantDiseaseDataset(Dataset):
    """Custom dataset for plant disease images with albumentations."""
    
    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Get classes from directory structure
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        
        if class_to_idx is None:
            self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.class_to_idx = class_to_idx
        
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Collect all image paths and labels
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        self.targets = [s[1] for s in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Read image with OpenCV (BGR) and convert to RGB
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, label
    
    def get_class_weights(self):
        """Compute class weights for imbalanced dataset."""
        class_counts = Counter(self.targets)
        total = len(self.targets)
        num_classes = len(self.classes)
        weights = [total / (num_classes * class_counts[i]) for i in range(num_classes)]
        return torch.FloatTensor(weights)


def get_transforms(config, is_train=True):
    """Create albumentations transforms from config."""
    img_size = config['data']['image_size']
    mean = config['data']['mean']
    std = config['data']['std']
    
    if is_train:
        aug = config['data']['augmentation']
        transforms = [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=aug.get('horizontal_flip', 0.5)),
            A.VerticalFlip(p=aug.get('vertical_flip', 0.2)),
            A.Rotate(limit=aug.get('rotation', 30), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=aug.get('brightness', 0.2),
                contrast_limit=aug.get('contrast', 0.2),
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=int(aug.get('hue', 0.1) * 180),
                sat_shift_limit=int(aug.get('saturation', 0.2) * 255),
                val_shift_limit=0,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=aug.get('cutout_holes', 8),
                max_height=aug.get('cutout_size', 16),
                max_width=aug.get('cutout_size', 16),
                p=0.5 if aug.get('cutout', True) else 0
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(img_size, img_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms)


def get_tta_transforms(config):
    """Test-time augmentation transforms."""
    img_size = config['data']['image_size']
    mean = config['data']['mean']
    std = config['data']['std']
    
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def create_dataloaders(config):
    """Create train, val, test dataloaders."""
    train_transform = get_transforms(config, is_train=True)
    val_transform = get_transforms(config, is_train=False)
    
    # Load datasets
    train_dataset = PlantDiseaseDataset(
        config['data']['train_dir'], 
        transform=train_transform
    )
    
    val_dataset = PlantDiseaseDataset(
        config['data']['val_dir'], 
        transform=val_transform,
        class_to_idx=train_dataset.class_to_idx
    )
    
    test_dataset = None
    if os.path.exists(config['data']['test_dir']):
        test_dataset = PlantDiseaseDataset(
            config['data']['test_dir'], 
            transform=val_transform,
            class_to_idx=train_dataset.class_to_idx
        )
    
    # Class weights for imbalanced data
    class_weights = train_dataset.get_class_weights()
    
    # Weighted sampler for training
    sample_weights = [class_weights[label] for _, label in train_dataset.samples]
    sampler = WeightedRandomSampler(
        sample_weights, 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = None
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['batch_size'],
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory']
        )
    
    return train_loader, val_loader, test_loader, train_dataset.classes, class_weights


def prepare_dataset_from_split(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Split a flat directory of class folders into train/val/test."""
    import shutil
    from pathlib import Path
    
    np.random.seed(seed)
    classes = sorted([d for d in os.listdir(source_dir) 
                     if os.path.isdir(os.path.join(source_dir, d))])
    
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)
    
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        np.random.shuffle(images)
        
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train + n_val],
            'test': images[n_train + n_val:],
        }
        
        for split_name, split_images in splits.items():
            for img in split_images:
                src = os.path.join(cls_dir, img)
                dst = os.path.join(output_dir, split_name, cls, img)
                shutil.copy2(src, dst)
    
    print(f"Dataset prepared at {output_dir}")
    for split in ['train', 'val', 'test']:
        total = sum(len(os.listdir(os.path.join(output_dir, split, c))) for c in classes)
        print(f"  {split}: {total} images")