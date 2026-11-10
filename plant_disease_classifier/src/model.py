"""
Model architecture for Plant Disease Classification.
Supports multiple backbones via timm.
"""

import torch
import torch.nn as nn
import timm
from torchvision import models


def create_model(config):
    """Create a model based on config."""
    model_name = config['model']['name']
    num_classes = config['model']['num_classes']
    pretrained = config['model']['pretrained']
    dropout = config['model']['dropout']
    freeze_backbone = config['model']['freeze_backbone']
    
    # Use timm for most models
    if model_name.startswith(('resnet', 'efficientnet', 'vit', 'convnext', 'densenet')):
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    if freeze_backbone:
        # Freeze all layers except the classifier
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze the classifier
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif hasattr(model, 'head'):
            for param in model.head.parameters():
                param.requires_grad = True
    
    return model


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, epoch, loss, accuracy, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint