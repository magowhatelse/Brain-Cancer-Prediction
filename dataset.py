import cv2 as cv
from torch.utils.data import Dataset
import numpy as np
import torch
from torchvision import transforms
from PIL import Image

def read_mri(path):
    """
    Read and preprocess MRI image.
    Returns image in format (H, W, C) for compatibility with transforms
    """
    mri = cv.imread(path)
    if mri is None:
        raise ValueError(f"Failed to load image at {path}")
    
    # Convert BGR to RGB
    mri = cv.cvtColor(mri, cv.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    mri = mri.astype(np.float32) / 255.0
    
    return mri

class MRI_dataset(Dataset):
    def __init__(self, dataset, is_training=False):
        self.dataset = dataset
        self.is_training = is_training
        
        # Base transforms that are always applied
        self.base_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(

                # calculated with helper.py get_normalization_parameters()
                mean=[-0.0217, -0.0217, -0.0217],
                std=[0.1669, 0.1669, 0.1669] 
            )
        ])
        
        # Additional augmentations for training
        self.train_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2
            )
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def apply_transforms(self, img):
        """Apply transforms to image based on training/validation mode"""
        # Convert numpy array to PIL Image for transforms
        img = Image.fromarray((img * 255).astype(np.uint8))
        
        # Apply augmentations if in training mode
        if self.is_training:
            img = self.train_transforms(img)
            
        # Apply base transforms (ToTensor and Normalize)
        img = self.base_transforms(img)                       
        return img
    
    def __getitem__(self, idx):
        # Read image
        img = read_mri(self.dataset["Path"].iloc[idx])
        
        # Apply transforms
        img = self.apply_transforms(img)
        
        # Get target
        target = self.dataset["Cancer"].iloc[idx]
        
        return {
            "Name": self.dataset["Name"].iloc[idx],
            "img": img,
            "target": target
        }
