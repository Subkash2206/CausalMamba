import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import albumentations as A

class CVCDataset(Dataset):
    def __init__(self, img_dir, mask_dir, split='train', img_size=352, seed=42):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.split = split
        
        all_imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.tif', '.tiff'))])
        if not all_imgs:
            raise ValueError(f"No TIFF images found in {img_dir}. Please ensure the files have been placed.")
        
        train_imgs, val_imgs = train_test_split(all_imgs, test_size=0.2, random_state=seed)
        self.images = train_imgs if split == 'train' else val_imgs
        
        # Apply Flips, Rotations, and mild color jitter only during Training
        if self.split == 'train':
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, p=0.5)
            ])
        # Validation strictly resizes
        else:
            self.transform = A.Compose([
                A.Resize(self.img_size, self.img_size)
            ])
            
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply albumentations transforms
        augmented = self.transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        
        # Normalize image to [0,1]
        img = img.astype(np.float32) / 255.0
        
        # Binarize mask SAFELY after augmentations to prevent interpolation artifacts
        mask = (mask > 127).astype(np.float32)
        
        img_tensor = torch.FloatTensor(img).permute(2, 0, 1)
        mask_tensor = torch.FloatTensor(mask).unsqueeze(0)
        
        return img_tensor, mask_tensor
