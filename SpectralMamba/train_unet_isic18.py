import os
import glob
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ISICDataset(Dataset):
    def __init__(self, img_paths, mask_dir, size=256, is_train=True):
        self.img_paths = img_paths
        self.mask_dir = mask_dir
        self.size = size
        self.is_train = is_train
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = os.path.basename(img_path)
        base = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, base + '_segmentation.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        image = image.resize((self.size, self.size), Image.BILINEAR)
        mask = mask.resize((self.size, self.size), Image.NEAREST)
        
        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            angle = random.uniform(-30, 30)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()
        
        return image, mask

def compute_dice(pred, target):
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset split
    img_dir = 'data/isic18/train/images/'
    mask_dir = 'data/isic18/train/masks/'
    all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    
    # Shuffle and split
    random.shuffle(all_imgs)
    split_idx = int(0.8 * len(all_imgs))
    train_imgs = all_imgs[:split_idx]
    val_imgs = all_imgs[split_idx:]
    
    train_dataset = ISICDataset(train_imgs, mask_dir, size=256, is_train=True)
    val_dataset = ISICDataset(val_imgs, mask_dir, size=256, is_train=False)
    
    print("Dataset initialized")
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    # Model, Loss, Optimizer
    print("Loading model...")
    model = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet', in_channels=3, classes=1).to(device)
    print("Model loaded")
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    os.makedirs('best-ckpt', exist_ok=True)
    best_val_dice = 0.0
    epochs = 100
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                preds = torch.sigmoid(outputs)
                val_dice += compute_dice(preds, masks).item() * images.size(0)
        val_dice /= len(val_loader.dataset)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), 'best-ckpt/best-unet-isic18.pth')
            
        with open('unet_progress.txt', 'a') as f:
            f.write(f"Epoch {epoch:03d}/{epochs} - Train Loss: {train_loss:.4f} - Val Dice: {val_dice:.4f}\n")
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} - Train Loss: {train_loss:.4f} - Val Dice: {val_dice:.4f}")

if __name__ == '__main__':
    main()
