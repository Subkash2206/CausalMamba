"""UNet-ResNet50 retrain on CVC-ClinicDB at 256x256 (paper-consistent with ISIC)."""
import sys, os, torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # tta_boundary_study
sys.path.insert(0, _ROOT)

from src.datasets.cvc_dataset import CVCDataset
from src.models.unet import get_unet

IMG_SIZE, EPOCHS, BATCH_SIZE, LR, SEED = 256, 100, 8, 1e-4, 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = os.path.join(_ROOT, 'checkpoints', 'unet_cvc_best_256.pth')


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_ds = CVCDataset(os.path.join(_ROOT, 'cvc_clinicdb', 'original'),
                          os.path.join(_ROOT, 'cvc_clinicdb', 'ground_truth'),
                          split='train', img_size=IMG_SIZE)
    val_ds = CVCDataset(os.path.join(_ROOT, 'cvc_clinicdb', 'original'),
                        os.path.join(_ROOT, 'cvc_clinicdb', 'ground_truth'),
                        split='val', img_size=IMG_SIZE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    model = get_unet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    best_val_loss = float('inf')
    print(f"Training UNet-ResNet50 CVC at {IMG_SIZE}x{IMG_SIZE} on {DEVICE} -> {SAVE_PATH}")
    for epoch in range(EPOCHS):
        model.train(); train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad(); loss = criterion(model(imgs), masks)
            loss.backward(); optimizer.step(); train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval(); val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                val_loss += criterion(model(imgs), masks).item()
        val_loss /= len(val_loader); scheduler.step()
        ok = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH); ok = ' --> saved best'
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train {train_loss:.4f} | Val {val_loss:.4f}{ok}")


if __name__ == '__main__':
    main()