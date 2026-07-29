import torch
from torch.utils.data import DataLoader
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets.brats_dataset import BraTSDataset

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
BATCH_SIZE = 1
SAVE_PATH = 'checkpoints/segresnet_brats_best.pth'

def main():
    print(f"Initializing 3D SegResNet on {DEVICE}...")
    model = SegResNet(in_channels=4, out_channels=1, init_filters=16).to(DEVICE)
    criterion = DiceCELoss(sigmoid=True)
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    train_loader = DataLoader(BraTSDataset('brats2021', 'train'), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(BraTSDataset('brats2021', 'val'), batch_size=1, shuffle=False, num_workers=0)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            # Bullet-proof 3D Train-Time Augmentation (Flips along H, W, D)
            if torch.rand(1) > 0.5: imgs, masks = torch.flip(imgs, [2]), torch.flip(masks, [2])
            if torch.rand(1) > 0.5: imgs, masks = torch.flip(imgs, [3]), torch.flip(masks, [3])
            if torch.rand(1) > 0.5: imgs, masks = torch.flip(imgs, [4]), torch.flip(masks, [4])
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f'Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}', end="")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(' --> Saved best model')
        else:
            print()

if __name__ == '__main__':
    main()
