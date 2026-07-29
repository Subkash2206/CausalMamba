import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.datasets.cvc_dataset import CVCDataset
from src.models.swin_unetr_cvc import get_swin_unetr

IMG_DIR = 'cvc_clinicdb/original'
MASK_DIR = 'cvc_clinicdb/ground_truth'
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_PATH = 'checkpoints/swinunetr_cvc_best.pth'

def main():
    train_ds = CVCDataset(IMG_DIR, MASK_DIR, split='train')
    val_ds = CVCDataset(IMG_DIR, MASK_DIR, split='val')
    
    # num_workers=4 as per manual, but can be set to 0 if Windows multiprocessing complains
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    model = get_swin_unetr().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    best_val_loss = float('inf')
    
    print(f"Starting training on {DEVICE}...")
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss = 0.0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        scheduler.step()
        
        print(f"Epoch {epoch+1:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}", end="")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print(" --> Saved best model")
        else:
            print()

if __name__ == '__main__':
    main()
