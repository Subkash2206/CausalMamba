"""
train_vmunet_isic18_cvcrecipe.py — ISIC2018 VM-UNet retrain with the CVC recipe,
canonical VSSM implementation.

Why: the existing ISIC VM-UNet was trained with the *repo* VSSM (JIT per-timestep
loop); the CVC VM-UNet used the *canonical* VSSM (vectorized scan). We proved the
two implementations give different outputs for identical weights. To make the SSM
leg a clean, implementation-matched comparison, retrain ISIC with the canonical
implementation and the exact CVC training recipe (from scratch, AdamW 1e-4 wd 1e-4,
BCE + soft-Dice, cosine, best-val-loss, seed 42). Only dataset + normalization
(ImageNet, per the plan) differ from CVC.

Feasibility note: at 256px, micro-batch 2 + accumulation 4 on the 6GB laptop GPU,
the ISIC train split (~2,075 imgs) projects to roughly 3-4 h/epoch -> ~2-3 weeks
for 100 epochs. This run is therefore intended for a cloud GPU (the plan's
fallback). Script is self-contained: `python train_vmunet_isic18_cvcrecipe.py`.

Output: interventions/checkpoints/vmunet_isic_cvcrecipe_best.pth
"""

import sys, os, json, random, time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
import torchvision.transforms.functional as TF

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SPECTRAL = os.path.join(_REPO, 'SpectralMamba')
sys.path.insert(0, _SPECTRAL)

IMG_DIR = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'images')
MASK_DIR = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train', 'masks')
SPLIT_JSON = os.path.join(_REPO, 'interventions', 'results', 'splits', 'isic_split.json')
CKPT_DIR = os.path.join(_REPO, 'interventions', 'checkpoints')
os.makedirs(CKPT_DIR, exist_ok=True)
SAVE_PATH = os.path.join(CKPT_DIR, 'vmunet_isic_cvcrecipe_best.pth')

IMG_SIZE, EPOCHS, LR, SEED = 256, 100, 1e-4, 42
MICRO_BATCH, ACC_STEPS = 2, 4           # effective batch 8, matching CVC recipe
AMP, CHECKPOINTING = True, True


class ISICDataset(Dataset):
    """ISIC2018 with ImageNet normalization; CVC-style flip/rotate augmentation."""
    def __init__(self, names, img_dir, mask_dir, img_size=256, is_train=True):
        self.names, self.img_dir, self.mask_dir = names, img_dir, mask_dir
        self.img_size, self.is_train = img_size, is_train

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        base = os.path.splitext(name)[0]
        img = Image.open(os.path.join(self.img_dir, name)).convert('RGB').resize(
            (self.img_size, self.img_size), Image.BILINEAR)
        mask = Image.open(os.path.join(self.mask_dir, base + '_segmentation.png')).convert('L').resize(
            (self.img_size, self.img_size), Image.NEAREST)
        if self.is_train:
            if random.random() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            if random.random() > 0.5:
                angle = random.uniform(-30, 30)
                img, mask = TF.rotate(img, angle), TF.rotate(mask, angle)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0
        mask = (mask > 0.5).float()
        return img, mask


def soft_dice_loss(probs, masks, smooth=1.0):
    inter = (probs * masks).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + smooth) / (union + smooth)).mean()


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    from models.vmunet.vmunet import VMUNet  # canonical VSSM (vectorized scan)

    with open(SPLIT_JSON) as f:
        split = json.load(f)
    train_ds = ISICDataset(split['train'], IMG_DIR, MASK_DIR, is_train=True)
    val_ds = ISICDataset(split['val'], IMG_DIR, MASK_DIR, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True,
                              num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VMUNet(num_classes=1, input_channels=3,
                   depths=[2, 2, 9, 2], depths_decoder=[2, 9, 2, 2],
                   drop_path_rate=0.2, load_ckpt_path=None,
                   use_checkpoint=CHECKPOINTING).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'VM-UNet (canonical) {n_params:.1f}M params | train={len(train_ds)} '
          f'val={len(val_ds)} | eff-batch {MICRO_BATCH * ACC_STEPS} | {device}')

    bce = nn.BCELoss()

    def combined_loss(preds, masks):
        return bce(preds, masks) + soft_dice_loss(preds, masks)

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=AMP)
    autocast = torch.cuda.amp.autocast(enabled=AMP)

    best_val_loss = float('inf')
    t_start = time.time()
    print(f'Training {EPOCHS} epochs ...')
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, (imgs, masks) in enumerate(train_loader, start=1):
            imgs, masks = imgs.to(device), masks.to(device)
            with autocast:
                probs = model(imgs)  # VMUNet applies sigmoid internally
            loss = combined_loss(probs.float(), masks.float()) / ACC_STEPS
            scaler.scale(loss).backward()
            train_loss += loss.item() * ACC_STEPS
            if batch_idx % ACC_STEPS == 0 or batch_idx == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with autocast:
                    probs = model(imgs)
                val_loss += combined_loss(probs.float(), masks.float()).item()
                pred = (probs > 0.5).float()
                inter = (pred * masks).sum().item()
                union = pred.sum().item() + masks.sum().item()
                val_dice += 2 * inter / max(union, 1e-8)
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        scheduler.step()

        saved = ''
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            saved = ' --> Saved best'
        el = (time.time() - t_start) / 3600
        print(f'Epoch {epoch:03d}/{EPOCHS} | Train {train_loss:.4f} | '
              f'Val {val_loss:.4f} | Dice {val_dice:.4f} | {el:.1f}h{saved}')
    print(f'Done. Best val loss {best_val_loss:.4f} -> {SAVE_PATH}')


if __name__ == '__main__':
    main()
