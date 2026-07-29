import torch
from src.datasets.brats_dataset import BraTSDataset
from monai.networks.nets import SegResNet
from monai.losses import DiceCELoss

print("1. Loading one patient...")
ds = BraTSDataset('brats2021', split='train')
img, mask = ds[0]
print(f"Image shape: {img.shape} | Mask shape: {mask.shape}")

print("\n2. Pushing to GPU...")
img = img.unsqueeze(0).cuda()
mask = mask.unsqueeze(0).cuda()
model = SegResNet(in_channels=4, out_channels=1, init_filters=16).cuda()
criterion = DiceCELoss(sigmoid=True)

print("\n3. Testing Forward/Backward pass...")
out = model(img)
loss = criterion(out, mask)
loss.backward()

print(f"Loss computed: {loss.item():.4f}")
print(f"Max VRAM used: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
print("\n DRY RUN SUCCESSFUL. YOU ARE CLEARED FOR 24-HOUR TRAINING!")
