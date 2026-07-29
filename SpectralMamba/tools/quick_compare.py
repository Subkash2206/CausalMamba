import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion

# Import VM-UNet
sys.path.append(os.path.join(os.getcwd(), 'VM-UNet'))
from models.vmunet.vmunet import VMUNet

# Import Swin-Unet
sys.path.append(os.path.join(os.getcwd(), '..', 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = '../Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml'
        self.opts = None
        self.batch_size = 8
        self.zip = False
        self.cache_mode = 'part'
        self.resume = None
        self.accumulation_steps = None
        self.use_checkpoint = False
        self.amp_opt_level = 'O0'
        self.tag = 'test'
        self.eval = False
        self.throughput = False

def compute_dice(pred, target):
    smooth = 1e-5
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def compute_bf1(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(np.uint8).squeeze()
    target = target.cpu().numpy().astype(np.uint8).squeeze()
    
    # Erode to get boundaries
    def get_boundary(mask):
        eroded = binary_erosion(mask, structure=np.ones((5,5)))
        return mask ^ eroded
    
    b_pred = get_boundary(pred)
    b_target = get_boundary(target)
    
    intersection = (b_pred & b_target).sum()
    precision = intersection / (b_pred.sum() + 1e-5)
    recall = intersection / (b_target.sum() + 1e-5)
    
    return (2 * precision * recall) / (precision + recall + 1e-5)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. VM-UNet (Mamba)
    vmunet = VMUNet().to(device)
    ckpt_v = torch.load('best-ckpt/best-vmunet-scratch-isic18.pth', map_location=device)
    v_state = ckpt_v['model'] if 'model' in ckpt_v else ckpt_v
    vmunet.load_state_dict(v_state, strict=True)
    vmunet.eval()
    
    # 2. UNet (Final trained)
    unet = smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device)
    unet.load_state_dict(torch.load('best-ckpt/best-unet-isic18.pth', map_location=device))
    unet.eval()

    # 3. Swin-UNet (Current best)
    args = MockArgs()
    config = get_config(args)
    swin = SwinUnet(config, img_size=224, num_classes=1).to(device)
    swin.load_state_dict(torch.load('best-ckpt/best-swinunet-isic18.pth', map_location=device))
    swin.eval()
    
    # Data
    img_dir = 'data/isic18/train/images/'
    mask_dir = 'data/isic18/train/masks/'
    all_imgs = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    
    # Use SAME shuffle and split as training scripts
    import random
    random.seed(42)
    random.shuffle(all_imgs)
    split_idx = int(0.8 * len(all_imgs))
    val_imgs = all_imgs[split_idx:] # Take Val Set
    
    transform_256 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_224 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    with torch.no_grad():
        for img_path in val_imgs:
            img_name = os.path.basename(img_path)
            mask_path = os.path.join(mask_dir, os.path.splitext(img_name)[0] + '_segmentation.png')
            
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Input Tensors
            in_256 = transform_256(image).unsqueeze(0).to(device)
            in_224 = transform_224(image).unsqueeze(0).to(device)
            
            # Targets
            tg_256 = (torch.from_numpy(np.array(mask.resize((256, 256)))).float().to(device) / 255.0 > 0.5).float()
            tg_224 = (torch.from_numpy(np.array(mask.resize((224, 224)))).float().to(device) / 255.0 > 0.5).float()
            
            # Predictions
            out_v = vmunet(in_256) # model already applies sigmoid
            out_u = torch.sigmoid(unet(in_256))
            out_s = torch.sigmoid(swin(in_224))
            
            results.append({
                'v_dice': compute_dice(out_v, tg_256).item(),
                'v_bf1': compute_bf1(out_v, tg_256),
                'u_dice': compute_dice(out_u, tg_256).item(),
                'u_bf1': compute_bf1(out_u, tg_256),
                's_dice': compute_dice(out_s, tg_224).item(),
                's_bf1': compute_bf1(out_s, tg_224)
            })
            
    v_dice = np.mean([r['v_dice'] for r in results])
    v_bf1 = np.mean([r['v_bf1'] for r in results])
    u_dice = np.mean([r['u_dice'] for r in results])
    u_bf1 = np.mean([r['u_bf1'] for r in results])
    s_dice = np.mean([r['s_dice'] for r in results])
    s_bf1 = np.mean([r['s_bf1'] for r in results])
    
    print(f"Comparison (50 Images):")
    print(f"VM-UNet (Pretrained): Dice={v_dice:.4f}, BF1={v_bf1:.4f}")
    print(f"UNet (Final):         Dice={u_dice:.4f}, BF1={u_bf1:.4f}")
    print(f"Swin-UNet (Current):  Dice={s_dice:.4f}, BF1={s_bf1:.4f}")

if __name__ == '__main__':
    main()
