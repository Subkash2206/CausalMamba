import torch
import numpy as np
import os, sys, glob
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion

# Add paths
sys.path.append(os.path.join(os.getcwd(), 'VM-UNet'))
from models.vmunet.vmunet import VMUNet
sys.path.append(os.path.join(os.getcwd(), '..', 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = '../Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml'
        self.opts = None; self.batch_size = 1; self.zip = False; self.cache_mode = 'part'; self.resume = None; self.accumulation_steps = None; self.use_checkpoint = False; self.amp_opt_level = 'O0'; self.tag = 'test'; self.eval = False; self.throughput = False

def compute_bf1(pred, target):
    pred = (pred > 0.5).cpu().numpy().astype(np.uint8).squeeze()
    target = target.cpu().numpy().astype(np.uint8).squeeze()
    def get_boundary(mask):
        eroded = binary_erosion(mask, structure=np.ones((5,5)))
        return mask ^ eroded
    b_pred = get_boundary(pred); b_target = get_boundary(target)
    intersection = (b_pred & b_target).sum()
    precision = intersection / (b_pred.sum() + 1e-5)
    recall = intersection / (b_target.sum() + 1e-5)
    return (2 * precision * recall) / (precision + recall + 1e-5)

def main():
    device = 'cuda'
    vmunet = VMUNet().to(device); vmunet.load_state_dict(torch.load('best-ckpt/best-vmunet-scratch-isic18.pth', map_location=device), strict=False); vmunet.eval()
    args = MockArgs(); config = get_config(args); swin = SwinUnet(config, img_size=224, num_classes=1).to(device); swin.load_state_dict(torch.load('best-ckpt/best-swinunet-isic18.pth', map_location=device)); swin.eval()

    img_paths = sorted(glob.glob('data/isic18/train/images/*.jpg'))
    import random; random.seed(42); random.shuffle(img_paths)
    val_imgs = img_paths[int(0.8*len(img_paths)):]

    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    wins_v, wins_s = 0, 0
    scores_v, scores_s = [], []
    
    print('Analyzing per-image consistency...')
    for p in val_imgs:
        img = Image.open(p).convert('RGB')
        mask_p = p.replace('images','masks').replace('.jpg','_segmentation.png')
        mask = Image.open(mask_p).convert('L')
        with torch.no_grad():
            out_v = vmunet(t256(img).unsqueeze(0).to(device))
            out_s = torch.sigmoid(swin(t224(img).unsqueeze(0).to(device)))
            tg_256 = (torch.from_numpy(np.array(mask.resize((256, 256)))).float().to(device)/255.0 > 0.5).float()
            tg_224 = (torch.from_numpy(np.array(mask.resize((224, 224)))).float().to(device)/255.0 > 0.5).float()
            bf1_v = compute_bf1(out_v, tg_256)
            bf1_s = compute_bf1(out_s, tg_224)
            scores_v.append(bf1_v)
            scores_s.append(bf1_s)
            if bf1_v > bf1_s: wins_v += 1
            else: wins_s += 1
            
    print(f'\nConsistency Report (BF1):')
    print(f'Mamba Wins: {wins_v}/50')
    print(f'Swin  Wins: {wins_s}/50')
    print(f'Mamba Mean: {np.mean(scores_v):.4f} (±{np.std(scores_v):.4f})')
    print(f'Swin  Mean: {np.mean(scores_s):.4f} (±{np.std(scores_s):.4f})')

if __name__ == '__main__':
    main()
