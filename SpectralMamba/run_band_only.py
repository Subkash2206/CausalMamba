import sys
import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import pearsonr

# Set clean academic style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 12,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

COLORS = {'UNet': '#1f77b4', 'Swin': '#2ca02c', 'Mamba': '#ff7f0e'}
BANDS = {'Low': '#4c72b0', 'Mid': '#dd8452', 'High': '#c44e52'}

# Add paths for models
sys.path.append(os.getcwd())
from models.vmunet.vmunet import VMUNet
sys.path.append(os.path.join(os.getcwd(), 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

class MockArgs:
    def __init__(self):
        self.cfg = 'Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml'
        self.opts = None; self.batch_size = 1; self.zip = False; self.cache_mode = 'part'
        self.resume = None; self.accumulation_steps = None; self.use_checkpoint = False
        self.amp_opt_level = 'O0'; self.tag = 'test'; self.eval = False; self.throughput = False

def flexible_load(model, ckpt_path):
    print(f"Loading weights from {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'model' in state_dict: state_dict = state_dict['model']
    model_dict = model.state_dict()
    has_vmunet_prefix = any(k.startswith('vmunet.') for k in state_dict.keys())
    model_has_vmunet_prefix = any(k.startswith('vmunet.') for k in model_dict.keys())
    clean_state_dict = {k: v for k, v in state_dict.items() if not k.endswith('total_ops') and not k.endswith('total_params')}
    new_state_dict = {}
    for k, v in clean_state_dict.items():
        new_k = k
        if has_vmunet_prefix and not model_has_vmunet_prefix: new_k = k.replace('vmunet.', '')
        elif not has_vmunet_prefix and model_has_vmunet_prefix: new_k = 'vmunet.' + k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict, strict=True)
    print("  SUCCESS: Model loaded with strict=True.")
    return model

def compute_bands(fmap):
    fmap = fmap.cpu().float()
    fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_shifted) ** 2
    
    cy, cx = H // 2, W // 2
    y = torch.arange(H).view(1, 1, H, 1)
    x = torch.arange(W).view(1, 1, 1, W)
    
    dist_y = torch.abs(y - cy) / (H / 2)
    dist_x = torch.abs(x - cx) / (W / 2)
    freq_ratio = torch.max(dist_y.expand(B, C, H, W), dist_x.expand(B, C, H, W))
    
    mask_low = freq_ratio <= 0.25
    mask_mid = (freq_ratio > 0.25) & (freq_ratio <= 0.75)
    mask_high = freq_ratio > 0.75
    
    total = power.sum().item()
    if total == 0: return 0., 0., 0., 0.
    
    low = (power * mask_low).sum().item() / total
    mid = (power * mask_mid).sum().item() / total
    high = (power * mask_high).sum().item() / total
    
    # Standard AVR definition (freq > 0.5)
    avr_mask = freq_ratio > 0.5
    avr = (power * avr_mask).sum().item() / total
    
    return low, mid, high, avr

def compute_boundary_f1(pred, gt, iterations=2):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    pred_eroded = binary_erosion(pred, iterations=iterations)
    gt_eroded = binary_erosion(gt, iterations=iterations)
    pred_bound = pred & ~pred_eroded
    gt_bound = gt & ~gt_eroded
    tp = (pred_bound & gt_bound).sum()
    fp = (pred_bound & ~gt_bound).sum()
    fn = (~pred_bound & gt_bound).sum()
    if tp + fp + fn == 0: return 1.0
    return 2.0 * tp / (2.0 * tp + fp + fn)

def get_fft_power(fmap):
    fmap = fmap.cpu().float().mean(dim=1, keepdim=True) # average across channels
    fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)
    fft = torch.fft.fft2(fmap)
    fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_shifted)**2
    return power.squeeze().numpy()

def calculate_iou(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    target = (torch.sigmoid(target) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    if union == 0: return 1.0
    return (intersection / union).item()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Starting visualization generation on {device}...')
    
    out_dir = 'results/figures/'
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load models
    ckpt_dir = 'VM-UNet/best-ckpt/'
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    unet = flexible_load(smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device), os.path.join(ckpt_dir, 'best-unet-isic18.pth')).eval()
    
    args = MockArgs(); config = get_config(args); swin = flexible_load(SwinUnet(config, img_size=224, num_classes=1).to(device), os.path.join(ckpt_dir, 'best-swinunet-isic18.pth')).eval()
    
    vmunet = flexible_load(VMUNet().to(device), os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth')).eval()

    features = defaultdict(dict)
    def get_hook(model_name, level):
        def hook(module, input, output):
            if isinstance(output, tuple): features[model_name][level] = output[0].detach()
            else: features[model_name][level] = output.detach()
        return hook

    for i in range(1, 5):
        getattr(unet.encoder, f'layer{i}').register_forward_hook(get_hook('UNet', i))
        swin.swin_unet.layers[i-1].blocks[-1].register_forward_hook(get_hook('Swin', i))
        vmunet.vmunet.layers[i-1].blocks[-1].register_forward_hook(get_hook('Mamba', i))

    img_dir = 'VM-UNet/data/isic18/train/images/'
    mask_dir = 'VM-UNet/data/isic18/train/masks/'
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) + glob.glob(os.path.join(img_dir, '*.png')))
    import random; random.seed(42); random.shuffle(img_paths)
    split_idx = int(0.8 * len(img_paths))
    val_imgs = img_paths[split_idx:] # Full validation set

    # Data collection
    band_data = defaultdict(lambda: defaultdict(list))
    avr_data = defaultdict(lambda: defaultdict(list))
    bf1_data = defaultdict(list)
    shift_data = defaultdict(lambda: defaultdict(list))
    fft_images = defaultdict(dict) # For Image 0

    print("Running inference to collect full raw data points...")
    for idx, img_path in enumerate(val_imgs):
        base = os.path.splitext(os.path.basename(img_path))[0]
        mask_path = os.path.join(mask_dir, base + '_segmentation.png')
        
        img_pil = Image.open(img_path).convert('RGB')
        mask_pil = Image.open(mask_path).convert('L')
        
        i256 = t256(img_pil).unsqueeze(0).to(device)
        i224 = t224(img_pil).unsqueeze(0).to(device)
        gt_256 = np.array(mask_pil.resize((256, 256), Image.NEAREST)) > 127
        gt_224 = np.array(mask_pil.resize((224, 224), Image.NEAREST)) > 127

        with torch.no_grad():
            features.clear()
            base_unet = unet(i256)
            base_swin = swin(i224)
            base_mamba = vmunet(i256)
            
            # BF1
            bf1_data['UNet'].append(compute_boundary_f1((torch.sigmoid(base_unet).squeeze().cpu().numpy() > 0.5), gt_256))
            bf1_data['Swin'].append(compute_boundary_f1((torch.sigmoid(base_swin).squeeze().cpu().numpy() > 0.5), gt_224))
            bf1_data['Mamba'].append(compute_boundary_f1((torch.sigmoid(base_mamba).squeeze().cpu().numpy() > 0.5), gt_256))

            # Spectra
            for model_name in ['UNet', 'Swin', 'Mamba']:
                img_avrs = []
                for level in range(1, 5):
                    f = features[model_name][level]
                    if model_name == 'Swin':
                        if f.dim() == 3:
                            B, L, C = f.shape; H = W = int(np.sqrt(L)); f = f.transpose(1, 2).reshape(B, C, H, W)
                        elif f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]:
                            f = f.permute(0, 3, 1, 2)
                    elif model_name == 'Mamba':
                        if f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]:
                            f = f.permute(0, 3, 1, 2)
                    
                    if idx == 0:
                        fft_images[model_name][level] = get_fft_power(f)
                        
                    l, m, h, a = compute_bands(f)
                    band_data[model_name][level].append((l, m, h))
                    img_avrs.append(a)
                    avr_data[model_name][level].append(a)
                avr_data[model_name]['mean'].append(np.mean(img_avrs))

            # Shifts
            for s in range(1, 6):
                shift_256 = torch.roll(i256, shifts=s, dims=-1)
                shift_224 = torch.roll(i224, shifts=s, dims=-1)
                
                pu = unet(shift_256); ps = swin(shift_224); pm = vmunet(shift_256)
                pu_u = torch.roll(pu, shifts=-s, dims=-1)
                ps_u = torch.roll(ps, shifts=-s, dims=-1)
                pm_u = torch.roll(pm, shifts=-s, dims=-1)
                
                shift_data['UNet'][s].append(calculate_iou(base_unet, pu_u))
                shift_data['Swin'][s].append(calculate_iou(base_swin, ps_u))
                shift_data['Mamba'][s].append(calculate_iou(base_mamba, pm_u))

    # Plot 1: Band Decomposition
    print("Generating Figure 1: Band Decomposition...")
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['UNet', 'Swin', 'Mamba']
    levels = [1, 2, 3, 4]
    x = np.arange(len(levels))
    width = 0.25
    for i, model in enumerate(models):
        lows = [np.mean([x[0] for x in band_data[model][l]]) for l in levels]
        mids = [np.mean([x[1] for x in band_data[model][l]]) for l in levels]
        highs = [np.mean([x[2] for x in band_data[model][l]]) for l in levels]
        
        pos = x + (i - 1) * width
        ax.bar(pos, lows, width, label='Low' if i==0 else "", color=BANDS['Low'], edgecolor='white')
        ax.bar(pos, mids, width, bottom=lows, label='Mid' if i==0 else "", color=BANDS['Mid'], edgecolor='white')
        ax.bar(pos, highs, width, bottom=np.array(lows)+np.array(mids), label='High' if i==0 else "", color=BANDS['High'], edgecolor='white')
        
        # Add architecture labels (U, S, M) above each bar
        short_names = {'UNet': 'U', 'Swin': 'S', 'Mamba': 'M'}
        for p in pos:
            ax.text(p, 1.02, short_names[model], ha='center', va='bottom', fontsize=10, fontweight='bold')
            
    ax.set_xticks(x)
    ax.set_xticklabels(['Level 1\n(~64x64)', 'Level 2\n(~32x32)', 'Level 3\n(~16x16)', 'Level 4\n(~8x8)'])
    ax.set_ylabel('Energy Ratio')
    ax.set_ylim(0, 1.1) # Room for labels
    ax.set_title('Frequency Band Decomposition (U: UNet, S: Swin, M: Mamba)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'band_decomposition.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, 'band_decomposition.pdf'))
    plt.close()

    # Save band decomposition raw numbers to CSV
    print("Saving Figure 1 data to band_decomposition_results.csv...")
    csv_rows = []
    res_names = {1: '64x64', 2: '32x32', 3: '16x16', 4: '8x8'}
    for model in models:
        for l in levels:
            lows = [x[0] for x in band_data[model][l]]
            mids = [x[1] for x in band_data[model][l]]
            highs = [x[2] for x in band_data[model][l]]
            csv_rows.append(f"{model},{l},{res_names[l]},{np.mean(lows)},{np.mean(mids)},{np.mean(highs)},{np.std(lows)},{np.std(mids)},{np.std(highs)}")
    
    band_csv_path = 'VM-UNet/results/band_decomposition_results.csv'
    os.makedirs(os.path.dirname(band_csv_path), exist_ok=True)
    with open(band_csv_path, 'w') as f:
        f.write('model,resolution_level,resolution,low_band_ratio,mid_band_ratio,high_band_ratio,low_std,mid_std,high_std\n')
        f.write('\n'.join(csv_rows) + '\n')

    print(f"Band decomposition data successfully saved to {band_csv_path}")

if __name__ == '__main__':
    main()
