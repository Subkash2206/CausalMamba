"""
Regenerate ONLY the avr_bf1_scatter figure using corrected BF1 values.
No model inference is performed (Actually, this script DOES perform inference to get hooks).
"""
import sys, os, glob, torch, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.ndimage import binary_erosion, distance_transform_edt
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import segmentation_models_pytorch as smp

# Add paths for models
ROOT = os.getcwd()
sys.path.append(ROOT)
from models.vmunet.vmunet import VMUNet
sys.path.append(os.path.join(ROOT, 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

plt.style.use('default')
plt.rcParams.update({
    'font.size': 12, 'axes.grid': False,
    'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42
})
COLORS = {'UNet': '#1f77b4', 'Swin': '#2ca02c', 'Mamba': '#ff7f0e'}

class MockArgs:
    def __init__(self):
        self.cfg = os.path.join(ROOT, 'Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml')
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

def compute_avr(fmap):
    fmap = fmap.cpu().float()
    fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)
    B, C, H, W = fmap.shape
    fft = torch.fft.fft2(fmap); fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1)); power = torch.abs(fft_shifted) ** 2
    cy, cx = H // 2, W // 2; y = torch.arange(H).view(1, 1, H, 1); x = torch.arange(W).view(1, 1, 1, W)
    mask = (torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4); mask = mask.expand(B, C, H, W)
    return (power * mask).sum().item() / power.sum().item() if power.sum() > 0 else 0.0

def compute_boundary_f1(pred, gt, tolerance=2):
    # Cast both pred and gt to boolean arrays
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    
    # Extract strictly 1-pixel thick boundaries
    pred_boundary = pred ^ binary_erosion(pred, iterations=1)
    gt_boundary = gt ^ binary_erosion(gt, iterations=1)
    
    total_pred_boundary_pixels = pred_boundary.sum()
    total_gt_boundary_pixels = gt_boundary.sum()
    
    # Safety Check
    if total_pred_boundary_pixels == 0 and total_gt_boundary_pixels == 0:
        return 1.0
    if total_pred_boundary_pixels == 0 or total_gt_boundary_pixels == 0:
        return 0.0
        
    # Calculate distance maps (invert boundaries so they equal 0)
    pred_dist_map = distance_transform_edt(~pred_boundary)
    gt_dist_map = distance_transform_edt(~gt_boundary)
    
    # Calculate True Positives with tolerance
    tp_pred = (pred_boundary & (gt_dist_map <= tolerance)).sum()
    tp_gt = (gt_boundary & (pred_dist_map <= tolerance)).sum()
    
    # Calculate Precision and Recall
    precision = tp_pred / total_pred_boundary_pixels
    recall = tp_gt / total_gt_boundary_pixels
    
    # Return standard F1 score
    if precision + recall == 0:
        return 0.0
        
    return 2.0 * precision * recall / (precision + recall)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Regenerating AVR-BF1 scatter on {device}...')
    
    ckpt_dir = os.path.join(ROOT, 'VM-UNet/best-ckpt/')
    t256 = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    t224 = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    print("Loading models...")
    unet = flexible_load(smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device), os.path.join(ckpt_dir, 'best-unet-isic18.pth')).eval()
    args = MockArgs(); config = get_config(args); swin = flexible_load(SwinUnet(config, img_size=224, num_classes=1).to(device), os.path.join(ckpt_dir, 'best-swinunet-isic18.pth')).eval()
    # FIX: explicit depths=[2, 2, 9, 2] to match trained checkpoint
    vmunet = VMUNet(
        input_channels=3,
        num_classes=1,
        depths=[2, 2, 9, 2],
        depths_decoder=[2, 9, 2, 2]
    ).to(device)
    vmunet = flexible_load(vmunet, os.path.join(ckpt_dir, 'best-vmunet-scratch-isic18.pth')).eval()

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
    val_imgs = img_paths[int(0.8*len(img_paths)):]

    avr_data = defaultdict(list)
    bf1_data = defaultdict(list)

    print(f'Computing AVR and BF1 for {len(val_imgs)} images...')
    for idx, img_path in enumerate(val_imgs):
        if idx % 10 == 0: print(f"  Image {idx}/50...")
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
            out_unet = unet(i256); out_swin = swin(i224); out_mamba = vmunet(i256)
            pred_unet = (torch.sigmoid(out_unet).squeeze().cpu().numpy() > 0.5)
            pred_swin = (torch.sigmoid(out_swin).squeeze().cpu().numpy() > 0.5)
            pred_mamba = (out_mamba.squeeze().cpu().numpy() > 0.5)
            
            bf1_data['UNet'].append(compute_boundary_f1(pred_unet, gt_256))
            bf1_data['Swin'].append(compute_boundary_f1(pred_swin, gt_224))
            bf1_data['Mamba'].append(compute_boundary_f1(pred_mamba, gt_256))

            for model_name in ['UNet', 'Swin', 'Mamba']:
                avrs = []
                for i in range(1, 5):
                    f = features[model_name][i]
                    if model_name == 'Swin':
                        if f.dim() == 3: B, L, C = f.shape; H = W = int(np.sqrt(L)); f = f.transpose(1, 2).reshape(B, C, H, W)
                        elif f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]: f = f.permute(0, 3, 1, 2)
                    elif model_name == 'Mamba':
                        if f.dim() == 4 and f.shape[-1] in [96, 192, 384, 768]: f = f.permute(0, 3, 1, 2)
                    avrs.append(compute_avr(f))
                avr_data[model_name].append(np.mean(avrs))

    # Generate the scatter plot
    print("Generating avr_bf1_scatter...")
    models = ['UNet', 'Swin', 'Mamba']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    all_x, all_y = [], []
    
    for i, model in enumerate(models):
        ax = axes[i]
        x_pts = np.array(avr_data[model]); y_pts = np.array(bf1_data[model])
        all_x.extend(x_pts); all_y.extend(y_pts)
        ax.scatter(x_pts, y_pts, color=COLORS[model], alpha=0.6)
        m, b = np.polyfit(x_pts, y_pts, 1); r, p = pearsonr(x_pts, y_pts)
        x_line = np.linspace(min(x_pts), max(x_pts), 100); y_line = m * x_line + b
        ax.plot(x_line, y_line, color='black', linewidth=2)
        n = len(x_pts); y_hat = m * x_pts + b; std_err = np.sqrt(np.sum((y_pts - y_hat)**2) / (n - 2))
        margin = 1.96 * std_err * np.sqrt(1/n + (x_line - np.mean(x_pts))**2 / np.sum((x_pts - np.mean(x_pts))**2))
        ax.fill_between(x_line, y_line - margin, y_line + margin, color='black', alpha=0.1)
        ax.set_title(f'{model} (r={r:.2f}, p={p:.2e})'); ax.set_xlabel('Mean AVR'); ax.set_ylabel('Boundary F1')

    ax = axes[3]
    for model in models: ax.scatter(avr_data[model], bf1_data[model], color=COLORS[model], alpha=0.6, label=model)
    all_x = np.array(all_x); all_y = np.array(all_y); m, b = np.polyfit(all_x, all_y, 1); r, p = pearsonr(all_x, all_y)
    x_line = np.linspace(min(all_x), max(all_x), 100); y_line = m * x_line + b
    n = len(all_x); y_hat = m * all_x + b; std_err = np.sqrt(np.sum((all_y - y_hat)**2) / (n - 2))
    margin = 1.96 * std_err * np.sqrt(1/n + (x_line - np.mean(all_x))**2 / np.sum((all_x - np.mean(all_x))**2))
    ax.fill_between(x_line, y_line - margin, y_line + margin, color='black', alpha=0.1)
    ax.plot(x_line, y_line, color='black', linewidth=2)
    ax.set_title(f'Pooled (r={r:.2f}, p={p:.2e})'); ax.set_xlabel('Mean AVR'); ax.set_ylabel('Boundary F1'); ax.legend()
    plt.tight_layout()
    
    out_dir = 'results/figures/'
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'avr_bf1_scatter.png'), dpi=300)
    plt.savefig(os.path.join(out_dir, 'avr_bf1_scatter.pdf'))
    plt.close()
    print(f"Saved avr_bf1_scatter to {out_dir}")

if __name__ == '__main__':
    main()
