"""
reviewer1_avr_sensitivity.py
-----------------------------
AVR cutoff sensitivity analysis for Reviewer 1.

Tests 5 radial cutoff thresholds (0.25, 0.375, 0.50, 0.625, 0.75 of Nyquist)
and reports per-model mean AVR + AVR[OK][OK][OK]BF1 Pearson r for each cutoff.

Output:
  VM-UNet/results/avr_sensitivity_results.csv
  results/figures/avr_sensitivity.png  (line plot: cutoff vs r per model)

Usage:
  cd c:\\Users\\subka\\Documents\\SpectralMamba
  python reviewer1_avr_sensitivity.py
"""
import sys, os, glob, torch, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from collections import defaultdict
import segmentation_models_pytorch as smp
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.stats import pearsonr

ROOT = os.getcwd()
sys.path.append(ROOT)
from models.vmunet.vmunet import VMUNet
sys.path.append(os.path.join(ROOT, 'Swin-Unet'))
from config import get_config
from networks.vision_transformer import SwinUnet

# [OK][OK][OK][OK][OK][OK] Cutoffs to test (fraction of Nyquist half-spectrum) [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
# H/4 corresponds to 0.50 (the published cutoff).
# We test [OK][OK]1 and [OK][OK]2 steps around it.
CUTOFFS = [0.25, 0.375, 0.50, 0.625, 0.75]   # normalized freq > this = "high"
# Published cutoff = 0.50  (mask: |y-cy| > H/4, i.e., |y-cy|/(H/2) > 0.5)

COLORS = {'UNet': '#1f77b4', 'Swin': '#2ca02c', 'Mamba': '#ff7f0e'}


class MockArgs:
    cfg = os.path.join(ROOT, 'Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml')
    opts = None; batch_size = 1; zip = False; cache_mode = 'part'
    resume = None; accumulation_steps = None; use_checkpoint = False
    amp_opt_level = 'O0'; tag = 'test'; eval = False; throughput = False


def flexible_load(model, path):
    sd = torch.load(path, map_location='cpu')
    if 'model' in sd: sd = sd['model']
    sd = {k: v for k, v in sd.items()
          if not k.endswith('total_ops') and not k.endswith('total_params')}
    has_pfx = any(k.startswith('vmunet.') for k in sd)
    mdl_pfx = any(k.startswith('vmunet.') for k in model.state_dict())
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if has_pfx and not mdl_pfx:  nk = k.replace('vmunet.', '')
        elif not has_pfx and mdl_pfx: nk = 'vmunet.' + k
        new_sd[nk] = v
    model.load_state_dict(new_sd, strict=True)
    print(f"  [OK][OK][OK] {os.path.basename(path)}")
    return model


def compute_avr_cutoff(fmap, cutoff_norm):
    """
    cutoff_norm: fraction of the half-spectrum [0, 1].
      0.50 = original (mask: |freq| > 0.5 [OK][OK] Nyquist)
      Lower values = more energy classified as "high-freq"
      Higher values = less energy classified as "high-freq"
    """
    fmap = fmap.cpu().float()
    fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)   # DC correction ON
    B, C, H, W = fmap.shape
    fft   = torch.fft.fft2(fmap)
    fft_s = torch.fft.fftshift(fft, dim=(-2, -1))
    power = torch.abs(fft_s) ** 2
    cy, cx = H // 2, W // 2
    y = torch.arange(H).view(1, 1, H, 1).float()
    x = torch.arange(W).view(1, 1, 1, W).float()
    # Normalize distance to [0, 1] where 1 = edge of spectrum (Nyquist)
    dy = torch.abs(y - cy) / (H / 2)
    dx = torch.abs(x - cx) / (W / 2)
    freq_ratio = torch.max(dy.expand(B, C, H, W), dx.expand(B, C, H, W))
    mask = freq_ratio > cutoff_norm
    return (power * mask).sum().item() / power.sum().item() if power.sum() > 0 else 0.0


def compute_bf1(pred, gt, tol=2):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    pb = pred ^ binary_erosion(pred, iterations=1)
    gb = gt   ^ binary_erosion(gt,   iterations=1)
    if pb.sum() == 0 and gb.sum() == 0: return 1.0
    if pb.sum() == 0 or  gb.sum() == 0: return 0.0
    pd = distance_transform_edt(~pb)
    gd = distance_transform_edt(~gb)
    tp_p = (pb & (gd <= tol)).sum(); tp_g = (gb & (pd <= tol)).sum()
    prec = tp_p / pb.sum(); rec = tp_g / gb.sum()
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"AVR CUTOFF SENSITIVITY ANALYSIS  |  Device: {device}")
    print(f"Cutoffs tested: {CUTOFFS}")
    print(f"Published cutoff (0.50) is the 3rd entry.")
    print(f"{'='*70}\n")

    ckpt = os.path.join(ROOT, 'VM-UNet', 'best-ckpt')
    t256 = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t224 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    print("Loading models...")
    unet = flexible_load(
        smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device),
        os.path.join(ckpt, 'best-unet-isic18.pth')).eval()
    config = get_config(MockArgs())
    swin = flexible_load(
        SwinUnet(config, img_size=224, num_classes=1).to(device),
        os.path.join(ckpt, 'best-swinunet-isic18.pth')).eval()
    vmunet = flexible_load(
        VMUNet(input_channels=3, num_classes=1, depths=[2,2,9,2], depths_decoder=[2,9,2,2]).to(device),
        os.path.join(ckpt, 'best-vmunet-scratch-isic18.pth')).eval()

    feats = defaultdict(dict)
    def hook(mname, lvl):
        def _h(m, inp, out):
            feats[mname][lvl] = (out[0] if isinstance(out, tuple) else out).detach()
        return _h
    for i in range(1, 5):
        getattr(unet.encoder, f'layer{i}').register_forward_hook(hook('UNet', i))
        swin.swin_unet.layers[i-1].blocks[-1].register_forward_hook(hook('Swin', i))
        vmunet.vmunet.layers[i-1].blocks[-1].register_forward_hook(hook('Mamba', i))

    img_dir  = os.path.join(ROOT, 'VM-UNet', 'data', 'isic18', 'train', 'images')
    mask_dir = os.path.join(ROOT, 'VM-UNet', 'data', 'isic18', 'train', 'masks')
    import random
    paths = sorted(glob.glob(os.path.join(img_dir,'*.jpg')) +
                   glob.glob(os.path.join(img_dir,'*.png')))
    random.seed(42); random.shuffle(paths)
    val_paths = paths[int(0.8*len(paths)):]

    # Collect raw feature tensors + BF1 once (BF1 is cutoff-independent)
    print(f"Collecting feature maps and BF1 from {len(val_paths)} images...")
    raw_feats = {'UNet':[], 'Swin':[], 'Mamba':[]}   # list of dicts {level: tensor}
    bf1_data  = {'UNet':[], 'Swin':[], 'Mamba':[]}

    for idx, ip in enumerate(val_paths):
        if idx % 20 == 0: print(f"  {idx}/{len(val_paths)}", flush=True)
        base = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(mask_dir, base + '_segmentation.png')
        if not os.path.exists(mp): continue
        img = Image.open(ip).convert('RGB'); msk = Image.open(mp).convert('L')
        i256 = t256(img).unsqueeze(0).to(device)
        i224 = t224(img).unsqueeze(0).to(device)
        gt256 = np.array(msk.resize((256,256), Image.NEAREST)) > 127
        gt224 = np.array(msk.resize((224,224), Image.NEAREST)) > 127

        with torch.no_grad():
            feats.clear()
            ou = unet(i256); os_ = swin(i224); om = vmunet(i256)
            pu = (torch.sigmoid(ou).squeeze().cpu().numpy() > 0.5)
            ps = (torch.sigmoid(os_).squeeze().cpu().numpy() > 0.5)
            pm = (om.squeeze().cpu().numpy() > 0.5)
            bf1_data['UNet'].append(compute_bf1(pu, gt256))
            bf1_data['Swin'].append(compute_bf1(ps, gt224))
            bf1_data['Mamba'].append(compute_bf1(pm, gt256))

            for mn in ['UNet','Swin','Mamba']:
                snap = {}
                for i in range(1,5):
                    f = feats[mn][i].cpu()
                    if mn in ('Swin','Mamba') and f.dim()==4 and f.shape[-1] in [96,192,384,768]:
                        f = f.permute(0,3,1,2)
                    elif mn == 'Swin' and f.dim()==3:
                        B,L,C = f.shape; H=W=int(np.sqrt(L)); f=f.transpose(1,2).reshape(B,C,H,W)
                    snap[i] = f
                raw_feats[mn].append(snap)

    # [OK][OK][OK][OK][OK][OK] Sweep cutoffs WITHOUT re-running inference [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    print("\nSweeping cutoffs (no further inference required)...")
    sensitivity = {}   # {cutoff: {model: {'mean_avr': float, 'r': float, 'p': float}}}

    for cutoff in CUTOFFS:
        sensitivity[cutoff] = {}
        for mn in ['UNet','Swin','Mamba']:
            avr_list = []
            for snap in raw_feats[mn]:
                img_avrs = [compute_avr_cutoff(snap[i], cutoff) for i in range(1,5)]
                avr_list.append(np.mean(img_avrs))
            avr_arr = np.array(avr_list)
            bf1_arr = np.array(bf1_data[mn])
            r, p = pearsonr(avr_arr, bf1_arr)
            sensitivity[cutoff][mn] = {'mean_avr': float(np.mean(avr_arr)), 'r': r, 'p': p}

    # [OK][OK][OK][OK][OK][OK] Print table [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    print(f"\n{'='*80}")
    print(f"{'Cutoff':>8} | {'Model':<10} | {'Mean AVR':>10} | {'r':>8} | {'p':>10}")
    print('-'*80)
    csv_rows = ['cutoff,model,mean_avr,pearson_r,p_value']
    for cutoff in CUTOFFS:
        marker = ' [OK][OK][OK] published' if cutoff == 0.50 else ''
        for mn in ['UNet','Swin','Mamba']:
            d = sensitivity[cutoff][mn]
            print(f"{cutoff:>8.3f} | {mn:<10} | {d['mean_avr']:>10.4f} | {d['r']:>8.4f} | {d['p']:>10.4e}{marker}")
            csv_rows.append(f"{cutoff},{mn},{d['mean_avr']},{d['r']},{d['p']}")
        print()
    print('='*80)

    out_csv = os.path.join('VM-UNet','results','avr_sensitivity_results.csv')
    with open(out_csv,'w') as f:
        f.write("\n".join(csv_rows)+"\n")
    print(f"[OK][OK][OK] Saved CSV to {out_csv}")

    # [OK][OK][OK][OK][OK][OK] Figure: cutoff vs Pearson r per model [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                          'pdf.fonttype': 42, 'ps.fonttype': 42})

    ax = axes[0]
    for mn in ['UNet','Swin','Mamba']:
        rs = [sensitivity[c][mn]['r'] for c in CUTOFFS]
        ax.plot(CUTOFFS, rs, 'o-', color=COLORS[mn], label=mn, linewidth=2, markersize=7)
    ax.axvline(0.50, color='gray', linestyle='--', alpha=0.6, label='Published cutoff')
    ax.axhline(0.0,  color='black', linestyle='-',  alpha=0.2)
    ax.set_xlabel('Radial cutoff (fraction of Nyquist)')
    ax.set_ylabel("Pearson r  (AVR[OK][OK][OK]BF1)")
    ax.set_title("Correlation vs. Radial Cutoff")
    ax.legend(); ax.set_xticks(CUTOFFS)

    ax = axes[1]
    for mn in ['UNet','Swin','Mamba']:
        avrs = [sensitivity[c][mn]['mean_avr'] for c in CUTOFFS]
        ax.plot(CUTOFFS, avrs, 'o-', color=COLORS[mn], label=mn, linewidth=2, markersize=7)
    ax.axvline(0.50, color='gray', linestyle='--', alpha=0.6, label='Published cutoff')
    ax.set_xlabel('Radial cutoff (fraction of Nyquist)')
    ax.set_ylabel("Mean AVR")
    ax.set_title("Mean AVR vs. Radial Cutoff")
    ax.legend(); ax.set_xticks(CUTOFFS)

    plt.tight_layout()
    os.makedirs('results/figures', exist_ok=True)
    plt.savefig('results/figures/avr_sensitivity.png', dpi=300)
    plt.savefig('results/figures/avr_sensitivity.pdf')
    plt.close()
    print("[OK][OK][OK] Saved figure to results/figures/avr_sensitivity.{png,pdf}")


if __name__ == '__main__':
    main()

