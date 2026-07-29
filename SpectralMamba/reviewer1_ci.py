"""
reviewer1_ci.py
---------------
Computes 95% confidence intervals for all reported Pearson correlations.

MODE 1 (--fast, default): Fisher z-transform CI from existing CSV + known n.
  No inference. Runs in <5 seconds.

MODE 2 (--bootstrap N): Bootstrap CI by re-running inference to collect
  per-image (avr, bf1) pairs, then bootstrapping. Requires GPU, ~20 min.

Usage:
  python reviewer1_ci.py              # Fisher z, instant
  python reviewer1_ci.py --bootstrap 2000
"""

import argparse
import os
import sys
import numpy as np
from scipy.stats import pearsonr

# [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
# FISHER Z-TRANSFORM CI
# [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]

def fisher_ci(r, n, alpha=0.05):
    """Return (lo, hi) 95% CI for Pearson r via Fisher z-transform."""
    z  = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = 1.959964  # two-tailed 95%
    return np.tanh(z - z_crit * se), np.tanh(z + z_crit * se)


def run_fisher():
    csv_path = os.path.join('VM-UNet', 'results', 'correlation_results.csv')
    if not os.path.exists(csv_path):
        sys.exit(f"ERROR: {csv_path} not found. Run per_image_correlation.py first.")

    # n per model = 50; pooled / partial n = 150 (matching per_image_correlation run data)
    N_PER_MODEL = 50
    N_POOLED    = 150

    n_map = {'UNet': N_PER_MODEL, 'Swin': N_PER_MODEL, 'Mamba': N_PER_MODEL,
             'Pooled': N_POOLED, 'Partial': N_POOLED}

    print("\n" + "="*78)
    print(f"{'Model/Pool':<15} | {'r':>8} | {'p':>10} | {'95% CI lower':>13} | {'95% CI upper':>13}")
    print("-"*78)

    rows = []
    with open(csv_path) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(',')
            name, r_val, p_val = parts[0], float(parts[1]), float(parts[2])
            n = n_map.get(name, N_POOLED)
            lo, hi = fisher_ci(r_val, n)
            print(f"{name:<15} | {r_val:>8.4f} | {p_val:>10.4e} | {lo:>13.4f} | {hi:>13.4f}")
            rows.append(f"{name},{r_val},{p_val},{lo},{hi},{n}")

    print("="*78)
    out = os.path.join('VM-UNet', 'results', 'correlation_results_with_ci.csv')
    with open(out, 'w') as f:
        f.write("model,pearson_r,p_value,ci_95_lo,ci_95_hi,n\n")
        f.write("\n".join(rows) + "\n")
    print(f"\n[OK][OK][OK] Saved to {out}")


# [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
# BOOTSTRAP CI (requires inference)
# [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]

def collect_per_image_data():
    """Re-run inference to collect per-image (avr, bf1) lists."""
    import glob, random, torch
    from PIL import Image
    from torchvision import transforms
    from collections import defaultdict
    from scipy.ndimage import binary_erosion, distance_transform_edt
    import segmentation_models_pytorch as smp

    ROOT = os.getcwd()
    sys.path.append(ROOT)
    from models.vmunet.vmunet import VMUNet
    sys.path.append(os.path.join(ROOT, 'Swin-Unet'))
    from config import get_config
    from networks.vision_transformer import SwinUnet

    # [OK][OK][OK][OK][OK][OK] helpers (identical to per_image_correlation.py) [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
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
        has_pfx  = any(k.startswith('vmunet.') for k in sd)
        mdl_pfx  = any(k.startswith('vmunet.') for k in model.state_dict())
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if has_pfx and not mdl_pfx: nk = k.replace('vmunet.', '')
            elif not has_pfx and mdl_pfx: nk = 'vmunet.' + k
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=True)
        return model

    def compute_avr(fmap, dc_correct=True):
        fmap = fmap.cpu().float()
        if dc_correct:
            fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)
        B, C, H, W = fmap.shape
        fft  = torch.fft.fft2(fmap)
        fft_s = torch.fft.fftshift(fft, dim=(-2, -1))
        power = torch.abs(fft_s) ** 2
        cy, cx = H // 2, W // 2
        y = torch.arange(H).view(1, 1, H, 1)
        x = torch.arange(W).view(1, 1, 1, W)
        mask = ((torch.abs(y - cy) > H / 4) | (torch.abs(x - cx) > W / 4)).expand(B, C, H, W)
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

    # [OK][OK][OK][OK][OK][OK] models [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Collecting per-image data on {device}...")
    ckpt = os.path.join(ROOT, 'VM-UNet', 'best-ckpt')
    unet = flexible_load(
        smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device),
        os.path.join(ckpt, 'best-unet-isic18.pth')).eval()
    args = MockArgs(); config = get_config(args)
    swin = flexible_load(
        SwinUnet(config, img_size=224, num_classes=1).to(device),
        os.path.join(ckpt, 'best-swinunet-isic18.pth')).eval()
    vmunet = flexible_load(
        VMUNet(input_channels=3, num_classes=1, depths=[2,2,9,2], depths_decoder=[2,9,2,2]).to(device),
        os.path.join(ckpt, 'best-vmunet-scratch-isic18.pth')).eval()

    # [OK][OK][OK][OK][OK][OK] hooks [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    feats = defaultdict(dict)
    def hook(mname, lvl):
        def _h(m, inp, out):
            feats[mname][lvl] = (out[0] if isinstance(out, tuple) else out).detach()
        return _h
    for i in range(1, 5):
        getattr(unet.encoder, f'layer{i}').register_forward_hook(hook('UNet', i))
        swin.swin_unet.layers[i-1].blocks[-1].register_forward_hook(hook('Swin', i))
        vmunet.vmunet.layers[i-1].blocks[-1].register_forward_hook(hook('Mamba', i))

    # [OK][OK][OK][OK][OK][OK] data split [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    t256 = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t224 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    img_dir  = os.path.join(ROOT, 'VM-UNet', 'data', 'isic18', 'train', 'images')
    mask_dir = os.path.join(ROOT, 'VM-UNet', 'data', 'isic18', 'train', 'masks')
    paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')) +
                   glob.glob(os.path.join(img_dir, '*.png')))
    random.seed(42); random.shuffle(paths)
    val_paths = paths[int(0.8*len(paths)):]

    res = {'UNet': {'avr':[], 'bf1':[]}, 'Swin': {'avr':[], 'bf1':[]}, 'Mamba': {'avr':[], 'bf1':[]}}
    for idx, ip in enumerate(val_paths):
        if idx % 20 == 0: print(f"  {idx}/{len(val_paths)}")
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
            res['UNet']['bf1'].append(compute_bf1(pu, gt256))
            res['Swin']['bf1'].append(compute_bf1(ps, gt224))
            res['Mamba']['bf1'].append(compute_bf1(pm, gt256))
            for mn in ['UNet','Swin','Mamba']:
                avrs = []
                for i in range(1,5):
                    f = feats[mn][i]
                    if mn in ('Swin','Mamba') and f.dim()==4 and f.shape[-1] in [96,192,384,768]:
                        f = f.permute(0,3,1,2)
                    elif mn == 'Swin' and f.dim()==3:
                        B,L,C = f.shape; H=W=int(np.sqrt(L)); f=f.transpose(1,2).reshape(B,C,H,W)
                    avrs.append(compute_avr(f))
                res[mn]['avr'].append(np.mean(avrs))
    return res


def bootstrap_ci(x, y, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(x); boot_rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x[idx], y[idx]
        if np.std(xb) == 0 or np.std(yb) == 0: continue
        boot_rs.append(pearsonr(xb, yb)[0])
    boot_rs = np.array(boot_rs)
    lo = np.percentile(boot_rs, 100 * alpha/2)
    hi = np.percentile(boot_rs, 100 * (1 - alpha/2))
    return lo, hi


def run_bootstrap(n_boot=2000):
    res = collect_per_image_data()
    all_avr, all_bf1 = [], []
    out_rows = []
    print("\n" + "="*78)
    print(f"{'Model/Pool':<15} | {'r':>8} | {'p':>10} | {'Boot CI lo':>11} | {'Boot CI hi':>11}")
    print("-"*78)
    for mn in ['UNet','Swin','Mamba']:
        x = np.array(res[mn]['avr']); y = np.array(res[mn]['bf1'])
        r, p = pearsonr(x, y)
        lo, hi = bootstrap_ci(x, y, n_boot)
        print(f"{mn:<15} | {r:>8.4f} | {p:>10.4e} | {lo:>11.4f} | {hi:>11.4f}")
        out_rows.append(f"{mn},{r},{p},{lo},{hi},{len(x)}")
        all_avr.extend(x); all_bf1.extend(y)
    ax, ay = np.array(all_avr), np.array(all_bf1)
    r_p, p_p = pearsonr(ax, ay)
    lo_p, hi_p = bootstrap_ci(ax, ay, n_boot)
    print(f"{'Pooled':<15} | {r_p:>8.4f} | {p_p:>10.4e} | {lo_p:>11.4f} | {hi_p:>11.4f}")
    print("="*78)
    out_rows.append(f"Pooled,{r_p},{p_p},{lo_p},{hi_p},{len(ax)}")
    out = os.path.join('VM-UNet','results','correlation_results_bootstrap_ci.csv')
    with open(out,'w') as f:
        f.write("model,pearson_r,p_value,boot_ci_lo,boot_ci_hi,n\n")
        f.write("\n".join(out_rows)+"\n")
    print(f"\n[OK][OK][OK] Saved bootstrap CIs to {out}")


# [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bootstrap', type=int, default=0, metavar='N',
                        help='Run bootstrap with N resamples (requires inference). 0 = Fisher z only.')
    args = parser.parse_args()
    if args.bootstrap > 0:
        run_bootstrap(args.bootstrap)
    else:
        run_fisher()

