"""
reviewer1_pre_dc_correlation.py
--------------------------------
Runs the AVR[OK][OK][OK]BF1 Pearson correlation WITHOUT DC correction (mean-centering),
as requested by Reviewer 1. This is identical to per_image_correlation.py
except compute_avr() has dc_correct=False.

The result should show a spuriously high correlation driven by the DC component
(mean intensity) [OK][OK][OK] supporting the methodological argument for DC correction in
the main paper.

Usage:
  cd c:\\Users\\subka\\Documents\\SpectralMamba
  python reviewer1_pre_dc_correlation.py
"""
import sys, os, glob, torch, numpy as np
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
    print(f"  [OK][OK][OK] Loaded {path}")
    return model


# [OK][OK][OK][OK][OK][OK] KEY CHANGE: dc_correct=False removes the mean-centering step [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
def compute_avr(fmap, dc_correct=False):          # <-- FALSE = pre-DC
    """
    Compute Alias Volume Ratio.
    dc_correct=False: raw FFT including DC component  (pre-correction)
    dc_correct=True : mean-centered FFT               (post-correction, original)
    """
    fmap = fmap.cpu().float()
    if dc_correct:                                 # line that was previously always executed
        fmap = fmap - fmap.mean(dim=(-2, -1), keepdim=True)
    B, C, H, W = fmap.shape
    fft   = torch.fft.fft2(fmap)
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


def fisher_ci(r, n, alpha=0.05):
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    return np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print("PRE-DC CORRECTION: AVR[OK][OK][OK]BF1 Correlation (Reviewer 1 Request)")
    print(f"DC correction: DISABLED  |  Device: {device}")
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

    res = {'UNet':{'avr':[],'bf1':[]},'Swin':{'avr':[],'bf1':[]},'Mamba':{'avr':[],'bf1':[]}}

    print(f"Running inference on {len(val_paths)} validation images (DC correction OFF)...")
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
                    avrs.append(compute_avr(f, dc_correct=False))   # [OK][OK][OK] PRE-DC
                res[mn]['avr'].append(np.mean(avrs))

    # [OK][OK][OK][OK][OK][OK] Report [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    print(f"\n{'='*70}")
    print("PRE-DC CORRECTION RESULTS  (compare to post-DC in correlation_results.csv)")
    print(f"{'Model':<15} | {'r':>8} | {'p':>10} | {'CI lo':>8} | {'CI hi':>8} | {'n':>5}")
    print('-'*70)
    csv_rows = []
    all_avr, all_bf1 = [], []
    for mn in ['UNet','Swin','Mamba']:
        x = np.array(res[mn]['avr']); y = np.array(res[mn]['bf1'])
        r, p = pearsonr(x, y); n = len(x)
        lo, hi = fisher_ci(r, n)
        print(f"{mn:<15} | {r:>8.4f} | {p:>10.4e} | {lo:>8.4f} | {hi:>8.4f} | {n:>5}")
        csv_rows.append(f"{mn},{r},{p},{lo},{hi},{n}")
        all_avr.extend(x); all_bf1.extend(y)

    ax, ay = np.array(all_avr), np.array(all_bf1)
    r_p, p_p = pearsonr(ax, ay); n_p = len(ax)
    lo_p, hi_p = fisher_ci(r_p, n_p)
    print(f"{'Pooled':<15} | {r_p:>8.4f} | {p_p:>10.4e} | {lo_p:>8.4f} | {hi_p:>8.4f} | {n_p:>5}")
    print(f"{'='*70}")
    csv_rows.append(f"Pooled,{r_p},{p_p},{lo_p},{hi_p},{n_p}")

    out = os.path.join('VM-UNet','results','pre_dc_correlation_results.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'w') as f:
        f.write("model,pearson_r,p_value,ci_95_lo,ci_95_hi,n\n")
        f.write("\n".join(csv_rows)+"\n")
    print(f"\n[OK][OK][OK] Saved to {out}")
    print("NOTE: High pre-DC correlations are expected [OK][OK][OK] they are an artifact of")
    print("      the DC (mean intensity) term dominating the energy spectrum.")


if __name__ == '__main__':
    main()

