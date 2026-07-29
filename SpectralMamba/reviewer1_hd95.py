"""
reviewer1_hd95.py
------------------
Adds HD95 (Hausdorff Distance 95th percentile) to the existing
boundary evaluation pipeline for Reviewer 1.

Extends tools/boundary_eval.py with:
  - compute_hd95()   using scipy distance_transform_edt (no medpy dependency)
  - HD95 per image collection
  - Mean + 95% CI reporting

Output:
  VM-UNet/results/hd95_results.csv
    model, mean_dice, mean_bf1, mean_hd95, ci_dice_lo, ci_dice_hi,
    ci_bf1_lo, ci_bf1_hi, ci_hd95_lo, ci_hd95_hi, n

Usage:
  cd c:\\Users\\subka\\Documents\\SpectralMamba
  python reviewer1_hd95.py
"""
import sys, os, glob, torch, numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from scipy.ndimage import binary_erosion, distance_transform_edt
import segmentation_models_pytorch as smp

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
    print(f"  [OK][OK][OK] {os.path.basename(path)}")
    return model


# [OK][OK][OK][OK][OK][OK] Metrics [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]

def compute_dice(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    return 2.0 * inter / total if total > 0 else 1.0


def compute_bf1(pred, gt, tol=2):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    pb = pred ^ binary_erosion(pred, iterations=1)
    gb = gt   ^ binary_erosion(gt,   iterations=1)
    if pb.sum() == 0 and gb.sum() == 0: return 1.0
    if pb.sum() == 0 or  gb.sum() == 0: return 0.0
    pd_map = distance_transform_edt(~pb)
    gd_map = distance_transform_edt(~gb)
    tp_p = (pb & (gd_map <= tol)).sum()
    tp_g = (gb & (pd_map <= tol)).sum()
    prec = tp_p / pb.sum(); rec = tp_g / gb.sum()
    return 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0


def compute_hd95(pred, gt):
    """
    Hausdorff Distance 95th percentile [OK][OK][OK] pure scipy implementation.

    Returns the 95th percentile of the symmetric directed Hausdorff distance
    between the boundary of pred and the boundary of gt.

    Returns 0.0 if both masks are empty (perfect agreement).
    Returns np.nan if only one mask is empty (undefined).
    """
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)

    # Edge case: both empty [OK][OK][OK] perfect (0.0)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    # Edge case: one empty [OK][OK][OK] undefined [OK][OK][OK] penalize with image diagonal
    if pred.sum() == 0 or gt.sum() == 0:
        H, W = pred.shape
        return float(np.sqrt(H**2 + W**2))

    # Distance from every pred pixel to nearest gt boundary pixel
    gt_dist  = distance_transform_edt(~gt)   # dist to GT foreground boundary
    pred_dist = distance_transform_edt(~pred)  # dist to pred foreground boundary

    # Distances from pred boundary to GT, and GT boundary to pred
    pred_boundary = pred ^ binary_erosion(pred, iterations=1)
    gt_boundary   = gt   ^ binary_erosion(gt,   iterations=1)

    # For each pred boundary pixel, distance to nearest GT boundary pixel
    # We use distance_transform_edt(~gt_boundary) which gives dist to GT boundary
    dist_pred_to_gt = distance_transform_edt(~gt_boundary)[pred_boundary]
    dist_gt_to_pred = distance_transform_edt(~pred_boundary)[gt_boundary]

    # Symmetric: take union of both directed distances, then 95th percentile
    all_distances = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_distances, 95))


def fisher_ci(r, n, alpha=0.05):
    z = np.arctanh(np.clip(r, -0.9999, 0.9999))
    se = 1.0 / np.sqrt(n - 3)
    return float(np.tanh(z - 1.96*se)), float(np.tanh(z + 1.96*se))


def mean_ci(arr, alpha=0.05):
    """Return (mean, lo, hi) via normal approximation CI."""
    a = np.array(arr)
    n = len(a)
    m = np.mean(a)
    se = np.std(a, ddof=1) / np.sqrt(n)
    z = 1.959964
    return m, m - z*se, m + z*se


# [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"HD95 EVALUATION  |  Reviewer 1 Request  |  Device: {device}")
    print(f"{'='*70}\n")

    ckpt = os.path.join(ROOT, 'VM-UNet', 'best-ckpt')
    t256 = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    t224 = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    print("Loading models...")
    vmunet = flexible_load(
        VMUNet(input_channels=3, num_classes=1, depths=[2,2,9,2], depths_decoder=[2,9,2,2]).to(device),
        os.path.join(ckpt, 'best-vmunet-scratch-isic18.pth')).eval()
    unet = flexible_load(
        smp.Unet(encoder_name='resnet50', encoder_weights=None, in_channels=3, classes=1).to(device),
        os.path.join(ckpt, 'best-unet-isic18.pth')).eval()
    config = get_config(MockArgs())
    swin = flexible_load(
        SwinUnet(config, img_size=224, num_classes=1).to(device),
        os.path.join(ckpt, 'best-swinunet-isic18.pth')).eval()

    models = {'VM-UNet': (vmunet, 256, t256),
              'UNet-ResNet50': (unet, 256, t256),
              'Swin-Tiny':     (swin, 224, t224)}
    results = {n: {'dice':[], 'bf1':[], 'hd95':[]} for n in models}

    # [OK][OK][OK][OK][OK][OK] Data split (same seed as all other scripts) [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    img_dir  = os.path.join(ROOT, 'VM-UNet', 'data', 'isic18', 'train', 'images')
    mask_dir = os.path.join(ROOT, 'VM-UNet', 'data', 'isic18', 'train', 'masks')
    import random
    all_imgs = sorted(glob.glob(os.path.join(img_dir,'*.jpg')) +
                      glob.glob(os.path.join(img_dir,'*.png')))
    random.seed(42); random.shuffle(all_imgs)
    val_imgs = all_imgs[int(0.8*len(all_imgs)):]
    print(f"Validation set: {len(val_imgs)} images\n")

    # [OK][OK][OK][OK][OK][OK] Inference loop [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    for img_path in tqdm(val_imgs, desc='HD95 Evaluation'):
        base = os.path.splitext(os.path.basename(img_path))[0]
        mp   = os.path.join(mask_dir, base + '_segmentation.png')
        if not os.path.exists(mp): continue
        img_pil = Image.open(img_path).convert('RGB')
        msk_pil = Image.open(mp).convert('L')

        for m_name, (model, size, tfm) in models.items():
            x  = tfm(img_pil).unsqueeze(0).to(device)
            gt = np.array(msk_pil.resize((size, size), Image.NEAREST)) > 127

            with torch.no_grad():
                out = model(x)
                if isinstance(out, tuple): out = out[0]
                # VM-UNet applies sigmoid internally; others need it
                prob = out.squeeze().cpu().numpy() \
                    if m_name == 'VM-UNet' \
                    else torch.sigmoid(out).squeeze().cpu().numpy()

            pred_bin = prob > 0.5
            results[m_name]['dice'].append(compute_dice(pred_bin, gt))
            results[m_name]['bf1'].append(compute_bf1(pred_bin, gt))
            results[m_name]['hd95'].append(compute_hd95(pred_bin, gt))

    # [OK][OK][OK][OK][OK][OK] Report [OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK][OK]
    print(f"\n{'='*90}")
    print(f"{'Model':<18} | {'Dice':>6} | {'Dice 95% CI':>15} | "
          f"{'BF1':>6} | {'BF1 95% CI':>15} | {'HD95 (px)':>10} | {'HD95 95% CI':>18}")
    print('-'*90)
    csv_rows = ['model,mean_dice,dice_ci_lo,dice_ci_hi,'
                'mean_bf1,bf1_ci_lo,bf1_ci_hi,'
                'mean_hd95,hd95_ci_lo,hd95_ci_hi,n']

    for m_name in models:
        n = len(results[m_name]['dice'])
        dm, dlo, dhi = mean_ci(results[m_name]['dice'])
        bm, blo, bhi = mean_ci(results[m_name]['bf1'])
        hm, hlo, hhi = mean_ci(results[m_name]['hd95'])
        print(f"{m_name:<18} | {dm:>6.4f} | [{dlo:.4f}, {dhi:.4f}] | "
              f"{bm:>6.4f} | [{blo:.4f}, {bhi:.4f}] | {hm:>10.2f} | [{hlo:.2f}, {hhi:.2f}]")
        csv_rows.append(
            f"{m_name},{dm:.4f},{dlo:.4f},{dhi:.4f},"
            f"{bm:.4f},{blo:.4f},{bhi:.4f},"
            f"{hm:.4f},{hlo:.4f},{hhi:.4f},{n}")

    print('='*90)
    print("Note: HD95 in pixels. Multiply by image spacing (mm/px) for clinical units.")

    out = os.path.join('VM-UNet','results','hd95_results.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out,'w') as f:
        f.write("\n".join(csv_rows)+"\n")
    print(f"\n[OK][OK][OK] Saved to {out}")


if __name__ == '__main__':
    main()

