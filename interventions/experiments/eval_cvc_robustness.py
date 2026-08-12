"""
eval_cvc_robustness.py â€” Comprehensive robustness evaluation on CVC-ClinicDB.

For the 4 CVC checkpoints (VM-UNet, TSA-VM-UNet, ResNet50-UNet@256, Swin-UNETR),
evaluates at 256x256 under:
  A. Feature-space low-pass dose-response : cutoffs [0.10,0.15,0.20,0.25,0.30,0.40]
  B. Input-space FFT low-pass dose-response: same cutoffs (apply_tsa)
  C. Realistic input degradations          : Gaussian blur (sigma=2,4,8),
                                            JPEG (q=80,50,20), motion blur (len=5,10)
  D. Boundary metrics (clean + feat-LP 0.25): boundary-F1, HD95 (per-image)

Per-image Dice is recorded for downstream paired statistics.
Results -> interventions/results/cvc_robustness_eval.json

Usage:
    python interventions/experiments/eval_cvc_robustness.py
"""

import sys, os, argparse, json, datetime, io

import torch
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'tta_boundary_study'))
sys.path.insert(0, os.path.join(_REPO, 'SpectralMamba'))
sys.path.insert(0, _REPO)

from interventions.train_vmunet_cvc_tsa import apply_tsa

CVC_IMG_DIR = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
CVC_MASK_DIR = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')
IMG_SIZE = 256

FEAT_CUTOFFS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
GAUSS_SIGMAS = [2, 4, 8]
JPEG_QS = [80, 50, 20]
MOTION_LENS = [5, 10]


def compute_segmentation_metrics(preds_flat, gts_flat, threshold=0.5):
    y_pre = np.where(preds_flat >= threshold, 1, 0)
    y_true = np.where(gts_flat >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]
    total = float(np.sum(confusion))
    return {
        'dice': float(2 * TP) / float(2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0,
        'iou': float(TP) / float(TP + FP + FN) if (TP + FP + FN) > 0 else 0.0,
        'accuracy': float(TN + TP) / total if total > 0 else 0.0,
        'sensitivity': float(TP) / float(TP + FN) if (TP + FN) > 0 else 0.0,
        'specificity': float(TN) / float(TN + FP) if (TN + FP) > 0 else 0.0,
    }


def per_image_dice(pred_np, gt_np):
    pb = (pred_np >= 0.5).astype(np.uint8)
    gb = (gt_np >= 0.5).astype(np.uint8)
    inter = float((pb & gb).sum())
    denom = float(pb.sum() + gb.sum())
    return 1.0 if denom == 0 else 2.0 * inter / denom


def build_model(arch, ckpt, device):
    if arch == 'vmunet':
        from models.vmunet.vmunet import VMUNet
        _NC = {'num_classes': 1, 'input_channels': 3,
               'depths': [2, 2, 9, 2], 'depths_decoder': [2, 9, 2, 2],
               'drop_path_rate': 0.2}
        model = VMUNet(**_NC).to(device)
    elif arch == 'resnet50':
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                         in_channels=3, classes=1).to(device)
    elif arch == 'swinunetr':
        from src.models.swin_unetr_cvc import get_swin_unetr
        model = get_swin_unetr().to(device)
    else:
        raise ValueError(f'Unknown arch: {arch}')
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)
    model.eval()
    return model


def _resolve(model, path):
    if '[-1]' in path:
        base, _ = path.rsplit('[-1]', 1)
        m = model
        for part in base.split('.'):
            m = getattr(m, part)
        return m[-1]
    m = model
    for ch in path.split('.'):
        if '[' in ch:
            n, i = ch.split('[')
            m = getattr(m, n)[int(i.rstrip(']'))]
        else:
            m = getattr(m, ch)
    return m


def make_lp_hook(interv, layout):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if layout == 'nhwc':
            fmap = out.permute(0, 3, 1, 2).contiguous()
            modified = interv(fmap)
            return modified.permute(0, 2, 3, 1).contiguous()
        return interv(out)
    return hook


def attach_lp(model, arch, interv):
    handles = []
    if arch == 'vmunet':
        for nm, mod in model.named_modules():
            if 'VSSBlock' in type(mod).__name__:
                handles.append(mod.register_forward_hook(make_lp_hook(interv, 'nhwc')))
    elif arch == 'resnet50':
        targets = ['encoder.layer1[-1]', 'encoder.layer2[-1]', 'encoder.layer3[-1]',
                   'encoder.layer4[-1]', 'decoder.blocks[0]', 'decoder.blocks[1]',
                   'decoder.blocks[2]', 'decoder.blocks[3]', 'decoder.blocks[4]']
        for t in targets:
            handles.append(_resolve(model, t).register_forward_hook(make_lp_hook(interv, 'nchw')))
    elif arch == 'swinunetr':
        swin = model.swinViT
        for i in range(1, 5):
            layer = getattr(swin, f'layers{i}')[0]
            for blk in layer.blocks:
                handles.append(blk.register_forward_hook(make_lp_hook(interv, 'nhwc')))
    return handles


# ---------------------------------------------------------------------------
# Input degradations (operate on [0,1] batch tensors (B,3,H,W))
# ---------------------------------------------------------------------------
def degrade_gaussian(images, sigma):
    from scipy.ndimage import gaussian_filter
    arr = images.cpu().numpy()
    out = np.stack([np.stack([gaussian_filter(arr[b, c], sigma=sigma) for c in range(3)])
                    for b in range(arr.shape[0])]).astype(np.float32)
    return torch.from_numpy(out).to(images.device)


def degrade_jpeg(images, quality):
    B = images.shape[0]
    arrs = (images.cpu().numpy().transpose(0, 2, 3, 1) * 255.0).clip(0, 255).astype(np.uint8)
    outs = []
    for b in range(B):
        im = Image.fromarray(arrs[b])
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        outs.append(np.array(Image.open(buf).convert('RGB'), dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(outs).transpose(0, 3, 1, 2)).to(images.device)


def degrade_motion(images, length):
    from scipy.ndimage import convolve
    kernel = np.zeros((length, length))
    kernel[int((length - 1) / 2), :] = 1.0 / length
    arr = images.cpu().numpy()
    out = np.stack([np.stack([convolve(arr[b, c], kernel, mode='nearest') for c in range(3)])
                    for b in range(arr.shape[0])]).astype(np.float32)
    return torch.from_numpy(out).to(images.device)


def evaluate(model, loader, device, needs_sigmoid, degrade=None, boundary=False):
    """Pooled metrics + per-image Dice; optional input degrade fn; optional boundary metrics."""
    from src.metrics.boundary_metrics import boundary_f1, hausdorff_95
    preds, gts, per_dice = [], [], []
    bf1, hd95 = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            if degrade is not None:
                imgs = degrade(imgs)
            out = model(imgs)
            probs = torch.sigmoid(out) if needs_sigmoid else out
            p_np = probs.squeeze(1).cpu().numpy()
            g_np = masks.squeeze(1).cpu().numpy()
            for i in range(p_np.shape[0]):
                per_dice.append(per_image_dice(p_np[i], g_np[i]))
                if boundary:
                    pb = (p_np[i] >= 0.5).astype(bool)
                    gb = (g_np[i] >= 0.5).astype(bool)
                    if gb.sum() > 0:
                        # BF1 handles empty predictions internally (returns 0.0).
                        bf1.append(boundary_f1(p_np[i], g_np[i], threshold=0.5, thickness=2))
                        if pb.sum() > 0:
                            hd95.append(hausdorff_95(p_np[i], g_np[i], threshold=0.5))
                        else:
                            # Empty prediction -> HD95 undefined. Record NaN and
                            # drop from the mean via nanmean (no NaN propagation).
                            hd95.append(np.nan)
            preds.append(p_np.ravel())
            gts.append(g_np.ravel())
    pooled = compute_segmentation_metrics(np.concatenate(preds), np.concatenate(gts))
    res = {'pooled': pooled,
           'dice_mean': float(np.mean(per_dice)), 'dice_std': float(np.std(per_dice)),
           'per_dice': per_dice}
    if boundary:
        res['boundary_f1_mean'] = float(np.mean(bf1)) if bf1 else float('nan')
        _hd = np.asarray(hd95, dtype=float)
        res['hd95_mean'] = (float(np.nanmean(_hd))
                            if _hd.size and not bool(np.isnan(_hd).all())
                            else float('nan'))
    return res


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_json', default=os.path.join(
        _REPO, 'interventions', 'results', 'cvc_robustness_eval.json'))
    ap.add_argument('--seed', type=int, default=42)
    args, _ = ap.parse_known_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from src.datasets.cvc_dataset import CVCDataset
    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 90)
    print('CVC robustness suite: dose-response + degradations + boundary  |  @256')
    print('=' * 90)
    print(f'Device: {device}')

    val_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='val', img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'Val: {len(val_ds)} images @ {IMG_SIZE}x{IMG_SIZE}')

    MODELS = [
        ('VM-UNet', 'SSM', 'vmunet',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'best-vmunet-cvc.pth'), False),
        ('VM-UNet-TSA', 'SSM', 'vmunet',
         os.path.join(_REPO, 'interventions', 'results', 'best-vmunet-cvc-tsa-finetune',
                      'best-vmunet-cvc.pth'), False),
        ('ResNet50-UNet', 'CNN', 'resnet50',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best_256.pth'), True),
        ('Swin-UNETR', 'ViT', 'swinunetr',
         os.path.join(_REPO, 'interventions', 'results', 'best-swinunetr-cvc-256', 'best-swinunetr-cvc-256.pth'), True),
    ]

    results = {'experiment': 'CVC robustness suite (dose-response/degradations/boundary)',
               'img_size': IMG_SIZE, 'num_images': len(val_ds), 'cutoff_set': FEAT_CUTOFFS,
               'device': str(device), 'timestamp': datetime.datetime.now().isoformat(),
               'models': []}

    for name, family, arch, ckpt, needs_sigmoid in MODELS:
        print(f'\n===== {name} ({family}) =====')
        model = build_model(arch, ckpt, device)
        model_row = {'name': name, 'family': family, 'architecture': arch,
                     'checkpoint': ckpt, 'clean': {}, 'feature_dose': {}, 'input_dose': {},
                     'degradations': {}, 'boundary': {}}

        clean = evaluate(model, val_loader, device, needs_sigmoid, boundary=True)
        model_row['clean'] = {'dice': clean['pooled']['dice'], 'iou': clean['pooled']['iou'],
                              'dice_mean': clean['dice_mean'], 'dice_std': clean['dice_std'],
                              'boundary_f1': clean['boundary_f1_mean'], 'hd95': clean['hd95_mean'],
                              'per_dice': clean['per_dice']}
        print(f"  Clean: Dice={clean['pooled']['dice']:.4f} BF1={clean['boundary_f1_mean']:.4f} "
              f"HD95={clean['hd95_mean']:.1f}")

        print('  Feature-space LP:', end='')
        for c in FEAT_CUTOFFS:
            lp_fn = lambda h, w, dev, dt, c=c: lowpass_mask(h, w, c, device=dev, dtype=dt)
            handles = attach_lp(model, arch, FrequencyIntervention(lp_fn, check_nan=True))
            r = evaluate(model, val_loader, device, needs_sigmoid)
            for h in handles:
                h.remove()
            model_row['feature_dose'][str(c)] = {
                'dice': r['pooled']['dice'], 'dice_mean': r['dice_mean'], 'dice_std': r['dice_std'],
                'per_dice': r['per_dice']}
            print(f' {c:.2f}:{r["pooled"]["dice"]:.3f}', end='', flush=True)
        print()

        print('  Input-space LP:', end='')
        for c in FEAT_CUTOFFS:
            r = evaluate(model, val_loader, device, needs_sigmoid,
                         degrade=lambda im, c=c: apply_tsa(im, cutoff=c, p=1.0))
            model_row['input_dose'][str(c)] = {
                'dice': r['pooled']['dice'], 'dice_mean': r['dice_mean'], 'dice_std': r['dice_std'],
                'per_dice': r['per_dice']}
            print(f' {c:.2f}:{r["pooled"]["dice"]:.3f}', end='', flush=True)
        print()
        # Realistic degradations
        for s in GAUSS_SIGMAS:
            r = evaluate(model, val_loader, device, needs_sigmoid,
                         degrade=lambda im, s=s: degrade_gaussian(im, s))
            model_row['degradations'][f'gauss_{s}'] = r['pooled']['dice']
            print(f"  Gauss sigma={s}: Dice={r['pooled']['dice']:.4f}")
        for q in JPEG_QS:
            r = evaluate(model, val_loader, device, needs_sigmoid,
                         degrade=lambda im, q=q: degrade_jpeg(im, q))
            model_row['degradations'][f'jpeg_{q}'] = r['pooled']['dice']
            print(f"  JPEG q={q}: Dice={r['pooled']['dice']:.4f}")
        for ln in MOTION_LENS:
            r = evaluate(model, val_loader, device, needs_sigmoid,
                         degrade=lambda im, ln=ln: degrade_motion(im, ln))
            model_row['degradations'][f'motion_{ln}'] = r['pooled']['dice']
            print(f"  Motion len={ln}: Dice={r['pooled']['dice']:.4f}")

        # Boundary metrics under feature-LP 0.25
        lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
        handles = attach_lp(model, arch, FrequencyIntervention(lp_fn, check_nan=True))
        r = evaluate(model, val_loader, device, needs_sigmoid, boundary=True)
        for h in handles:
            h.remove()
        model_row['boundary'] = {'dice': r['pooled']['dice'], 'boundary_f1': r['boundary_f1_mean'],
                                 'hd95': r['hd95_mean']}
        print(f"  Feat-LP0.25: Dice={r['pooled']['dice']:.4f} BF1={r['boundary_f1_mean']:.4f} "
              f"HD95={r['hd95_mean']:.1f}")

        results['models'].append(model_row)

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {args.output_json}')

    print('\n' + '=' * 90)
    print('Feature-space low-pass dose-response (pooled Dice)')
    print('=' * 90)
    print('| Model | Clean | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|')
    for m in results['models']:
        row = f'| {m["name"]} | {m["clean"]["dice"]:.3f}'
        for c in FEAT_CUTOFFS:
            row += f' | {m["feature_dose"][str(c)]["dice"]:.3f}'
        print(row + ' |')

    print('\nInput-space degradations (pooled Dice)')
    print('| Model | Clean | LP0.25 | G2 | G4 | G8 | J80 | J50 | J20 | M5 | M10 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for m in results['models']:
        print(f'| {m["name"]} | {m["clean"]["dice"]:.3f} '
              f'| {m["input_dose"]["0.25"]["dice"]:.3f} '
              f'| {m["degradations"]["gauss_2"]:.3f} | {m["degradations"]["gauss_4"]:.3f} '
              f'| {m["degradations"]["gauss_8"]:.3f} '
              f'| {m["degradations"]["jpeg_80"]:.3f} | {m["degradations"]["jpeg_50"]:.3f} '
              f'| {m["degradations"]["jpeg_20"]:.3f} '
              f'| {m["degradations"]["motion_5"]:.3f} | {m["degradations"]["motion_10"]:.3f} |')

    print('\nBoundary metrics (clean vs feature-LP 0.25)')
    print('| Model | Clean BF1 | Clean HD95 | LP BF1 | LP HD95 |')
    print('|---|---:|---:|---:|---:|')
    for m in results['models']:
        print(f'| {m["name"]} | {m["clean"]["boundary_f1"]:.3f} | {m["clean"]["hd95"]:.1f} '
              f'| {m["boundary"]["boundary_f1"]:.3f} | {m["boundary"]["hd95"]:.1f} |')


if __name__ == '__main__':
    main()

