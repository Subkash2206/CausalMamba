"""
eval_isic_cross_arch.py — Cross-architecture Fourier low-pass benchmark on ISIC2018.

Evaluates the 3 trained ISIC2018 baselines under two protocols:
  (1) Feature-space low-pass  (FFT->mask->IFFT on every semantic block, cutoff=0.25)
  (2) Input-space low-pass    (FFT blur on the INPUT image only, cutoff=0.25,
                               using the same apply_tsa logic as the TSA training)

Models (native training resolution, ImageNet normalization):
  - VM-UNet       (SSM): best-vmunet-scratch-isic18.pth  @256 (30-block, matches CVC arch)
  - ResNet50-UNet (CNN): best-unet-isic18.pth            @256 (smp, logits)
  - Swin-UNet     (ViT): best-swinunet-isic18.pth        @224 (logits)

Per-image Dice is recorded for downstream paired statistics.
Results -> interventions/results/isic_cross_arch_eval.json

Usage:
    python interventions/experiments/eval_isic_cross_arch.py
"""

import sys, os, argparse, json, datetime, glob

import torch
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Canonical path order: repo root, SpectralMamba, tta_boundary_study. Keep
# SpectralMamba BEFORE SpectralMamba/VM-UNet so `models.vmunet` resolves to the
# canonical 30-block VMUNet (same class the Phase-3 CVC experiments use).
sys.path.insert(0, os.path.join(_REPO, 'tta_boundary_study'))
sys.path.insert(0, os.path.join(_REPO, 'SpectralMamba'))
sys.path.insert(0, _REPO)

# Same FFT input-space augmentation as the TSA training/eval.
from interventions.train_vmunet_cvc_tsa import apply_tsa

CUTOFF = 0.25
ISIC_ROOT = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'data', 'isic18', 'train')
IMG_DIR = os.path.join(ISIC_ROOT, 'images')
MASK_DIR = os.path.join(ISIC_ROOT, 'masks')
N_VAL = 50                       # match exp2 / avr_analysis.py protocol
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        # CRITICAL: the ISIC VM-UNet checkpoints were trained with the VM-UNet
        # repo's VSSM implementation (SpectralMamba/VM-UNet/models/vmunet), NOT
        # the canonical SpectralMamba/models copy. The two produce different
        # outputs with the same weights, so we must use the repo module here.
        _repo_vmunet = os.path.join(_REPO, 'SpectralMamba', 'VM-UNet')
        if _repo_vmunet not in sys.path:
            sys.path.insert(0, _repo_vmunet)
        from models.vmunet.vmunet import VMUNet
        _NC = {'num_classes': 1, 'input_channels': 3,
               'depths': [2, 2, 9, 2], 'depths_decoder': [2, 9, 2, 2],
               'drop_path_rate': 0.2}
        model = VMUNet(**_NC).to(device)
    elif arch == 'resnet50':
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                         in_channels=3, classes=1).to(device)
    elif arch == 'swinunet':
        swin_dir = os.path.join(_REPO, 'SpectralMamba', 'Swin-Unet')
        if swin_dir not in sys.path:
            sys.path.insert(0, swin_dir)      # config.py, networks/, configs/
        from config import get_config
        from networks.vision_transformer import SwinUnet
        class _MockArgs:
            cfg = 'configs/swin_tiny_patch4_window7_224_lite.yaml'
            opts = None; batch_size = 8; zip = False; cache_mode = 'part'
            resume = None; accumulation_steps = None; use_checkpoint = False
            amp_opt_level = 'O0'; tag = 'test'; eval = False; throughput = False
        cwd = os.getcwd()
        os.chdir(swin_dir)                    # config path is relative to Swin-Unet
        try:
            cfg = get_config(_MockArgs())
            model = SwinUnet(cfg, img_size=224, num_classes=1).to(device)
        finally:
            os.chdir(cwd)
    else:
        raise ValueError(f'Unknown arch: {arch}')

    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)
    model.eval()
    return model


def make_lp_hook(interv, layout):
    """layout: 'nhwc' (B,H,W,C), 'nchw', or 'tokens' (B,H*W,C)."""
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if layout == 'tokens':
            B, L, C = out.shape
            H = int(round(L ** 0.5))
            if H * H != L:
                return out
            fmap = out.permute(0, 2, 1).reshape(B, C, H, H).contiguous()
            modified = interv(fmap)
            return modified.reshape(B, H * H, C).contiguous()
        if layout == 'nhwc':
            fmap = out.permute(0, 3, 1, 2).contiguous()
            modified = interv(fmap)
            return modified.permute(0, 2, 3, 1).contiguous()
        # nchw
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
        for t in targets:
            handles.append(_resolve(model, t).register_forward_hook(make_lp_hook(interv, 'nchw')))
    elif arch == 'swinunet':
        for nm, mod in model.named_modules():
            if 'SwinTransformerBlock' in type(mod).__name__:
                handles.append(mod.register_forward_hook(make_lp_hook(interv, 'tokens')))
    return handles


def load_isic_pairs():
    """Return [(raw_img [0,1] (3,H,W) float, gt mask (1,H,W) float), ...] for the val subset."""
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, '*.jpg')))[:N_VAL]
    pairs = []
    for p in img_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        mask_p = os.path.join(MASK_DIR, base + '_segmentation.png')
        if not os.path.exists(mask_p):
            raise FileNotFoundError(f'Missing mask for {p}')
        img = np.array(Image.open(p).convert('RGB'), dtype=np.float32) / 255.0   # (H,W,3) [0,1]
        msk = np.array(Image.open(mask_p).convert('L'), dtype=np.float32) / 255.0
        pairs.append((torch.from_numpy(img).permute(2, 0, 1),
                      (torch.from_numpy(msk) > 0.5).float().unsqueeze(0)))
    return pairs


def normalize(img):
    t = img.clone()
    for c in range(3):
        t[c] = (t[c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
    return t


def evaluate(model, pairs, device, size, needs_sigmoid, mode='clean'):
    """mode: 'clean' | 'blur' (input-space) | 'feat' (feature-space hooks pre-attached).
    Resize to model resolution first (matches training preprocessing), then blur/normalize."""
    import torch.nn.functional as F
    per_dice, preds, gts = [], [], []
    with torch.no_grad():
        for img01, gt in pairs:
            img_rs = F.interpolate(img01.unsqueeze(0), size=(size, size),
                                   mode='bilinear', align_corners=False).squeeze(0)
            gt_rs = F.interpolate(gt.unsqueeze(0), size=(size, size),
                                  mode='nearest').squeeze(0)
            if mode == 'blur':
                img_rs = apply_tsa(img_rs.unsqueeze(0), cutoff=CUTOFF, p=1.0).squeeze(0)
            inp = normalize(img_rs).unsqueeze(0).to(device)
            out = model(inp)
            probs = torch.sigmoid(out) if needs_sigmoid else out
            p_np = probs.squeeze(1).squeeze(0).cpu().numpy()
            gt_np = gt_rs.squeeze(0).cpu().numpy()
            per_dice.append(per_image_dice(p_np, gt_np))
            preds.append(p_np.ravel())
            gts.append(gt_np.ravel())
    pooled = compute_segmentation_metrics(np.concatenate(preds), np.concatenate(gts))
    return pooled, float(np.mean(per_dice)), float(np.std(per_dice)), per_dice


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_json', default=os.path.join(
        _REPO, 'interventions', 'results', 'isic_cross_arch_eval.json'))
    ap.add_argument('--seed', type=int, default=42)
    args, _ = ap.parse_known_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 84)
    print('Cross-architecture Fourier benchmark  |  ISIC2018, cutoff=0.25')
    print('=' * 84)
    print(f'Device: {device}')

    pairs = load_isic_pairs()
    print(f'Val: {len(pairs)} images (first {N_VAL} of ISIC18)')

    MODELS = [
        ('VM-UNet', 'SSM', 'vmunet',
         os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                      'best-vmunet-scratch-isic18.pth'), 256, False),
        ('ResNet50-UNet', 'CNN', 'resnet50',
         os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                      'best-unet-isic18.pth'), 256, True),
        ('Swin-UNet', 'ViT', 'swinunet',
         os.path.join(_REPO, 'SpectralMamba', 'VM-UNet', 'best-ckpt',
                      'best-swinunet-isic18.pth'), 224, True),
    ]

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, CUTOFF, device=dev, dtype=dt)
    interv_lp = FrequencyIntervention(lp_fn, check_nan=True)

    results = {'experiment': 'Cross-architecture Fourier benchmark (ISIC2018)',
               'cutoff': CUTOFF, 'num_images': len(pairs),
               'device': str(device), 'timestamp': datetime.datetime.now().isoformat(),
               'models': []}

    for name, family, arch, ckpt, size, needs_sigmoid in MODELS:
        print(f'\n--- {name} ({family}, {size}x{size}) ---')
        model = build_model(arch, ckpt, device)
        print(f'  checkpoint: {os.path.basename(ckpt)} loaded (strict=True)')

        # Clean
        clean, cmean, cstd, cper = evaluate(model, pairs, device, size, needs_sigmoid, 'clean')
        print(f"  Clean Dice: {clean['dice']:.4f} (per-img {cmean:.4f}±{cstd:.4f})")

        # Feature-space low-pass
        handles = attach_lp(model, arch, interv_lp)
        feat, fmean, fstd, fper = evaluate(model, pairs, device, size, needs_sigmoid, 'feat')
        for h in handles:
            h.remove()
        print(f"  Feat-LP Dice: {feat['dice']:.4f} (per-img {fmean:.4f}±{fstd:.4f})")

        # Input-space blur
        blur, bmean, bstd, bper = evaluate(model, pairs, device, size, needs_sigmoid, 'blur')
        print(f"  Input-LP Dice:{blur['dice']:.4f} (per-img {bmean:.4f}±{bstd:.4f})")

        results['models'].append({
            'name': name, 'family': family, 'architecture': arch,
            'checkpoint': ckpt, 'size': size, 'hooks': len(handles),
            'clean': clean, 'feature_lp': feat, 'input_blur': blur,
            'per_image_dice': {'clean': cper, 'feature_lp': fper, 'input_blur': bper},
            'delta_dice_feature': feat['dice'] - clean['dice'],
            'delta_pct_feature': (feat['dice'] - clean['dice']) / (clean['dice'] + 1e-12) * 100.0,
            'delta_dice_input': blur['dice'] - clean['dice'],
            'delta_pct_input': (blur['dice'] - clean['dice']) / (clean['dice'] + 1e-12) * 100.0,
        })

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved -> {args.output_json}')

    print('\n' + '=' * 84)
    print('ISIC feature-space low-pass benchmark (cutoff=0.25)')
    print('=' * 84)
    print('| Architecture | Family | Clean Dice | Feat-LP Dice | Abs Drop | Rel % |')
    print('|---|---|---:|---:|---:|---:|')
    for m in results['models']:
        print(f'| {m["name"]} | {m["family"]} | {m["clean"]["dice"]:.4f} | '
              f'{m["feature_lp"]["dice"]:.4f} | {m["delta_dice_feature"]:.4f} | '
              f'{m["delta_pct_feature"]:+.1f}% |')

    print('\nISIC input-space low-pass benchmark (cutoff=0.25)')
    print('| Architecture | Family | Clean Dice | Input-LP Dice | Abs Drop | Rel % |')
    print('|---|---|---:|---:|---:|---:|')
    for m in results['models']:
        print(f'| {m["name"]} | {m["family"]} | {m["clean"]["dice"]:.4f} | '
              f'{m["input_blur"]["dice"]:.4f} | {m["delta_dice_input"]:.4f} | '
              f'{m["delta_pct_input"]:+.1f}% |')


if __name__ == '__main__':
    main()
