"""
cross_arch_cvc_eval.py â€” Cross-architecture Fourier Low-Pass benchmark on CVC-ClinicDB.

Evaluates 4 trained checkpoints under Clean and Low-Pass (cutoff=0.25) conditions
on the CVC-ClinicDB validation set (123 images, 256x256):

  1. VM-UNet          (SSM)  - best-vmunet-cvc.pth
  2. VM-UNet TSA      (SSM)  - best-vmunet-cvc-tsa-finetune/best-vmunet-cvc.pth (Defense)
  3. ResNet50-UNet    (CNN)  - unet_cvc_best_256.pth
  4. Swin-UNETR       (ViT)  - swinunetr_cvc_best.pth

The low-pass intervention applies the repo's FFT->mask->IFFT pipeline to every
semantic block of each architecture (VM-UNet: 30 VSSBlocks; ResNet50-UNet: 9
stages; Swin-UNETR: 8 transformer blocks). Results -> cross_arch_cvc_eval.json
and a markdown summary table.

Usage:
    python interventions/experiments/cross_arch_cvc_eval.py
"""

import sys, os, argparse, json, datetime

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
for p in [_REPO, os.path.join(_REPO, 'SpectralMamba'),
          os.path.join(_REPO, 'tta_boundary_study')]:
    sys.path.insert(0, p)

# Robust console output on Windows (cp1252 default): allow delta symbols.
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

CVC_IMG_DIR = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
CVC_MASK_DIR = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')

CUTOFF = 0.25
IMG_SIZE = 256


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


def _resolve_path(model, path):
    if '[-1]' in path:
        base, _ = path.rsplit('[-1]', 1)
        module = model
        for part in base.split('.'):
            module = getattr(module, part)
        return module[-1]
    module = model
    for chunk in path.split('.'):
        if '[' in chunk:
            name, idx = chunk.split('[')
            module = getattr(module, name)[int(idx.rstrip(']'))]
        else:
            module = getattr(module, chunk)
    return module


# ResNet50-UNet semantic stages (identical to Experiments 9/10/12)
HOOK_TARGETS = [
    ('encoder.block1', 'encoder.layer1[-1]'),
    ('encoder.block2', 'encoder.layer2[-1]'),
    ('encoder.block3', 'encoder.layer3[-1]'),
    ('encoder.block4', 'encoder.layer4[-1]'),
    ('bridge',         'decoder.blocks[0]'),
    ('decoder.block1', 'decoder.blocks[1]'),
    ('decoder.block2', 'decoder.blocks[2]'),
    ('decoder.block3', 'decoder.blocks[3]'),
    ('decoder.block4', 'decoder.blocks[4]'),
]


def make_lp_hook(interv, nhwc):
    """Forward hook applying the frequency intervention (NHWC permuted if needed)."""
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if nhwc:  # (B, H, W, C) -> (B, C, H, W) for the FFT
            fmap = out.permute(0, 3, 1, 2).contiguous()
        else:
            fmap = out
        modified = interv(fmap)
        return modified.permute(0, 2, 3, 1).contiguous() if nhwc else modified
    return hook


def load_ckpt(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
    clean = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=True)
    return model


def build_model(arch, ckpt_path, device):
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
    load_ckpt(model, ckpt_path, device)
    model.eval()
    return model


def attach_lp(model, arch, interv):
    """Hook every semantic block of the given architecture with the intervention."""
    handles = []
    if arch == 'vmunet':
        for nm, mod in model.named_modules():
            if 'VSSBlock' in type(mod).__name__:
                handles.append(mod.register_forward_hook(make_lp_hook(interv, nhwc=True)))
    elif arch == 'resnet50':
        for name, path in HOOK_TARGETS:
            handles.append(_resolve_path(model, path)
                           .register_forward_hook(make_lp_hook(interv, nhwc=False)))
    elif arch == 'swinunetr':
        swin = model.swinViT
        for i in range(1, 5):
            layer = getattr(swin, f'layers{i}')[0]
            for blk in layer.blocks:
                handles.append(blk.register_forward_hook(make_lp_hook(interv, nhwc=True)))
    return handles


def evaluate(model, loader, device, needs_sigmoid):
    """Pooled segmentation metrics over the whole val set."""
    preds, gts = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            probs = torch.sigmoid(out) if needs_sigmoid else out
            preds.append(probs.squeeze(1).cpu().numpy())
            gts.append(masks.squeeze(1).cpu().numpy())
    return compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in preds]),
        np.concatenate([g.ravel() for g in gts]))


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_json', default=os.path.join(
        _REPO, 'interventions', 'results', 'cross_arch_cvc_eval.json'))
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
    print('Cross-architecture Fourier Low-Pass benchmark  (CVC-ClinicDB, cutoff=0.25, 256)')
    print('=' * 90)
    print(f'Device: {device}')

    val_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='val', img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'Val: {len(val_ds)} images @ {IMG_SIZE}x{IMG_SIZE}')

    # (name, family, arch, checkpoint, needs_sigmoid)
    CKPTS = [
        ('VM-UNet', 'SSM', 'vmunet',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'best-vmunet-cvc.pth'),
         False),   # VMUNet returns sigmoid probabilities internally
        ('VM-UNet-TSA (Defense)', 'SSM', 'vmunet',
         os.path.join(_REPO, 'interventions', 'results', 'best-vmunet-cvc-tsa-finetune',
                      'best-vmunet-cvc.pth'),
         False),
        ('ResNet50-UNet', 'CNN', 'resnet50',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best_256.pth'),
         True),    # logits
        ('Swin-UNETR', 'ViT', 'swinunetr',
         os.path.join(_REPO, 'interventions', 'results', 'best-swinunetr-cvc-256', 'best-swinunetr-cvc-256.pth'),
         True),    # logits
    ]

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, CUTOFF, device=dev, dtype=dt)
    interv_lp = FrequencyIntervention(lp_fn, check_nan=True)

    results = {'experiment': 'Cross-architecture Fourier low-pass benchmark (CVC-ClinicDB)',
               'cutoff': CUTOFF, 'img_size': IMG_SIZE, 'num_images': len(val_ds),
               'device': str(device), 'timestamp': datetime.datetime.now().isoformat(),
               'models': []}

    for name, family, arch, ckpt, needs_sigmoid in CKPTS:
        print(f'\n--- {name} ({family}) ---')
        model = build_model(arch, ckpt, device)
        print(f'  checkpoint: {os.path.basename(ckpt)} loaded (strict=True)')

        clean_metrics = evaluate(model, val_loader, device, needs_sigmoid)
        print(f"  Clean Dice:     {clean_metrics['dice']:.4f}   IoU: {clean_metrics['iou']:.4f}")

        handles = attach_lp(model, arch, interv_lp)
        print(f'  LP hooks: {len(handles)}')
        lp_metrics = evaluate(model, val_loader, device, needs_sigmoid)
        for h in handles:
            h.remove()
        print(f"  Low-Pass Dice:  {lp_metrics['dice']:.4f}   IoU: {lp_metrics['iou']:.4f}")

        delta = lp_metrics['dice'] - clean_metrics['dice']
        pct = (delta / (clean_metrics['dice'] + 1e-12)) * 100.0
        print(f'  Dice delta: {delta:+.4f} ({pct:+.2f}%)')

        results['models'].append({
            'name': name, 'family': family, 'architecture': arch,
            'checkpoint': ckpt, 'hooks': len(handles),
            'clean': clean_metrics,
            'lowpass': lp_metrics,
            'delta_dice': delta,
            'delta_pct': pct,
        })

    # ------------------------------------------------------------------
    # Save + markdown table
    # ------------------------------------------------------------------
    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved structured results -> {args.output_json}')

    print('\n' + '=' * 90)
    print('Cross-architecture benchmark (CVC-ClinicDB val, low-pass cutoff=0.25)')
    print('=' * 90)
    hdr = f'{"Architecture":<22}{"Family":<7}{"Clean Dice":>12}{"LP Dice":>12}{"Abs Drop":>10}{"Rel %":>9}'
    print(hdr)
    print('-' * len(hdr))
    for m in results['models']:
        print(f'{m["name"]:<22}{m["family"]:<7}{m["clean"]["dice"]:>12.4f}'
              f'{m["lowpass"]["dice"]:>12.4f}{m["delta_dice"]:>10.4f}{m["delta_pct"]:>+8.2f}%')

    print('\n--- Markdown (manuscript-ready) ---')
    print('| Architecture | Model Family | Clean Dice | Low-Pass Dice (0.25) | Absolute Drop |')
    print('|---|---|---:|---:|---:|')
    for m in results['models']:
        print(f'| {m["name"]} | {m["family"]} | {m["clean"]["dice"]:.4f} | '
              f'{m["lowpass"]["dice"]:.4f} | {m["delta_dice"]:.4f} |')


if __name__ == '__main__':
    main()
