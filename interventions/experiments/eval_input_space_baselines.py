"""
eval_input_space_baselines.py â€” Input-space (blur) evaluation of the CNN and ViT
baselines, completing the CVC-ClinicDB architectural benchmark.

Applies the EXACT same FFT input-space low-pass (cutoff=0.25, apply_tsa from the
TSA training script, p=1.0 -> every image blurred) as eval_input_space_tsa.py,
with a STANDARD forward pass (no internal block hooks), on:

  1. ResNet50-UNet (CNN): tta_boundary_study/checkpoints/unet_cvc_best_256.pth
  2. Swin-UNETR   (ViT): tta_boundary_study/checkpoints/swinunetr_cvc_best.pth

Results -> interventions/results/input_space_baselines_eval.json + markdown table.

Usage:
    python interventions/experiments/eval_input_space_baselines.py
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

# Same FFT low-pass input augmentation used in the TSA evaluation / training.
from interventions.train_vmunet_cvc_tsa import apply_tsa

CUTOFF = 0.25
IMG_SIZE = 256
CVC_IMG_DIR = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
CVC_MASK_DIR = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')


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


def build_model(arch, ckpt_path, device):
    if arch == 'resnet50':
        import segmentation_models_pytorch as smp
        model = smp.Unet(encoder_name='resnet50', encoder_weights=None,
                         in_channels=3, classes=1).to(device)
    elif arch == 'swinunetr':
        from src.models.swin_unetr_cvc import get_swin_unetr
        model = get_swin_unetr().to(device)
    else:
        raise ValueError(f'Unknown arch: {arch}')
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)
    model.eval()
    return model


def evaluate(model, loader, device, blur=False):
    """Pooled metrics; if blur=True, low-pass the input with apply_tsa (p=1.0)."""
    preds, gts = [], []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            if blur:
                imgs = apply_tsa(imgs, cutoff=CUTOFF, p=1.0)
            logits = model(imgs)          # both baselines output logits
            probs = torch.sigmoid(logits)
            preds.append(probs.squeeze(1).cpu().numpy())
            gts.append(masks.squeeze(1).cpu().numpy())
    return compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in preds]),
        np.concatenate([g.ravel() for g in gts]))


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_json', default=os.path.join(
        _REPO, 'interventions', 'results', 'input_space_baselines_eval.json'))
    ap.add_argument('--seed', type=int, default=42)
    args, _ = ap.parse_known_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from src.datasets.cvc_dataset import CVCDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('Input-space low-pass (blur) baselines  |  CVC-ClinicDB, cutoff=0.25, 256')
    print('=' * 80)
    print(f'Device: {device}')

    val_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='val', img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'Val: {len(val_ds)} images @ {IMG_SIZE}x{IMG_SIZE}')

    MODELS = [
        ('ResNet50-UNet', 'CNN', 'resnet50',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best_256.pth')),
        ('Swin-UNETR', 'ViT', 'swinunetr',
         os.path.join(_REPO, 'interventions', 'results', 'best-swinunetr-cvc-256', 'best-swinunetr-cvc-256.pth')),
    ]

    results = {'experiment': 'Input-space low-pass (blur) evaluation - CNN/ViT baselines',
               'cutoff': CUTOFF, 'img_size': IMG_SIZE, 'num_images': len(val_ds),
               'device': str(device), 'timestamp': datetime.datetime.now().isoformat(),
               'models': []}

    for name, family, arch, ckpt in MODELS:
        print(f'\n--- {name} ({family}) ---')
        model = build_model(arch, ckpt, device)
        print(f'  checkpoint: {os.path.basename(ckpt)} loaded (strict=True)')

        clean = evaluate(model, val_loader, device, blur=False)
        blurred = evaluate(model, val_loader, device, blur=True)
        print(f"  Clean Dice:         {clean['dice']:.4f}")
        print(f"  Input-Blurred Dice: {blurred['dice']:.4f}")

        delta = blurred['dice'] - clean['dice']
        pct = (delta / (clean['dice'] + 1e-12)) * 100.0
        print(f'  Dice delta: {delta:+.4f} ({pct:+.2f}%)')

        results['models'].append({
            'name': name, 'family': family, 'architecture': arch, 'checkpoint': ckpt,
            'clean': clean, 'blurred': blurred,
            'delta_dice': delta, 'delta_pct': pct,
        })

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved structured results -> {args.output_json}')

    print('\n' + '=' * 80)
    print('Input-space low-pass benchmark (cutoff=0.25 on inputs only)')
    print('=' * 80)
    print('| Architecture | Clean Dice | Input-Blurred Dice | Absolute Drop |')
    print('|---|---|---:|---:|')
    for m in results['models']:
        print(f'| {m["name"]} | {m["clean"]["dice"]:.4f} | {m["blurred"]["dice"]:.4f} | '
              f'{m["delta_dice"]:.4f} |')
    for m in results['models']:
        print(f'  {m["name"]}: relative drop {m["delta_pct"]:+.2f}%')


if __name__ == '__main__':
    main()
