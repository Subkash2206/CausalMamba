"""
eval_input_space_tsa.py — Input-space (blur) robustness evaluation of the TSA defense.

The TSA defense is an INPUT-SPACE augmentation, so its true clinical robustness
must be tested against INPUT-level degradation (blurring), NOT the internal
feature-space ablation. This script:

  1. Loads the CVC-ClinicDB validation set (123 images, 256x256).
  2. Applies the FFT low-pass filter (cutoff=0.25) STRICTLY to the INPUT images
     using the SAME apply_tsa() logic from training (p=1.0 -> every image
     blurred, deterministic).
  3. Runs a STANDARD forward pass (no internal block hooks).
  4. Evaluates the Baseline VM-UNet and the TSA-Defense VM-UNet.

Results -> interventions/results/input_space_tsa_eval.json + markdown table.

Usage:
    python interventions/experiments/eval_input_space_tsa.py
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

# Same FFT low-pass input augmentation used during TSA training (with clamp).
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


def build_vmunet(ckpt_path, device):
    from models.vmunet.vmunet import VMUNet
    _NC = {'num_classes': 1, 'input_channels': 3,
           'depths': [2, 2, 9, 2], 'depths_decoder': [2, 9, 2, 2],
           'drop_path_rate': 0.2}
    model = VMUNet(**_NC).to(device)
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
                imgs = apply_tsa(imgs, cutoff=CUTOFF, p=1.0)   # every image blurred
            probs = model(imgs)  # VMUNet returns sigmoid probabilities
            preds.append(probs.squeeze(1).cpu().numpy())
            gts.append(masks.squeeze(1).cpu().numpy())
    return compute_segmentation_metrics(
        np.concatenate([p.ravel() for p in preds]),
        np.concatenate([g.ravel() for g in gts]))


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument('--output_json', default=os.path.join(
        _REPO, 'interventions', 'results', 'input_space_tsa_eval.json'))
    ap.add_argument('--seed', type=int, default=42)
    args, _ = ap.parse_known_args()

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from src.datasets.cvc_dataset import CVCDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('=' * 80)
    print('Input-space low-pass (blur) evaluation  |  CVC-ClinicDB, cutoff=0.25, 256')
    print('=' * 80)
    print(f'Device: {device}')

    val_ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='val', img_size=IMG_SIZE)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    print(f'Val: {len(val_ds)} images @ {IMG_SIZE}x{IMG_SIZE}')

    MODELS = [
        ('Baseline VM-UNet', os.path.join(_REPO, 'tta_boundary_study', 'checkpoints',
                                          'best-vmunet-cvc.pth')),
        ('TSA-Defense VM-UNet', os.path.join(_REPO, 'interventions', 'results',
                                             'best-vmunet-cvc-tsa-finetune',
                                             'best-vmunet-cvc.pth')),
    ]

    results = {'experiment': 'Input-space low-pass (blur) evaluation - TSA defense',
               'cutoff': CUTOFF, 'img_size': IMG_SIZE, 'num_images': len(val_ds),
               'device': str(device), 'timestamp': datetime.datetime.now().isoformat(),
               'models': []}

    for name, ckpt in MODELS:
        print(f'\n--- {name} ---')
        model = build_vmunet(ckpt, device)
        print(f'  checkpoint: {os.path.basename(ckpt)} loaded (strict=True)')

        clean = evaluate(model, val_loader, device, blur=False)
        blurred = evaluate(model, val_loader, device, blur=True)
        print(f"  Clean Dice:         {clean['dice']:.4f}")
        print(f"  Input-Blurred Dice: {blurred['dice']:.4f}")

        delta = blurred['dice'] - clean['dice']
        pct = (delta / (clean['dice'] + 1e-12)) * 100.0
        print(f'  Dice delta: {delta:+.4f} ({pct:+.2f}%)')

        results['models'].append({
            'name': name, 'checkpoint': ckpt,
            'clean': clean, 'blurred': blurred,
            'delta_dice': delta, 'delta_pct': pct,
        })

    with open(args.output_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved structured results -> {args.output_json}')

    print('\n' + '=' * 80)
    print('Input-space low-pass benchmark (cutoff=0.25 on inputs only)')
    print('=' * 80)
    print('| Model | Clean Dice | Input-Blurred Dice | Absolute Drop |')
    print('|---|---|---:|---:|')
    for m in results['models']:
        print(f'| {m["name"]} | {m["clean"]["dice"]:.4f} | {m["blurred"]["dice"]:.4f} | '
              f'{m["delta_dice"]:.4f} |')
    for m in results['models']:
        print(f'  {m["name"]}: relative drop {m["delta_pct"]:+.2f}%')


if __name__ == '__main__':
    main()
