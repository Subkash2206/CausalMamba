"""
make_robustness_figures.py â€” Qualitative figures for the spectral-robustness paper.

For a few representative CVC-ClinicDB validation images, renders:
  (1) Input / GT / prediction triplets under Clean, Input-space blur (cutoff 0.25)
      and Feature-space low-pass (cutoff 0.25) for each model.
  (2) An FFT log-magnitude spectrum of a first-VSSBlock feature map before and
      after the low-pass intervention (VM-UNet) to visualise the ablation.

Outputs PNGs to interventions/results/figures/.

Usage:
    python interventions/experiments/make_robustness_figures.py
"""

import sys, os

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(_REPO, 'tta_boundary_study'))
sys.path.insert(0, os.path.join(_REPO, 'SpectralMamba'))
sys.path.insert(0, _REPO)

from interventions.train_vmunet_cvc_tsa import apply_tsa

CVC_IMG_DIR = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'original')
CVC_MASK_DIR = os.path.join(_REPO, 'tta_boundary_study', 'cvc_clinicdb', 'ground_truth')
FIG_DIR = os.path.join(_REPO, 'interventions', 'results', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

SAMPLE_IDX = [10, 40, 80]   # representative val images (diverse sizes)


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
        raise ValueError(arch)
    sd = torch.load(ckpt, map_location=device)
    if isinstance(sd, dict):
        sd = sd.get('model_state_dict') or sd.get('state_dict') or sd
    model.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()}, strict=True)
    model.eval()
    return model


def attach_lp(model, arch, interv):
    def make_hook(layout):
        def hook(module, inp, out):
            if isinstance(out, tuple):
                out = out[0]
            if layout == 'nhwc':
                fmap = out.permute(0, 3, 1, 2).contiguous()
                return interv(fmap).permute(0, 2, 3, 1).contiguous()
            return interv(out)
        return hook

    handles = []
    if arch == 'vmunet':
        for nm, mod in model.named_modules():
            if 'VSSBlock' in type(mod).__name__:
                handles.append(mod.register_forward_hook(make_hook('nhwc')))
    elif arch == 'resnet50':
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
        targets = ['encoder.layer1[-1]', 'encoder.layer2[-1]', 'encoder.layer3[-1]',
                   'encoder.layer4[-1]', 'decoder.blocks[0]', 'decoder.blocks[1]',
                   'decoder.blocks[2]', 'decoder.blocks[3]', 'decoder.blocks[4]']
        for t in targets:
            handles.append(_resolve(model, t).register_forward_hook(make_hook('nchw')))
    elif arch == 'swinunetr':
        swin = model.swinViT
        for i in range(1, 5):
            layer = getattr(swin, f'layers{i}')[0]
            for blk in layer.blocks:
                handles.append(blk.register_forward_hook(make_hook('nhwc')))
    return handles


def main():
    from src.datasets.cvc_dataset import CVCDataset
    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Generating qualitative figures...')

    ds = CVCDataset(CVC_IMG_DIR, CVC_MASK_DIR, split='val', img_size=256)
    MODELS = [
        ('VM-UNet', 'vmunet',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'best-vmunet-cvc.pth'), False),
        ('TSA-VM-UNet', 'vmunet',
         os.path.join(_REPO, 'interventions', 'results', 'best-vmunet-cvc-tsa-finetune',
                      'best-vmunet-cvc.pth'), False),
        ('ResNet50-UNet', 'resnet50',
         os.path.join(_REPO, 'tta_boundary_study', 'checkpoints', 'unet_cvc_best_256.pth'), True),
        ('Swin-UNETR', 'swinunetr',
         os.path.join(_REPO, 'interventions', 'results', 'best-swinunetr-cvc-256', 'best-swinunetr-cvc-256.pth'), True),
    ]
    models = [(name, build_model(arch, ckpt, device), sig) for name, arch, ckpt, sig in MODELS]

    lp_fn = lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt)
    interv = FrequencyIntervention(lp_fn, check_nan=True)

    for idx in SAMPLE_IDX:
        img, mask = ds[idx]
        img = img.unsqueeze(0).to(device)
        gt = mask.squeeze(0).cpu().numpy()
        rows = []
        for cond, blur, feat in [('Clean', False, False), ('Input-LP', True, False),
                                 ('Feature-LP', False, True)]:
            row = [cond]
            disp = img
            for name, model, sig in models:
                arch = [m[1] for m in MODELS if m[0] == name][0]
                handles = attach_lp(model, arch, interv) if feat else []
                with torch.no_grad():
                    inp = apply_tsa(img, cutoff=0.25, p=1.0) if blur else img
                    out = model(inp)
                for h in handles:
                    h.remove()
                if blur:
                    disp = inp
                probs = torch.sigmoid(out) if sig else out
                row.append(probs.squeeze(1).squeeze(0).cpu().numpy())
            rows.append((cond, disp, row[1:]))

        n_cols = 1 + 4
        fig, axes = plt.subplots(3, n_cols, figsize=(3.0 * n_cols, 9.5))
        for r, (cond, disp, preds) in enumerate(rows):
            img_np = disp.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            axes[r][0].imshow(np.clip(img_np, 0, 1))
            axes[r][0].set_ylabel(cond, rotation=0, labelpad=28, fontsize=11)
            axes[r][0].axis('off')
            for c, p in enumerate(preds):
                axes[r][c + 1].imshow(p, cmap='gray', vmin=0, vmax=1)
                axes[r][c + 1].set_title([n for n, *_ in MODELS][c], fontsize=11)
                axes[r][c + 1].axis('off')
        axes[0][0].set_title('Input', fontsize=11)
        plt.tight_layout()
        out_p = os.path.join(FIG_DIR, f'cvc_sample_{idx}.png')
        plt.savefig(out_p, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'  saved {out_p}')

    # FFT spectrum of first VSSBlock (VM-UNet) before/after intervention
    print('  generating feature-spectrum figure...')
    vm = models[0][1]   # VM-UNet
    img, _ = ds[SAMPLE_IDX[0]]
    img = img.unsqueeze(0).to(device)
    store = {}
    def cap(mod, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        store['feat'] = out.permute(0, 3, 1, 2).detach()
        return out
    h = vm.vmunet.layers[0].blocks[0].register_forward_hook(cap)
    with torch.no_grad():
        _ = vm(img)
    h.remove()
    f0 = store['feat'].float()
    h2 = attach_lp(vm, 'vmunet', interv)
    with torch.no_grad():
        _ = vm(img)
    for hh in h2:
        hh.remove()
    f1 = store['feat'].float()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, f, ttl in [(axes[0], f0, 'Before LP'), (axes[1], f1, 'After LP')]:
        spec = torch.log1p(torch.fft.fftshift(torch.fft.fft2(f.mean(1), norm='ortho')).abs())
        ax.imshow(spec.squeeze(0).cpu().numpy(), cmap='magma')
        ax.set_title(ttl, fontsize=11)
        ax.axis('off')
    out_s = os.path.join(FIG_DIR, 'vmunet_feature_spectrum.png')
    plt.tight_layout()
    plt.savefig(out_s, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  saved {out_s}')


if __name__ == '__main__':
    main()
