"""Exp 13 valley: radius nesting + multi-seed determinism check."""
import sys, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [_ROOT, os.path.join(_ROOT, "SpectralMamba"), os.path.join(_ROOT, "tta_boundary_study"),
          os.path.join(_ROOT, "interventions", "experiments")]:
    sys.path.insert(0, p)
from experiment13_resnet50_cvc_cutoff import resolve, mk_hook, dice_iou, HOOKS


def main():
    from src.datasets.cvc_dataset import CVCDataset
    import segmentation_models_pytorch as smp
    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(_ROOT, "tta_boundary_study", "checkpoints", "unet_cvc_best.pth")
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "encoder.conv1.weight" not in sd:
        sd = sd.get("model_state_dict") or sd.get("state_dict") or sd
    img_dir = os.path.join(_ROOT, "tta_boundary_study", "cvc_clinicdb", "original")
    mask_dir = os.path.join(_ROOT, "tta_boundary_study", "cvc_clinicdb", "ground_truth")

    print("radius_px @352 (max_radius=176):")
    for c in [0.10, 0.15, 0.20, 0.25]:
        print(f"  cutoff {c:.2f} -> {c*176:.1f}px, nested monotone: {c*176 <= 0.25*176}")

    print("\nvalley cutoffs x 2 python seeds:")
    for c in [0.10, 0.20, 0.25]:
        out = []
        for s in [2, 999]:
            random.seed(s); torch.manual_seed(s); np.random.seed(s)
            ds = CVCDataset(img_dir, mask_dir, split="val", img_size=352)
            loader = DataLoader(ds, batch_size=1, shuffle=False)
            m = smp.Unet(encoder_name="resnet50", encoder_weights=None,
                         in_channels=3, classes=1).to(device)
            m.load_state_dict(sd, strict=True); m.eval()
            iv = FrequencyIntervention(
                lambda h, w, dev, dt, cc=c: lowpass_mask(h, w, cc, device=dev, dtype=dt))
            hs = [resolve(m, p).register_forward_hook(mk_hook(iv)) for _, p in HOOKS]
            pr, gt = [], []
            with torch.no_grad():
                for im, mk in loader:
                    im, mk = im.to(device), mk.to(device)
                    pr.append(torch.sigmoid(m(im)).squeeze(1).cpu().numpy())
                    gt.append(mk.squeeze(1).cpu().numpy())
            for h in hs:
                h.remove()
            del m
            d, _ = dice_iou(np.concatenate([x.ravel() for x in pr]),
                            np.concatenate([x.ravel() for x in gt]))
            out.append(round(d, 6))
        print(f"  cutoff {c:.2f}: s2={out[0]} s999={out[1]} identical={out[0]==out[1]}")


if __name__ == "__main__":
    main()