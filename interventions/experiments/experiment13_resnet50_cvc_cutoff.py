"""Exp 13: UNet-ResNet50 / CVC-ClinicDB cutoff sweep (0.10-0.80), analogue of Exp 4."""
import sys, os, json, csv, datetime, argparse, random
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in [_REPO, os.path.join(_REPO, "SpectralMamba"), os.path.join(_REPO, "tta_boundary_study")]:
    sys.path.insert(0, p)

HOOKS = [
    ("e1", "encoder.layer1[-1]"), ("e2", "encoder.layer2[-1]"),
    ("e3", "encoder.layer3[-1]"), ("e4", "encoder.layer4[-1]"),
    ("br", "decoder.blocks[0]"),
    ("d1", "decoder.blocks[1]"), ("d2", "decoder.blocks[2]"),
    ("d3", "decoder.blocks[3]"), ("d4", "decoder.blocks[4]"),
]


def resolve(model, path):
    if "[-1]" in path:
        base, _ = path.rsplit("[-1]", 1)
        m = model
        for part in base.split("."):
            m = getattr(m, part)
        return m[-1]
    m = model
    for chunk in path.split("."):
        if "[" in chunk:
            name, idx = chunk.split("[")
            m = getattr(m, name)[int(idx.rstrip("]"))]
        else:
            m = getattr(m, chunk)
    return m


def mk_hook(interv):
    def hook(m, i, o):
        if isinstance(o, tuple):
            o = o[0]
        return interv(o)
    return hook


def dice_iou(p_flat, g_flat):
    yp = (p_flat >= 0.5).astype(int)
    yg = (g_flat >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yg, yp, labels=[0, 1]).ravel()
    d = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    i = tp / (tp + fp + fn + 1e-8)
    return d, i


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt_path", default=None)
    ap.add_argument("--img_size", type=int, default=256)
    args, _ = ap.parse_known_args()
    random.seed(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)

    from interventions.intervention import FrequencyIntervention
    from interventions.masks import lowpass_mask
    from src.datasets.cvc_dataset import CVCDataset
    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Exp 13: UNet-ResNet50 CVC cutoff sweep |", device)

    ckpt = args.ckpt_path or os.path.join(_REPO, "tta_boundary_study", "checkpoints", "unet_cvc_best.pth")
    img_dir = os.path.join(_REPO, "tta_boundary_study", "cvc_clinicdb", "original")
    mask_dir = os.path.join(_REPO, "tta_boundary_study", "cvc_clinicdb", "ground_truth")
    ds = CVCDataset(img_dir, mask_dir, split="val", img_size=args.img_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    sd = torch.load(ckpt, map_location="cpu")
    if isinstance(sd, dict) and "encoder.conv1.weight" not in sd:
        sd = sd.get("model_state_dict") or sd.get("state_dict") or sd

    rows = []
    for c in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1).to(device)
        model.load_state_dict(sd, strict=True)
        model.eval()
        interv = FrequencyIntervention(
            lambda h, w, dev, dt, cc=c: lowpass_mask(h, w, cc, device=dev, dtype=dt))
        handles = [resolve(model, p).register_forward_hook(mk_hook(interv)) for _, p in HOOKS]
        preds, gts = [], []
        with torch.no_grad():
            for imgs, masks in loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds.append(torch.sigmoid(model(imgs)).squeeze(1).cpu().numpy())
                gts.append(masks.squeeze(1).cpu().numpy())
        for h in handles:
            h.remove()
        del model
        d, i = dice_iou(np.concatenate([p.ravel() for p in preds]),
                        np.concatenate([g.ravel() for g in gts]))
        rows.append({"cutoff": c, "dice": d, "iou": i})
        print(f"  cutoff={c:.2f}  Dice={d:.4f}  IoU={i:.4f}")

    ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = args.output_dir or os.path.join(_REPO, "interventions", "results", "experiment13_resnet50_cvc_cutoff")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, f"cutoff_sweep_{ts}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cutoff", "dice", "iou"])
        w.writeheader(); w.writerows(rows)
    meta = {"experiment": "Exp 13: UNet-ResNet50 CVC cutoff sweep", "dataset": "CVC-ClinicDB",
            "img_size": args.img_size, "num_images": len(ds), "cutoffs": [r["cutoff"] for r in rows],
            "device": str(device), "timestamp": ts}
    with open(os.path.join(out, f"metadata_{ts}.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()