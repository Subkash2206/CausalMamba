"""Audit verification for the CVC-ClinicDB UNet-ResNet50 results (asks 1b-1d).

1b: Identity-mask reconstruction test on the exact CVC pipeline.
1c: Save 5 (input, GT, baseline-pred, LP-pred) PNG pairs for visual inspection.
1d: Report unet_cvc_best.pth checkpoint metadata / structural introspection.
"""
import sys, os, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_REPO, os.path.join(_REPO, "SpectralMamba"), os.path.join(_REPO, "tta_boundary_study")):
    sys.path.insert(0, p)

OUT = os.path.join(_REPO, "interventions", "results", "cvc_audit")
os.makedirs(OUT, exist_ok=True)

from src.datasets.cvc_dataset import CVCDataset
from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask
import segmentation_models_pytorch as smp


def dice(pred, gt):
    inter = 2 * (pred * gt).sum()
    return (inter / (pred.sum() + gt.sum() + 1e-8)).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = os.path.join(_REPO, "tta_boundary_study", "checkpoints", "unet_cvc_best.pth")

    # ---- 1d: checkpoint metadata / introspection -------------------------
    ck = torch.load(ckpt, map_location="cpu")
    print("=" * 70)
    print("1d: unet_cvc_best.pth introspection")
    print("=" * 70)
    print("top-level type:", type(ck).__name__)
    if isinstance(ck, dict):
        meta_keys = [k for k in ck.keys() if not isinstance(ck[k], torch.Tensor)]
        print("non-tensor keys:", meta_keys)
        for k in meta_keys:
            v = ck[k]
            if isinstance(v, dict):
                print(" ", k, "->", {kk: (str(vv)[:60]) for kk, vv in list(v.items())[:6]})
            else:
                print(" ", k, "->", str(v)[:80])
    sd = (ck.get("model_state_dict") or ck.get("state_dict") or ck) if isinstance(ck, dict) else ck
    if not isinstance(sd, dict):
        sd = ck
    print("state_dict keys:", len(sd))
    print("sample:", list(sd.keys())[:4])

    # ---- build dataset / model -------------------------------------------
    img_dir = os.path.join(_REPO, "tta_boundary_study", "cvc_clinicdb", "original")
    mask_dir = os.path.join(_REPO, "tta_boundary_study", "cvc_clinicdb", "ground_truth")
    ds = CVCDataset(img_dir, mask_dir, split="val", img_size=352)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print(f"\nval images: {len(ds)}")

    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    print("model loaded (strict=True), params:", sum(p.numel() for p in model.parameters()))

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

    identity = FrequencyIntervention(lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt), check_nan=True)
    lowpass = FrequencyIntervention(lambda h, w, dev, dt: lowpass_mask(h, w, 0.25, device=dev, dtype=dt), check_nan=True)

    # ---- 1b: identity reconstruction over full val split ------------------
    print("\n" + "=" * 70)
    print("1b: Identity-mask reconstruction (full val)")
    print("=" * 70)
    model_id = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1).to(device)
    model_id.load_state_dict(sd, strict=True); model_id.eval()
    maxdiff = 0.0
    preds, gts = [], []
    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        with torch.no_grad():
            logits = model_id(imgs)
            pbase = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        preds.append(pbase); gts.append(masks.squeeze(1).cpu().numpy())
    p_flat = np.concatenate([x.ravel() for x in preds])
    g_flat = np.concatenate([x.ravel() for x in gts])
    y_p = (p_flat >= 0.5).astype(int); y_g = (g_flat >= 0.5).astype(int)
    cm = confusion_matrix(y_g, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    d = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    print(f"identity-run Dice = {d:.6f}  (baseline recorded = 0.9286)")
    print(f"|delta Dice| = {abs(d - 0.928592):.6f}")

    # ---- 1c: 5 visual pairs (baseline + low-pass) ------------------------
    print("\n" + "=" * 70)
    print("1c: saving 5 visual samples")
    print("=" * 70)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_bl = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1).to(device)
    model_bl.load_state_dict(sd, strict=True); model_bl.eval()
    model_lp = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1).to(device)
    model_lp.load_state_dict(sd, strict=True); model_lp.eval()

    saved = 0
    with torch.no_grad():
        for i, (imgs, masks) in enumerate(loader):
            imgs0 = imgs  # raw [0,1] tensor (352x352)
            imgs, masks = imgs.to(device), masks.to(device)
            pb = torch.sigmoid(model_bl(imgs)).squeeze(1).cpu().numpy()
            for name, path in HOOKS:
                resolve(model_lp, path).register_forward_hook(
                    lambda mod, inp, out, path=path: None
                )
            # low-pass model with actual hooks (fresh handles irrelevant for single forward)
            handles = []
            def mk_hook(iv):
                def hook(m, i, o):
                    if isinstance(o, tuple):
                        o = o[0]
                    return iv(o)
                return hook
            for name, path in HOOKS:
                handles.append(resolve(model_lp, path).register_forward_hook(mk_hook(lowpass)))
            pl = torch.sigmoid(model_lp(imgs)).squeeze(1).cpu().numpy()
            for h in handles:
                h.remove()

            inp_np = imgs0.squeeze(0).permute(1, 2, 0).numpy()  # (352,352,3) [0,1]
            gt_np = masks.squeeze(1).cpu().numpy()[0]
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            ax[0].imshow(inp_np); ax[0].set_title("input")
            ax[1].imshow(gt_np, cmap="gray"); ax[1].set_title("GT")
            ax[2].imshow(pb[0], cmap="gray"); ax[2].set_title(f"baseline d={dice((pb[0]>0.5).astype(int), gt_np):.3f}")
            ax[3].imshow(pl[0], cmap="gray"); ax[3].set_title(f"lowpass d={dice((pl[0]>0.5).astype(int), gt_np):.3f}")
            for a in ax:
                a.axis("off")
            fname = os.path.join(OUT, f"sample_{saved:03d}.png")
            plt.tight_layout(); plt.savefig(fname, dpi=120); plt.close()
            print(f"  saved {fname} (lp dice {dice((pl[0]>0.5).astype(int), gt_np):.3f})")
            saved += 1
            if saved >= 5:
                break

    print(f"\nvisuals saved to {OUT}")
    summary = {"identity_dice": d, "baseline_recorded": 0.928592,
               "delta": abs(d - 0.928592), "samples_saved": saved}
    with open(os.path.join(OUT, "audit_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()