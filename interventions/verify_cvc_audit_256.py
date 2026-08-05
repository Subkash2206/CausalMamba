"""256x256 CVC audit: identity reconstruction + exact-zero collapse check + 5 PNGs."""
import sys, os, math
import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [ROOT, os.path.join(ROOT, "SpectralMamba"), os.path.join(ROOT, "tta_boundary_study"),
          os.path.join(ROOT, "interventions", "experiments")]:
    sys.path.insert(0, p)

from src.datasets.cvc_dataset import CVCDataset
import segmentation_models_pytorch as smp
from interventions.intervention import FrequencyIntervention
from interventions.masks import lowpass_mask

HOOKS = [
    ("e1", "encoder.layer1[-1]"), ("e2", "encoder.layer2[-1]"),
    ("e3", "encoder.layer3[-1]"), ("e4", "encoder.layer4[-1]"),
    ("br", "decoder.blocks[0]"),
    ("d1", "decoder.blocks[1]"), ("d2", "decoder.blocks[2]"),
    ("d3", "decoder.blocks[3]"), ("d4", "decoder.blocks[4]"),
]


def resolve(m, path):
    if "[-1]" in path:
        base, _ = path.rsplit("[-1]", 1)
        mm = m
        for part in base.split("."):
            mm = getattr(mm, part)
        return mm[-1]
    mm = m
    for chunk in path.split("."):
        if "[" in chunk:
            name, idx = chunk.split("[")
            mm = getattr(mm, name)[int(idx.rstrip("]"))]
        else:
            mm = getattr(mm, chunk)
    return mm


def dice(p, g):
    inter = 2 * (p * g).sum()
    return (inter / (p.sum() + g.sum() + 1e-8)).item()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = os.path.join(ROOT, "tta_boundary_study", "checkpoints", "unet_cvc_best_256.pth")
    sd = torch.load(ck, map_location="cpu")
    if isinstance(sd, dict) and "encoder.conv1.weight" not in sd:
        sd = sd.get("model_state_dict") or sd.get("state_dict") or sd
    print("checkpoint:", os.path.getsize(ck), "bytes")

    ds = CVCDataset(os.path.join(ROOT, "tta_boundary_study", "cvc_clinicdb", "original"),
                    os.path.join(ROOT, "tta_boundary_study", "cvc_clinicdb", "ground_truth"),
                    split="val", img_size=256)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    print("val images:", len(ds))

    def new_model():
        m = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1).to(device)
        m.load_state_dict(sd, strict=True)
        m.eval()
        return m

    # ---- 1b: identity reconstruction over full val ----
    m = new_model()
    ident = FrequencyIntervention(lambda h, w, dev, dt: torch.ones(1, 1, h, w, device=dev, dtype=dt))
    hs = [resolve(m, p).register_forward_hook(_hook(ident)) for _, p in HOOKS]
    preds, gts = [], []
    with torch.no_grad():
        for im, mk in loader:
            im, mk = im.to(device), mk.to(device)
            preds.append(torch.sigmoid(m(im)).squeeze(1).cpu().numpy())
            gts.append(mk.squeeze(1).cpu().numpy())
    for h in hs:
        h.remove()
    del m
    d_id = dice(np.concatenate([p.ravel() for p in preds]) >= 0.5,
                np.concatenate([g.ravel() for g in gts]) >= 0.5)
    print(f"identity Dice = {d_id:.6f}  (baseline 0.931966)  |delta| = {abs(d_id - 0.931966):.6f}")

    # ---- 1c: low-pass 0.20 exact-zero + empty-mask check ----
    m = new_model()
    lp = FrequencyIntervention(lambda h, w, dev, dt: lowpass_mask(h, w, 0.20, device=dev, dtype=dt))
    hs = [resolve(m, p).register_forward_hook(_hook(lp)) for _, p in HOOKS]
    pf, gf = [], []
    with torch.no_grad():
        for im, mk in loader:
            im, mk = im.to(device), mk.to(device)
            pr = torch.sigmoid(m(im)).squeeze(1).cpu().numpy()
            pf.append(pr.ravel())
            gf.append((mk.squeeze(1).cpu().numpy() > 0).ravel())
    for h in hs:
        h.remove()
    del m
    pfa = np.concatenate(pf)
    gfa = np.concatenate(gf)
    d_lp = dice(pfa >= 0.5, gfa)
    print(f"lowpass0.20 Dice = {d_lp:.6f}; predicted-foreground fraction = {(pfa >= 0.5).mean():.8f}")

    # ---- 1c (visual): 5 PNGs ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(ROOT, "interventions", "results", "cvc_audit_256")
    os.makedirs(out, exist_ok=True)
    mb = new_model()
    ml = new_model()
    hsl = [resolve(ml, p).register_forward_hook(_hook(lp)) for _, p in HOOKS]
    saved = 0
    with torch.no_grad():
        for im, mk in loader:
            im0 = im.squeeze(0).permute(1, 2, 0).numpy()
            im = im.to(device); mk = mk.to(device)
            pb = torch.sigmoid(mb(im)).squeeze(1).cpu().numpy()[0]
            pl = torch.sigmoid(ml(im)).squeeze(1).cpu().numpy()[0]
            gt = mk.squeeze(1).cpu().numpy()[0]
            fig, ax = plt.subplots(1, 4, figsize=(16, 4))
            ax[0].imshow(im0); ax[0].set_title("input")
            ax[1].imshow(gt, cmap="gray"); ax[1].set_title("GT")
            ax[2].imshow(pb, cmap="gray"); ax[2].set_title(f"base d={dice(pb > 0.5, gt):.3f}")
            ax[3].imshow(pl, cmap="gray"); ax[3].set_title(f"LP0.20 d={dice(pl > 0.5, gt):.3f} fore={(pl > 0.5).mean():.3f}")
            for a in ax:
                a.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out, f"sample_{saved:03d}.png"), dpi=120)
            plt.close()
            saved += 1
            if saved >= 5:
                break
    for h in hsl:
        h.remove()
    print(f"saved {saved} PNGs -> {out}")


def _hook(iv):
    def hook(m, i, o):
        if isinstance(o, tuple):
            o = o[0]
        return iv(o)
    return hook


if __name__ == "__main__":
    main()