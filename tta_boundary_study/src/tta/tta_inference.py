import torch
import numpy as np

def apply_tta_augment(img_tensor, aug_idx):
    # img_tensor: [C, H, W]
    augmentations = [
        lambda x: x,  # 0: original
        lambda x: torch.flip(x, dims=[2]),  # 1: horizontal flip
        lambda x: torch.flip(x, dims=[1]),  # 2: vertical flip
        lambda x: torch.rot90(x, 1, dims=[1,2]),  # 3: 90 degrees
        lambda x: torch.rot90(x, 2, dims=[1,2]),  # 4: 180 degrees
        lambda x: torch.rot90(x, 3, dims=[1,2]),  # 5: 270 degrees
        lambda x: torch.flip(torch.rot90(x, 1, [1,2]), [2]),  # 6: rot90+hflip
        lambda x: torch.flip(torch.rot90(x, 1, [1,2]), [1])   # 7: rot90+vflip
    ]
    return augmentations[aug_idx](img_tensor)

def reverse_tta_augment(pred_tensor, aug_idx):
    # pred_tensor: [1, H, W]
    reverses = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[1]),
        lambda x: torch.rot90(x, -1, dims=[1,2]),
        lambda x: torch.rot90(x, -2, dims=[1,2]),
        lambda x: torch.rot90(x, -3, dims=[1,2]),
        # FIXED: Must un-flip FIRST, then un-rotate
        lambda x: torch.rot90(torch.flip(x, dims=[2]), -1, dims=[1,2]),  
        lambda x: torch.rot90(torch.flip(x, dims=[1]), -1, dims=[1,2])   
    ]
    return reverses[aug_idx](pred_tensor)

def tta_predict(model, img_tensor, device, n_views=8):
    model.eval()
    prob_maps = []
    with torch.no_grad():
        for i in range(n_views):
            aug_img = apply_tta_augment(img_tensor[0], i).unsqueeze(0).to(device)
            logits = model(aug_img)
            prob = torch.sigmoid(logits)
            prob_rev = reverse_tta_augment(prob[0], i)
            prob_maps.append(prob_rev.cpu().numpy())
    return np.mean(prob_maps, axis=0)[0]  # [H, W]

def baseline_predict(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        return torch.sigmoid(logits)[0, 0].cpu().numpy()

def apply_tta_3d(vol_tensor, aug_idx):
    # vol_tensor: [C, H, W, D]
    augmentations = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[1]),  # flip H
        lambda x: torch.flip(x, dims=[2]),  # flip W
        lambda x: torch.flip(x, dims=[3]),  # flip D
        lambda x: torch.rot90(x, 1, dims=[1,2]),  # rot90 HW
        lambda x: torch.rot90(x, 2, dims=[1,2]),  # rot180 HW
        lambda x: torch.flip(x, dims=[1,2]),      # flip HW
        lambda x: torch.rot90(x, 1, dims=[1,3])   # rot90 HD
    ]
    return augmentations[aug_idx](vol_tensor)

def reverse_tta_3d(pred_tensor, aug_idx):
    # pred_tensor: [1, H, W, D]
    reverses = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[1]),
        lambda x: torch.flip(x, dims=[2]),
        lambda x: torch.flip(x, dims=[3]),
        lambda x: torch.rot90(x, -1, dims=[1,2]),
        lambda x: torch.rot90(x, -2, dims=[1,2]),
        lambda x: torch.flip(x, dims=[1,2]),
        lambda x: torch.rot90(x, -1, dims=[1,3])
    ]
    return reverses[aug_idx](pred_tensor)

def tta_predict_3d(model, vol_tensor, device, n_views=8):
    model.eval()
    prob_maps = []
    with torch.no_grad():
        for i in range(n_views):
            aug = apply_tta_3d(vol_tensor[0], i).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(aug))
            prob_maps.append(reverse_tta_3d(prob[0], i).cpu().numpy())
    return np.mean(prob_maps, axis=0)[0]  # [H, W, D]

def baseline_predict_3d(model, vol_tensor, device):
    model.eval()
    with torch.no_grad():
        logits = model(vol_tensor.to(device))
        return torch.sigmoid(logits)[0, 0].cpu().numpy()
