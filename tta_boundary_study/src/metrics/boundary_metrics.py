import numpy as np 
from scipy.ndimage import binary_erosion, binary_dilation 
from scipy.spatial.distance import directed_hausdorff 

def dice_score(pred, target, threshold=0.5): 
    pred_bin = (pred > threshold).astype(np.float32) 
    target = target.astype(np.float32) 
    intersection = (pred_bin * target).sum() 
    return (2.0 * intersection) / (pred_bin.sum() + target.sum() + 1e-8) 

def get_boundary(mask, thickness=1):
    eroded = binary_erosion(mask, iterations=thickness) 
    return mask.astype(bool) ^ eroded 

def boundary_f1(pred, target, threshold=0.5, thickness=2): 
    pred_bin = (pred > threshold).astype(bool) 
    target_bin = target.astype(bool) 
    if pred_bin.sum() == 0 and target_bin.sum() == 0: return 1.0 
    if pred_bin.sum() == 0 or target_bin.sum() == 0: return 0.0 
    pred_bound = get_boundary(pred_bin, thickness) 
    target_bound = get_boundary(target_bin, thickness) 
    pred_dilated = binary_dilation(pred_bound, iterations=thickness) 
    target_dilated = binary_dilation(target_bound, iterations=thickness) 
    precision = (pred_bound & target_dilated).sum() / (pred_bound.sum() + 1e-8) 
    recall = (target_bound & pred_dilated).sum() / (target_bound.sum() + 1e-8) 
    if precision + recall == 0: return 0.0 
    return 2 * precision * recall / (precision + recall) 

def hausdorff_95(pred, target, threshold=0.5): 
    pred_bin = (pred > threshold).astype(bool) 
    target_bin = target.astype(bool)
    if pred_bin.sum() == 0 or target_bin.sum() == 0: return float('nan') 
    pred_pts = np.argwhere(pred_bin) 
    target_pts = np.argwhere(target_bin) 
    d1 = directed_hausdorff(pred_pts, target_pts)[0] 
    d2 = directed_hausdorff(target_pts, pred_pts)[0] 
    return max(d1, d2)
