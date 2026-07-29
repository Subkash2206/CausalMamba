import os
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class BraTSDataset(Dataset):
    def __init__(self, root_dir, split='train', seed=42):
        # Ignore hidden Mac files like .DS_Store
        self.cases = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        train_cases, val_cases = train_test_split(self.cases, test_size=0.2, random_state=seed)
        self.cases = train_cases if split == 'train' else val_cases
        self.root_dir = root_dir

    def load_volume(self, path):
        vol = nib.load(path).get_fdata().astype(np.float32)
        p1, p99 = np.percentile(vol[vol>0], [1,99]) if vol.max() > 0 else (0,1)
        vol = np.clip(vol, p1, p99)
        return (vol - p1) / (p99 - p1 + 1e-8)

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        case = self.cases[idx]
        case_dir = os.path.join(self.root_dir, case)
        
        modalities = []
        for mod in ['flair', 't1', 't1ce', 't2']:
            f = [x for x in os.listdir(case_dir) if mod in x.lower() and x.endswith('.nii.gz')][0]
            modalities.append(self.load_volume(os.path.join(case_dir, f)))
        
        img = np.stack(modalities, axis=0)  # [4, H, W, D]
        
        seg_f = [x for x in os.listdir(case_dir) if 'seg' in x.lower()][0]
        seg = nib.load(os.path.join(case_dir, seg_f)).get_fdata()
        
        # Label 4 is Enhancing Tumor
        mask = (seg == 4).astype(np.float32) 
        
        # --- FIX: EXACT CENTER CROP (128x128x128) ---
        _, H, W, D = img.shape
        
        h_start = max((H - 128) // 2, 0)
        w_start = max((W - 128) // 2, 0)
        d_start = max((D - 128) // 2, 0)
        
        img = img[:, h_start:h_start+128, w_start:w_start+128, d_start:d_start+128]
        mask = mask[h_start:h_start+128, w_start:w_start+128, d_start:d_start+128]
        
        # Pad with zeros if any dimension is somehow smaller than 128
        pad_h = max(128 - img.shape[1], 0)
        pad_w = max(128 - img.shape[2], 0)
        pad_d = max(128 - img.shape[3], 0)
        
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            img = np.pad(img, ((0,0), (0,pad_h), (0,pad_w), (0,pad_d)), mode='constant')
            mask = np.pad(mask, ((0,pad_h), (0,pad_w), (0,pad_d)), mode='constant')
        
        # .float() ensures no DoubleTensor crashes
        return torch.from_numpy(img).float(), torch.from_numpy(mask).unsqueeze(0).float()
