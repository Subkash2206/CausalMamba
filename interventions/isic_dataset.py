"""
isic_dataset.py — ISIC2018 dataset with a persisted, thread-built cache.

Native ISIC images are ~29 MP; decoding+resizing them every epoch dominated wall-clock
time. This dataset decodes each image ONCE (thread-parallel), caches the 256x256
[0,1]-normalized RGB tensor + binarized mask in RAM, and persists to a single .pt file
per split so subsequent runs load in seconds.

Windows DataLoader note: spawn workers each get a pickled copy of the dataset, so the
RAM cache would multiply per worker. Use num_workers=0 with this dataset — cache hits
are fast enough that workers are unnecessary.

Usage:
    from interventions.isic_dataset import ISICCacheDataset
    ds = ISICCacheDataset(split['train'], is_train=True)
"""

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

CACHE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', 'isic256')


class ISICCacheDataset(Dataset):
    """Train/val/test ISIC18 at 256px with an on-disk + in-RAM cache.

    Cached tensors: img (3,256,256) float32 in [0,1], mask (1,256,256) float32 in {0,1}.
    Normalization to ImageNet stats and augmentation are applied per-access in
    __getitem__ (so augmentations stay fresh every epoch).
    """

    def __init__(self, names, img_dir, mask_dir, img_size=256, is_train=True,
                 cache_name='split'):
        self.names = names
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.is_train = is_train
        os.makedirs(CACHE_ROOT, exist_ok=True)
        cache_path = os.path.join(CACHE_ROOT, f'{cache_name}.pt')
        self._imgs, self._masks = self._load_cache(cache_path, names)

    # -- cache build ---------------------------------------------------------
    def _load_cache(self, cache_path, names):
        if os.path.exists(cache_path):
            blob = torch.load(cache_path, map_location='cpu')
            if blob['names'] == names:
                return blob['imgs'], blob['masks']
        return self._build_and_save(cache_path, names)

    def _build_and_save(self, cache_path, names):
        print(f'[cache] building {len(names)} ISIC images @ {self.img_size}px '
              f'(one-time, thread-parallel) ...')
        imgs = [None] * len(names)
        masks = [None] * len(names)

        def work(i, name):
            base = os.path.splitext(name)[0]
            img = Image.open(os.path.join(self.img_dir, name)).convert('RGB').resize(
                (self.img_size, self.img_size), Image.BILINEAR)
            msk = Image.open(os.path.join(self.mask_dir, base + '_segmentation.png')).convert(
                'L').resize((self.img_size, self.img_size), Image.NEAREST)
            return i, TF.to_tensor(img), torch.from_numpy(np.array(msk)).float().unsqueeze(0) / 255.0

        with ThreadPoolExecutor(max_workers=6) as ex:
            for i, img_t, msk_t in ex.map(lambda p: work(*p), enumerate(names)):
                imgs[i] = img_t
                masks[i] = (msk_t > 0.5).float()
        imgs = torch.stack(imgs)    # (N,3,H,W)
        masks = torch.stack(masks)  # (N,1,H,W)
        torch.save({'names': names, 'imgs': imgs, 'masks': masks}, cache_path)
        print(f'[cache] saved -> {cache_path}')
        return imgs, masks

    # -- dataset API ---------------------------------------------------------
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = self._imgs[idx].clone()
        mask = self._masks[idx].clone()
        if self.is_train:
            if np.random.rand() > 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if np.random.rand() > 0.5:
                img, mask = TF.vflip(img), TF.vflip(mask)
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-30, 30)
                img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img, mask
