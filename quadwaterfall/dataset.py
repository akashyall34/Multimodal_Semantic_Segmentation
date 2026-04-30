"""
MCubeS Dataset — PyTorch DataLoader
=====================================
Folders
-------
  polL_color/       RGB images          .png  uint8   (1024, 1224, 3)
  polL_aolp_cos/    AoLP cos component  .npy  float32 (1024, 1224)
  polL_aolp_sin/    AoLP sin component  .npy  float32 (1024, 1224)
  polL_dolp/        Degree of LP        .npy  float32 (1024, 1224)
  NIR_warped/       Near-infrared       .png  uint16  (1024, 1224)
  NIR_warped_mask/  Valid NIR pixels    .png  uint8   (1024, 1224)
  SSGT4MS/          Segmentation labels .png  uint8   (1024, 1224)
  list_folder/      train/val/test.txt  stems only (no extension)

Preprocessing
-------------
  RGB  : uint8 / 255  → ImageNet mean/std normalisation
  AoLP : arctan2(sin, cos) → [−π, π] → [0, 1]
  DoLP : clip [0, 1]  (raw measurements contain small negatives)
  NIR  : uint16 / 65535  then NIR-specific mean/std normalisation
  Label: uint8 long tensor, class indices 0–19

Splits: 301 train / 95 val / 102 test
"""

from __future__ import annotations
import os
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ── Normalisation constants ────────────────────────────────────────────────────
_RGB_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_RGB_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# NIR statistics estimated from the MCubeS training set (single channel)
_NIR_MEAN = torch.tensor([0.299]).view(1, 1, 1)
_NIR_STD  = torch.tensor([0.211]).view(1, 1, 1)


class MCubeSDataset(Dataset):
    """
    PyTorch Dataset for the MCubeS multimodal material segmentation dataset.

    Args
    ----
      root      : path to the dataset root (contains polL_color/, GT/, …)
      split     : 'train', 'val', or 'test'
      crop_size : random-crop size during training (ignored for val/test)
    """

    def __init__(self, root: str, split: str = 'train', crop_size: int = 512):
        assert split in ('train', 'val', 'test'), f"Unknown split: {split}"
        self.root      = root
        self.split     = split
        self.crop_size = crop_size

        list_file = os.path.join(root, 'list_folder', f'{split}.txt')
        with open(list_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

    # ── path helpers ──────────────────────────────────────────────────────────

    def _p(self, folder: str, name: str, ext: str) -> str:
        return os.path.join(self.root, folder, name + ext)

    # ── per-modality loaders ──────────────────────────────────────────────────

    def _load_rgb(self, name: str) -> torch.Tensor:
        """Returns (3, H, W) float32 in ImageNet-normalised space."""
        arr = np.array(Image.open(self._p('polL_color', name, '.png')).convert('RGB'))
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        return (t - _RGB_MEAN) / _RGB_STD

    def _load_aolp(self, name: str) -> torch.Tensor:
        """
        AoLP reconstructed from cos/sin Stokes components via arctan2.
        Returns (1, H, W) float32 in [0, 1].
        """
        cos = np.load(self._p('polL_aolp_cos', name, '.npy'))
        sin = np.load(self._p('polL_aolp_sin', name, '.npy'))
        angle = np.arctan2(sin, cos).astype(np.float32)   # [-π, π]
        angle = (angle + np.pi) / (2 * np.pi)             # [0, 1]
        return torch.from_numpy(angle).unsqueeze(0)

    def _load_dolp(self, name: str) -> torch.Tensor:
        """Returns (1, H, W) float32 in [0, 1] (noise-clipped)."""
        arr = np.load(self._p('polL_dolp', name, '.npy')).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)
        return torch.from_numpy(arr).unsqueeze(0)

    def _load_nir(self, name: str) -> torch.Tensor:
        """
        Loads 16-bit NIR image. Invalid pixels (outside NIR_warped_mask) are
        set to zero before normalisation so they contribute a neutral value.
        Returns (1, H, W) float32 normalised.
        """
        arr  = np.array(Image.open(self._p('NIR_warped',      name, '.png'))).astype(np.float32)
        mask = np.array(Image.open(self._p('NIR_warped_mask', name, '.png')))
        arr[mask == 0] = 0.0
        t = torch.from_numpy(arr / 65535.0).unsqueeze(0)
        return (t - _NIR_MEAN) / _NIR_STD

    def _load_label(self, name: str) -> torch.Tensor:
        """Returns (H, W) int64 class-index map."""
        arr = np.array(Image.open(self._p('SSGT4MS', name, '.png')))
        return torch.from_numpy(arr.astype(np.int64))

    # ── joint spatial transforms ───────────────────────────────────────────────

    def _random_crop(
        self,
        modalities: list[torch.Tensor],
        label: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        _, H, W = modalities[0].shape
        cs  = self.crop_size
        top  = random.randint(0, H - cs)
        left = random.randint(0, W - cs)
        modalities = [m[:, top:top + cs, left:left + cs] for m in modalities]
        label      = label[top:top + cs, left:left + cs]
        return modalities, label

    def _random_hflip(
        self,
        modalities: list[torch.Tensor],
        label: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        if random.random() < 0.5:
            modalities = [TF.hflip(m) for m in modalities]
            label      = TF.hflip(label.unsqueeze(0)).squeeze(0)
        return modalities, label

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx: int) -> dict:
        name  = self.ids[idx]
        rgb   = self._load_rgb(name)
        aolp  = self._load_aolp(name)
        dolp  = self._load_dolp(name)
        nir   = self._load_nir(name)
        label = self._load_label(name)

        modalities = [rgb, aolp, dolp, nir]

        if self.split == 'train':
            modalities, label = self._random_crop(modalities, label)
            modalities, label = self._random_hflip(modalities, label)

        rgb, aolp, dolp, nir = modalities
        return {
            'rgb':   rgb,    # (3, H, W)
            'aolp':  aolp,   # (1, H, W)
            'dolp':  dolp,   # (1, H, W)
            'nir':   nir,    # (1, H, W)
            'label': label,  # (H, W)  int64
            'name':  name,
        }

    def __len__(self) -> int:
        return len(self.ids)


# ── Convenience factory ────────────────────────────────────────────────────────

def build_loaders(
    root:       str,
    crop_size:  int = 512,
    batch_size: int = 2,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds = MCubeSDataset(root, split='train', crop_size=crop_size)
    val_ds   = MCubeSDataset(root, split='val')
    test_ds  = MCubeSDataset(root, split='test')

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,          # full 1024×1224 image — fits at bs=1
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ── Smoke test ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'multimodal_dataset'

    for split in ('train', 'val', 'test'):
        ds = MCubeSDataset(root, split=split)
        sample = ds[0]
        print(f'\n{split} ({len(ds)} samples):')
        for k, v in sample.items():
            if isinstance(v, torch.Tensor):
                print(f'  {k:6s} {tuple(v.shape)}  dtype={v.dtype}  '
                      f'min={v.float().min():.3f}  max={v.float().max():.3f}')
            else:
                print(f'  {k:6s} {v}')
