from glob import glob
import os
import torch
from typing import List, Optional, Callable, Sequence, Union
import numpy as np
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandGaussianSmoothd, RandShiftIntensityd, RandScaleIntensityd,
    RandAdjustContrastd, Rand3DElasticd, Zoomd, CenterSpatialCropd,
    ToTensord, Resized, Lambdad, ThresholdIntensityd
)
import matplotlib.pyplot as plt
from monai.transforms import Compose
import torch
import numpy as np
from monai.data import Dataset
from my_dataset import BratsDataModule
from my_transform import get_transforms



if __name__ == "__main__":
    data_dir = "/work/cuc.buithi/brats_challenge/BraTS2021"
    spatial_size = (128, 128, 128)

    dm = BratsDataModule(
        data_dir=data_dir,
        spatial_size=spatial_size,
        batch_size=2,
        num_workers=2,
        train_percent=0.8,
        transform_fn=get_transforms
    )
    dm.setup()

    # Lấy 1 batch từ train loader
    train_loader = dm.train_dataloader()
    for batch in train_loader:
        images = batch["image"]   # (B, C, H, W, D)
        labels = batch["label"]   # (B, 1, H, W, D) hoặc (B, H, W, D)
        print("✅ Image shape:", images.shape)
        print("✅ Label shape:", labels.shape)
        print("✅ Label dtype:", labels.dtype)
        print("✅ Label unique values:", torch.unique(labels))

        # Lấy 1 sample trong batch
        img = images[0].detach().cpu().numpy()   # shape: (4, 128, 128, 128)
        lbl = labels[0].detach().cpu().numpy()   # shape: (128, 128, 128) hoặc (1, 128, 128, 128)

        # Nếu label có shape (1, H, W, D) thì squeeze đi
        if lbl.ndim == 4:
            lbl = lbl[0]

        # Chọn lát cắt giữa (trục D)
        z = img.shape[-1] // 2

        # Plot 4 ảnh và 1 mask
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        titles = ['T1', 'T1ce', 'T2', 'FLAIR', 'Label']
        for i in range(4):
            axes[i].imshow(img[i, :, :, z], cmap='gray')
            axes[i].set_title(titles[i])
            axes[i].axis('off')

        axes[4].imshow(lbl[:, :, z], cmap='nipy_spectral', vmin=0, vmax=3)  # hiển thị label nhiều class
        axes[4].set_title("Label")
        axes[4].axis('off')

        plt.tight_layout()
        plt.savefig('check_img.png')
        plt.show()
        break
