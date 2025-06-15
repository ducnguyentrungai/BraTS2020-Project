from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandGaussianSmoothd, RandShiftIntensityd, RandScaleIntensityd,
    RandAdjustContrastd, Rand3DElasticd, Zoomd, CenterSpatialCropd,
    ToTensord, Resized, Lambdad, ThresholdIntensityd, CastToTyped
)
from typing import Union, Sequence
import numpy as np
import torch

def zscore_clip(img):
    if isinstance(img, torch.Tensor):
        img = img.clone()
        mask = img > 0
        if mask.any():
            mean = img[mask].mean()
            std = img[mask].std()
            img[mask] = (img[mask] - mean) / (std + 1e-8)
            img[mask] = torch.clamp(img[mask], -5, 5)
        return img
    elif isinstance(img, np.ndarray):
        img = img.copy()
        mask = img > 0
        if np.any(mask):
            mean = img[mask].mean()
            std = img[mask].std()
            img[mask] = (img[mask] - mean) / (std + 1e-8)
            img[mask] = np.clip(img[mask], -5, 5)
        return img
    else:
        raise TypeError(f"Unsupported input type: {type(img)}")


def remap_label(label):
    label = label.clone()
    label[label == 4] = 3
    return label

def get_transforms(spatial_size: Union[Sequence[int], int] = (128, 128, 128), is_train: bool = True):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ThresholdIntensityd(keys=["image"], threshold=0.0, above=True, cval=0.0),
        Lambdad(keys=["image"], func=zscore_clip),
        Lambdad(keys="label", func=remap_label),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]

    if is_train:
        transforms += [
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.5),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.3,
                spatial_size=spatial_size,
                rotate_range=(0.1, 0.1, 0.1),
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(10, 10, 10),
                padding_mode="border"
            ),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.5),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.2),
            Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 8), magnitude_range=(50, 100), prob=0.2),
        ]
    else:
        transforms += [
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
        ]

    transforms += [
        CastToTyped(keys=["label"], dtype=np.uint8),
        ToTensord(keys=["image", "label"]),
        # Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
    ]

    return Compose(transforms)




# if __name__ == "__main__":
#     data = torch.randn(1, 1, 96, 96, 96)
#     dicts = {'image': data, 'label': data}
#     lambd = Lambdad(keys=['image'], func=zscore_clip)
#     out = lambd(dicts)
#     print(out['image'].shape)
#     print(f"min: {out['image'].min().item():.4f}, max: {out['image'].max().item():.4f}")
