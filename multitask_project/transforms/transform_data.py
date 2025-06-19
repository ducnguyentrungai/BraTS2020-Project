from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    CropForegroundd, SpatialPadd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandAffined, RandGaussianNoised,
    RandGaussianSmoothd, RandShiftIntensityd, RandScaleIntensityd,
    RandAdjustContrastd, Rand3DElasticd, Zoomd, CenterSpatialCropd,
    ToTensord, Resized, Lambdad, ThresholdIntensityd, CastToTyped, ResizeWithPadOrCrop, MapTransform
)
from typing import Union, Sequence, Dict, Optional
import numpy as np
import torch
import pandas as pd

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


# Custom transform to convert tabular dict → vector tensor
def compute_minmax_stats(csv_path: str) -> Dict[str, Dict[str, float]]:
    df = pd.read_csv(csv_path)
    numeric_cols = [
        "Age", "tumor_volume", "ncr_net_volume", "ed_volume", "et_volume",
        "tumor_pct", "ncr_net_pct", "ed_pct", "et_pct"
    ]
    stats = {}
    for col in numeric_cols:
        stats[col] = {
            "min": df[col].min(),
            "max": df[col].max()
        }
    return stats


def minmax_scale_tabular(tab: dict, stats: Dict[str, Dict[str, float]]) -> torch.Tensor:
    def scale(val, vmin, vmax):
        val = float(val)
        x = (val - vmin) / (vmax - vmin + 1e-8)  # scale về [0, 1]
        return x * 10.0 - 5.0  # scale về [-5, 5]

    resection_str = tab.get("Extent_of_Resection", "NA")
    resection_code = (
        5.0 if resection_str == "GTR" else
        -5.0 if resection_str == "STR" else
        0.0
    )

    features = [
        scale(tab["Age"], *stats["Age"].values()),
        resection_code,
        scale(tab["tumor_volume"], *stats["tumor_volume"].values()),
        scale(tab["ncr_net_volume"], *stats["ncr_net_volume"].values()),
        scale(tab["ed_volume"], *stats["ed_volume"].values()),
        scale(tab["et_volume"], *stats["et_volume"].values()),
        scale(tab["tumor_pct"], *stats["tumor_pct"].values()),
        scale(tab["ncr_net_pct"], *stats["ncr_net_pct"].values()),
        scale(tab["ed_pct"], *stats["ed_pct"].values()),
        scale(tab["et_pct"], *stats["et_pct"].values()),
    ]

    return torch.tensor(features, dtype=torch.float)

class TabularToTensor(MapTransform):
    def __init__(self, keys, stats: Dict[str, Dict[str, float]]):
        super().__init__(keys)
        self.stats = stats

    def __call__(self, data):
        d = dict(data)
        d["tabular"] = minmax_scale_tabular(d["tabular"], self.stats)
        d["label_class"] = torch.tensor(d["label_class"], dtype=torch.long)
        return d


def get_multitask_transforms(
    spatial_size: Union[Sequence[int], int] = (128, 128, 128),
    is_train: bool = True,
    tabular_stats: Optional[Dict[str, Dict[str, float]]] = None
):
    transforms = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        ThresholdIntensityd(keys=["image"], threshold=0.0, above=True, cval=0.0),
        Lambdad(keys=["image"], func=zscore_clip),
        Lambdad(keys="label", func=remap_label),
        CropForegroundd(keys=["image", "label"], source_key="image", return_coords=False),
    ]

    if is_train:
        transforms += [
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1, neg=1, num_samples=1,
                image_key="image", image_threshold=0,
            ),
            # RandFlipd(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
            # RandRotate90d(keys=["image", "label"], prob=0.3, max_k=1),
            # RandAffined(  # chỉ giữ nhẹ
            #     keys=["image", "label"],
            #     mode=("bilinear", "nearest"),
            #     prob=0.1,
            #     spatial_size=spatial_size,
            #     rotate_range=(0.05, 0.05, 0.05),
            #     scale_range=(0.05, 0.05, 0.05),
            #     translate_range=(2, 2, 2),
            #     padding_mode="border"
            # ),
            # RandScaleIntensityd(keys=["image"], factors=0.05, prob=0.3),
            # RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.03),
        ]

    else:
        transforms += [
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size),
            CenterSpatialCropd(keys=["image", "label"], roi_size=spatial_size),
        ]

    transforms += [
        CastToTyped(keys=["label"], dtype=np.uint8),
        ToTensord(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=spatial_size),
    ]

    if tabular_stats is not None:
        transforms += [
            TabularToTensor(keys=["tabular"], stats=tabular_stats),
        ]

    return Compose(transforms)


