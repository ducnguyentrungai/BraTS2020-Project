import os
import glob
from tqdm import tqdm
from typing import Sequence, Union

import torch.distributed as dist
from torch.utils.data._utils.collate import default_collate

from pytorch_lightning import LightningDataModule
from monai.data.utils import partition_dataset
from monai.data import list_data_collate
from monai.data import CacheDataset, SmartCacheDataset, Dataset, DataLoader


def dict_collate(batch):
    return default_collate(batch)

class BratsDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        spatial_size: Union[Sequence[int], int] = (128, 128, 128),
        batch_size: int = 4,
        num_workers: int = 2,
        cache_num: int = 100,
        cache_num_val_all: bool = True,
        train_percent: float = 0.9,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.spatial_size = spatial_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_num = cache_num
        self.cache_num_val_all = cache_num_val_all
        self.train_percent = train_percent

    def setup(self, stage=None):
        self.train_dicts, self.val_dicts = extract_data_dicts(self.data_dir, self.train_percent)

        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        train_partitioned = partition_dataset(
            data=self.train_dicts,
            num_partitions=world_size,
            shuffle=True,
            even_divisible=True
        )
        my_train_data = train_partitioned[rank]

        self.train_dataset = SmartCacheDataset(
            data=my_train_data,
            transform=get_transforms(self.spatial_size, is_train=True),
            cache_num=min(self.cache_num, len(my_train_data)),
            replace_rate=0.1
        )

        val_cache_num = len(self.val_dicts) if self.cache_num_val_all else min(self.cache_num, len(self.val_dicts))

        self.val_dataset = SmartCacheDataset(
            data=self.val_dicts,
            transform=get_transforms(self.spatial_size, is_train=False),
            cache_num=val_cache_num,
            replace_rate=1.0
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=list_data_collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=list_data_collate 
        )


def extract_data_dicts(data_dir:str, train_percent:float=0.8):
    images = sorted(glob.glob(os.path.join(data_dir, "imageTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    from sklearn.model_selection import train_test_split
    train_dicts, val_dicts = train_test_split(data_dicts, train_size=train_percent, random_state=42, shuffle=True)
    return train_dicts, val_dicts


def get_transforms(spatial_size=(128, 128, 128), is_train=True):
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
        ScaleIntensityRanged, CropForegroundd, Resized,
        RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
        ToTensord, Compose, RandGaussianNoised, RandGaussianSmoothd,
        Zoomd, Rand3DElasticd, CenterSpatialCropd, RandShiftIntensityd,
        RandAdjustContrastd, RandScaleIntensityd, SpatialPadd, 
        RandSpatialCropd, RandAffined, NormalizeIntensityd, 
        ThresholdIntensityd
    )

    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        # Intensity preprocessing
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        ThresholdIntensityd(keys=["image"], threshold=0.0, above=True, cval=0.0),  # Remove negative values
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),  # Z-score normalization
    ]

    if is_train:
        base += [
            # Spatial preprocessing
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            SpatialPadd(keys=["image", "label"], spatial_size=spatial_size, mode='constant'),
            
            # Cropping augmentations
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandSpatialCropd(keys=["image", "label"], roi_size=(int(spatial_size[0]*0.9), 
                                                               int(spatial_size[1]*0.9), 
                                                               int(spatial_size[2]*0.9)), 
                           random_center=True, random_size=True),
            
            # Geometric augmentations
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.3),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.3),
            RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.2),
            RandRotate90d(keys=["image", "label"], prob=0.3, max_k=3),
            RandAffined(
                keys=["image", "label"],
                mode=("bilinear", "nearest"),
                prob=0.4,
                spatial_size=spatial_size,
                rotate_range=(0.2, 0.2, 0.2),
                scale_range=(0.1, 0.1, 0.1),
                translate_range=(10, 10, 10),
                padding_mode="border"
            ),
            Zoomd(keys=["image", "label"], zoom=1.4, mode=['area', 'nearest']),
            CenterSpatialCropd(keys=["image", "label"], roi_size=(96, 96, 96)),
            Rand3DElasticd(keys=["image", "label"], sigma_range=(5, 7), magnitude_range=(100, 200), prob=0.3),
            
            # Intensity augmentations
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.4),
            RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.4),
            RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7, 1.5)),
            
            # Noise augmentations
            RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.05),
            RandGaussianSmoothd(keys=["image"], prob=0.3, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)),
        ]

    # Add final transforms
    base += [
        ToTensord(keys=["image", "label"]),  
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest"))
    ]
    
    return Compose(base)


# if __name__ == "__main__":
#     data_dir = '/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2'
#     train_list, val_list = extract_data_dicts(data_dir=data_dir, train_percent=0.8)
#     print(len(train_list))
#     print(train_list[0])
    