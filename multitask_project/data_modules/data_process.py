import glob
import os

def extract_data_dicts(data_dir:str, train_percent:float=0.8):
    images = sorted(glob.glob(os.path.join(data_dir, "imageTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    from sklearn.model_selection import train_test_split
    train_dicts, test_dicts = train_test_split(data_dicts, train_size=train_percent, random_state=42, shuffle=True)
    return train_dicts, test_dicts


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
        ThresholdIntensityd(keys=["image"], threshold=0.0, above=True, cval=0.0),   # Remove negative values
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),       # Z-score normalization
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


if __name__ == "__main__":
    import pandas as pd
    tab_path = "/work/cuc.buithi/brats_challenge/code/multitask_project/data/suvivaldays_info.csv"
    df = pd.read_csv(tab_path)
    print(df.head(5))
    